#!/usr/bin/env python3
"""Solve one primary MoE placement against multiple routing traces.

Every input keeps its own exact Top-K hypergraph objective. Dataset counts are
normalized before they are combined, so a large trace cannot dominate merely
because it contains more tokens. After each short refinement, multiplicative
reweighting focuses the next round on datasets with the worst normalized
communication. The best placement is selected using::

    compute-cap violation, mean_normalized_objective + rho * worst_normalized_objective

Example::

    PYTHONPATH=python python benchmark/solve_robust_co_routing_graph.py \
        --input traces/sharegpt.pt \
        --input traces/humaneval.pt \
        --input traces/xsum.pt \
        --num-ranks 32 --device cuda \
        --robust-rounds 8 --swaps-per-round 2 \
        --no-compute-limit \
        --save-placement traces/robust.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from solve_co_routing_graph import (
    _TensorTrace,
    _balanced_initial_placement,
    _baseline_homes,
    _compute_loads,
    _evaluate_hypergraph_objective_tensor,
    _evaluate_primary_remote_tensor,
    _format_table,
    _graph_placement_from_rank_map,
    _layer_number,
    _load_input,
    _materialize_layer_tokens,
    _max_over_average,
    _save_placements,
    _summarize_tensor_trace,
    _summarize_tokens,
    _tensorize_compact_layer,
    _tensorize_entries,
    _torch,
    _wait_tensor_trace,
)
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    GraphPlacement,
    RoutedToken,
    build_co_routing_graph,
    evaluate_hypergraph_objective,
    evaluate_primary_remote,
    refine_hypergraph_placement,
)


@dataclass(frozen=True)
class _InputTrace:
    path: str
    label: str
    layers: tuple[tuple[str, Any], ...]
    compact: bool
    source_ep: int


@dataclass(frozen=True)
class _LayerTrace:
    label: str
    path: str
    value: Any
    tensor: bool
    experts: tuple[int, ...]
    demand: Mapping[int, int]
    tokens: int
    bundles: int


def _unique_labels(paths: Sequence[str]) -> list[str]:
    stems = [Path(path).stem for path in paths]
    totals = {stem: stems.count(stem) for stem in stems}
    return [
        stem if totals[stem] == 1 else f"{index}:{stem}"
        for index, stem in enumerate(stems)
    ]


def _load_traces(args: argparse.Namespace) -> list[_InputTrace]:
    labels = _unique_labels(args.input)
    result = []
    expected_gates = None
    for path, label in zip(args.input, labels):
        raw, layers, compact = _load_input(path)
        gates = tuple(name for name, _ in layers)
        if expected_gates is None:
            expected_gates = gates
        elif gates != expected_gates:
            raise ValueError(
                f"trace {path} has different gates; expected {expected_gates}, "
                f"got {gates}"
            )
        source_ep = args.source_ep
        if source_ep is None and isinstance(raw, dict):
            source_ep = raw.get("num_ranks")
        if source_ep is None:
            source_ep = args.num_ranks
        source_ep = int(source_ep)
        if source_ep < 1 or args.num_ranks < source_ep or args.num_ranks % source_ep:
            raise ValueError(
                f"target EP {args.num_ranks} must be a multiple of source EP "
                f"{source_ep} for {path}"
            )
        result.append(_InputTrace(path, label, tuple(layers), compact, source_ep))
    return result


def _prepare_layer(
    trace: _InputTrace,
    layer_index: int,
    args: argparse.Namespace,
) -> _LayerTrace:
    name, entries = trace.layers[layer_index]
    use_tensor = args.device != "cpu"
    if use_tensor:
        if trace.compact:
            value = _tensorize_compact_layer(
                name,
                entries,
                device=args.device,
                source_ep=trace.source_ep,
                target_ep=args.num_ranks,
            )
            _wait_tensor_trace(value)
        else:
            value = _tensorize_entries(
                name,
                entries,
                device=args.device,
                source_ep=trace.source_ep,
                target_ep=args.num_ranks,
            )
        experts, demand = _summarize_tensor_trace(
            value, chunk_size=args.tensor_chunk_size
        )
        token_count = int(value.count.sum().item())
        bundle_count = value.count.numel()
    else:
        value = _materialize_layer_tokens(
            name,
            entries,
            compact_input=trace.compact,
            source_ep=trace.source_ep,
            target_ep=args.num_ranks,
        )
        experts, demand = _summarize_tokens(value)
        token_count = sum(token.count for token in value)
        bundle_count = len(value)
    return _LayerTrace(
        trace.label,
        trace.path,
        value,
        use_tensor,
        experts,
        demand,
        token_count,
        bundle_count,
    )


def _remote(
    trace: _LayerTrace,
    placement: Mapping[int, int],
    chunk_size: int,
) -> int:
    if trace.tensor:
        return _evaluate_primary_remote_tensor(
            trace.value, [placement], chunk_size=chunk_size
        )[0]
    return evaluate_primary_remote(trace.value, placement)


def _objective(
    trace: _LayerTrace,
    placement: Mapping[int, int],
    objective: str,
    chunk_size: int,
) -> int:
    if objective == "source-aware":
        return _remote(trace, placement, chunk_size)
    if trace.tensor:
        return _evaluate_hypergraph_objective_tensor(
            trace.value,
            [placement],
            objective=objective,
            chunk_size=chunk_size,
        )[0]
    return evaluate_hypergraph_objective(trace.value, placement, objective=objective)


def _normalizer(
    trace: _LayerTrace,
    baseline_objective: int,
    mode: str,
) -> int:
    if mode == "baseline" and baseline_objective > 0:
        return baseline_objective
    return max(trace.tokens, 1)


def _weighted_scale_factors(
    weights: Sequence[float], normalizers: Sequence[int], precision: int
) -> list[float]:
    coefficients = [
        weight / normalizer for weight, normalizer in zip(weights, normalizers)
    ]
    maximum = max(coefficients)
    return [coefficient / maximum * precision for coefficient in coefficients]


def _weighted_cpu_trace(
    traces: Sequence[_LayerTrace],
    weights: Sequence[float],
    normalizers: Sequence[int],
    precision: int,
) -> list[RoutedToken]:
    factors = _weighted_scale_factors(weights, normalizers, precision)
    merged = []
    for trace, factor in zip(traces, factors):
        for token in trace.value:
            count = max(1, int(round(token.count * factor)))
            merged.append(RoutedToken(token.source_rank, token.topk_experts, count))
    return merged


def _weighted_tensor_trace(
    traces: Sequence[_LayerTrace],
    weights: Sequence[float],
    normalizers: Sequence[int],
    precision: int,
) -> _TensorTrace:
    torch = _torch()
    factors = _weighted_scale_factors(weights, normalizers, precision)
    sources = []
    topks = []
    counts = []
    top_k = None
    for trace, factor in zip(traces, factors):
        value = trace.value
        if top_k is None:
            top_k = value.topk_experts.shape[1]
        elif value.topk_experts.shape[1] != top_k:
            raise ValueError("all robust inputs must use the same Top-K")
        scaled = (
            torch.round(value.count.to(torch.float64) * factor)
            .to(torch.int64)
            .clamp_min_(1)
        )
        sources.append(value.source_rank)
        topks.append(value.topk_experts)
        counts.append(scaled)
    return _TensorTrace(
        torch.cat(sources),
        torch.cat(topks),
        torch.cat(counts),
    )


def _weighted_trace(
    traces: Sequence[_LayerTrace],
    weights: Sequence[float],
    normalizers: Sequence[int],
    precision: int,
) -> Any:
    if traces[0].tensor:
        return _weighted_tensor_trace(traces, weights, normalizers, precision)
    return _weighted_cpu_trace(traces, weights, normalizers, precision)


def _aggregate_demand(
    traces: Sequence[_LayerTrace], weights: Sequence[float]
) -> tuple[tuple[int, ...], dict[int, float]]:
    experts = tuple(sorted({expert for trace in traces for expert in trace.experts}))
    result = {expert: 0.0 for expert in experts}
    for trace, weight in zip(traces, weights):
        total = max(sum(trace.demand.values()), 1)
        for expert, demand in trace.demand.items():
            result[expert] += weight * demand / total
    return experts, result


def _dataset_metrics(
    traces: Sequence[_LayerTrace],
    placement: Mapping[int, int],
    baselines: Sequence[Mapping[str, Any]],
    normalizers: Sequence[int],
    num_ranks: int,
    chunk_size: int,
    objective: str,
) -> list[dict[str, Any]]:
    result = []
    for trace, baseline, normalizer in zip(traces, baselines, normalizers):
        remote = _remote(trace, placement, chunk_size)
        objective_value = (
            remote
            if objective == "source-aware"
            else _objective(trace, placement, objective, chunk_size)
        )
        compute_load = _compute_loads(trace.demand, placement, num_ranks)
        compute_imbalance = _max_over_average(compute_load)
        baseline_compute = baseline["compute_imbalance"]
        result.append(
            {
                "label": trace.label,
                "path": trace.path,
                "tokens": trace.tokens,
                "bundles": trace.bundles,
                "baseline_remote": baseline["remote"],
                "remote": remote,
                "baseline_objective": baseline["objective"],
                "objective": objective_value,
                "normalized_objective": objective_value / normalizer,
                "objective_delta": (
                    objective_value / baseline["objective"] - 1.0
                    if baseline["objective"]
                    else None
                ),
                "remote_delta": (
                    remote / baseline["remote"] - 1.0 if baseline["remote"] else None
                ),
                "baseline_compute_imbalance": baseline_compute,
                "compute_imbalance": compute_imbalance,
                "compute_inflation": (
                    compute_imbalance / baseline_compute if baseline_compute else 1.0
                ),
            }
        )
    return result


def _quality(
    metrics: Sequence[Mapping[str, Any]],
    *,
    worst_weight: float,
    max_compute_inflation: float | None,
) -> tuple[tuple[float, ...], dict[str, float]]:
    normalized = [float(metric["normalized_objective"]) for metric in metrics]
    mean_objective = sum(normalized) / len(normalized)
    worst_objective = max(normalized)
    max_inflation = max(float(metric["compute_inflation"]) for metric in metrics)
    violation = (
        max(0.0, max_inflation - max_compute_inflation)
        if max_compute_inflation is not None
        else 0.0
    )
    objective = mean_objective + worst_weight * worst_objective
    return (
        (
            float(violation > 1e-12),
            violation,
            objective,
            worst_objective,
            mean_objective,
        ),
        {
            "mean_normalized_objective": mean_objective,
            "worst_normalized_objective": worst_objective,
            "robust_objective": objective,
            "max_compute_inflation": max_inflation,
            "compute_cap_violation": violation,
        },
    )


def _updated_weights(
    weights: Sequence[float],
    metrics: Sequence[Mapping[str, Any]],
    rate: float,
) -> list[float]:
    costs = [float(metric["normalized_objective"]) for metric in metrics]
    average = sum(costs) / len(costs)
    if average <= 0 or rate == 0:
        return list(weights)
    updated = [
        weight * math.exp(max(-20.0, min(20.0, rate * (cost / average - 1.0))))
        for weight, cost in zip(weights, costs)
    ]
    total = sum(updated)
    return [weight / total for weight in updated]


def _candidate(
    placement: Mapping[int, int],
    traces: Sequence[_LayerTrace],
    baselines: Sequence[Mapping[str, Any]],
    normalizers: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = _dataset_metrics(
        traces,
        placement,
        baselines,
        normalizers,
        args.num_ranks,
        args.tensor_chunk_size,
        args.hypergraph_objective,
    )
    key, summary = _quality(
        metrics,
        worst_weight=args.robust_worst_weight,
        max_compute_inflation=args.max_compute_inflation,
    )
    return {
        "rank_by_expert": dict(placement),
        "datasets": metrics,
        "quality_key": key,
        **summary,
    }


def _initial_graph_placement(
    weighted: Any,
    balanced: GraphPlacement,
    *,
    capacity: int,
    args: argparse.Namespace,
    cuda_ops: tuple[Any, ...] | None,
) -> GraphPlacement:
    if args.graph_rounds == 0:
        return balanced
    if cuda_ops is not None:
        build_cuda_graph, solve_cuda_graph, _, _ = cuda_ops
        graph = build_cuda_graph(weighted.topk_experts, weighted.count)
        return solve_cuda_graph(
            graph,
            num_ranks=args.num_ranks,
            slots_per_rank=capacity,
            max_rounds=args.graph_rounds,
            balance_weight=args.balance_weight,
        )
    graph = build_co_routing_graph(weighted)
    return CoRoutingGraphSolver(
        num_ranks=args.num_ranks,
        slots_per_rank=capacity,
        max_rounds=args.graph_rounds,
        balance_weight=args.balance_weight,
    ).solve(graph)


def _refine(
    weighted: Any,
    placement: Mapping[int, int],
    *,
    args: argparse.Namespace,
    cuda_ops: tuple[Any, ...] | None,
) -> tuple[Mapping[int, int], int, int, float]:
    start = time.perf_counter()
    balance_rounds = None if args.balance_rounds == -1 else args.balance_rounds
    if cuda_ops is not None:
        _, _, _, refine_cuda = cuda_ops
        result = refine_cuda(
            weighted.source_rank,
            weighted.topk_experts,
            weighted.count,
            placement,
            num_ranks=args.num_ranks,
            max_rounds=args.swaps_per_round,
            balance_rounds=balance_rounds,
            objective=args.hypergraph_objective,
        )
        elapsed = result.solve_seconds
    else:
        result = refine_hypergraph_placement(
            weighted,
            placement,
            num_ranks=args.num_ranks,
            max_rounds=args.swaps_per_round,
            balance_rounds=balance_rounds,
            objective=args.hypergraph_objective,
        )
        elapsed = time.perf_counter() - start
    return (
        result.rank_by_expert,
        result.iterations,
        result.balance_iterations,
        elapsed,
    )


def _solve_layer(
    traces: Sequence[_LayerTrace],
    *,
    args: argparse.Namespace,
    cuda_ops: tuple[Any, ...] | None,
) -> dict[str, Any]:
    layer_start = time.perf_counter()
    experts, aggregate_demand = _aggregate_demand(
        traces, [1.0 / len(traces)] * len(traces)
    )
    capacity = args.slots_per_rank
    if capacity is None:
        capacity = (len(experts) + args.num_ranks - 1) // args.num_ranks
    baseline_homes = _baseline_homes(experts, args.num_ranks, args.baseline)
    baselines = []
    for trace in traces:
        compute_load = _compute_loads(trace.demand, baseline_homes, args.num_ranks)
        baseline_remote = _remote(trace, baseline_homes, args.tensor_chunk_size)
        baseline_objective = (
            baseline_remote
            if args.hypergraph_objective == "source-aware"
            else _objective(
                trace,
                baseline_homes,
                args.hypergraph_objective,
                args.tensor_chunk_size,
            )
        )
        baselines.append(
            {
                "remote": baseline_remote,
                "objective": baseline_objective,
                "compute_imbalance": _max_over_average(compute_load),
            }
        )
    normalizers = [
        _normalizer(trace, baseline["objective"], args.robust_normalizer)
        for trace, baseline in zip(traces, baselines)
    ]
    equal_weights = [1.0 / len(traces)] * len(traces)
    balanced = _balanced_initial_placement(
        experts,
        aggregate_demand,
        num_ranks=args.num_ranks,
        slots_per_rank=capacity,
    )
    best = _candidate(balanced.rank_by_expert, traces, baselines, normalizers, args)
    weights = list(equal_weights)
    weighted = _weighted_trace(
        traces, weights, normalizers, args.normalization_precision
    )
    graph_placement = _initial_graph_placement(
        weighted,
        balanced,
        capacity=capacity,
        args=args,
        cuda_ops=cuda_ops,
    )
    graph_candidate = _candidate(
        graph_placement.rank_by_expert, traces, baselines, normalizers, args
    )
    if graph_candidate["quality_key"] < best["quality_key"]:
        best = graph_candidate

    total_iterations = 0
    total_balance_iterations = 0
    solve_seconds = 0.0
    rounds_run = 0
    for robust_round in range(args.robust_rounds):
        if robust_round > 0:
            weighted = _weighted_trace(
                traces, weights, normalizers, args.normalization_precision
            )
        rank_map, iterations, balance_iterations, elapsed = _refine(
            weighted,
            best["rank_by_expert"],
            args=args,
            cuda_ops=cuda_ops,
        )
        candidate = _candidate(rank_map, traces, baselines, normalizers, args)
        total_iterations += iterations
        total_balance_iterations += balance_iterations
        solve_seconds += elapsed
        rounds_run += 1
        if candidate["quality_key"] < best["quality_key"]:
            best = candidate
        updated_weights = _updated_weights(
            weights, best["datasets"], args.robust_reweight_rate
        )
        weights_changed = max(
            abs(new - old) for new, old in zip(updated_weights, weights)
        )
        weights = updated_weights
        if iterations == 0 and balance_iterations == 0 and weights_changed < 1e-12:
            break

    final_placement = _graph_placement_from_rank_map(
        best["rank_by_expert"], args.num_ranks
    )
    return {
        **best,
        "experts": len(experts),
        "normalizers": normalizers,
        "final_dataset_weights": weights,
        "robust_rounds": rounds_run,
        "iterations": total_iterations,
        "balance_iterations": total_balance_iterations,
        "solve_seconds": solve_seconds,
        "layer_seconds": time.perf_counter() - layer_start,
        "experts_by_rank": {
            str(rank): list(rank_experts)
            for rank, rank_experts in final_placement.experts_by_rank.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    if args.device != "cpu":
        torch = _torch()
        if not args.device.startswith("cuda"):
            raise ValueError("--device must be cpu, cuda, or cuda:<index>")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for --device {args.device}")
        from solve_co_routing_graph import _cuda_fast_planner_ops

        cuda_ops = _cuda_fast_planner_ops()
    else:
        cuda_ops = None
    load_start = time.perf_counter()
    inputs = _load_traces(args)
    input_load_seconds = time.perf_counter() - load_start
    layers = []
    for layer_index, (gate, _) in enumerate(inputs[0].layers):
        prepared = [_prepare_layer(trace, layer_index, args) for trace in inputs]
        layer = _solve_layer(prepared, args=args, cuda_ops=cuda_ops)
        layer["gate"] = gate
        layer["layer"] = _layer_number(gate, layer_index)
        layers.append(layer)
    return {
        "inputs": [trace.path for trace in inputs],
        "labels": [trace.label for trace in inputs],
        "num_ranks": args.num_ranks,
        "device": args.device,
        "baseline": args.baseline,
        "robust_normalizer": args.robust_normalizer,
        "robust_worst_weight": args.robust_worst_weight,
        "hypergraph_objective": args.hypergraph_objective,
        "max_compute_inflation": args.max_compute_inflation,
        "input_load_seconds": input_load_seconds,
        "total_wall_seconds": time.perf_counter() - start,
        "layers": layers,
    }


def _print_result(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    rows = [["layer", "mean", "worst", "max_comp", "rounds", "swaps", "solve", "total"]]
    for layer in result["layers"]:
        rows.append(
            [
                str(layer["layer"]),
                f"{layer['mean_normalized_objective']:.3f}",
                f"{layer['worst_normalized_objective']:.3f}",
                f"{layer['max_compute_inflation']:.2f}x",
                str(layer["robust_rounds"]),
                str(layer["iterations"]),
                f"{layer['solve_seconds']:.3f}s",
                f"{layer['layer_seconds']:.3f}s",
            ]
        )
    print("Robust multi-dataset hypergraph placement")
    compute_cap = (
        "disabled"
        if result["max_compute_inflation"] is None
        else f"{result['max_compute_inflation']:.2f}x"
    )
    print(
        f"datasets={len(result['inputs'])} device={result['device']} "
        f"objective={result['hypergraph_objective']} "
        f"normalizer={result['robust_normalizer']} "
        f"worst_weight={result['robust_worst_weight']} "
        f"compute_cap={compute_cap}"
    )
    print(_format_table(rows))

    totals = {
        label: {
            "baseline_remote": 0,
            "remote": 0,
            "baseline_objective": 0,
            "objective": 0,
            "comp": [],
            "inflation": [],
        }
        for label in result["labels"]
    }
    for layer in result["layers"]:
        for metric in layer["datasets"]:
            item = totals[metric["label"]]
            item["baseline_remote"] += metric["baseline_remote"]
            item["remote"] += metric["remote"]
            item["baseline_objective"] += metric["baseline_objective"]
            item["objective"] += metric["objective"]
            item["comp"].append(metric["compute_imbalance"])
            item["inflation"].append(metric["compute_inflation"])
    dataset_rows = [
        [
            "dataset",
            "base_remote",
            "robust_remote",
            "remote_delta",
            "base_obj",
            "robust_obj",
            "obj_delta",
            "avg_comp",
            "max_infl",
        ]
    ]
    for label, item in totals.items():
        remote_delta = (
            item["remote"] / item["baseline_remote"] - 1.0
            if item["baseline_remote"]
            else 0.0
        )
        objective_delta = (
            item["objective"] / item["baseline_objective"] - 1.0
            if item["baseline_objective"]
            else 0.0
        )
        dataset_rows.append(
            [
                label,
                str(item["baseline_remote"]),
                str(item["remote"]),
                f"{remote_delta:+.1%}",
                str(item["baseline_objective"]),
                str(item["objective"]),
                f"{objective_delta:+.1%}",
                f"{sum(item['comp']) / len(item['comp']):.2f}",
                f"{max(item['inflation']):.2f}x",
            ]
        )
    print("\nPer-dataset replay")
    print(_format_table(dataset_rows))
    print(
        f"total solve={sum(layer['solve_seconds'] for layer in result['layers']):.3f}s "
        f"wall={result['total_wall_seconds']:.3f}s"
    )
    if args.show_placement:
        for layer in result["layers"]:
            print(f"\n{layer['gate']} experts_by_rank")
            for rank, experts in layer["experts_by_rank"].items():
                print(f"  rank {rank}: {experts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="training trace; repeat once per dataset",
    )
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--source-ep", type=int, default=None)
    parser.add_argument("--slots-per-rank", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-chunk-size", type=int, default=262144)
    parser.add_argument("--graph-rounds", type=int, default=8)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--hypergraph-objective",
        choices=["source-aware", "source-agnostic"],
        default="source-aware",
        help=(
            "exact objective: remote destination ranks relative to each source, "
            "or source-independent distinct destination ranks (default: source-aware)"
        ),
    )
    parser.add_argument("--robust-rounds", type=int, default=8)
    parser.add_argument(
        "--swaps-per-round",
        type=int,
        default=2,
        help="hypergraph swaps before reweighting datasets (default: 2)",
    )
    parser.add_argument(
        "--balance-rounds",
        type=int,
        default=0,
        help="communication-neutral balance swaps per robust round; -1 converges",
    )
    parser.add_argument(
        "--robust-normalizer",
        choices=["baseline", "tokens"],
        default="baseline",
    )
    parser.add_argument(
        "--robust-worst-weight",
        type=float,
        default=1.0,
        help="rho in mean normalized remote + rho * worst (default: 1)",
    )
    parser.add_argument(
        "--robust-reweight-rate",
        type=float,
        default=1.0,
        help="multiplicative update rate for poorly served datasets (default: 1)",
    )
    parser.add_argument(
        "--normalization-precision",
        type=int,
        default=1024,
        help="integer precision for normalized bundle weights (default: 1024)",
    )
    parser.add_argument(
        "--max-compute-inflation",
        type=float,
        default=None,
        help="prefer placements whose max/avg is at most this times baseline",
    )
    parser.add_argument(
        "--no-compute-limit",
        action="store_true",
        help="optimize robust communication without a compute imbalance limit",
    )
    parser.add_argument(
        "--baseline", choices=["round-robin", "contiguous"], default="round-robin"
    )
    parser.add_argument("--save-placement", default=None)
    parser.add_argument("--show-placement", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("repeat --input for at least two datasets")
    if args.num_ranks < 1:
        parser.error("--num-ranks must be positive")
    if args.source_ep is not None and args.source_ep < 1:
        parser.error("--source-ep must be positive")
    if args.slots_per_rank is not None and args.slots_per_rank < 1:
        parser.error("--slots-per-rank must be positive")
    if args.tensor_chunk_size < 1 or args.normalization_precision < 1:
        parser.error("chunk size and normalization precision must be positive")
    if args.graph_rounds < 0 or args.robust_rounds < 1 or args.swaps_per_round < 0:
        parser.error("round counts must be non-negative and robust rounds positive")
    if args.balance_rounds < -1:
        parser.error("--balance-rounds must be -1 or non-negative")
    if args.balance_weight < 0 or args.robust_worst_weight < 0:
        parser.error("objective weights must be non-negative")
    if args.robust_reweight_rate < 0:
        parser.error("--robust-reweight-rate must be non-negative")
    if args.max_compute_inflation is not None and args.max_compute_inflation < 1:
        parser.error("--max-compute-inflation must be at least 1")
    if args.no_compute_limit and args.max_compute_inflation is not None:
        parser.error("--no-compute-limit conflicts with --max-compute-inflation")
    if args.no_compute_limit:
        args.max_compute_inflation = None
    result = run(args)
    if args.save_placement:
        _save_placements(args.save_placement, result)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    _print_result(result, args)
    if args.save_placement:
        print(f"saved placement={args.save_placement}")


if __name__ == "__main__":
    main()
