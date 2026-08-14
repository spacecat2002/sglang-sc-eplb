#!/usr/bin/env python3
"""Solve an offline Top-K bundle trace with a co-routing graph.

Input JSON can be either a list of bundle records or an object with a
``layers`` list.  A bundle record has the following shape::

    {"source_rank": 0, "topk_experts": [3, 7, 9], "count": 128}

For the layered form, each layer may additionally contain ``gate``.

Example::

    python benchmark/solve_co_routing_graph.py \
        --input bundles.json --num-ranks 8 --slots-per-rank 32 \
        --device cuda --planner cuda-fast
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import orjson

from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    RoutedToken,
)
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraph,
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_primary_remote,
)


def _raw_layers(raw: Any) -> list[tuple[str, list[Any]]]:
    if isinstance(raw, list):
        layers = [("layer0", raw)]
    elif isinstance(raw, dict) and isinstance(raw.get("layers"), list):
        layers = []
        for index, layer in enumerate(raw["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"layers[{index}] must be an object")
            layers.append(
                (str(layer.get("gate", f"layer{index}")), layer.get("bundles", []))
            )
    else:
        raise ValueError("input must be a bundle list or an object containing layers")

    result = []
    for name, entries in layers:
        if not isinstance(entries, list):
            raise ValueError(f"{name}.bundles must be a list")
        if entries:
            result.append((name, entries))
    if not result:
        raise ValueError("input contains no non-empty layers")
    return result


def _tokens_from_entries(name: str, entries: Sequence[Any]) -> list[RoutedToken]:
    tokens = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}.bundles[{index}] must be an object")
        try:
            source = int(entry["source_rank"])
            experts = tuple(sorted(int(expert) for expert in entry["topk_experts"]))
            count = int(entry.get("count", 1))
            tokens.append(RoutedToken(source, experts, count))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle in {name}: {entry!r}") from exc
    return tokens


@dataclass(frozen=True)
class _TensorTrace:
    source_rank: Any
    topk_experts: Any
    count: Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--device cuda requires PyTorch") from exc
    return torch


def _cuda_fast_planner_ops():
    try:
        from sglang.srt.eplb.cuda_fast_co_routing_planner import (
            build_co_routing_graph_cuda,
            plan_communication_replicas_cuda,
            solve_co_routing_graph_cuda,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--planner cuda-fast requires PyTorch, CUDA, and Triton"
        ) from exc
    return (
        build_co_routing_graph_cuda,
        solve_co_routing_graph_cuda,
        plan_communication_replicas_cuda,
    )


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        _torch().cuda.synchronize(device)


def _tensorize_entries(
    name: str,
    entries: Sequence[Any],
    *,
    device: str,
    source_ep: int,
    target_ep: int,
) -> _TensorTrace:
    torch = _torch()
    sources = []
    topks = []
    counts = []
    topk_size = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}.bundles[{index}] must be an object")
        try:
            source = int(entry["source_rank"])
            experts = tuple(int(expert) for expert in entry["topk_experts"])
            count = int(entry.get("count", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle in {name}: {entry!r}") from exc
        if not experts:
            raise ValueError(f"invalid bundle in {name}: {entry!r}")
        if topk_size is None:
            topk_size = len(experts)
        elif len(experts) != topk_size:
            raise ValueError(
                f"--device cuda requires a fixed Top-K within {name}; "
                f"expected {topk_size}, got {len(experts)}"
            )
        sources.append(source)
        topks.append(experts)
        counts.append(count)

    source_tensor = torch.tensor(sources, dtype=torch.int64, device=device)
    topk_tensor = torch.tensor(topks, dtype=torch.int64, device=device)
    count_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
    invalid = (
        (source_tensor < 0).any()
        | (source_tensor >= source_ep).any()
        | (count_tensor < 1).any()
        | (topk_tensor < 0).any()
    )
    sorted_topk = topk_tensor.sort(dim=1).values
    duplicates = (sorted_topk[:, 1:] == sorted_topk[:, :-1]).any()
    if bool(invalid.item()) or bool(duplicates.item()):
        raise ValueError(f"invalid source, count, or expert ids in {name}")
    topk_tensor = sorted_topk
    if source_ep == target_ep:
        return _TensorTrace(source_tensor, topk_tensor, count_tensor)
    if target_ep < source_ep or target_ep % source_ep:
        raise ValueError("target EP must be a multiple of source EP")

    fanout = target_ep // source_ep
    offsets = torch.arange(fanout, device=device)
    split_counts = count_tensor[:, None] // fanout + (
        offsets[None, :] < (count_tensor % fanout)[:, None]
    )
    keep = split_counts > 0
    return _TensorTrace(
        (source_tensor[:, None] * fanout + offsets[None, :]).expand_as(split_counts)[
            keep
        ],
        topk_tensor[:, None, :]
        .expand(-1, fanout, -1)[keep]
        .reshape(-1, topk_tensor.shape[1]),
        split_counts[keep],
    )


def _build_graph_tensor(trace: _TensorTrace, *, chunk_size: int) -> CoRoutingGraph:
    torch = _torch()
    max_expert = int(trace.topk_experts.max().item())
    num_experts = max_expert + 1
    demand = torch.zeros(
        num_experts, dtype=torch.int64, device=trace.topk_experts.device
    )
    edge_weights = torch.zeros(
        num_experts * num_experts,
        dtype=torch.int64,
        device=trace.topk_experts.device,
    )
    topk_size = trace.topk_experts.shape[1]
    pair_indices = torch.triu_indices(
        topk_size,
        topk_size,
        offset=1,
        device=trace.topk_experts.device,
    )
    for start in range(0, trace.count.numel(), chunk_size):
        stop = min(start + chunk_size, trace.count.numel())
        topk = trace.topk_experts[start:stop]
        count = trace.count[start:stop]
        demand.scatter_add_(
            0,
            topk.reshape(-1),
            count[:, None].expand_as(topk).reshape(-1),
        )
        left = topk[:, pair_indices[0]]
        right = topk[:, pair_indices[1]]
        pair_ids = torch.minimum(left, right) * num_experts + torch.maximum(left, right)
        edge_weights.scatter_add_(
            0,
            pair_ids.reshape(-1),
            count[:, None].expand_as(pair_ids).reshape(-1),
        )

    observed = torch.nonzero(demand, as_tuple=False).flatten()
    nonzero_edges = torch.nonzero(edge_weights, as_tuple=False).flatten()
    observed_cpu = observed.cpu().tolist()
    demand_cpu = demand[observed].cpu().tolist()
    edge_ids_cpu = nonzero_edges.cpu().tolist()
    edge_values_cpu = edge_weights[nonzero_edges].cpu().tolist()
    demand_map = dict(zip(observed_cpu, demand_cpu))
    edges = {
        (edge_id // num_experts, edge_id % num_experts): weight
        for edge_id, weight in zip(edge_ids_cpu, edge_values_cpu)
    }
    adjacency = {expert: {} for expert in observed_cpu}
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(tuple(observed_cpu), demand_map, edges, adjacency)


def _evaluate_primary_remote_tensor(
    trace: _TensorTrace,
    placements: Sequence[Mapping[int, int]],
    *,
    chunk_size: int,
) -> list[int]:
    torch = _torch()
    max_expert = int(trace.topk_experts.max().item())
    homes = torch.full(
        (len(placements), max_expert + 1),
        -1,
        dtype=torch.int64,
        device=trace.topk_experts.device,
    )
    for placement_index, placement in enumerate(placements):
        expert_ids = torch.tensor(
            list(placement), dtype=torch.int64, device=trace.topk_experts.device
        )
        rank_ids = torch.tensor(
            list(placement.values()),
            dtype=torch.int64,
            device=trace.topk_experts.device,
        )
        homes[placement_index, expert_ids] = rank_ids
    if bool((homes[:, trace.topk_experts.reshape(-1)] < 0).any().item()):
        raise ValueError("placement is missing an observed expert")

    totals = torch.zeros(
        len(placements), dtype=torch.int64, device=trace.topk_experts.device
    )
    for start in range(0, trace.count.numel(), chunk_size):
        stop = min(start + chunk_size, trace.count.numel())
        topk = trace.topk_experts[start:stop]
        destination = homes[:, topk].sort(dim=-1).values
        is_new = torch.ones_like(destination, dtype=torch.bool)
        is_new[:, :, 1:] = destination[:, :, 1:] != destination[:, :, :-1]
        is_remote = destination != trace.source_rank[start:stop][None, :, None]
        remote_per_bundle = (is_new & is_remote).sum(dim=-1)
        totals += (remote_per_bundle * trace.count[start:stop][None, :]).sum(dim=1)
    return totals.cpu().tolist()


def _baseline_homes(
    experts: Sequence[int], num_ranks: int, mode: str
) -> dict[int, int]:
    if mode == "round-robin":
        return {expert: expert % num_ranks for expert in experts}
    per_rank = (max(experts, default=-1) + 1 + num_ranks - 1) // num_ranks
    per_rank = max(per_rank, 1)
    return {expert: min(expert // per_rank, num_ranks - 1) for expert in experts}


def _format_table(rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _compute_loads(
    demand: Mapping[int, int],
    rank_by_expert: Mapping[int, int],
    num_ranks: int,
) -> list[int]:
    loads = [0] * num_ranks
    for expert, count in demand.items():
        loads[rank_by_expert[expert]] += count
    return loads


def _max_over_average(values: Sequence[int]) -> float:
    average = sum(values) / len(values) if values else 0.0
    return max(values, default=0) / average if average else 0.0


def _reshard_tokens(
    tokens: Sequence[RoutedToken], source_ep: int, target_ep: int
) -> list[RoutedToken]:
    """Deterministically split source-rank traffic across virtual target ranks."""

    if source_ep == target_ep:
        return list(tokens)
    if target_ep < source_ep or target_ep % source_ep:
        raise ValueError("target EP must be a multiple of source EP")
    fanout = target_ep // source_ep
    result: list[RoutedToken] = []
    for token in tokens:
        base, remainder = divmod(token.count, fanout)
        for offset in range(fanout):
            count = base + int(offset < remainder)
            if count:
                result.append(
                    RoutedToken(
                        token.source_rank * fanout + offset,
                        token.topk_experts,
                        count,
                    )
                )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.perf_counter()
    use_tensor_backend = args.device != "cpu"
    if use_tensor_backend:
        torch = _torch()
        if not args.device.startswith("cuda"):
            raise ValueError("--device must be cpu, cuda, or cuda:<index>")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for --device {args.device}")
    if args.planner == "cuda-fast" and not use_tensor_backend:
        raise ValueError("--planner cuda-fast requires --device cuda")
    cuda_fast_ops = _cuda_fast_planner_ops() if args.planner == "cuda-fast" else None
    input_start = time.perf_counter()
    raw = orjson.loads(Path(args.input).read_bytes())
    layers = _raw_layers(raw)
    input_load_seconds = time.perf_counter() - input_start
    trace_ep = args.source_ep
    if trace_ep is None and isinstance(raw, dict):
        trace_ep = raw.get("num_ranks")
    if trace_ep is None:
        trace_ep = args.num_ranks
    trace_ep = int(trace_ep)
    if trace_ep < 1:
        raise ValueError("source EP must be positive")
    output_layers = []
    for name, entries in layers:
        layer_start = time.perf_counter()
        prepare_start = time.perf_counter()
        tensorize_start = time.perf_counter()
        tokens = None
        tensor_trace = None
        if use_tensor_backend:
            tensor_trace = _tensorize_entries(
                name,
                entries,
                device=args.device,
                source_ep=trace_ep,
                target_ep=args.num_ranks,
            )
            _synchronize(args.device)
        else:
            tokens = _reshard_tokens(
                _tokens_from_entries(name, entries), trace_ep, args.num_ranks
            )
        tensorize_seconds = time.perf_counter() - tensorize_start
        graph_build_start = time.perf_counter()
        cuda_graph = None
        if tensor_trace is not None:
            if cuda_fast_ops is not None:
                build_cuda_graph, _, _ = cuda_fast_ops
                cuda_graph = build_cuda_graph(
                    tensor_trace.topk_experts, tensor_trace.count
                )
                graph = cuda_graph.to_cpu_graph()
            else:
                graph = _build_graph_tensor(
                    tensor_trace, chunk_size=args.tensor_chunk_size
                )
            _synchronize(args.device)
        else:
            graph = build_co_routing_graph(tokens)
        graph_build_seconds = time.perf_counter() - graph_build_start
        prepare_seconds = time.perf_counter() - prepare_start
        capacity = args.slots_per_rank
        if capacity is None:
            capacity = (len(graph.experts) + args.num_ranks - 1) // args.num_ranks
        solver = CoRoutingGraphSolver(
            num_ranks=args.num_ranks,
            slots_per_rank=capacity,
            max_rounds=args.max_rounds,
            balance_weight=args.balance_weight,
        )
        graph_start = time.perf_counter()
        if cuda_graph is not None:
            _, solve_cuda_graph, _ = cuda_fast_ops
            placement = solve_cuda_graph(
                cuda_graph,
                num_ranks=args.num_ranks,
                slots_per_rank=capacity,
                max_rounds=args.max_rounds,
                balance_weight=args.balance_weight,
                cpu_graph=graph,
            )
            _synchronize(args.device)
        else:
            placement = solver.solve(graph)
        graph_solve_seconds = time.perf_counter() - graph_start
        baseline_homes = _baseline_homes(graph.experts, args.num_ranks, args.baseline)
        baseline_compute_load = _compute_loads(
            graph.demand, baseline_homes, args.num_ranks
        )
        graph_compute_load = _compute_loads(
            graph.demand, placement.rank_by_expert, args.num_ranks
        )
        primary_replay_start = time.perf_counter()
        if tensor_trace is not None:
            baseline_remote, graph_remote = _evaluate_primary_remote_tensor(
                tensor_trace,
                [baseline_homes, placement.rank_by_expert],
                chunk_size=args.tensor_chunk_size,
            )
            _synchronize(args.device)
        else:
            baseline_remote = evaluate_primary_remote(tokens, baseline_homes)
            graph_remote = evaluate_primary_remote(tokens, placement.rank_by_expert)
        primary_replay_seconds = time.perf_counter() - primary_replay_start
        planned_remote = graph_remote
        action_count = 0
        replica_action_details = []
        replica_materialize_seconds = 0.0
        replica_solve_seconds = 0.0
        replica_replay_seconds = 0.0
        if args.replica_slots_per_rank:
            if cuda_fast_ops is not None:
                _, _, plan_cuda_replicas = cuda_fast_ops
                replica_plan = plan_cuda_replicas(
                    tensor_trace.source_rank,
                    tensor_trace.topk_experts,
                    tensor_trace.count,
                    placement.rank_by_expert,
                    num_ranks=args.num_ranks,
                    replica_slots_per_rank=args.replica_slots_per_rank,
                    max_candidates=args.fast_max_candidates,
                    max_bundle_size=args.max_bundle_size,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    chunk_size=min(args.tensor_chunk_size, 65536),
                )
                planned_remote = replica_plan.unique_remote_rank_copies
                action_count = len(replica_plan.actions)
                replica_action_details = [
                    {
                        "destination_rank": action.destination_rank,
                        "experts": list(action.experts),
                        "kind": action.kind,
                    }
                    for action in replica_plan.actions
                ]
                replica_solve_seconds = replica_plan.solve_seconds
                replica_replay_seconds = replica_plan.replay_seconds
            elif tokens is None:
                materialize_start = time.perf_counter()
                tokens = _reshard_tokens(
                    _tokens_from_entries(name, entries), trace_ep, args.num_ranks
                )
                replica_materialize_seconds = time.perf_counter() - materialize_start
            if cuda_fast_ops is None:
                replica_planner = BundleAwareReplicaPlanner(
                    num_ranks=args.num_ranks,
                    baseline_rank_by_expert=placement.rank_by_expert,
                    replica_slots_per_rank=args.replica_slots_per_rank,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    compute_weight=0.0,
                    communication_weight=1.0,
                    max_bundle_size=args.max_bundle_size,
                )
                start = time.perf_counter()
                if args.planner == "fast":
                    replica_plan = replica_planner.plan_fast(
                        tokens, max_candidates=args.fast_max_candidates
                    )
                else:
                    replica_plan = replica_planner.plan(
                        tokens, max_actions=args.max_actions
                    )
                replica_solve_seconds = time.perf_counter() - start
                replay_start = time.perf_counter()
                replayed = replica_planner.evaluate_placement(
                    tokens, replica_plan.replicas_by_rank
                )
                replica_replay_seconds = time.perf_counter() - replay_start
                planned_remote = replayed.unique_remote_rank_copies
                action_count = len(replica_plan.actions)
                replica_action_details = [
                    {
                        "destination_rank": action.destination_rank,
                        "experts": list(action.experts),
                        "kind": action.kind,
                    }
                    for action in replica_plan.actions
                ]
        if tensor_trace is not None:
            token_count = int(tensor_trace.count.sum().item())
            bundle_count = tensor_trace.count.numel()
        else:
            token_count = sum(token.count for token in tokens)
            bundle_count = len(tokens)
        output_layers.append(
            {
                "gate": name,
                "tokens": token_count,
                "bundles": bundle_count,
                "experts": len(graph.experts),
                "edges": len(graph.edges),
                "baseline_remote": baseline_remote,
                "graph_remote": graph_remote,
                "planned_remote": planned_remote,
                "baseline_compute_load": baseline_compute_load,
                "graph_compute_load": graph_compute_load,
                "baseline_compute_imbalance": _max_over_average(baseline_compute_load),
                "graph_compute_imbalance": _max_over_average(graph_compute_load),
                "graph_delta": (graph_remote / baseline_remote - 1.0)
                if baseline_remote
                else 0.0,
                "planned_delta": (planned_remote / baseline_remote - 1.0)
                if baseline_remote
                else 0.0,
                "initial_cut": placement.initial_cut,
                "final_cut": placement.final_cut,
                "iterations": placement.iterations,
                "prepare_seconds": prepare_seconds,
                "tensorize_seconds": tensorize_seconds,
                "graph_build_seconds": graph_build_seconds,
                "primary_replay_seconds": primary_replay_seconds,
                "layer_seconds": time.perf_counter() - layer_start,
                "graph_solve_seconds": graph_solve_seconds,
                "replica_actions": action_count,
                "replica_action_details": replica_action_details,
                "replica_materialize_seconds": replica_materialize_seconds,
                "replica_solve_seconds": replica_solve_seconds,
                "replica_replay_seconds": replica_replay_seconds,
                "experts_by_rank": {
                    str(rank): list(experts)
                    for rank, experts in placement.experts_by_rank.items()
                },
            }
        )
    return {
        "source_ep": trace_ep,
        "num_ranks": args.num_ranks,
        "baseline": args.baseline,
        "planner": args.planner,
        "device": args.device,
        "input_load_seconds": input_load_seconds,
        "total_wall_seconds": time.perf_counter() - run_start,
        "layers": output_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="offline bundle JSON")
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument(
        "--source-ep",
        type=int,
        default=None,
        help="source EP of the trace; defaults to trace JSON num_ranks",
    )
    parser.add_argument("--slots-per-rank", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "backend for graph statistics and primary replay: cpu, cuda, or "
            "cuda:<index> (default: cpu)"
        ),
    )
    parser.add_argument(
        "--tensor-chunk-size",
        type=int,
        default=262144,
        help="maximum bundles per CUDA graph/replay chunk (default: 262144)",
    )
    parser.add_argument(
        "--replica-slots-per-rank",
        type=int,
        default=0,
        help="optional additional copies after graph placement (default: 0)",
    )
    parser.add_argument("--ranks-per-node", type=int, default=None)
    parser.add_argument("--rdma-cost", type=float, default=4.0)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-bundle-size", type=int, default=None)
    parser.add_argument("--fast-max-candidates", type=int, default=32)
    parser.add_argument(
        "--planner", choices=["exact", "fast", "cuda-fast"], default="fast"
    )
    parser.add_argument(
        "--baseline",
        choices=["round-robin", "contiguous"],
        default="round-robin",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-placement",
        action="store_true",
        help="print expert ids assigned to every rank",
    )
    args = parser.parse_args()
    if args.num_ranks < 1:
        parser.error("--num-ranks must be positive")
    if args.source_ep is not None and args.source_ep < 1:
        parser.error("--source-ep must be positive")
    if args.slots_per_rank is not None and args.slots_per_rank < 1:
        parser.error("--slots-per-rank must be positive")
    if args.replica_slots_per_rank < 0:
        parser.error("--replica-slots-per-rank must be non-negative")
    if args.ranks_per_node is not None and args.ranks_per_node < 1:
        parser.error("--ranks-per-node must be positive")
    if args.rdma_cost < 1:
        parser.error("--rdma-cost must be at least 1")
    if args.fast_max_candidates < 1:
        parser.error("--fast-max-candidates must be positive")
    if args.tensor_chunk_size < 1:
        parser.error("--tensor-chunk-size must be positive")
    result = run(args)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    rows = [
        [
            "gate",
            "tokens",
            "bundles",
            "edges",
            "baseline",
            "graph",
            "planned",
            "graph_delta",
            "planned_delta",
            "cut",
            "comp",
            "rounds",
            "tensorize",
            "graph_build",
            "primary_replay",
            "layer_total",
            "actions",
            "graph_solve",
            "replica_input",
            "replica_solve",
            "replica_replay",
        ]
    ]
    for layer in result["layers"]:
        rows.append(
            [
                layer["gate"],
                str(layer["tokens"]),
                str(layer["bundles"]),
                str(layer["edges"]),
                str(layer["baseline_remote"]),
                str(layer["graph_remote"]),
                str(layer["planned_remote"]),
                f"{layer['graph_delta']:+.1%}",
                f"{layer['planned_delta']:+.1%}",
                f"{layer['initial_cut']}->{layer['final_cut']}",
                f"{layer['baseline_compute_imbalance']:.2f}->"
                f"{layer['graph_compute_imbalance']:.2f}",
                str(layer["iterations"]),
                f"{layer['tensorize_seconds']:.3f}s",
                f"{layer['graph_build_seconds']:.3f}s",
                f"{layer['primary_replay_seconds']:.3f}s",
                f"{layer['layer_seconds']:.3f}s",
                str(layer["replica_actions"]),
                f"{layer['graph_solve_seconds']:.3f}s",
                f"{layer['replica_materialize_seconds']:.3f}s",
                f"{layer['replica_solve_seconds']:.3f}s",
                f"{layer['replica_replay_seconds']:.3f}s",
            ]
        )
    print("Co-routing graph placement replay")
    print(
        f"device={result['device']} planner={result['planner']} "
        f"input_load={result['input_load_seconds']:.3f}s"
    )
    print(_format_table(rows))
    print(
        "total tensorize="
        f"{sum(layer['tensorize_seconds'] for layer in result['layers']):.3f}s "
        "graph_build="
        f"{sum(layer['graph_build_seconds'] for layer in result['layers']):.3f}s "
        "primary_replay="
        f"{sum(layer['primary_replay_seconds'] for layer in result['layers']):.3f}s "
        "graph_solve="
        f"{sum(layer['graph_solve_seconds'] for layer in result['layers']):.3f}s "
        "replica_input="
        f"{sum(layer['replica_materialize_seconds'] for layer in result['layers']):.3f}s "
        "replica_solve="
        f"{sum(layer['replica_solve_seconds'] for layer in result['layers']):.3f}s "
        "replica_replay="
        f"{sum(layer['replica_replay_seconds'] for layer in result['layers']):.3f}s "
        "wall="
        f"{result['total_wall_seconds']:.3f}s",
    )
    if args.show_placement:
        for layer in result["layers"]:
            print(f"\n{layer['gate']} experts_by_rank")
            for rank, experts in layer["experts_by_rank"].items():
                print(f"  rank {rank}: {experts}")
            if layer["replica_action_details"]:
                print(f"{layer['gate']} replica actions")
                for action in layer["replica_action_details"]:
                    print(
                        f"  {action['kind']}: experts={action['experts']} "
                        f"-> rank {action['destination_rank']}"
                    )


if __name__ == "__main__":
    main()
