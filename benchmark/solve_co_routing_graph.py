#!/usr/bin/env python3
"""Solve an offline Top-K bundle trace with a co-routing graph.

Input JSON can be either a list of bundle records or an object with a
``layers`` list.  A bundle record has the following shape::

    {"source_rank": 0, "topk_experts": [3, 7, 9], "count": 128}

For the layered form, each layer may additionally contain ``gate``.

Example::

    python benchmark/solve_co_routing_graph.py \
        --input bundles.json --num-ranks 8 --slots-per-rank 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    RoutedToken,
)
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_primary_remote,
)


def _records(raw: Any) -> list[tuple[str, list[RoutedToken]]]:
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

    result: list[tuple[str, list[RoutedToken]]] = []
    for name, entries in layers:
        if not isinstance(entries, list):
            raise ValueError(f"{name}.bundles must be a list")
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
        if tokens:
            result.append((name, tokens))
    if not result:
        raise ValueError("input contains no non-empty layers")
    return result


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
    raw = json.loads(Path(args.input).read_text())
    layers = _records(raw)
    trace_ep = args.source_ep
    if trace_ep is None and isinstance(raw, dict):
        trace_ep = raw.get("num_ranks")
    if trace_ep is None:
        trace_ep = args.num_ranks
    if trace_ep < 1:
        raise ValueError("source EP must be positive")
    output_layers = []
    for name, tokens in layers:
        layer_start = time.perf_counter()
        prepare_start = time.perf_counter()
        tokens = _reshard_tokens(tokens, trace_ep, args.num_ranks)
        graph = build_co_routing_graph(tokens)
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
        placement = solver.solve(graph)
        graph_solve_seconds = time.perf_counter() - graph_start
        baseline_homes = _baseline_homes(graph.experts, args.num_ranks, args.baseline)
        baseline_remote = evaluate_primary_remote(tokens, baseline_homes)
        graph_remote = evaluate_primary_remote(tokens, placement.rank_by_expert)
        planned_remote = graph_remote
        action_count = 0
        replica_solve_seconds = 0.0
        replica_replay_seconds = 0.0
        if args.replica_slots_per_rank:
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
        output_layers.append(
            {
                "gate": name,
                "tokens": sum(token.count for token in tokens),
                "bundles": len(tokens),
                "experts": len(graph.experts),
                "edges": len(graph.edges),
                "baseline_remote": baseline_remote,
                "graph_remote": graph_remote,
                "planned_remote": planned_remote,
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
                "layer_seconds": time.perf_counter() - layer_start,
                "graph_solve_seconds": graph_solve_seconds,
                "replica_actions": action_count,
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
    parser.add_argument("--planner", choices=["exact", "fast"], default="fast")
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
    if args.fast_max_candidates < 1:
        parser.error("--fast-max-candidates must be positive")
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
            "rounds",
            "prepare",
            "layer_total",
            "actions",
            "graph_solve",
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
                str(layer["iterations"]),
                f"{layer['prepare_seconds']:.3f}s",
                f"{layer['layer_seconds']:.3f}s",
                str(layer["replica_actions"]),
                f"{layer['graph_solve_seconds']:.3f}s",
                f"{layer['replica_solve_seconds']:.3f}s",
                f"{layer['replica_replay_seconds']:.3f}s",
            ]
        )
    print("Co-routing graph placement replay")
    print(_format_table(rows))
    print(
        "total graph_solve="
        f"{sum(layer['graph_solve_seconds'] for layer in result['layers']):.3f}s "
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


if __name__ == "__main__":
    main()
