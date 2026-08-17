#!/usr/bin/env python3
"""Compare offline MoE primary-expert placement algorithms on one trace.

Example:

    PYTHONPATH=python python benchmark/compare_offline_expert_placement.py \
        --input trace.pt --num-ranks 8 --ranks-per-node 4 --json
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from typing import Any, Mapping

from solve_co_routing_graph import (
    _balanced_initial_placement,
    _baseline_homes,
    _compute_loads,
    _format_table,
    _layer_number,
    _load_input,
    _materialize_layer_tokens,
    _max_over_average,
    _save_placements,
    _summarize_tokens,
)
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_destination_rank_copies,
    evaluate_primary_remote,
    refine_hypergraph_placement,
)
from sglang.srt.eplb.offline_expert_placement import (
    evaluate_topology_hypergraph_objective,
    grace_hierarchical_placement,
    refine_load_constrained_hypergraph_placement,
)


def _metrics(
    tokens,
    demand: Mapping[int, int],
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
) -> dict[str, Any]:
    loads = _compute_loads(demand, placement, num_ranks)
    counts = [0] * num_ranks
    for rank in placement.values():
        counts[rank] += 1
    return {
        "remote": evaluate_primary_remote(tokens, placement),
        "destination_copies": evaluate_destination_rank_copies(tokens, placement),
        "topology_objective": evaluate_topology_hypergraph_objective(
            tokens,
            placement,
            ranks_per_node=ranks_per_node,
            rdma_cost=rdma_cost,
        ),
        "compute_imbalance": _max_over_average(loads),
        "compute_load": loads,
        "expert_count": counts,
    }


def _seed_key(
    metrics: Mapping[str, Any], demand: Mapping[int, int], max_load_ratio: float
):
    cap = max(
        max(demand.values(), default=0),
        sum(demand.values()) / len(metrics["compute_load"]) * max_load_ratio,
    )
    violation = sum(max(0, load - cap) for load in metrics["compute_load"])
    return (
        violation,
        metrics["topology_objective"],
        sum(load * load for load in metrics["compute_load"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw, layers, compact = _load_input(args.input)
    source_ep = args.source_ep
    if source_ep is None and isinstance(raw, dict):
        source_ep = raw.get("num_ranks")
    source_ep = int(source_ep or args.num_ranks)
    output_layers = []

    for position, (gate, entries) in enumerate(layers):
        tokens = _materialize_layer_tokens(
            gate,
            entries,
            compact_input=compact,
            source_ep=source_ep,
            target_ep=args.num_ranks,
        )
        experts, demand = _summarize_tokens(tokens)
        capacity = args.slots_per_rank or ceil(len(experts) / args.num_ranks)
        baseline = _baseline_homes(experts, args.num_ranks, args.baseline)
        balanced = _balanced_initial_placement(
            experts,
            demand,
            num_ranks=args.num_ranks,
            slots_per_rank=capacity,
        )
        graph = build_co_routing_graph(tokens, experts=experts)
        pairwise = CoRoutingGraphSolver(
            num_ranks=args.num_ranks,
            slots_per_rank=capacity,
            max_rounds=args.rounds,
        ).solve(graph)
        current = refine_hypergraph_placement(
            tokens,
            pairwise.rank_by_expert,
            num_ranks=args.num_ranks,
            max_rounds=args.rounds,
            objective="source-agnostic",
        )
        grace = grace_hierarchical_placement(
            graph,
            num_ranks=args.num_ranks,
            ranks_per_node=args.ranks_per_node,
            nonuniform_ratio=args.grace_ratio,
        )

        candidates = {
            "balanced": balanced.rank_by_expert,
            "pairwise": pairwise.rank_by_expert,
            "grace": grace.rank_by_expert,
        }
        candidate_metrics = {
            name: _metrics(
                tokens,
                demand,
                placement,
                num_ranks=args.num_ranks,
                ranks_per_node=args.ranks_per_node,
                rdma_cost=args.rdma_cost,
            )
            for name, placement in candidates.items()
        }
        seed_name = min(
            candidates,
            key=lambda name: _seed_key(
                candidate_metrics[name], demand, args.max_load_ratio
            ),
        )
        grace_counts = [len(experts) for experts in grace.experts_by_rank.values()]
        proposed = refine_load_constrained_hypergraph_placement(
            tokens,
            candidates[seed_name],
            num_ranks=args.num_ranks,
            ranks_per_node=args.ranks_per_node,
            rdma_cost=args.rdma_cost,
            max_load_ratio=args.max_load_ratio,
            min_experts_per_rank=min(grace_counts),
            max_experts_per_rank=max(grace_counts),
            max_rounds=args.rounds,
        )
        placements = {
            "baseline": baseline,
            "pairwise": pairwise.rank_by_expert,
            "current_hypergraph": current.rank_by_expert,
            "grace": grace.rank_by_expert,
            "proposed": proposed.rank_by_expert,
        }
        metrics = {
            name: _metrics(
                tokens,
                demand,
                placement,
                num_ranks=args.num_ranks,
                ranks_per_node=args.ranks_per_node,
                rdma_cost=args.rdma_cost,
            )
            for name, placement in placements.items()
        }
        output_layers.append(
            {
                "gate": gate,
                "layer": _layer_number(gate, position),
                "tokens": sum(token.count for token in tokens),
                "bundles": len(tokens),
                "experts": len(experts),
                "proposed_seed": seed_name,
                "proposed_iterations": proposed.iterations,
                "metrics": metrics,
                "rank_by_expert": dict(proposed.rank_by_expert),
                "experts_by_rank": {
                    str(rank): list(rank_experts)
                    for rank, rank_experts in proposed.experts_by_rank.items()
                },
            }
        )

    return {
        "input": args.input,
        "source_ep": source_ep,
        "num_ranks": args.num_ranks,
        "ranks_per_node": args.ranks_per_node,
        "grace_ratio": args.grace_ratio,
        "max_load_ratio": args.max_load_ratio,
        "rdma_cost": args.rdma_cost,
        "layers": output_layers,
    }


def _print(result: Mapping[str, Any]) -> None:
    rows = [["layer", "method", "remote", "dest", "topo", "comp", "experts/rank"]]
    for layer in result["layers"]:
        for method, metric in layer["metrics"].items():
            rows.append(
                [
                    str(layer["layer"]),
                    method,
                    str(metric["remote"]),
                    str(metric["destination_copies"]),
                    f"{metric['topology_objective']:.0f}",
                    f"{metric['compute_imbalance']:.2f}x",
                    f"{min(metric['expert_count'])}-{max(metric['expert_count'])}",
                ]
            )
    print(_format_table(rows))
    seeds = ", ".join(
        f"L{layer['layer']}={layer['proposed_seed']}" for layer in result["layers"]
    )
    print(f"proposed seeds: {seeds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--source-ep", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=None)
    parser.add_argument("--slots-per-rank", type=int, default=None)
    parser.add_argument("--grace-ratio", type=float, default=0.15)
    parser.add_argument("--max-load-ratio", type=float, default=1.2)
    parser.add_argument("--rdma-cost", type=float, default=4.0)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument(
        "--baseline", choices=["round-robin", "contiguous"], default="round-robin"
    )
    parser.add_argument("--save-placement", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.ranks_per_node = args.ranks_per_node or args.num_ranks
    if args.num_ranks < 1 or args.num_ranks % args.ranks_per_node:
        parser.error("--ranks-per-node must be positive and divide --num-ranks")
    if args.source_ep is not None and args.source_ep < 1:
        parser.error("--source-ep must be positive")
    if args.slots_per_rank is not None and args.slots_per_rank < 1:
        parser.error("--slots-per-rank must be positive")
    if not 0 <= args.grace_ratio < 1:
        parser.error("--grace-ratio must be in [0, 1)")
    if args.max_load_ratio < 1 or args.rdma_cost < 1 or args.rounds < 0:
        parser.error("load ratio and RDMA cost must be >= 1; rounds must be >= 0")

    result = run(args)
    if args.save_placement:
        _save_placements(args.save_placement, result)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print(result)


if __name__ == "__main__":
    main()
