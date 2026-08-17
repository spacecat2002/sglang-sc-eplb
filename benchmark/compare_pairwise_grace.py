#!/usr/bin/env python3
"""Compare Pairwise, GRACE-MoE, and source-aware hypergraph placement.

Input is the JSON or compact ``.pt`` trace produced by ``benchmark_ep_trace.py``.

Example:

    PYTHONPATH=python python benchmark/compare_pairwise_grace.py \
        --input trace.pt --num-ranks 8 --ranks-per-node 4
"""

from __future__ import annotations

import argparse
import json
import re
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    RoutedToken,
    build_co_routing_graph,
    evaluate_pairwise_cut,
    evaluate_primary_remote,
    evaluate_weighted_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_hierarchical_placement
from sglang.srt.eplb.hypergraph_expert_placement import SourceAwareHypergraphSolver


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _json_layers(raw: Any) -> list[tuple[str, Any, bool]]:
    if isinstance(raw, list):
        return [("layer0", raw, False)]
    if not isinstance(raw, dict) or not isinstance(raw.get("layers"), list):
        raise ValueError("input must be a bundle list or an object containing layers")
    result = []
    for index, layer in enumerate(raw["layers"]):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{index}] must be an object")
        if layer.get("bundles"):
            result.append(
                (
                    str(layer.get("gate", f"layer{index}")),
                    layer["bundles"],
                    False,
                )
            )
    if not result:
        raise ValueError("input contains no non-empty layers")
    return result


def _load(path: str) -> tuple[Any, list[tuple[str, Any, bool]]]:
    if Path(path).suffix == ".pt":
        from sglang.srt.eplb.moe_bundle_trace import load_compact_trace

        raw = load_compact_trace(path)
        return raw, [(str(layer["gate"]), layer, True) for layer in raw["layers"]]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw, _json_layers(raw)


def _tokens(name: str, value: Any, compact: bool) -> list[RoutedToken]:
    if compact:
        entries = zip(
            value["source_rank"].tolist(),
            value["topk_experts"].tolist(),
            value["count"].tolist(),
        )
        return [
            RoutedToken(int(source), tuple(sorted(map(int, experts))), int(count))
            for source, experts, count in entries
        ]
    result = []
    for index, entry in enumerate(value):
        try:
            result.append(
                RoutedToken(
                    int(entry["source_rank"]),
                    tuple(sorted(int(expert) for expert in entry["topk_experts"])),
                    int(entry.get("count", 1)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle at {name}[{index}]: {entry!r}") from exc
    return result


def _reshard(
    tokens: Sequence[RoutedToken], source_ep: int, target_ep: int
) -> list[RoutedToken]:
    if source_ep < 1 or any(token.source_rank >= source_ep for token in tokens):
        raise ValueError("trace contains a source rank outside source EP")
    if source_ep == target_ep:
        return list(tokens)
    if source_ep < 1 or target_ep < source_ep or target_ep % source_ep:
        raise ValueError("target EP must be a multiple of source EP")
    fanout = target_ep // source_ep
    result = []
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


def _metrics(
    tokens: Sequence[RoutedToken],
    graph,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
) -> dict[str, Any]:
    loads = [0] * num_ranks
    counts = [0] * num_ranks
    for expert, rank in placement.items():
        loads[rank] += graph.demand[expert]
        counts[rank] += 1
    average = sum(loads) / num_ranks
    return {
        "remote": evaluate_primary_remote(tokens, placement),
        "weighted_remote": evaluate_weighted_remote(
            tokens,
            placement,
            ranks_per_node=ranks_per_node,
            rdma_cost=rdma_cost,
        ),
        "pairwise_cut": evaluate_pairwise_cut(graph, placement),
        "compute_imbalance": max(loads, default=0) / average if average else 0.0,
        "compute_load": loads,
        "expert_count": counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw, layers = _load(args.input)
    source_ep = args.source_ep
    if source_ep is None and isinstance(raw, dict):
        source_ep = raw.get("num_ranks")
    source_ep = int(source_ep or args.num_ranks)
    results = []
    for position, (gate, value, compact) in enumerate(layers):
        tokens = _reshard(_tokens(gate, value, compact), source_ep, args.num_ranks)
        graph = build_co_routing_graph(tokens)
        capacity = args.slots_per_rank or ceil(len(graph.experts) / args.num_ranks)
        placements = {}
        pairwise_iterations = None
        hypergraph_result = None
        if args.method in ("both", "all", "pairwise"):
            pairwise = CoRoutingGraphSolver(
                num_ranks=args.num_ranks,
                slots_per_rank=capacity,
                max_rounds=args.max_rounds,
                rerank_candidates=args.pairwise_candidates,
                max_compute_imbalance=args.pairwise_max_imbalance,
            ).solve(
                graph,
                routed_tokens=tokens,
                ranks_per_node=args.ranks_per_node,
                rdma_cost=args.rdma_cost,
            )
            placements["pairwise"] = pairwise.rank_by_expert
            pairwise_iterations = pairwise.iterations
        if args.method in ("both", "all", "grace"):
            grace = grace_hierarchical_placement(
                graph,
                num_ranks=args.num_ranks,
                ranks_per_node=args.ranks_per_node,
                nonuniform_ratio=args.grace_ratio,
            )
            placements["grace"] = grace.rank_by_expert
        if args.method in ("all", "hypergraph"):
            hypergraph_result = SourceAwareHypergraphSolver(
                num_ranks=args.num_ranks,
                slots_per_rank=capacity,
                max_rounds=args.hypergraph_rounds,
                restarts=args.hypergraph_restarts,
                candidates=args.hypergraph_candidates,
                load_weight=args.hypergraph_load_weight,
                max_compute_imbalance=args.pairwise_max_imbalance,
                seed=args.hypergraph_seed,
            ).solve(
                graph,
                tokens,
                ranks_per_node=args.ranks_per_node,
                rdma_cost=args.rdma_cost,
            )
            placements["hypergraph"] = hypergraph_result.rank_by_expert
        result = {
            "gate": gate,
            "layer": (
                int(match.group(1))
                if (match := _LAYER_PATTERN.search(gate))
                else position
            ),
            "tokens": sum(token.count for token in tokens),
            "bundles": len(tokens),
            "metrics": {
                name: _metrics(
                    tokens,
                    graph,
                    placement,
                    num_ranks=args.num_ranks,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                )
                for name, placement in placements.items()
            },
            "placements": placements,
        }
        if pairwise_iterations is not None:
            result["pairwise_iterations"] = pairwise_iterations
        if hypergraph_result is not None:
            result["hypergraph_iterations"] = hypergraph_result.iterations
            result["hypergraph_restarts"] = hypergraph_result.restarts
        results.append(result)
    return {
        "input": args.input,
        "source_ep": source_ep,
        "num_ranks": args.num_ranks,
        "ranks_per_node": args.ranks_per_node,
        "method": args.method,
        "pairwise_candidates": args.pairwise_candidates,
        "pairwise_max_imbalance": args.pairwise_max_imbalance,
        "grace_ratio": args.grace_ratio,
        "layers": results,
    }


def _table(rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _print(result: Mapping[str, Any]) -> None:
    rows = [["layer", "method", "remote", "weighted", "cut", "comp", "experts/rank"]]
    for layer in result["layers"]:
        for method, metric in layer["metrics"].items():
            rows.append(
                [
                    str(layer["layer"]),
                    method,
                    str(metric["remote"]),
                    f"{metric['weighted_remote']:.0f}",
                    str(metric["pairwise_cut"]),
                    f"{metric['compute_imbalance']:.2f}x",
                    f"{min(metric['expert_count'])}-{max(metric['expert_count'])}",
                ]
            )
    print(_table(rows))


def _save(path: str, result: Mapping[str, Any], method: str) -> None:
    payload = {
        layer["gate"]: {
            str(expert): rank
            for expert, rank in sorted(layer["placements"][method].items())
        }
        for layer in result["layers"]
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--source-ep", type=int)
    parser.add_argument("--ranks-per-node", type=int)
    parser.add_argument(
        "--method",
        choices=("both", "all", "pairwise", "grace", "hypergraph"),
        default="both",
    )
    parser.add_argument("--slots-per-rank", type=int)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--pairwise-candidates", type=int, default=32)
    parser.add_argument("--pairwise-max-imbalance", type=float, default=1.2)
    parser.add_argument("--hypergraph-rounds", type=int, default=8)
    parser.add_argument("--hypergraph-restarts", type=int, default=2)
    parser.add_argument("--hypergraph-candidates", type=int, default=128)
    parser.add_argument("--hypergraph-load-weight", type=float, default=0.5)
    parser.add_argument("--hypergraph-seed", type=int, default=0)
    parser.add_argument("--grace-ratio", type=float, default=0.15)
    parser.add_argument("--rdma-cost", type=float, default=4.0)
    parser.add_argument("--save-pairwise")
    parser.add_argument("--save-grace")
    parser.add_argument("--save-hypergraph")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.ranks_per_node = args.ranks_per_node or args.num_ranks
    if args.num_ranks < 1 or args.num_ranks % args.ranks_per_node:
        parser.error("--ranks-per-node must be positive and divide --num-ranks")
    if args.slots_per_rank is not None and args.slots_per_rank < 1:
        parser.error("--slots-per-rank must be positive")
    if args.max_rounds < 0 or not 0 <= args.grace_ratio < 1:
        parser.error("invalid --max-rounds or --grace-ratio")
    if args.pairwise_candidates < 1 or args.pairwise_max_imbalance < 1:
        parser.error("invalid Pairwise candidate count or imbalance limit")
    if args.rdma_cost < 1:
        parser.error("--rdma-cost must be at least 1")
    if args.hypergraph_rounds < 0 or args.hypergraph_restarts < 1:
        parser.error("invalid hypergraph rounds or restarts")
    if args.hypergraph_candidates < 1 or args.hypergraph_load_weight < 0:
        parser.error("invalid hypergraph candidates or load weight")
    if args.save_pairwise and args.method not in ("both", "all", "pairwise"):
        parser.error("--save-pairwise requires --method pairwise, both, or all")
    if args.save_grace and args.method not in ("both", "all", "grace"):
        parser.error("--save-grace requires --method grace, both, or all")
    if args.save_hypergraph and args.method not in ("all", "hypergraph"):
        parser.error("--save-hypergraph requires --method hypergraph or all")

    result = run(args)
    if args.save_pairwise:
        _save(args.save_pairwise, result, "pairwise")
    if args.save_grace:
        _save(args.save_grace, result, "grace")
    if args.save_hypergraph:
        _save(args.save_hypergraph, result, "hypergraph")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print(result)


if __name__ == "__main__":
    main()
