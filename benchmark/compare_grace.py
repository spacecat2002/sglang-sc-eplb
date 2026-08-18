#!/usr/bin/env python3
"""Run GRACE and optional CABLE offline expert placement on a routing trace."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sglang.srt.eplb.cable_expert_placement import (
    cable_expert_placement,
    evaluate_cable_placement,
)
from sglang.srt.eplb.expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    build_co_routing_graph,
    evaluate_primary_remote,
    evaluate_weighted_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_hierarchical_placement
from sglang.srt.eplb.hypergraph_expert_placement import hypergraph_expert_placement


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
                (str(layer.get("gate", f"layer{index}")), layer["bundles"], False)
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


def _tokens(
    name: str, value: Any, compact: bool, *, limit: int, seed: int
) -> tuple[list[RoutedToken] | RoutedArrays, int]:
    if compact:
        size = int(value["source_rank"].shape[0])
        if limit and size > limit:
            import torch

            index = torch.tensor(
                sorted(random.Random(seed).sample(range(size), limit)),
                dtype=torch.int64,
            )
            source = value["source_rank"].index_select(0, index)
            topk = value["topk_experts"].index_select(0, index)
            count = value["count"].index_select(0, index)
        else:
            source, topk, count = (
                value["source_rank"],
                value["topk_experts"],
                value["count"],
            )
        return RoutedArrays(source.numpy(), topk.numpy(), count.numpy()), size
    result = []
    for index, entry in enumerate(value):
        try:
            result.append(
                RoutedToken(
                    int(entry["source_rank"]),
                    tuple(sorted(int(e) for e in entry["topk_experts"])),
                    int(entry.get("count", 1)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle at {name}[{index}]: {entry!r}") from exc
    return result, len(result)


def _reshard(
    tokens: Sequence[RoutedToken] | RoutedArrays, source_ep: int, target_ep: int
) -> list[RoutedToken] | RoutedArrays:
    if isinstance(tokens, RoutedArrays):
        if source_ep < 1 or tokens.source_rank.max() >= source_ep:
            raise ValueError("trace contains a source rank outside source EP")
        if source_ep == target_ep:
            return tokens
        if target_ep < source_ep or target_ep % source_ep:
            raise ValueError("target EP must be a multiple of source EP")
        fanout = target_ep // source_ep
        offsets = np.tile(np.arange(fanout), len(tokens))
        counts = np.repeat(tokens.count // fanout, fanout)
        counts += offsets < np.repeat(tokens.count % fanout, fanout)
        keep = counts != 0
        return RoutedArrays(
            (np.repeat(tokens.source_rank, fanout) * fanout + offsets)[keep],
            np.repeat(tokens.topk_experts, fanout, axis=0)[keep],
            counts[keep],
        )
    if source_ep < 1 or any(token.source_rank >= source_ep for token in tokens):
        raise ValueError("trace contains a source rank outside source EP")
    if source_ep == target_ep:
        return list(tokens)
    if target_ep < source_ep or target_ep % source_ep:
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
                        token.source_rank * fanout + offset, token.topk_experts, count
                    )
                )
    return result


def _metrics(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
    compute_load: Sequence[int] | None = None,
) -> dict[str, Any]:
    cable_metrics = evaluate_cable_placement(tokens, placement, num_ranks=num_ranks)
    loads = (
        list(compute_load)
        if compute_load is not None
        else list(cable_metrics.compute_load)
    )
    counts = [0] * len(loads)
    for expert, rank in placement.items():
        counts[rank] += 1
    average = sum(loads) / len(loads)
    return {
        "remote": evaluate_primary_remote(tokens, placement),
        "weighted_remote": evaluate_weighted_remote(
            tokens, placement, ranks_per_node=ranks_per_node, rdma_cost=rdma_cost
        ),
        "compute_imbalance": max(loads, default=0) / average if average else 0.0,
        "compute_load": loads,
        "expert_count": counts,
        "max_pair_traffic": cable_metrics.max_pair_traffic,
        "max_ingress": cable_metrics.max_ingress,
        "max_egress": cable_metrics.max_egress,
    }


def _baseline_placement(experts: Sequence[int], num_ranks: int) -> dict[int, int]:
    return {
        expert: min(index * num_ranks // len(experts), num_ranks - 1)
        for index, expert in enumerate(experts)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw, layers = _load(args.input)
    source_ep = args.source_ep
    if source_ep is None and isinstance(raw, dict):
        source_ep = raw.get("num_ranks")
    source_ep = int(source_ep or args.num_ranks)
    results = []
    for position, (gate, value, compact) in enumerate(layers):
        tokens, bundle_count = _tokens(
            gate,
            value,
            compact,
            limit=args.optimizer_bundles if compact else 0,
            seed=args.seed + position,
        )
        tokens = _reshard(tokens, source_ep, args.num_ranks)
        experts = (
            range(int(value["topk_experts"].max().item()) + 1) if compact else None
        )
        if experts is None:
            experts = tuple(
                sorted({expert for token in tokens for expert in token.topk_experts})
            )
        experts = tuple(experts)
        baseline_placement = _baseline_placement(experts, args.num_ranks)
        needs_grace = not (args.cable_only or args.hypergraph_only)
        grace_started = time.perf_counter() if needs_grace else None
        graph = (
            build_co_routing_graph(tokens, experts=experts)
            if needs_grace
            else None
        )
        placement = None
        grace_seconds = None
        if needs_grace:
            placement = grace_hierarchical_placement(
                graph,
                num_ranks=args.num_ranks,
                ranks_per_node=args.ranks_per_node,
                nonuniform_ratio=args.grace_ratio,
                equal_experts=args.grace_equal_experts,
            )
            grace_seconds = time.perf_counter() - grace_started
        started = time.perf_counter()
        grace_refined = (
            hypergraph_expert_placement(
                tokens,
                experts=tuple(placement.rank_by_expert),
                num_ranks=args.num_ranks,
                capacity_ratio=(
                    0.0
                    if args.grace_equal_experts
                    else args.grace_refine_capacity_ratio
                    if args.grace_refine_capacity_ratio is not None
                    else args.grace_ratio
                ),
                compute_imbalance_limit=args.hypergraph_compute_limit,
                starts=1,
                refine_rounds=args.grace_refine_rounds,
                remote_budget=0.0,
                initial_placement=placement.rank_by_expert,
                align_groups=True,
                swap_rounds=args.grace_refine_swaps,
                swap_candidate_partners=args.grace_refine_partners,
                swap_allow_load_worsening=(
                    args.grace_refine_allow_load_worsening
                ),
                swap_max_compute_imbalance=(
                    args.grace_refine_swap_compute_limit
                ),
                swap_exhaustive=args.grace_refine_exhaustive_swaps,
            )
            if args.grace_refine
            else None
        )
        grace_refine_seconds = time.perf_counter() - started if grace_refined else None
        started = time.perf_counter()
        cable = (
            cable_expert_placement(
                tokens,
                experts=graph.experts if graph is not None else experts,
                num_ranks=args.num_ranks,
                congestion_weight=args.cable_congestion_weight,
                load_weight=args.cable_load_weight,
                refine_swaps=args.cable_refine_swaps,
                refine_strategy=args.cable_refine_strategy,
                remote_budget=args.cable_remote_budget,
                capacity_ratio=args.cable_capacity_ratio,
                compute_refine_moves=args.cable_compute_moves,
                compute_imbalance_limit=args.cable_compute_limit,
            )
            if args.cable
            else None
        )
        cable_seconds = time.perf_counter() - started if cable is not None else None
        started = time.perf_counter()
        hypergraph = (
            hypergraph_expert_placement(
                tokens,
                experts=experts,
                num_ranks=args.num_ranks,
                capacity_ratio=args.hypergraph_capacity_ratio,
                compute_imbalance_limit=args.hypergraph_compute_limit,
                starts=args.hypergraph_starts,
                refine_rounds=args.hypergraph_refine_rounds,
                remote_budget=args.hypergraph_remote_budget,
            )
            if args.hypergraph
            else None
        )
        hypergraph_seconds = (
            time.perf_counter() - started if hypergraph is not None else None
        )
        total_tokens = (
            int(value["count"].sum().item())
            if compact
            else sum(t.count for t in tokens)
        )
        result = {
            "gate": gate,
            "layer": int(match.group(1))
            if (match := _LAYER_PATTERN.search(gate))
            else position,
            "tokens": total_tokens,
            "bundles": bundle_count,
            "optimizer_bundles": len(tokens),
            "baseline": {
                "metrics": _metrics(
                    tokens,
                    baseline_placement,
                    num_ranks=args.num_ranks,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                ),
                "placement": baseline_placement,
                "solve_seconds": 0.0,
            },
        }
        if placement is not None:
            result.update(
                {
                    "metrics": _metrics(
                        tokens,
                        placement.rank_by_expert,
                        num_ranks=args.num_ranks,
                        ranks_per_node=args.ranks_per_node,
                        rdma_cost=args.rdma_cost,
                    ),
                    "placement": placement.rank_by_expert,
                    "solve_seconds": grace_seconds,
                }
            )
        if cable is not None:
            result["cable"] = {
                "objective": cable.objective,
                "metrics": _metrics(
                    tokens,
                    cable.rank_by_expert,
                    num_ranks=args.num_ranks,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    compute_load=cable.metrics.compute_load,
                ),
                "placement": cable.rank_by_expert,
                "solve_seconds": cable_seconds,
            }
        if hypergraph is not None:
            result["hypergraph"] = {
                "metrics": _metrics(
                    tokens,
                    hypergraph["rank_by_expert"],
                    num_ranks=args.num_ranks,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    compute_load=hypergraph["metrics"].compute_load,
                ),
                "placement": hypergraph["rank_by_expert"],
                "solve_seconds": hypergraph_seconds,
            }
        if grace_refined is not None:
            result["grace_refine"] = {
                "metrics": _metrics(
                    tokens,
                    grace_refined["rank_by_expert"],
                    num_ranks=args.num_ranks,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    compute_load=grace_refined["metrics"].compute_load,
                ),
                "placement": grace_refined["rank_by_expert"],
                "solve_seconds": grace_seconds + grace_refine_seconds,
                "refine_seconds": grace_refine_seconds,
            }
        results.append(result)
    return {
        "input": args.input,
        "method": (
            "cable"
            if args.cable_only
            else "hypergraph"
            if args.hypergraph_only
            else "+".join(
                [
                    "grace",
                    *(["grace-refine"] if args.grace_refine else []),
                    *(["cable"] if args.cable else []),
                    *(["hypergraph"] if args.hypergraph else []),
                ]
            )
        ),
        "source_ep": source_ep,
        "num_ranks": args.num_ranks,
        "ranks_per_node": args.ranks_per_node,
        "grace_ratio": args.grace_ratio,
        "optimizer_bundles": args.optimizer_bundles,
        "layers": results,
    }


def _table(rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows
    )


def _print(result: Mapping[str, Any]) -> None:
    if any(layer["optimizer_bundles"] < layer["bundles"] for layer in result["layers"]):
        print(
            f"[trace] placements optimized on at most "
            f"{result['optimizer_bundles']} bundles/layer"
        )
    rows = [
        [
            "layer",
            "method",
            "remote",
            "weighted",
            "max-pair",
            "max-ingress",
            "max-egress",
            "comp",
            "experts/rank",
            "solve-ms",
        ]
    ]
    for layer in result["layers"]:
        metrics = [("baseline", layer["baseline"]["metrics"], 0.0)]
        if "metrics" in layer:
            metrics.append(("grace", layer["metrics"], layer["solve_seconds"]))
        if "cable" in layer:
            metrics.append(
                (
                    "cable",
                    layer["cable"]["metrics"],
                    layer["cable"]["solve_seconds"],
                )
            )
        if "hypergraph" in layer:
            metrics.append(
                (
                    "hypergraph",
                    layer["hypergraph"]["metrics"],
                    layer["hypergraph"]["solve_seconds"],
                )
            )
        if "grace_refine" in layer:
            metrics.append(
                (
                    "grace-refine",
                    layer["grace_refine"]["metrics"],
                    layer["grace_refine"]["solve_seconds"],
                )
            )
        for method, metric, solve_seconds in metrics:
            rows.append(
                [
                    str(layer["layer"]),
                    method,
                    str(metric["remote"]),
                    f"{metric['weighted_remote']:.0f}",
                    str(metric["max_pair_traffic"]),
                    str(metric["max_ingress"]),
                    str(metric["max_egress"]),
                    f"{metric['compute_imbalance']:.2f}x",
                    f"{min(metric['expert_count'])}-{max(metric['expert_count'])}",
                    f"{solve_seconds * 1000:.1f}",
                ]
            )
    print(_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--source-ep", type=int)
    parser.add_argument("--ranks-per-node", type=int)
    parser.add_argument("--grace-ratio", type=float, default=0.15)
    parser.add_argument("--grace-equal-experts", action="store_true")
    parser.add_argument("--optimizer-bundles", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rdma-cost", type=float, default=1.0)
    parser.add_argument("--cable", action="store_true")
    parser.add_argument("--cable-only", action="store_true")
    parser.add_argument("--hypergraph", action="store_true")
    parser.add_argument("--hypergraph-only", action="store_true")
    parser.add_argument("--grace-refine", action="store_true")
    parser.add_argument("--grace-refine-rounds", type=int, default=4)
    parser.add_argument("--grace-refine-swaps", type=int, default=2)
    parser.add_argument("--grace-refine-partners", type=int, default=8)
    parser.add_argument(
        "--grace-refine-allow-load-worsening", action="store_true"
    )
    parser.add_argument("--grace-refine-swap-compute-limit", type=float, default=1.25)
    parser.add_argument("--grace-refine-exhaustive-swaps", action="store_true")
    parser.add_argument("--grace-refine-capacity-ratio", type=float)
    parser.add_argument("--cable-congestion-weight", type=float, default=0.25)
    parser.add_argument("--cable-load-weight", type=float, default=0.25)
    parser.add_argument("--cable-refine-swaps", type=int, default=2)
    parser.add_argument(
        "--cable-refine-strategy",
        choices=("remote", "balanced"),
        default="balanced",
    )
    parser.add_argument("--cable-remote-budget", type=float, default=0.03)
    parser.add_argument("--cable-capacity-ratio", type=float, default=0.15)
    parser.add_argument("--cable-compute-moves", type=int, default=2)
    parser.add_argument("--cable-compute-limit", type=float, default=2.0)
    parser.add_argument("--hypergraph-capacity-ratio", type=float, default=0.15)
    parser.add_argument("--hypergraph-compute-limit", type=float, default=2.0)
    parser.add_argument("--hypergraph-starts", type=int, default=4)
    parser.add_argument("--hypergraph-refine-rounds", type=int, default=4)
    parser.add_argument("--hypergraph-remote-budget", type=float, default=0.01)
    parser.add_argument("--save-grace")
    parser.add_argument("--save-cable")
    parser.add_argument("--save-hypergraph")
    parser.add_argument("--save-grace-refine")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cable_only:
        args.cable = True
    if args.hypergraph_only:
        args.hypergraph = True
    args.ranks_per_node = args.ranks_per_node or args.num_ranks
    if args.num_ranks < 1 or args.num_ranks % args.ranks_per_node:
        parser.error("--ranks-per-node must be positive and divide --num-ranks")
    if (
        args.optimizer_bundles < 0
        or not 0 <= args.grace_ratio < 1
        or args.rdma_cost < 1
        or args.cable_refine_swaps < 0
        or args.cable_compute_moves < 0
        or not 0 <= args.cable_remote_budget <= 1
        or not 0 <= args.cable_capacity_ratio < 1
        or args.cable_compute_limit < 1
        or not 0 <= args.hypergraph_capacity_ratio < 1
        or args.hypergraph_compute_limit < 1
        or args.hypergraph_starts < 1
        or args.hypergraph_refine_rounds < 0
        or args.grace_refine_rounds < 0
        or args.grace_refine_swaps < 0
        or args.grace_refine_partners < 1
        or args.grace_refine_swap_compute_limit < 1
        or (
            args.grace_refine_capacity_ratio is not None
            and not 0 <= args.grace_refine_capacity_ratio < 1
        )
        or not 0 <= args.hypergraph_remote_budget <= 1
        or min(args.cable_congestion_weight, args.cable_load_weight) < 0
    ):
        parser.error("invalid placement parameters")
    if args.save_cable and not args.cable:
        parser.error("--save-cable requires --cable")
    if args.save_grace and (args.cable_only or args.hypergraph_only):
        parser.error("--save-grace requires the GRACE placement")
    if args.save_hypergraph and not args.hypergraph:
        parser.error("--save-hypergraph requires --hypergraph")
    if args.grace_refine and (args.cable_only or args.hypergraph_only):
        parser.error("--grace-refine requires the GRACE placement")
    if args.grace_equal_experts and (args.cable_only or args.hypergraph_only):
        parser.error("--grace-equal-experts requires the GRACE placement")
    if args.save_grace_refine and not args.grace_refine:
        parser.error("--save-grace-refine requires --grace-refine")
    result = run(args)
    if args.save_grace:
        Path(args.save_grace).write_text(
            json.dumps(
                {
                    layer["gate"]: {
                        str(e): r for e, r in sorted(layer["placement"].items())
                    }
                    for layer in result["layers"]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.save_cable:
        Path(args.save_cable).write_text(
            json.dumps(
                {
                    layer["gate"]: {
                        str(e): r
                        for e, r in sorted(layer["cable"]["placement"].items())
                    }
                    for layer in result["layers"]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.save_hypergraph:
        Path(args.save_hypergraph).write_text(
            json.dumps(
                {
                    layer["gate"]: {
                        str(e): r
                        for e, r in sorted(layer["hypergraph"]["placement"].items())
                    }
                    for layer in result["layers"]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.save_grace_refine:
        Path(args.save_grace_refine).write_text(
            json.dumps(
                {
                    layer["gate"]: {
                        str(e): r
                        for e, r in sorted(layer["grace_refine"]["placement"].items())
                    }
                    for layer in result["layers"]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print(result)


if __name__ == "__main__":
    main()
