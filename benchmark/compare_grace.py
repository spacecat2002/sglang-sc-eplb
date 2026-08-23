#!/usr/bin/env python3
"""Compare baseline, GRACE, and GRACE+ on an offline routing trace."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sglang.srt.eplb.expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    build_co_routing_graph,
    evaluate_primary_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_expert_placement
from sglang.srt.eplb.grace_plus_expert_placement import (
    grace_plus_expert_placement,
)
from sglang.srt.eplb.grace_plus_refinement import evaluate_placement
from sglang.srt.eplb.grace_plus_replication import (
    balance_replica_compute,
    replicate_hot_experts,
)


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
    if source_ep < 1:
        raise ValueError("source EP must be positive")
    if isinstance(tokens, RoutedArrays):
        if tokens.source_rank.max() >= source_ep:
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
    if any(token.source_rank >= source_ep for token in tokens):
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


def _baseline_placement(experts: Sequence[int], num_ranks: int) -> dict[int, int]:
    return {
        expert: min(index * num_ranks // len(experts), num_ranks - 1)
        for index, expert in enumerate(experts)
    }


def _metrics(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    compute_load: Sequence[int] | None = None,
) -> dict[str, Any]:
    placement_metrics = evaluate_placement(tokens, placement, num_ranks=num_ranks)
    loads = list(compute_load or placement_metrics.compute_load)
    counts = [0] * num_ranks
    for rank in placement.values():
        counts[rank] += 1
    average = sum(loads) / num_ranks
    return {
        "remote": evaluate_primary_remote(tokens, placement),
        "compute_imbalance": max(loads, default=0) / average if average else 0.0,
        "compute_load": loads,
        "expert_count": counts,
        "max_pair_traffic": placement_metrics.max_pair_traffic,
        "max_ingress": placement_metrics.max_ingress,
        "max_egress": placement_metrics.max_egress,
    }


def _replica_metrics(
    metrics: Any, placement: Mapping[int, Sequence[int]]
) -> dict[str, Any]:
    loads = list(metrics.compute_load)
    counts = [0] * len(loads)
    for ranks in placement.values():
        for rank in ranks:
            counts[rank] += 1
    average = sum(loads) / len(loads)
    return {
        "remote": metrics.remote,
        "compute_imbalance": max(loads, default=0) / average if average else 0.0,
        "compute_load": loads,
        "expert_count": counts,
        "max_pair_traffic": metrics.max_pair_traffic,
        "max_ingress": metrics.max_ingress,
        "max_egress": metrics.max_egress,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw, layers = _load(args.input)
    source_ep = int(
        args.source_ep
        or (raw.get("num_ranks") if isinstance(raw, dict) else args.num_ranks)
        or args.num_ranks
    )
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
            tuple(range(int(value["topk_experts"].max().item()) + 1))
            if compact
            else tuple(
                sorted({expert for token in tokens for expert in token.topk_experts})
            )
        )
        baseline_placement = _baseline_placement(experts, args.num_ranks)
        started = time.perf_counter()
        graph = build_co_routing_graph(
            tokens, experts=experts, source_affinity_weight=args.source_affinity_weight
        )
        grace = grace_expert_placement(
            graph,
            num_ranks=args.num_ranks,
            nonuniform_ratio=args.grace_ratio,
            equal_experts=args.equal_experts,
        )
        grace_seconds = time.perf_counter() - started
        started = time.perf_counter()
        grace_plus = grace_plus_expert_placement(
            tokens,
            experts=experts,
            num_ranks=args.num_ranks,
            capacity_ratio=0.0 if args.equal_experts else args.capacity_ratio,
            compute_imbalance_limit=args.compute_limit,
            refine_rounds=args.rounds,
            initial_placement=grace.rank_by_expert,
            align_groups=True,
            swap_rounds=args.swaps,
            swap_candidate_partners=args.partners,
            swap_allow_load_worsening=args.allow_load_worsening,
            swap_max_compute_imbalance=args.swap_compute_limit,
            objective=args.objective,
        )
        refinement_seconds = time.perf_counter() - started
        replication = replicate_hot_experts(
            tokens,
            grace_plus["rank_by_expert"],
            num_ranks=args.num_ranks,
            objective=args.objective,
            hot_experts=args.hot_experts,
            candidate_ranks=args.replica_candidates,
            compute_imbalance_limit=args.replica_compute_limit,
            max_extra_per_rank=args.max_comm_expert_per_rank,
        )
        balanced = balance_replica_compute(
            tokens,
            replication,
            num_ranks=args.num_ranks,
            max_extra_per_rank=args.max_comp_expert_per_rank,
            communication_budget_ratio=args.communication_budget_ratio,
        )
        plus_seconds = time.perf_counter() - started
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
        }
        result["baseline"] = {
            "metrics": _metrics(
                tokens,
                baseline_placement,
                num_ranks=args.num_ranks,
            ),
            "placement": baseline_placement,
            "solve_seconds": 0.0,
        }
        result["grace"] = {
            "metrics": _metrics(
                tokens,
                grace.rank_by_expert,
                num_ranks=args.num_ranks,
            ),
            "placement": grace.rank_by_expert,
            "solve_seconds": grace_seconds,
        }
        result["grace+"] = {
            "metrics": _replica_metrics(balanced.metrics, balanced.replicas_by_expert),
            "placement": balanced.replicas_by_expert,
            "routing": balanced.routing_by_source,
            "quota": balanced.quota_by_source,
            "extra_copies": replication.extra_copies,
            "balance_copies": balanced.balance_copies,
            "solve_seconds": grace_seconds + plus_seconds,
            "refine_seconds": refinement_seconds,
            "replication_seconds": plus_seconds - refinement_seconds,
        }
        results.append(result)
    return {
        "input": args.input,
        "method": "baseline+grace+grace+",
        "objective": args.objective,
        "communication_budget_ratio": args.communication_budget_ratio,
        "source_ep": source_ep,
        "num_ranks": args.num_ranks,
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
            f"[trace] placements optimized on at most {result['optimizer_bundles']} bundles/layer"
        )
    rows = [
        [
            "layer",
            "method",
            "remote",
            "max-pair",
            "max-ingress",
            "max-egress",
            "comp",
            "experts/rank",
            "solve-ms",
        ]
    ]
    for layer in result["layers"]:
        for method in ("baseline", "grace", "grace+"):
            metric = layer[method]["metrics"]
            rows.append(
                [
                    str(layer["layer"]),
                    method,
                    str(metric["remote"]),
                    str(metric["max_pair_traffic"]),
                    str(metric["max_ingress"]),
                    str(metric["max_egress"]),
                    f"{metric['compute_imbalance']:.2f}x",
                    f"{min(metric['expert_count'])}-{max(metric['expert_count'])}",
                    f"{layer[method]['solve_seconds'] * 1000:.1f}",
                ]
            )
    print(_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--source-ep", type=int)
    parser.add_argument("--grace-ratio", type=float, default=0.15)
    parser.add_argument("--source-affinity-weight", type=float, default=0.0)
    parser.add_argument("--equal-experts", action="store_true")
    parser.add_argument("--optimizer-bundles", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective", choices=("remote", "ingress-egress"), default="ingress-egress"
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--swaps", type=int, default=1)
    parser.add_argument("--partners", type=int, default=8)
    parser.add_argument("--capacity-ratio", type=float, default=0.15)
    parser.add_argument("--compute-limit", type=float, default=1.0)
    parser.add_argument("--allow-load-worsening", action="store_true")
    parser.add_argument("--swap-compute-limit", type=float)
    parser.add_argument("--hot-experts", type=int, default=16)
    parser.add_argument("--replica-candidates", type=int, default=4)
    parser.add_argument("--replica-compute-limit", type=float, default=1.25)
    parser.add_argument("--max-comm-expert-per-rank", type=int, default=0)
    parser.add_argument("--max-comp-expert-per-rank", type=int, default=0)
    parser.add_argument(
        "--communication-budget-ratio",
        type=float,
        help=(
            "maximum communication metrics as a multiple of the placement "
            "before compute balancing (disabled by default)"
        ),
    )
    parser.add_argument("--save-grace")
    parser.add_argument("--save-grace-plus")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if (
        args.num_ranks < 1
        or args.optimizer_bundles < 0
        or not 0 <= args.grace_ratio < 1
        or args.source_affinity_weight < 0
        or args.rounds < 0
        or args.swaps < 0
        or args.partners < 1
        or not 0 <= args.capacity_ratio < 1
        or args.compute_limit < 1
        or args.hot_experts < 0
        or args.replica_candidates < 1
        or args.max_comm_expert_per_rank < 0
        or args.max_comp_expert_per_rank < 0
        or (
            args.communication_budget_ratio is not None
            and (
                not np.isfinite(args.communication_budget_ratio)
                or args.communication_budget_ratio < 0
            )
        )
        or args.replica_compute_limit < 1
        or (args.swap_compute_limit is not None and args.swap_compute_limit < 1)
    ):
        parser.error("invalid placement parameters")
    result = run(args)
    if args.save_grace:
        Path(args.save_grace).write_text(
            json.dumps(
                {
                    layer["gate"]: {
                        str(e): r
                        for e, r in sorted(layer["grace"]["placement"].items())
                    }
                    for layer in result["layers"]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.save_grace_plus:
        Path(args.save_grace_plus).write_text(
            json.dumps(
                {
                    layer["gate"]: dict(
                        replicas={
                            str(e): list(ranks)
                            for e, ranks in sorted(layer["grace+"]["placement"].items())
                        },
                        routing=[list(row) for row in layer["grace+"]["routing"]],
                        **(
                            {"quota": layer["grace+"]["quota"]}
                            if layer["grace+"]["quota"] is not None
                            else {}
                        ),
                    )
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
