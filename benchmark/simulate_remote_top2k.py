#!/usr/bin/env python3
"""Simulate per-rank remote Top-2K replicas on a compact routing trace."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from compare_grace import _baseline_placement, _load, _table, _tokens
from sglang.srt.eplb.grace_plus_replication import (
    balance_replica_compute,
    evaluate_replicated_placement,
    replicate_source_top_experts,
)


def _rows(
    layer: int,
    method: str,
    metrics,
    copies: Sequence[int],
    elapsed: float,
    balance_copies: int = 0,
):
    average = sum(metrics.compute_load) / len(metrics.compute_load)
    return [
        str(layer),
        method,
        str(metrics.remote),
        str(metrics.max_pair_traffic),
        str(metrics.max_ingress),
        str(metrics.max_egress),
        f"{max(metrics.compute_load) / average if average else 0:.2f}x",
        f"{min(copies)}-{max(copies)}",
        str(balance_copies),
        f"{elapsed * 1000:.1f}",
    ]


def _plan_entry(placement):
    return {
        "replicas": {
            str(expert): list(ranks)
            for expert, ranks in sorted(placement.replicas_by_expert.items())
        },
        "routing": [list(row) for row in placement.routing_by_source],
        "quota": placement.quota_by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="compact routing trace (.pt)")
    parser.add_argument(
        "--max-extra-experts-per-rank",
        type=int,
        help="remote expert replica limit per rank (default: 2 * trace Top-K)",
    )
    parser.add_argument(
        "--compute-imbalance-limit",
        type=float,
        help="maximum rank load / average load; omit for original all-local Top-N",
    )
    parser.add_argument(
        "--max-compute-extra-experts-per-rank",
        type=int,
        default=0,
        help="additional replicas per rank used only to balance compute (default: 0)",
    )
    parser.add_argument(
        "--communication-budget-ratio",
        type=float,
        help=(
            "communication budget relative to the source-top placement; "
            "disabled by default"
        ),
    )
    parser.add_argument(
        "--output-plan",
        help="write replicas/routing/quota JSON for benchmark_a2a_plan.py",
    )
    args = parser.parse_args()
    if Path(args.input).suffix != ".pt":
        parser.error("--input must be a compact .pt routing trace")
    if (
        args.max_extra_experts_per_rank is not None
        and args.max_extra_experts_per_rank < 0
    ):
        parser.error("--max-extra-experts-per-rank must be non-negative")
    if args.compute_imbalance_limit is not None and (
        not math.isfinite(args.compute_imbalance_limit)
        or args.compute_imbalance_limit < 1
    ):
        parser.error("--compute-imbalance-limit must be at least 1")
    if args.max_compute_extra_experts_per_rank < 0:
        parser.error("--max-compute-extra-experts-per-rank must be non-negative")
    if args.communication_budget_ratio is not None and (
        not math.isfinite(args.communication_budget_ratio)
        or args.communication_budget_ratio < 0
    ):
        parser.error("--communication-budget-ratio must be non-negative")
    raw, layers = _load(args.input)
    num_ranks = int(raw["num_ranks"])
    top_k = int(raw["top_k"])
    max_extra = (
        args.max_extra_experts_per_rank
        if args.max_extra_experts_per_rank is not None
        else 2 * top_k
    )
    budget_ratio = args.communication_budget_ratio
    rows = [
        [
            "layer",
            "method",
            "remote",
            "max-pair",
            "max-ingress",
            "max-egress",
            "comp",
            "extra/rank",
            "compute-copies",
            "eval-ms",
        ]
    ]
    plan = {}
    for layer, (gate, value, compact) in enumerate(layers):
        tokens, _ = _tokens(gate, value, compact, limit=0, seed=0)
        experts = tuple(range(int(value["topk_experts"].max().item()) + 1))
        primary = _baseline_placement(experts, num_ranks)

        started = time.perf_counter()
        baseline = evaluate_replicated_placement(tokens, primary, num_ranks=num_ranks)
        rows.append(
            _rows(
                layer,
                "baseline",
                baseline,
                [0] * num_ranks,
                time.perf_counter() - started,
            )
        )

        started = time.perf_counter()
        optimized = replicate_source_top_experts(
            tokens,
            primary,
            num_ranks=num_ranks,
            max_extra_per_rank=max_extra,
            compute_imbalance_limit=args.compute_imbalance_limit,
        )
        if args.max_compute_extra_experts_per_rank:
            optimized = balance_replica_compute(
                tokens,
                optimized,
                num_ranks=num_ranks,
                max_extra_per_rank=args.max_compute_extra_experts_per_rank,
                communication_budget_ratio=budget_ratio,
            )
        if args.output_plan:
            plan[gate] = _plan_entry(optimized)
        copies = [0] * num_ranks
        for ranks in optimized.replicas_by_expert.values():
            for rank in ranks[1:]:
                copies[rank] += 1
        rows.append(
            _rows(
                layer,
                (
                    f"remote-top{max_extra}+compute"
                    if args.max_compute_extra_experts_per_rank
                    else f"remote-top{max_extra}"
                ),
                optimized.metrics,
                copies,
                time.perf_counter() - started,
                optimized.balance_copies,
            )
        )

    compute = (
        f"{args.compute_imbalance_limit:.2f}x"
        if args.compute_imbalance_limit is not None
        else "none (all-local)"
    )
    print(
        f"trace={args.input}  EP={num_ranks}  K={top_k}  "
        f"remote replica cap/rank={max_extra}  compute limit={compute}  "
        f"compute replica cap/rank={args.max_compute_extra_experts_per_rank}  "
        f"communication budget={budget_ratio if budget_ratio is not None else 'none'}"
    )
    print(_table(rows))
    if args.output_plan:
        Path(args.output_plan).write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        print(f"plan={args.output_plan}")


if __name__ == "__main__":
    main()
