#!/usr/bin/env python3
"""Simulate per-rank remote Top-2K replicas on a compact routing trace."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from compare_grace import _baseline_placement, _load, _reshard, _table, _tokens
from sglang.srt.eplb.grace_plus_replication import (
    evaluate_replicated_placement,
    replicate_source_top_experts,
)


def _rows(layer: int, method: str, metrics, copies: Sequence[int], elapsed: float):
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


def _communication_estimate(
    baseline_bottleneck: int,
    optimized_bottleneck: int,
    traffic_sensitive_fraction: float,
) -> tuple[float, float]:
    traffic_reduction = (
        1 - optimized_bottleneck / baseline_bottleneck
        if baseline_bottleneck
        else 0.0
    )
    return traffic_reduction, traffic_reduction * traffic_sensitive_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="compact routing trace (.pt)")
    parser.add_argument(
        "--num-ranks",
        type=int,
        help="target EP size; defaults to the trace EP size",
    )
    parser.add_argument(
        "--source-ep",
        type=int,
        help="source EP size; defaults to the trace metadata",
    )
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
        "--output-plan",
        help="write replicas/routing/quota JSON for benchmark_a2a_plan.py",
    )
    parser.add_argument(
        "--traffic-sensitive-fraction",
        type=float,
        default=0.5,
        help="fraction of communication time that scales with traffic (default: 0.5)",
    )
    parser.add_argument(
        "--baseline-communication-ms",
        type=float,
        help="measured baseline communication time for an equivalent prefill",
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
    if not 0 <= args.traffic_sensitive_fraction <= 1:
        parser.error("--traffic-sensitive-fraction must be between 0 and 1")
    if args.baseline_communication_ms is not None and (
        not math.isfinite(args.baseline_communication_ms)
        or args.baseline_communication_ms <= 0
    ):
        parser.error("--baseline-communication-ms must be positive")
    raw, layers = _load(args.input)
    source_ep = int(args.source_ep or raw["num_ranks"])
    num_ranks = int(args.num_ranks or source_ep)
    if source_ep < 1 or num_ranks < 1:
        parser.error("--source-ep and --num-ranks must be positive")
    top_k = int(raw["top_k"])
    max_extra = (
        args.max_extra_experts_per_rank
        if args.max_extra_experts_per_rank is not None
        else 2 * top_k
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
            "extra/rank",
            "eval-ms",
        ]
    ]
    plan = {}
    baseline_bottleneck = optimized_bottleneck = 0
    for layer, (gate, value, compact) in enumerate(layers):
        tokens, _ = _tokens(gate, value, compact, limit=0, seed=0)
        tokens = _reshard(tokens, source_ep, num_ranks)
        experts = tuple(range(int(value["topk_experts"].max().item()) + 1))
        primary = _baseline_placement(experts, num_ranks)

        started = time.perf_counter()
        baseline = evaluate_replicated_placement(tokens, primary, num_ranks=num_ranks)
        baseline_bottleneck += max(baseline.max_ingress, baseline.max_egress)
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
        optimized_bottleneck += max(
            optimized.metrics.max_ingress, optimized.metrics.max_egress
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
                f"remote-top{max_extra}",
                optimized.metrics,
                copies,
                time.perf_counter() - started,
            )
        )

    compute = (
        f"{args.compute_imbalance_limit:.2f}x"
        if args.compute_imbalance_limit is not None
        else "none (all-local)"
    )
    print(
        f"trace={args.input}  source EP={source_ep}  simulated EP={num_ranks}  "
        f"K={top_k}  "
        f"remote replica cap/rank={max_extra}  compute limit={compute}"
    )
    print(_table(rows))
    traffic_reduction, time_reduction = _communication_estimate(
        baseline_bottleneck,
        optimized_bottleneck,
        args.traffic_sensitive_fraction,
    )
    estimate = (
        f"[estimate] summed layer bottleneck {baseline_bottleneck} -> "
        f"{optimized_bottleneck} ({traffic_reduction:.2%} reduction); "
        f"communication time reduction ~= {time_reduction:.2%} "
        f"with traffic-sensitive fraction={args.traffic_sensitive_fraction:.2f}"
    )
    if args.baseline_communication_ms is not None:
        estimate += (
            f"; {args.baseline_communication_ms:.3f} ms -> "
            f"{args.baseline_communication_ms * (1 - time_reduction):.3f} ms"
        )
    print(estimate)
    if source_ep != num_ranks:
        print(
            "[estimate] source demand was split uniformly across virtual child ranks; "
            "this is an EP scaling estimate, not an EP hardware measurement"
        )
    if args.output_plan:
        Path(args.output_plan).write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        print(f"plan={args.output_plan}")


if __name__ == "__main__":
    main()
