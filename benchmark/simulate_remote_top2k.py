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
from sglang.srt.eplb.expert_affinity_graph import as_routed_arrays
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
    phase_ms: dict[str, float] | None = None,
):
    phase_ms = phase_ms or {}
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
        f"{phase_ms.get('communication_replication_ms', 0.0):.1f}",
        f"{phase_ms.get('compute_replication_ms', 0.0):.1f}",
        f"{phase_ms.get('quota_solve_ms', 0.0):.1f}",
        f"{phase_ms.get('quota_allocation_ms', 0.0):.1f}",
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
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="run the source-demand, Top-N replication and traffic evaluation on CUDA",
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
            "comm-repl-ms",
            "compute-repl-ms",
            "quota-solve-ms",
            "quota-alloc/eval-ms",
            "eval-ms",
        ]
    ]
    plan = {}
    for layer, (gate, value, compact) in enumerate(layers):
        tokens, _ = _tokens(gate, value, compact, limit=0, seed=0)
        # Materialize the compact arrays once per layer.  Passing the same
        # object through baseline, communication, and quota evaluation avoids
        # rebuilding three identical NumPy views without changing routing.
        tokens = as_routed_arrays(tokens)
        experts = tuple(range(int(value["topk_experts"].max().item()) + 1))
        primary = _baseline_placement(experts, num_ranks)

        started = time.perf_counter()
        if args.cuda:
            from sglang.srt.eplb.gpu_replication import (
                evaluate_replicated_placement_cuda,
            )

            baseline = evaluate_replicated_placement_cuda(
                tokens, primary, num_ranks=num_ranks
            )
        else:
            baseline = evaluate_replicated_placement(
                tokens, primary, num_ranks=num_ranks
            )
        baseline_elapsed = time.perf_counter() - started
        rows.append(
            _rows(
                layer,
                "baseline",
                baseline,
                [0] * num_ranks,
                baseline_elapsed,
            )
        )

        phase_ms: dict[str, float] = {}
        started = time.perf_counter()
        if args.cuda:
            from sglang.srt.eplb.gpu_replication import (
                replicate_source_top_experts_cuda,
            )

            optimized = replicate_source_top_experts_cuda(
                tokens,
                primary,
                num_ranks=num_ranks,
                max_extra_per_rank=max_extra,
                max_compute_extra_per_rank=args.max_compute_extra_experts_per_rank,
                compute_imbalance_limit=args.compute_imbalance_limit,
                communication_budget_ratio=budget_ratio,
                device="cuda",
                timing=phase_ms,
                materialize_quota=bool(args.output_plan),
            )
        else:
            optimized = replicate_source_top_experts(
                tokens,
                primary,
                num_ranks=num_ranks,
                max_extra_per_rank=max_extra,
                compute_imbalance_limit=args.compute_imbalance_limit,
                timing=phase_ms,
            )
        replication_elapsed = time.perf_counter() - started
        # The first call contains communication-oriented placement.  When
        # compute balancing is requested, the remainder is the quota/compute
        # stage; keep it separate from the communication candidate search.
        quota_elapsed = sum(
            phase_ms.get(key, 0.0) for key in ("quota_solve_ms", "quota_allocation_ms")
        )
        phase_ms["compute_replication_ms"] = (
            max(
                0.0,
                replication_elapsed * 1000.0
                - phase_ms.get("communication_replication_ms", 0.0)
                - quota_elapsed,
            )
            if args.compute_imbalance_limit is not None
            or args.max_compute_extra_experts_per_rank
            else 0.0
        )
        if args.max_compute_extra_experts_per_rank and not args.cuda:
            compute_started = time.perf_counter()
            before_quota = {
                key: value
                for key, value in phase_ms.items()
                if key.startswith("quota_")
            }
            optimized = balance_replica_compute(
                tokens,
                optimized,
                num_ranks=num_ranks,
                max_extra_per_rank=args.max_compute_extra_experts_per_rank,
                communication_budget_ratio=budget_ratio,
                timing=phase_ms,
            )
            quota_elapsed = sum(
                phase_ms.get(key, 0.0) - before_quota.get(key, 0.0)
                for key in ("quota_solve_ms", "quota_allocation_ms")
            )
            phase_ms["compute_replication_ms"] += max(
                0.0,
                (time.perf_counter() - compute_started) * 1000.0 - quota_elapsed,
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
                phase_ms,
            )
        )

    compute = (
        f"{args.compute_imbalance_limit:.2f}x"
        if args.compute_imbalance_limit is not None
        else "none (all-local)"
    )
    print(
        f"trace={args.input}  EP={num_ranks}  K={top_k}  "
        f"backend={'cuda' if args.cuda else 'cpu'}  "
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
