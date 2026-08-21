#!/usr/bin/env python3
"""Simulate per-rank remote Top-2K replicas on a compact routing trace."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from compare_grace import _baseline_placement, _load, _table, _tokens
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="compact routing trace (.pt)")
    args = parser.parse_args()
    if Path(args.input).suffix != ".pt":
        parser.error("--input must be a compact .pt routing trace")

    raw, layers = _load(args.input)
    num_ranks = int(raw["num_ranks"])
    top_k = int(raw["top_k"])
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
        replicas, copies = replicate_source_top_experts(
            tokens,
            primary,
            num_ranks=num_ranks,
            max_extra_per_rank=2 * top_k,
        )
        optimized = evaluate_replicated_placement(tokens, replicas, num_ranks=num_ranks)
        rows.append(
            _rows(
                layer, "remote-top2k", optimized, copies, time.perf_counter() - started
            )
        )

    print(
        f"trace={args.input}  EP={num_ranks}  K={top_k}  remote replica cap/rank={2 * top_k}"
    )
    print(_table(rows))


if __name__ == "__main__":
    main()
