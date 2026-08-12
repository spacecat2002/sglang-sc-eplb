#!/usr/bin/env python3
"""Microbenchmark exact and bounded fast bundle-aware replica planning.

This is a CPU/Python control-plane benchmark.  Use it to size the route
window and candidate cap before moving the runtime fast path into C++/CUDA.
"""

from __future__ import annotations

import argparse
import statistics
import time

from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    RoutedToken,
)


def _tokens(num_bundles: int, top_k: int, num_experts: int, num_ranks: int):
    return [
        RoutedToken(
            source_rank=index % num_ranks,
            topk_experts=tuple(
                sorted(
                    (index * 17 + offset * 31) % num_experts for offset in range(top_k)
                )
            ),
            count=1 + index % 8,
        )
        for index in range(num_bundles)
    ]


def _measure(call, warmup: int, repeats: int):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-bundles", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--fast-max-candidates", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--include-exact", action="store_true")
    args = parser.parse_args()

    homes = {expert: expert % args.num_ranks for expert in range(args.num_experts)}
    print("bundles  fast median  fast max  exact median  exact max")
    print("-------  -----------  --------  ------------  ---------")
    for num_bundles in args.num_bundles:
        tokens = _tokens(num_bundles, args.top_k, args.num_experts, args.num_ranks)
        planner = BundleAwareReplicaPlanner(
            num_ranks=args.num_ranks,
            baseline_rank_by_expert=homes,
            replica_slots_per_rank=args.top_k,
        )
        fast_median, fast_max = _measure(
            lambda: planner.plan_fast(tokens, max_candidates=args.fast_max_candidates),
            args.warmup,
            args.repeats,
        )
        if args.include_exact:
            exact_median, exact_max = _measure(
                lambda: planner.plan(tokens, max_actions=1), args.warmup, args.repeats
            )
            exact = f"{exact_median:9.3f}ms  {exact_max:7.3f}ms"
        else:
            exact = "       n/a         n/a"
        print(f"{num_bundles:7d}  {fast_median:9.3f}ms  {fast_max:7.3f}ms  {exact}")


if __name__ == "__main__":
    main()
