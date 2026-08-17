"""Fixed-terminal hypergraph placement with exact Top-K connectivity cost."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .cable_expert_placement import CableMetrics, evaluate_cable_placement
from .expert_affinity_graph import RoutedToken


def _capacity_bounds(
    expert_count: int, num_ranks: int, ratio: float
) -> tuple[int, int]:
    ideal = expert_count / num_ranks
    delta = round(ideal * ratio)
    minimum = max(1, int(np.floor(ideal - delta)))
    maximum = max(minimum, int(np.ceil(ideal + delta)))
    if minimum * num_ranks > expert_count:
        minimum = expert_count // num_ranks
    if maximum * num_ranks < expert_count:
        maximum = int(np.ceil(ideal))
    if minimum < 1 or maximum * num_ranks < expert_count:
        raise ValueError("expert count must be at least num_ranks")
    return minimum, maximum


def _prepare(
    tokens: Sequence[RoutedToken], experts: Sequence[int]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    expert_index = {expert: index for index, expert in enumerate(experts)}
    if any(
        expert not in expert_index for token in tokens for expert in token.topk_experts
    ):
        raise ValueError("tokens contain an expert outside experts")
    top_k = len(tokens[0].topk_experts)
    if any(len(token.topk_experts) != top_k for token in tokens):
        raise ValueError("all tokens must use the same Top-K")
    source = np.fromiter(
        (token.source_rank for token in tokens), dtype=np.intp, count=len(tokens)
    )
    topk = np.asarray(
        [[expert_index[expert] for expert in token.topk_experts] for token in tokens],
        dtype=np.intp,
    )
    count = np.fromiter(
        (token.count for token in tokens), dtype=np.int64, count=len(tokens)
    )
    flat = topk.ravel()
    bundle_ids = np.repeat(np.arange(len(tokens)), top_k)
    order = np.argsort(flat, kind="stable")
    frequencies = np.bincount(flat, minlength=len(experts))
    offsets = np.concatenate(([0], np.cumsum(frequencies)))
    token_indexes = [
        bundle_ids[order[offsets[e] : offsets[e + 1]]] for e in range(len(experts))
    ]
    repeated_count = np.repeat(count, top_k)
    demand = np.zeros(len(experts), dtype=np.int64)
    source_demand = np.zeros((len(experts), int(source.max()) + 1), dtype=np.int64)
    np.add.at(demand, flat, repeated_count)
    np.add.at(source_demand, (flat, np.repeat(source, top_k)), repeated_count)
    return source, topk, count, token_indexes, demand, source_demand


def _order(mode: int, demand: np.ndarray, source_demand: np.ndarray) -> np.ndarray:
    source_peak = source_demand.max(axis=1)
    index = np.arange(len(demand))
    if mode % 4 == 0:
        return np.lexsort((index, -demand, -source_peak))
    if mode % 4 == 1:
        return np.lexsort((index, -source_peak, -demand))
    if mode % 4 == 2:
        return np.lexsort((index, demand, -source_peak))
    return np.lexsort((index, -demand, source_peak))


def _greedy_seed(
    source: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    demand: np.ndarray,
    order: np.ndarray,
    *,
    num_ranks: int,
    minimum_capacity: int,
    maximum_capacity: int,
    compute_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_count = len(demand)
    ranks = np.full(expert_count, -1, dtype=np.intp)
    slots = np.zeros(num_ranks, dtype=np.intp)
    loads = np.zeros(num_ranks, dtype=np.int64)
    bundle_counts = np.zeros((len(source), num_ranks), dtype=np.uint8)
    average = demand.sum() / num_ranks
    minimum_needed = minimum_capacity * num_ranks
    for position, expert in enumerate(order):
        indexes = token_indexes[expert]
        sources = source[indexes]
        feasible = []
        candidates = []
        for rank in range(num_ranks):
            if slots[rank] >= maximum_capacity:
                continue
            if minimum_needed - int(slots[rank] < minimum_capacity) > (
                expert_count - position - 1
            ):
                continue
            delta = int(
                (
                    count[indexes]
                    * ((bundle_counts[indexes, rank] == 0) & (sources != rank))
                ).sum()
            )
            next_load = loads[rank] + demand[expert]
            score = (delta, next_load, slots[rank], rank)
            candidates.append((score, rank))
            if next_load <= compute_limit * average:
                feasible.append((score, rank))
        if not candidates:
            raise ValueError("capacity constraints cannot be satisfied")
        _, rank = min(feasible or candidates)
        if slots[rank] < minimum_capacity:
            minimum_needed -= 1
        ranks[expert] = rank
        slots[rank] += 1
        bundle_counts[indexes, rank] += 1
        loads[rank] += demand[expert]
    return ranks, slots, bundle_counts


def _refine_moves(
    source: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    ranks: np.ndarray,
    slots: np.ndarray,
    bundle_counts: np.ndarray,
    demand: np.ndarray,
    *,
    num_ranks: int,
    minimum_capacity: int,
    maximum_capacity: int,
    compute_limit: float,
    rounds: int,
    total_tokens: int,
    remote_budget: float,
) -> np.ndarray:
    loads = np.bincount(ranks, weights=demand, minlength=num_ranks).astype(np.int64)
    average = demand.sum() / num_ranks
    remote_budget_tokens = int(total_tokens * remote_budget)
    remote_delta_total = 0
    for _ in range(rounds):
        candidates = []
        for expert in np.argsort(-demand, kind="stable"):
            old = ranks[expert]
            if slots[old] <= minimum_capacity:
                continue
            indexes = token_indexes[expert]
            sources = source[indexes]
            remove = -count[indexes] * (
                (bundle_counts[indexes, old] == 1) & (sources != old)
            )
            for target in range(num_ranks):
                if target == old or slots[target] >= maximum_capacity:
                    continue
                next_loads = loads.copy()
                next_loads[old] -= demand[expert]
                next_loads[target] += demand[expert]
                if next_loads.max() > max(compute_limit * average, loads.max()):
                    continue
                add = count[indexes] * (
                    (bundle_counts[indexes, target] == 0) & (sources != target)
                )
                delta = int((remove + add).sum())
                gain = int(loads[old] ** 2 + loads[target] ** 2)
                gain -= int(next_loads[old] ** 2 + next_loads[target] ** 2)
                if delta < 0 or (
                    gain > 0 and remote_delta_total + delta <= remote_budget_tokens
                ):
                    candidates.append(
                        (0 if delta < 0 else 1, delta, -gain, int(expert), target)
                    )
        candidates.sort()
        changed = False
        locked = np.zeros(len(ranks), dtype=bool)
        for _, _, _, expert, target in candidates:
            old = ranks[expert]
            if locked[expert] or old == target:
                continue
            if slots[old] <= minimum_capacity or slots[target] >= maximum_capacity:
                continue
            next_old = loads[old] - demand[expert]
            next_target = loads[target] + demand[expert]
            if max(next_old, next_target) > max(compute_limit * average, loads.max()):
                continue
            indexes = token_indexes[expert]
            sources = source[indexes]
            old_counts = bundle_counts[indexes, old]
            target_counts = bundle_counts[indexes, target]
            remove = -count[indexes] * ((old_counts == 1) & (sources != old))
            add = count[indexes] * ((target_counts == 0) & (sources != target))
            delta = int((remove + add).sum())
            gain = loads[old] ** 2 + loads[target] ** 2
            gain -= next_old**2 + next_target**2
            if delta >= 0 and (
                gain <= 0 or remote_delta_total + delta > remote_budget_tokens
            ):
                continue
            bundle_counts[indexes, old] -= 1
            bundle_counts[indexes, target] += 1
            ranks[expert] = target
            slots[old] -= 1
            slots[target] += 1
            loads[old] = next_old
            loads[target] = next_target
            remote_delta_total += delta
            locked[expert] = True
            changed = True
        if not changed:
            break
    return loads


def hypergraph_expert_placement(
    tokens: Sequence[RoutedToken],
    *,
    experts: Sequence[int],
    num_ranks: int,
    capacity_ratio: float = 0.15,
    compute_imbalance_limit: float = 2.0,
    starts: int = 4,
    refine_rounds: int = 4,
    remote_budget: float = 0.0,
) -> dict[str, object]:
    """Solve fixed-terminal hypergraph connectivity placement.

    Each Top-K bundle is a weighted hyperedge containing its source terminal
    and expert vertices. Connectivity minus one is exactly bundle remote rank
    count. This implementation keeps the dependency-free multistart/FM core.
    """

    experts = tuple(sorted(experts))
    if not tokens or not experts:
        raise ValueError("tokens and experts must not be empty")
    if num_ranks < 1 or any(token.source_rank >= num_ranks for token in tokens):
        raise ValueError("invalid num_ranks or source rank")
    if not 0 <= capacity_ratio < 1 or compute_imbalance_limit < 1:
        raise ValueError("invalid capacity or compute limits")
    if starts < 1 or refine_rounds < 0:
        raise ValueError("starts and refine_rounds must be positive")
    if not 0 <= remote_budget <= 1:
        raise ValueError("remote_budget must be in [0, 1]")
    source, _, count, token_indexes, demand, source_demand = _prepare(tokens, experts)
    minimum, maximum = _capacity_bounds(len(experts), num_ranks, capacity_ratio)
    best = None
    for mode in range(starts):
        ranks, slots, bundle_counts = _greedy_seed(
            source,
            count,
            token_indexes,
            demand,
            _order(mode, demand, source_demand),
            num_ranks=num_ranks,
            minimum_capacity=minimum,
            maximum_capacity=maximum,
            compute_limit=compute_imbalance_limit,
        )
        loads = _refine_moves(
            source,
            count,
            token_indexes,
            ranks,
            slots,
            bundle_counts,
            demand,
            num_ranks=num_ranks,
            minimum_capacity=minimum,
            maximum_capacity=maximum,
            compute_limit=compute_imbalance_limit,
            rounds=refine_rounds,
            total_tokens=int(count.sum()),
            remote_budget=remote_budget,
        )
        placement = {expert: int(ranks[i]) for i, expert in enumerate(experts)}
        metrics = evaluate_cable_placement(tokens, placement, num_ranks=num_ranks)
        key = (
            metrics.remote,
            metrics.max_ingress,
            metrics.max_pair_traffic,
            metrics.compute_imbalance,
        )
        if best is None or key < best[0]:
            best = (key, placement, metrics, loads)
    _, placement, metrics, _ = best
    return {
        "rank_by_expert": placement,
        "experts_by_rank": {
            rank: tuple(expert for expert in experts if placement[expert] == rank)
            for rank in range(num_ranks)
        },
        "metrics": metrics,
    }
