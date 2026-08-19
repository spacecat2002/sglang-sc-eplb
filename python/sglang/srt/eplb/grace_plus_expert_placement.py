"""GRACE+ placement: GRACE grouping followed by exact traffic refinement."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .grace_plus_refinement import (
    _congestion_key,
    _refine_swaps,
    evaluate_placement,
)
from .expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    as_routed_arrays,
    index_topk_experts,
)


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
    tokens: Sequence[RoutedToken] | RoutedArrays, experts: Sequence[int]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
]:
    arrays = as_routed_arrays(tokens)
    source = arrays.source_rank
    topk = index_topk_experts(arrays, experts)
    count = arrays.count.astype(np.int64, copy=False)
    top_k = topk.shape[1]
    if top_k > np.iinfo(np.uint8).max:
        raise ValueError("Top-K must fit in bundle rank counts")
    flat = topk.ravel()
    order = np.argsort(flat, kind="stable")
    frequencies = np.bincount(flat, minlength=len(experts))
    offsets = np.concatenate(([0], np.cumsum(frequencies)))
    index_dtype = np.int32 if len(tokens) <= np.iinfo(np.int32).max else np.intp
    token_indexes = [
        (order[offsets[e] : offsets[e + 1]] // top_k).astype(index_dtype)
        for e in range(len(experts))
    ]
    demand = np.zeros(len(experts), dtype=np.int64)
    for column in range(top_k):
        np.add.at(demand, topk[:, column], count)
    return source, topk, count, token_indexes, demand


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


def _placement_state(
    placement: Mapping[int, int],
    experts: Sequence[int],
    source: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    demand: np.ndarray,
    *,
    num_ranks: int,
    minimum_capacity: int,
    maximum_capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact incremental state for a caller-provided seed."""

    if set(placement) != set(experts):
        raise ValueError("initial_placement must contain exactly all experts")
    ranks = np.asarray([placement[expert] for expert in experts], dtype=np.intp)
    if np.any((ranks < 0) | (ranks >= num_ranks)):
        raise ValueError("initial_placement contains an invalid rank")
    slots = np.bincount(ranks, minlength=num_ranks).astype(np.intp)
    if np.any(slots < minimum_capacity) or np.any(slots > maximum_capacity):
        raise ValueError("initial_placement violates expert capacity bounds")
    bundle_counts = np.zeros((len(source), num_ranks), dtype=np.uint8)
    for expert, indexes in enumerate(token_indexes):
        bundle_counts[indexes, ranks[expert]] += 1
    return ranks, slots, bundle_counts


def _linear_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the exact minimum-cost column for every row."""

    size = len(cost)
    u = np.zeros(size + 1, dtype=np.int64)
    v = np.zeros(size + 1, dtype=np.int64)
    matched_row = np.zeros(size + 1, dtype=np.intp)
    previous = np.zeros(size + 1, dtype=np.intp)
    for row in range(1, size + 1):
        matched_row[0] = row
        column = 0
        minimum = np.full(size + 1, np.iinfo(np.int64).max, dtype=np.int64)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.iinfo(np.int64).max
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            next_column = previous[column]
            matched_row[column] = matched_row[next_column]
            column = next_column
            if column == 0:
                break
    assignment = np.empty(size, dtype=np.intp)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


def _align_groups_to_ranks(
    source: np.ndarray,
    count: np.ndarray,
    ranks: np.ndarray,
    slots: np.ndarray,
    bundle_counts: np.ndarray,
) -> None:
    """Relabel fixed expert groups to minimize exact source-aware remote."""

    num_ranks = bundle_counts.shape[1]
    totals = np.zeros(num_ranks, dtype=np.int64)
    local = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for group in range(num_ranks):
        indexes = bundle_counts[:, group] != 0
        totals[group] = count[indexes].sum(dtype=np.int64)
        np.add.at(local[group], source[indexes], count[indexes])
    assignment = _linear_assignment(totals[:, None] - local)
    ranks[:] = assignment[ranks]
    old_slots = slots.copy()
    slots[assignment] = old_slots
    visited = np.zeros(num_ranks, dtype=bool)
    for start in range(num_ranks):
        if visited[start]:
            continue
        current = start
        values = bundle_counts[:, current].copy()
        while True:
            visited[current] = True
            target = assignment[current]
            if target == start:
                bundle_counts[:, target] = values
                break
            next_values = bundle_counts[:, target].copy()
            bundle_counts[:, target] = values
            values = next_values
            current = target


def _align_groups_to_congestion(
    source: np.ndarray,
    count: np.ndarray,
    ranks: np.ndarray,
    slots: np.ndarray,
    bundle_counts: np.ndarray,
) -> None:
    """Exactly minimize bottleneck, max-pair, then remote for fixed groups."""

    num_ranks = bundle_counts.shape[1]
    local = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for group in range(num_ranks):
        indexes = bundle_counts[:, group] != 0
        np.add.at(local[group], source[indexes], count[indexes])
    ingress = local.sum(axis=1)[:, None] - local
    egress = local.sum(axis=0)[None, :] - local
    bottleneck = np.maximum(ingress, egress)
    pair = np.empty_like(local)
    for rank in range(num_ranks):
        values = local.copy()
        values[:, rank] = 0
        pair[:, rank] = values.max(axis=1)

    def feasible(allowed: np.ndarray) -> bool:
        assignment = _linear_assignment((~allowed).astype(np.int64))
        return bool(np.all(allowed[np.arange(num_ranks), assignment]))

    allowed = np.ones_like(local, dtype=bool)
    for values in (bottleneck, pair):
        candidates = np.unique(values[allowed])
        low, high = 0, len(candidates) - 1
        while low < high:
            middle = (low + high) // 2
            if feasible(allowed & (values <= candidates[middle])):
                high = middle
            else:
                low = middle + 1
        allowed &= values <= candidates[low]
    cost = ingress.copy()
    cost[~allowed] = int(cost.max()) * num_ranks + 1
    assignment = _linear_assignment(cost)

    ranks[:] = assignment[ranks]
    old_slots = slots.copy()
    slots[assignment] = old_slots
    old_bundle_counts = bundle_counts.copy()
    bundle_counts[:, assignment] = old_bundle_counts


def _traffic(
    source: np.ndarray,
    count: np.ndarray,
    bundle_counts: np.ndarray,
) -> np.ndarray:
    num_ranks = bundle_counts.shape[1]
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for rank in range(num_ranks):
        indexes = (bundle_counts[:, rank] != 0) & (source != rank)
        np.add.at(traffic[:, rank], source[indexes], count[indexes])
    return traffic


def _refine_congestion_moves(
    source: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    ranks: np.ndarray,
    slots: np.ndarray,
    bundle_counts: np.ndarray,
    demand: np.ndarray,
    loads: np.ndarray,
    *,
    num_ranks: int,
    minimum_capacity: int,
    maximum_capacity: int,
    compute_limit: float,
    rounds: int,
    remote_limit: int,
) -> None:
    traffic = _traffic(source, count, bundle_counts)
    average = demand.sum() / num_ranks
    for _ in range(rounds):
        best = None
        for expert in np.argsort(-demand, kind="stable"):
            old = int(ranks[expert])
            if slots[old] <= minimum_capacity:
                continue
            indexes = token_indexes[expert]
            sources = source[indexes]
            old_counts = bundle_counts[indexes, old]
            remove = -count[indexes] * ((old_counts == 1) & (sources != old))
            for target in range(num_ranks):
                if target == old or slots[target] >= maximum_capacity:
                    continue
                next_loads = loads.copy()
                next_loads[old] -= demand[expert]
                next_loads[target] += demand[expert]
                if next_loads.max() > max(compute_limit * average, loads.max()):
                    continue
                target_counts = bundle_counts[indexes, target]
                add = count[indexes] * ((target_counts == 0) & (sources != target))
                next_traffic = traffic.copy()
                for rank, delta in ((old, remove), (target, add)):
                    changed = delta != 0
                    next_traffic[:, rank] += np.bincount(
                        sources[changed],
                        weights=delta[changed],
                        minlength=num_ranks,
                    ).astype(np.int64)
                key = _congestion_key(next_traffic)
                if (
                    remote_limit is None or int(next_traffic.sum()) <= remote_limit
                ) and key < _congestion_key(traffic):
                    choice = (
                        key,
                        int(next_loads.max()),
                        int(expert),
                        target,
                        next_traffic,
                    )
                    if best is None or choice[:4] < best[:4]:
                        best = choice
        if best is None:
            return
        _, _, expert, target, traffic = best
        old = int(ranks[expert])
        indexes = token_indexes[expert]
        bundle_counts[indexes, old] -= 1
        bundle_counts[indexes, target] += 1
        ranks[expert] = target
        slots[old] -= 1
        slots[target] += 1
        loads[old] -= demand[expert]
        loads[target] += demand[expert]


def grace_plus_expert_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    *,
    experts: Sequence[int],
    num_ranks: int,
    capacity_ratio: float = 0.15,
    compute_imbalance_limit: float = 2.0,
    refine_rounds: int = 4,
    initial_placement: Mapping[int, int],
    align_groups: bool = False,
    swap_rounds: int = 0,
    swap_candidate_partners: int = 4,
    swap_allow_load_worsening: bool = False,
    swap_max_compute_imbalance: float | None = None,
    objective: str = "ingress-egress",
) -> dict[str, object]:
    """Refine a GRACE placement using exact Top-K communication traffic.

    Each Top-K bundle is a weighted hyperedge containing its source terminal
    and expert vertices. Connectivity minus one is exactly bundle remote rank
    count. This implementation keeps the dependency-free multistart/FM core.
    """

    experts = tuple(sorted(experts))
    if not tokens or not experts:
        raise ValueError("tokens and experts must not be empty")
    arrays = as_routed_arrays(tokens)
    if num_ranks < 1 or arrays.source_rank.max() >= num_ranks:
        raise ValueError("invalid num_ranks or source rank")
    if not 0 <= capacity_ratio < 1 or compute_imbalance_limit < 1:
        raise ValueError("invalid capacity or compute limits")
    if min(refine_rounds, swap_rounds) < 0:
        raise ValueError("refinement rounds must be non-negative")
    if swap_candidate_partners < 1:
        raise ValueError("swap candidate partner count must be positive")
    if swap_max_compute_imbalance is not None and swap_max_compute_imbalance < 1:
        raise ValueError("swap compute imbalance limit must be at least 1")
    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid placement objective")
    source, topk, count, token_indexes, demand = _prepare(arrays, experts)
    minimum, maximum = _capacity_bounds(len(experts), num_ranks, capacity_ratio)
    best = None
    ranks, slots, bundle_counts = _placement_state(
        initial_placement,
        experts,
        source,
        token_indexes,
        demand,
        num_ranks=num_ranks,
        minimum_capacity=minimum,
        maximum_capacity=maximum,
    )
    for _ in range(1):
        if objective == "ingress-egress":
            _align_groups_to_congestion(source, count, ranks, slots, bundle_counts)
        elif align_groups:
            _align_groups_to_ranks(source, count, ranks, slots, bundle_counts)
        if objective == "ingress-egress":
            loads = np.bincount(ranks, weights=demand, minlength=num_ranks).astype(
                np.int64
            )
            _refine_congestion_moves(
                source,
                count,
                token_indexes,
                ranks,
                slots,
                bundle_counts,
                demand,
                loads,
                num_ranks=num_ranks,
                minimum_capacity=minimum,
                maximum_capacity=maximum,
                compute_limit=compute_imbalance_limit,
                rounds=refine_rounds,
                remote_limit=None,
            )
        else:
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
                remote_budget=0.0,
            )
        traffic = _traffic(source, count, bundle_counts)
        _refine_swaps(
            source=source,
            topk=topk,
            count=count,
            token_indexes=token_indexes,
            bundle_rank_counts=bundle_counts,
            ranks=ranks,
            demand=demand,
            compute_load=loads,
            traffic=traffic,
            num_ranks=num_ranks,
            rounds=swap_rounds,
            total_tokens=int(count.sum()),
            strategy=("congestion" if objective == "ingress-egress" else "remote"),
            remote_budget=0.0,
            candidate_partners=swap_candidate_partners,
            allow_load_worsening=swap_allow_load_worsening,
            max_compute_imbalance=swap_max_compute_imbalance,
            remote_limit=None,
        )
        placement = {expert: int(ranks[i]) for i, expert in enumerate(experts)}
        metrics = evaluate_placement(arrays, placement, num_ranks=num_ranks)
        key = (
            (
                max(metrics.max_ingress, metrics.max_egress),
                metrics.max_pair_traffic,
                metrics.remote,
            )
            if objective == "ingress-egress"
            else (
                metrics.remote,
                metrics.max_ingress,
                metrics.max_pair_traffic,
            )
        ) + (metrics.compute_imbalance,)
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
