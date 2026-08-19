"""Source-aware bundle and congestion optimized expert placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    as_routed_arrays,
    index_topk_experts,
)


@dataclass(frozen=True)
class CableMetrics:
    remote: int
    max_pair_traffic: int
    max_ingress: int
    max_egress: int
    compute_load: tuple[int, ...]

    @property
    def compute_imbalance(self) -> float:
        average = sum(self.compute_load) / len(self.compute_load)
        return max(self.compute_load, default=0) / average if average else 0.0


@dataclass(frozen=True)
class CablePlacement:
    rank_by_expert: dict[int, int]
    experts_by_rank: dict[int, tuple[int, ...]]
    metrics: CableMetrics
    objective: float


def _traffic_key(traffic: np.ndarray) -> tuple[int, int, int]:
    return (
        int(traffic.sum(axis=0).max()),
        int(traffic.max()),
        int(traffic.sum(axis=1).max()),
    )


def _congestion_key(traffic: np.ndarray) -> tuple[int, int, int, int, int]:
    ingress = int(traffic.sum(axis=0).max())
    egress = int(traffic.sum(axis=1).max())
    return (
        max(ingress, egress),
        int(traffic.max()),
        int(traffic.sum()),
        egress,
        ingress,
    )


def _move_deltas(
    source: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    bundle_rank_counts: np.ndarray,
    ranks: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    move_delta = np.zeros((len(ranks), num_ranks), dtype=np.int64)
    for expert, indexes in enumerate(token_indexes):
        old_rank = ranks[expert]
        sources = source[indexes]
        remove = -count[indexes] * (
            (bundle_rank_counts[indexes, old_rank] == 1) & (sources != old_rank)
        )
        for target in range(num_ranks):
            if target != old_rank:
                add = count[indexes] * (
                    (bundle_rank_counts[indexes, target] == 0) & (sources != target)
                )
                move_delta[expert, target] = int((remove + add).sum())
    return move_delta


def _refine_compute_moves(
    *,
    source: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    bundle_rank_counts: np.ndarray,
    ranks: np.ndarray,
    demand: np.ndarray,
    compute_load: np.ndarray,
    slots_used: np.ndarray,
    traffic: np.ndarray,
    num_ranks: int,
    minimum_capacity: int,
    maximum_capacity: int,
    rounds: int,
    total_tokens: int,
    compute_imbalance_limit: float,
    remote_budget: float,
) -> None:
    """Move hot experts to cooler ranks within capacity and traffic budgets."""

    if rounds < 1:
        return
    average_load = demand.sum() / num_ranks
    load_limit = compute_imbalance_limit * average_load
    remote_budget_tokens = int(total_tokens * remote_budget)
    remote_delta_total = 0
    expert_count = len(ranks)
    for _ in range(rounds):
        candidates = []
        for expert in np.argsort(-demand, kind="stable"):
            old_rank = ranks[expert]
            if slots_used[old_rank] <= minimum_capacity:
                continue
            indexes = token_indexes[expert]
            sources = source[indexes]
            old_counts = bundle_rank_counts[indexes, old_rank]
            remove = -count[indexes] * ((old_counts == 1) & (sources != old_rank))
            for target in range(num_ranks):
                if target == old_rank or slots_used[target] >= maximum_capacity:
                    continue
                next_load = compute_load.copy()
                next_load[old_rank] -= demand[expert]
                next_load[target] += demand[expert]
                if next_load.max() > max(load_limit, compute_load.max()):
                    continue
                add = count[indexes] * (
                    (bundle_rank_counts[indexes, target] == 0) & (sources != target)
                )
                delta = int((remove + add).sum())
                if remote_delta_total + delta > remote_budget_tokens:
                    continue
                old_squares = compute_load[old_rank] ** 2 + compute_load[target] ** 2
                new_squares = next_load[old_rank] ** 2 + next_load[target] ** 2
                load_gain = int(old_squares - new_squares)
                if load_gain > 0:
                    candidates.append((-load_gain, delta, int(expert), target))
        candidates.sort()
        changed = False
        locked = np.zeros(expert_count, dtype=bool)
        for _, _, expert, target in candidates:
            if locked[expert] or ranks[expert] == target:
                continue
            old_rank = ranks[expert]
            if slots_used[old_rank] <= minimum_capacity:
                continue
            if slots_used[target] >= maximum_capacity:
                continue
            indexes = token_indexes[expert]
            sources = source[indexes]
            old_counts = bundle_rank_counts[indexes, old_rank]
            target_counts = bundle_rank_counts[indexes, target]
            remove = -count[indexes] * ((old_counts == 1) & (sources != old_rank))
            add = count[indexes] * ((target_counts == 0) & (sources != target))
            delta = int((remove + add).sum())
            if remote_delta_total + delta > remote_budget_tokens:
                continue
            next_old = compute_load[old_rank] - demand[expert]
            next_target = compute_load[target] + demand[expert]
            if max(next_old, next_target) > max(load_limit, compute_load.max()):
                continue
            if compute_load[old_rank] ** 2 + compute_load[target] ** 2 <= (
                next_old**2 + next_target**2
            ):
                continue

            traffic[:, old_rank] += np.bincount(
                sources[remove != 0],
                weights=remove[remove != 0] * (sources[remove != 0] != old_rank),
                minlength=num_ranks,
            ).astype(np.int64)
            traffic[:, target] += np.bincount(
                sources[add != 0],
                weights=add[add != 0] * (sources[add != 0] != target),
                minlength=num_ranks,
            ).astype(np.int64)
            bundle_rank_counts[indexes, old_rank] -= 1
            bundle_rank_counts[indexes, target] += 1
            ranks[expert] = target
            slots_used[old_rank] -= 1
            slots_used[target] += 1
            compute_load[old_rank] = next_old
            compute_load[target] = next_target
            remote_delta_total += delta
            locked[expert] = True
            changed = True
        if not changed:
            break


def _exact_swap_delta(
    *,
    source: np.ndarray,
    topk: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    bundle_rank_counts: np.ndarray,
    ranks: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the exact bundle-connectivity delta for one expert swap."""

    left_rank, right_rank = ranks[left], ranks[right]
    indexes = np.union1d(token_indexes[left], token_indexes[right])
    rows = topk[indexes]
    has_left = np.any(rows == left, axis=1).astype(np.int8)
    has_right = np.any(rows == right, axis=1).astype(np.int8)
    sources = source[indexes]
    old_left = bundle_rank_counts[indexes, left_rank]
    old_right = bundle_rank_counts[indexes, right_rank]
    new_left = old_left - has_left + has_right
    new_right = old_right - has_right + has_left
    delta_left = (new_left > 0).astype(np.int8) - (old_left > 0).astype(np.int8)
    delta_right = (new_right > 0).astype(np.int8) - (old_right > 0).astype(np.int8)
    delta = int(
        (
            count[indexes]
            * (
                delta_left * (sources != left_rank)
                + delta_right * (sources != right_rank)
            )
        ).sum()
    )
    return indexes, sources, delta_left, delta_right, delta


def _update_swap_traffic(
    traffic: np.ndarray,
    *,
    count: np.ndarray,
    indexes: np.ndarray,
    sources: np.ndarray,
    rank_deltas: Sequence[tuple[int, np.ndarray]],
    num_ranks: int,
) -> None:
    """Update the two destination columns changed by an expert swap."""

    for rank, rank_delta in rank_deltas:
        changed = rank_delta != 0
        traffic[:, rank] += np.bincount(
            sources[changed],
            weights=count[indexes][changed]
            * rank_delta[changed]
            * (sources[changed] != rank),
            minlength=num_ranks,
        ).astype(np.int64)


def _swap_load_allowed(
    compute_load: np.ndarray,
    next_load: np.ndarray,
    *,
    allow_load_worsening: bool,
    max_compute_imbalance: float | None,
) -> bool:
    current_max = int(compute_load.max())
    next_max = int(next_load.max())
    if not allow_load_worsening:
        return next_max <= current_max
    if max_compute_imbalance is None:
        return True
    average = float(compute_load.sum()) / len(compute_load)
    return next_max <= max(current_max, average * max_compute_imbalance)


def _apply_swap(
    *,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    bundle_rank_counts: np.ndarray,
    ranks: np.ndarray,
    compute_load: np.ndarray,
    traffic: np.ndarray,
    num_ranks: int,
    left: int,
    right: int,
    next_left: int,
    next_right: int,
    delta_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
) -> None:
    left_rank, right_rank = int(ranks[left]), int(ranks[right])
    indexes, sources, delta_left, delta_right, _ = delta_state
    _update_swap_traffic(
        traffic,
        count=count,
        indexes=indexes,
        sources=sources,
        rank_deltas=((left_rank, delta_left), (right_rank, delta_right)),
        num_ranks=num_ranks,
    )
    bundle_rank_counts[token_indexes[left], left_rank] -= 1
    bundle_rank_counts[token_indexes[left], right_rank] += 1
    bundle_rank_counts[token_indexes[right], right_rank] -= 1
    bundle_rank_counts[token_indexes[right], left_rank] += 1
    ranks[left], ranks[right] = right_rank, left_rank
    compute_load[left_rank] = next_left
    compute_load[right_rank] = next_right


def _refine_swaps(
    *,
    source: np.ndarray,
    topk: np.ndarray,
    count: np.ndarray,
    token_indexes: Sequence[np.ndarray],
    bundle_rank_counts: np.ndarray,
    ranks: np.ndarray,
    demand: np.ndarray,
    compute_load: np.ndarray,
    traffic: np.ndarray,
    num_ranks: int,
    rounds: int,
    total_tokens: int,
    strategy: str,
    remote_budget: float,
    candidate_partners: int = 4,
    allow_load_worsening: bool = False,
    max_compute_imbalance: float | None = None,
    remote_limit: int | None = None,
) -> None:
    """Apply bounded remote- or load-improving swaps in-place."""

    if rounds < 1:
        return
    if strategy not in {"remote", "balanced", "congestion"}:
        raise ValueError("invalid refinement strategy")
    if strategy == "congestion" and remote_limit is None:
        raise ValueError("congestion refinement requires remote_limit")
    if max_compute_imbalance is not None and max_compute_imbalance < 1:
        raise ValueError("max_compute_imbalance must be at least 1")
    expert_count = len(ranks)
    remote_delta_total = 0
    remote_budget_tokens = int(total_tokens * remote_budget)
    for _ in range(rounds):
        move_delta = _move_deltas(
            source,
            count,
            token_indexes,
            bundle_rank_counts,
            ranks,
            num_ranks,
        )

        candidates = []
        for expert in np.argsort(-demand, kind="stable"):
            old_rank = ranks[expert]
            for target in range(num_ranks):
                if target == old_rank:
                    continue
                pool = np.flatnonzero(ranks == target)
                partners = sorted(
                    pool.tolist(),
                    key=lambda partner: (
                        (
                            demand[partner]
                            if strategy == "balanced"
                            else move_delta[partner, old_rank]
                        ),
                        move_delta[partner, old_rank],
                        abs(int(demand[expert] - demand[partner])),
                        int(partner),
                    ),
                )[:candidate_partners]
                for partner in partners:
                    if expert >= partner:
                        continue
                    estimate = int(
                        move_delta[expert, target] + move_delta[partner, old_rank]
                    )
                    load_gain = int(demand[expert] - demand[partner])
                    if (
                        estimate < 0
                        or (strategy == "remote" and estimate == 0)
                        or (strategy == "balanced" and load_gain > 0)
                        or strategy == "congestion"
                    ):
                        candidates.append(
                            (
                                -load_gain if strategy == "balanced" else estimate,
                                estimate,
                                abs(int(demand[expert] - demand[partner])),
                                int(expert),
                                partner,
                            )
                        )
        # ponytail: four partners bounds refinement cost; widen this pool when
        # placement quality matters more than offline solve latency.
        candidates.sort()
        locked = np.zeros(expert_count, dtype=bool)
        changed = False
        best_congestion = None
        for _, _, _, left, right in candidates:
            if locked[left] or locked[right] or ranks[left] == ranks[right]:
                continue
            left_rank, right_rank = ranks[left], ranks[right]
            next_left = compute_load[left_rank] + demand[right] - demand[left]
            next_right = compute_load[right_rank] + demand[left] - demand[right]
            next_load = compute_load.copy()
            next_load[left_rank] = next_left
            next_load[right_rank] = next_right
            load_improves = next_load.max() < compute_load.max()
            if not _swap_load_allowed(
                compute_load,
                next_load,
                allow_load_worsening=allow_load_worsening,
                max_compute_imbalance=max_compute_imbalance,
            ):
                continue

            delta_state = _exact_swap_delta(
                source=source,
                topk=topk,
                count=count,
                token_indexes=token_indexes,
                bundle_rank_counts=bundle_rank_counts,
                ranks=ranks,
                left=left,
                right=right,
            )
            indexes, sources, delta_left, delta_right, delta = delta_state
            remote_improves = delta < 0
            budget_allows = remote_delta_total + delta <= remote_budget_tokens
            if strategy == "congestion":
                next_traffic = traffic.copy()
                _update_swap_traffic(
                    next_traffic,
                    count=count,
                    indexes=indexes,
                    sources=sources,
                    rank_deltas=((left_rank, delta_left), (right_rank, delta_right)),
                    num_ranks=num_ranks,
                )
                if int(next_traffic.sum()) <= int(remote_limit) and _congestion_key(
                    next_traffic
                ) < _congestion_key(traffic):
                    candidate = (
                        _congestion_key(next_traffic),
                        int(next_load.max()),
                        left,
                        right,
                        next_left,
                        next_right,
                        delta_state,
                    )
                    if best_congestion is None or candidate[:4] < best_congestion[:4]:
                        best_congestion = candidate
                continue
            if strategy == "remote":
                accept = remote_improves
                if delta == 0:
                    next_traffic = traffic.copy()
                    for rank, rank_delta in (
                        (left_rank, delta_left),
                        (right_rank, delta_right),
                    ):
                        changed_rows = rank_delta != 0
                        next_traffic[:, rank] += np.bincount(
                            sources[changed_rows],
                            weights=count[indexes][changed_rows]
                            * rank_delta[changed_rows]
                            * (sources[changed_rows] != rank),
                            minlength=num_ranks,
                        ).astype(np.int64)
                    next_traffic_key = _traffic_key(next_traffic)
                    current_traffic_key = _traffic_key(traffic)
                    accept = next_traffic_key < current_traffic_key or (
                        next_traffic_key == current_traffic_key and load_improves
                    )
            else:
                accept = remote_improves or (load_improves and budget_allows)
            if not accept:
                continue

            _apply_swap(
                count=count,
                token_indexes=token_indexes,
                bundle_rank_counts=bundle_rank_counts,
                ranks=ranks,
                compute_load=compute_load,
                traffic=traffic,
                num_ranks=num_ranks,
                left=left,
                right=right,
                next_left=next_left,
                next_right=next_right,
                delta_state=delta_state,
            )
            remote_delta_total += delta
            locked[left] = locked[right] = True
            changed = True
        if best_congestion is not None:
            _, _, left, right, next_left, next_right, delta_state = best_congestion
            _apply_swap(
                count=count,
                token_indexes=token_indexes,
                bundle_rank_counts=bundle_rank_counts,
                ranks=ranks,
                compute_load=compute_load,
                traffic=traffic,
                num_ranks=num_ranks,
                left=left,
                right=right,
                next_left=next_left,
                next_right=next_right,
                delta_state=delta_state,
            )
            changed = True
        if not changed:
            break


def evaluate_cable_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
) -> CableMetrics:
    """Measure exact bundle traffic for a fixed single-copy placement."""

    if isinstance(tokens, RoutedArrays):
        size = max(int(tokens.topk_experts.max()), max(placement, default=-1)) + 1
        ranks = np.full(size, -1, dtype=np.intp)
        for expert, rank in placement.items():
            if expert < len(ranks):
                ranks[expert] = rank
        destinations = ranks[tokens.topk_experts]
        if np.any(destinations < 0):
            raise ValueError("placement is missing a routed expert")
        traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
        for rank in range(num_ranks):
            touched = np.any(destinations == rank, axis=1) & (
                tokens.source_rank != rank
            )
            traffic[:, rank] = np.bincount(
                tokens.source_rank[touched],
                weights=tokens.count[touched],
                minlength=num_ranks,
            ).astype(np.int64)
        demand = np.zeros(len(ranks), dtype=np.int64)
        weights = tokens.count.astype(np.int64, copy=False)
        for column in range(tokens.topk_experts.shape[1]):
            demand += np.bincount(
                tokens.topk_experts[:, column],
                weights=weights,
                minlength=len(ranks),
            ).astype(np.int64)
        compute_load = np.zeros(num_ranks, dtype=np.int64)
        for expert, rank in placement.items():
            compute_load[rank] += demand[expert]
        return CableMetrics(
            remote=int(traffic.sum()),
            max_pair_traffic=int(traffic.max()),
            max_ingress=int(traffic.sum(axis=0).max()),
            max_egress=int(traffic.sum(axis=1).max()),
            compute_load=tuple(int(load) for load in compute_load),
        )

    traffic = [[0] * num_ranks for _ in range(num_ranks)]
    demand = {expert: 0 for expert in placement}
    for token in tokens:
        destinations = {placement[expert] for expert in token.topk_experts}
        for expert in token.topk_experts:
            demand[expert] += token.count
        for rank in destinations - {token.source_rank}:
            traffic[token.source_rank][rank] += token.count
    ingress = [
        sum(traffic[source][rank] for source in range(num_ranks))
        for rank in range(num_ranks)
    ]
    egress = [sum(row) for row in traffic]
    compute_load = [0] * num_ranks
    for expert, rank in placement.items():
        compute_load[rank] += demand[expert]
    return CableMetrics(
        remote=sum(egress),
        max_pair_traffic=max((max(row, default=0) for row in traffic), default=0),
        max_ingress=max(ingress, default=0),
        max_egress=max(egress, default=0),
        compute_load=tuple(compute_load),
    )


def _objective(
    metrics: CableMetrics,
    *,
    total_tokens: int,
    congestion_weight: float,
    load_weight: float,
) -> float:
    average = sum(metrics.compute_load) / len(metrics.compute_load)
    load_variance = (
        sum((load / average - 1) ** 2 for load in metrics.compute_load)
        / len(metrics.compute_load)
        if average
        else 0.0
    )
    congestion = max(metrics.max_pair_traffic, metrics.max_ingress, metrics.max_egress)
    return (
        metrics.remote / total_tokens
        + congestion_weight * congestion / total_tokens
        + load_weight * load_variance
    )


def cable_expert_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    *,
    experts: Sequence[int],
    num_ranks: int,
    congestion_weight: float = 0.25,
    load_weight: float = 0.25,
    refine_swaps: int = 2,
    refine_strategy: str = "balanced",
    remote_budget: float = 0.03,
    capacity_ratio: float = 0.15,
    compute_refine_moves: int = 2,
    compute_imbalance_limit: float = 2.0,
) -> CablePlacement:
    """Greedily place experts using exact incremental bundle traffic."""

    experts = tuple(sorted(experts))
    if not tokens or not experts:
        raise ValueError("tokens and experts must not be empty")
    arrays = as_routed_arrays(tokens)
    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if min(congestion_weight, load_weight) < 0:
        raise ValueError("objective weights must be non-negative")
    if refine_swaps < 0:
        raise ValueError("refine_swaps must be non-negative")
    if compute_refine_moves < 0:
        raise ValueError("compute_refine_moves must be non-negative")
    if refine_strategy not in {"remote", "balanced"}:
        raise ValueError("refine_strategy must be 'remote' or 'balanced'")
    if not 0 <= remote_budget <= 1:
        raise ValueError("remote_budget must be in [0, 1]")
    if not 0 <= capacity_ratio < 1 or compute_imbalance_limit < 1:
        raise ValueError("invalid capacity or compute limits")
    if arrays.source_rank.max() >= num_ranks:
        raise ValueError("token source rank exceeds num_ranks")

    topk = index_topk_experts(arrays, experts)
    top_k = arrays.topk_experts.shape[1]
    if top_k > np.iinfo(np.uint8).max:
        raise ValueError("Top-K must fit in bundle rank counts")

    ideal_capacity = len(experts) / num_ranks
    capacity_delta = round(ideal_capacity * capacity_ratio)
    minimum_capacity = max(1, int(np.floor(ideal_capacity - capacity_delta)))
    maximum_capacity = max(
        minimum_capacity, int(np.ceil(ideal_capacity + capacity_delta))
    )
    if minimum_capacity * num_ranks > len(experts):
        minimum_capacity = len(experts) // num_ranks
    if maximum_capacity * num_ranks < len(experts):
        maximum_capacity = int(np.ceil(ideal_capacity))
    if not minimum_capacity or maximum_capacity * num_ranks < len(experts):
        raise ValueError("expert count must be at least num_ranks")

    source = arrays.source_rank
    count = arrays.count.astype(np.int64, copy=False)
    total_tokens = int(count.sum())
    flat_experts = topk.ravel()
    sorted_positions = np.argsort(flat_experts, kind="stable")
    expert_counts = np.bincount(flat_experts, minlength=len(experts))
    offsets = np.concatenate(([0], np.cumsum(expert_counts)))
    index_dtype = np.int32 if len(arrays) <= np.iinfo(np.int32).max else np.intp
    token_indexes = [
        (sorted_positions[offsets[expert] : offsets[expert + 1]] // top_k).astype(
            index_dtype
        )
        for expert in range(len(experts))
    ]

    demand = np.zeros(len(experts), dtype=np.int64)
    source_demand = np.zeros((len(experts), num_ranks), dtype=np.int64)
    for column in range(top_k):
        np.add.at(demand, topk[:, column], count)
        np.add.at(source_demand, (topk[:, column], source), count)
    order = np.lexsort((np.arange(len(experts)), -demand, -source_demand.max(axis=1)))

    ranks = np.full(len(experts), -1, dtype=np.intp)
    slots_used = np.zeros(num_ranks, dtype=np.intp)
    bundle_rank_counts = np.zeros((len(arrays), num_ranks), dtype=np.uint8)
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    ingress = np.zeros(num_ranks, dtype=np.int64)
    egress = np.zeros(num_ranks, dtype=np.int64)
    compute_load = np.zeros(num_ranks, dtype=np.int64)
    max_pair_traffic = 0
    max_ingress = 0
    max_egress = 0
    average_load = demand.sum() / num_ranks
    sum_load_squares = 0.0
    minimum_needed = minimum_capacity * num_ranks

    for position, expert in enumerate(order):
        indexes = token_indexes[expert]
        sources = source[indexes]
        weights = count[indexes]
        best = None
        best_feasible = None
        for rank in range(num_ranks):
            if slots_used[rank] >= maximum_capacity:
                continue
            needed_after = minimum_needed - int(slots_used[rank] < minimum_capacity)
            if needed_after > len(experts) - position - 1:
                continue
            new_remote = (sources != rank) & (bundle_rank_counts[indexes, rank] == 0)
            delta_by_source = np.bincount(
                sources[new_remote], weights=weights[new_remote], minlength=num_ranks
            ).astype(np.int64)
            delta_remote = int(delta_by_source.sum())
            next_pair = max(
                max_pair_traffic, int((traffic[:, rank] + delta_by_source).max())
            )
            next_ingress = max(max_ingress, int(ingress[rank] + delta_remote))
            next_egress = int((egress + delta_by_source).max())
            congestion = max(next_pair, next_ingress, next_egress)
            old_load = int(compute_load[rank])
            new_load = old_load + int(demand[expert])
            next_load_squares = sum_load_squares - old_load**2 + new_load**2
            load_variance = (
                next_load_squares / num_ranks / average_load**2 - 1
                if average_load
                else 0.0
            )
            score = (
                delta_remote / total_tokens
                + congestion_weight * congestion / total_tokens
                + load_weight * load_variance
            )
            candidate = (score, old_load, slots_used[rank], rank, delta_by_source)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
            if new_load <= compute_imbalance_limit * average_load and (
                best_feasible is None or candidate[:4] < best_feasible[:4]
            ):
                best_feasible = candidate

        assert best is not None
        _, old_load, _, rank, delta_by_source = best_feasible or best
        if slots_used[rank] < minimum_capacity:
            minimum_needed -= 1
        ranks[expert] = rank
        slots_used[rank] += 1
        bundle_rank_counts[indexes, rank] += 1
        traffic[:, rank] += delta_by_source
        delta_remote = int(delta_by_source.sum())
        ingress[rank] += delta_remote
        egress += delta_by_source
        max_pair_traffic = max(max_pair_traffic, int(traffic[:, rank].max()))
        max_ingress = max(max_ingress, int(ingress[rank]))
        max_egress = max(max_egress, int(egress.max()))
        new_load = old_load + int(demand[expert])
        sum_load_squares += new_load**2 - old_load**2
        compute_load[rank] = new_load

    _refine_compute_moves(
        source=source,
        count=count,
        token_indexes=token_indexes,
        bundle_rank_counts=bundle_rank_counts,
        ranks=ranks,
        demand=demand,
        compute_load=compute_load,
        slots_used=slots_used,
        traffic=traffic,
        num_ranks=num_ranks,
        minimum_capacity=minimum_capacity,
        maximum_capacity=maximum_capacity,
        rounds=compute_refine_moves,
        total_tokens=total_tokens,
        compute_imbalance_limit=compute_imbalance_limit,
        remote_budget=remote_budget,
    )

    _refine_swaps(
        source=source,
        topk=topk,
        count=count,
        token_indexes=token_indexes,
        bundle_rank_counts=bundle_rank_counts,
        ranks=ranks,
        demand=demand,
        compute_load=compute_load,
        traffic=traffic,
        num_ranks=num_ranks,
        rounds=refine_swaps,
        total_tokens=total_tokens,
        strategy=refine_strategy,
        remote_budget=remote_budget,
    )

    # ponytail: rank-pair and port loads proxy NVSwitch contention; use a path
    # incidence matrix when the deployment exposes stable physical routes.
    ranks = ranks.tolist()
    rank_by_expert = dict(zip(experts, ranks))
    metrics = CableMetrics(
        remote=int(traffic.sum()),
        max_pair_traffic=int(traffic.max()),
        max_ingress=int(traffic.sum(axis=0).max()),
        max_egress=int(traffic.sum(axis=1).max()),
        compute_load=tuple(int(load) for load in compute_load),
    )
    return CablePlacement(
        rank_by_expert=rank_by_expert,
        experts_by_rank={
            rank: tuple(expert for expert in experts if rank_by_expert[expert] == rank)
            for rank in range(num_ranks)
        },
        metrics=metrics,
        objective=_objective(
            metrics,
            total_tokens=total_tokens,
            congestion_weight=congestion_weight,
            load_weight=load_weight,
        ),
    )
