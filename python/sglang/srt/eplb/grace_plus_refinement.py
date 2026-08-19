"""Internal exact traffic accounting and swap refinement for GRACE+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
)


@dataclass(frozen=True)
class PlacementMetrics:
    remote: int
    max_pair_traffic: int
    max_ingress: int
    max_egress: int
    compute_load: tuple[int, ...]

    @property
    def compute_imbalance(self) -> float:
        average = sum(self.compute_load) / len(self.compute_load)
        return max(self.compute_load, default=0) / average if average else 0.0


def _traffic_key(traffic: np.ndarray) -> tuple[int, int, int]:
    return (
        int(traffic.sum(axis=0).max()),
        int(traffic.max()),
        int(traffic.sum(axis=1).max()),
    )


def _congestion_key(traffic: np.ndarray) -> tuple[int, int, int]:
    ingress = int(traffic.sum(axis=0).max())
    egress = int(traffic.sum(axis=1).max())
    return (
        max(ingress, egress),
        int(traffic.max()),
        int(traffic.sum()),
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
                if (
                    remote_limit is None or int(next_traffic.sum()) <= remote_limit
                ) and _congestion_key(next_traffic) < _congestion_key(traffic):
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


def evaluate_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
) -> PlacementMetrics:
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
        return PlacementMetrics(
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
    return PlacementMetrics(
        remote=sum(egress),
        max_pair_traffic=max((max(row, default=0) for row in traffic), default=0),
        max_ingress=max(ingress, default=0),
        max_egress=max(egress, default=0),
        compute_load=tuple(compute_load),
    )
