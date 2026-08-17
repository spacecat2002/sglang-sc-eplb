"""Source-aware bundle and congestion optimized expert placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .expert_affinity_graph import RoutedToken


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


def _refine_remote_swaps(
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
    candidate_partners: int = 4,
) -> None:
    """Apply bounded remote-decreasing swaps in-place."""

    if rounds < 1:
        return
    expert_count = len(ranks)
    for _ in range(rounds):
        move_delta = np.zeros((expert_count, num_ranks), dtype=np.int64)
        for expert, indexes in enumerate(token_indexes):
            old_rank = ranks[expert]
            sources = source[indexes]
            old_counts = bundle_rank_counts[indexes, old_rank]
            remove = -count[indexes] * (
                (old_counts == 1) & (sources != old_rank)
            )
            for target in range(num_ranks):
                if target == old_rank:
                    continue
                add = count[indexes] * (
                    (bundle_rank_counts[indexes, target] == 0)
                    & (sources != target)
                )
                move_delta[expert, target] = int((remove + add).sum())

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
                    if estimate < 0:
                        candidates.append(
                            (
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
        for _, _, left, right in candidates:
            if locked[left] or locked[right] or ranks[left] == ranks[right]:
                continue
            left_rank, right_rank = ranks[left], ranks[right]
            next_left = compute_load[left_rank] + demand[right] - demand[left]
            next_right = compute_load[right_rank] + demand[left] - demand[right]
            if max(next_left, next_right) > compute_load.max():
                continue

            indexes = np.union1d(token_indexes[left], token_indexes[right])
            rows = topk[indexes]
            has_left = np.any(rows == left, axis=1).astype(np.int8)
            has_right = np.any(rows == right, axis=1).astype(np.int8)
            sources = source[indexes]
            old_left = bundle_rank_counts[indexes, left_rank]
            old_right = bundle_rank_counts[indexes, right_rank]
            new_left = old_left - has_left + has_right
            new_right = old_right - has_right + has_left
            delta_left = (new_left > 0).astype(np.int8) - (
                old_left > 0
            ).astype(np.int8)
            delta_right = (new_right > 0).astype(np.int8) - (
                old_right > 0
            ).astype(np.int8)
            delta = int(
                (
                    count[indexes]
                    * (
                        delta_left * (sources != left_rank)
                        + delta_right * (sources != right_rank)
                    )
                ).sum()
            )
            if delta >= 0:
                continue

            # Only two destination columns change, so update the traffic matrix
            # without replaying every Top-K bundle after each accepted swap.
            traffic[:, left_rank] += np.bincount(
                sources[delta_left != 0],
                weights=count[indexes][delta_left != 0]
                * delta_left[delta_left != 0]
                * (sources[delta_left != 0] != left_rank),
                minlength=num_ranks,
            ).astype(np.int64)
            traffic[:, right_rank] += np.bincount(
                sources[delta_right != 0],
                weights=count[indexes][delta_right != 0]
                * delta_right[delta_right != 0]
                * (sources[delta_right != 0] != right_rank),
                minlength=num_ranks,
            ).astype(np.int64)
            bundle_rank_counts[token_indexes[left], left_rank] -= 1
            bundle_rank_counts[token_indexes[left], right_rank] += 1
            bundle_rank_counts[token_indexes[right], right_rank] -= 1
            bundle_rank_counts[token_indexes[right], left_rank] += 1
            ranks[left], ranks[right] = right_rank, left_rank
            compute_load[left_rank] = next_left
            compute_load[right_rank] = next_right
            locked[left] = locked[right] = True
            changed = True
        if not changed:
            break


def evaluate_cable_placement(
    tokens: Sequence[RoutedToken],
    placement: Mapping[int, int],
    *,
    num_ranks: int,
) -> CableMetrics:
    """Measure exact bundle traffic for a fixed single-copy placement."""

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
    tokens: Sequence[RoutedToken],
    *,
    experts: Sequence[int],
    num_ranks: int,
    congestion_weight: float = 0.25,
    load_weight: float = 0.25,
    refine_swaps: int = 2,
) -> CablePlacement:
    """Greedily place experts using exact incremental bundle traffic."""

    experts = tuple(sorted(experts))
    if not tokens or not experts:
        raise ValueError("tokens and experts must not be empty")
    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if min(congestion_weight, load_weight) < 0:
        raise ValueError("objective weights must be non-negative")
    if refine_swaps < 0:
        raise ValueError("refine_swaps must be non-negative")
    if any(token.source_rank >= num_ranks for token in tokens):
        raise ValueError("token source rank exceeds num_ranks")

    expert_index = {expert: index for index, expert in enumerate(experts)}
    if any(
        expert not in expert_index for token in tokens for expert in token.topk_experts
    ):
        raise ValueError("tokens contain an expert outside experts")
    top_k = len(tokens[0].topk_experts)
    if any(len(token.topk_experts) != top_k for token in tokens):
        raise ValueError("all tokens must use the same Top-K")

    quotient, remainder = divmod(len(experts), num_ranks)
    capacities = [quotient + int(rank < remainder) for rank in range(num_ranks)]
    if not quotient:
        raise ValueError("expert count must be at least num_ranks")

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
    total_tokens = int(count.sum())
    bundle_ids = np.repeat(np.arange(len(tokens)), top_k)
    flat_experts = topk.ravel()
    sorted_positions = np.argsort(flat_experts, kind="stable")
    expert_counts = np.bincount(flat_experts, minlength=len(experts))
    offsets = np.concatenate(([0], np.cumsum(expert_counts)))
    token_indexes = [
        bundle_ids[sorted_positions[offsets[expert] : offsets[expert + 1]]]
        for expert in range(len(experts))
    ]

    repeated_count = np.repeat(count, top_k)
    demand = np.zeros(len(experts), dtype=np.int64)
    source_demand = np.zeros((len(experts), num_ranks), dtype=np.int64)
    np.add.at(demand, flat_experts, repeated_count)
    np.add.at(
        source_demand,
        (flat_experts, np.repeat(source, top_k)),
        repeated_count,
    )
    order = np.lexsort((np.arange(len(experts)), -demand, -source_demand.max(axis=1)))

    ranks = np.full(len(experts), -1, dtype=np.intp)
    slots_used = np.zeros(num_ranks, dtype=np.intp)
    bundle_rank_counts = np.zeros((len(tokens), num_ranks), dtype=np.uint8)
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    ingress = np.zeros(num_ranks, dtype=np.int64)
    egress = np.zeros(num_ranks, dtype=np.int64)
    compute_load = np.zeros(num_ranks, dtype=np.int64)
    max_pair_traffic = 0
    max_ingress = 0
    max_egress = 0
    average_load = demand.sum() / num_ranks
    sum_load_squares = 0.0

    for expert in order:
        indexes = token_indexes[expert]
        sources = source[indexes]
        weights = count[indexes]
        best = None
        for rank in range(num_ranks):
            if slots_used[rank] >= capacities[rank]:
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

        assert best is not None
        _, old_load, _, rank, delta_by_source = best
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

    _refine_remote_swaps(
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
            total_tokens=sum(token.count for token in tokens),
            congestion_weight=congestion_weight,
            load_weight=load_weight,
        ),
    )
