"""Fast constrained hot-expert replication for offline routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .expert_affinity_graph import RoutedArrays, RoutedToken, as_routed_arrays


_CHUNK_SIZE = 200_000


@dataclass(frozen=True)
class ReplicaMetrics:
    remote: int
    weighted_remote: float
    max_pair_traffic: int
    max_ingress: int
    max_egress: int
    compute_load: tuple[int, ...]

    @property
    def compute_imbalance(self) -> float:
        average = sum(self.compute_load) / len(self.compute_load)
        return max(self.compute_load, default=0) / average if average else 0.0


@dataclass(frozen=True)
class ReplicaPlacement:
    replicas_by_expert: dict[int, tuple[int, ...]]
    metrics: ReplicaMetrics
    extra_copies: int


def _normalize_replicas(
    placement: Mapping[int, int | Sequence[int]], num_ranks: int
) -> dict[int, tuple[int, ...]]:
    replicas = {}
    for expert, value in placement.items():
        expert = int(expert)
        ranks = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
        if expert < 0:
            raise ValueError("replica placement contains an invalid expert")
        if not ranks or any(rank < 0 or rank >= num_ranks for rank in ranks):
            raise ValueError("replica placement contains an invalid rank")
        if len(set(ranks)) != len(ranks):
            raise ValueError("replica placement contains duplicate ranks")
        replicas[expert] = tuple(int(rank) for rank in ranks)
    return replicas


def _replica_arrays(
    arrays: RoutedArrays,
    replicas: Mapping[int, tuple[int, ...]],
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    if arrays.source_rank.max() >= num_ranks:
        raise ValueError("routing trace contains a source outside num_ranks")
    missing = set(np.unique(arrays.topk_experts).tolist()) - set(replicas)
    if missing:
        raise ValueError(f"replica placement is missing experts: {sorted(missing)}")
    num_experts = max(max(replicas, default=-1), int(arrays.topk_experts.max())) + 1
    primary = np.full(num_experts, -1, dtype=np.intp)
    replica_mask = np.zeros((num_experts, num_ranks), dtype=bool)
    for expert, ranks in replicas.items():
        primary[expert] = ranks[0]
        replica_mask[expert, ranks] = True
    return primary, replica_mask


def _destinations(
    source: np.ndarray,
    topk: np.ndarray,
    primary: np.ndarray,
    replica_mask: np.ndarray,
) -> np.ndarray:
    return np.where(
        replica_mask[topk, source[:, None]], source[:, None], primary[topk]
    )


def _route_arrays(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
) -> ReplicaMetrics:
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    compute_load = np.zeros(num_ranks, dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        destinations = _destinations(source, topk, primary, replica_mask)
        for column in range(topk.shape[1]):
            np.add.at(compute_load, destinations[:, column], count)
        for rank in range(num_ranks):
            routed = np.any(destinations == rank, axis=1) & (source != rank)
            np.add.at(traffic[:, rank], source[routed], count[routed])

    source_node = np.arange(num_ranks)[:, None] // ranks_per_node
    target_node = np.arange(num_ranks)[None, :] // ranks_per_node
    costs = np.where(source_node == target_node, 1.0, rdma_cost)
    np.fill_diagonal(costs, 0.0)
    return ReplicaMetrics(
        remote=int(traffic.sum()),
        weighted_remote=float((traffic * costs).sum()),
        max_pair_traffic=int(traffic.max()),
        max_ingress=int(traffic.sum(axis=0).max()),
        max_egress=int(traffic.sum(axis=1).max()),
        compute_load=tuple(int(load) for load in compute_load),
    )


def evaluate_replicated_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int | Sequence[int]],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
) -> ReplicaMetrics:
    """Evaluate deterministic source-local-secondary routing."""

    if num_ranks < 1 or ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if rdma_cost < 1:
        raise ValueError("rdma_cost must be at least 1")
    arrays = as_routed_arrays(tokens)
    replicas = _normalize_replicas(placement, num_ranks)
    primary, replica_mask = _replica_arrays(arrays, replicas, num_ranks)
    return _route_arrays(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )


def _source_demand(
    arrays: RoutedArrays, num_experts: int, num_ranks: int
) -> np.ndarray:
    demand = np.zeros((num_experts, num_ranks), dtype=np.int64)
    for column in range(arrays.topk_experts.shape[1]):
        np.add.at(
            demand,
            (arrays.topk_experts[:, column], arrays.source_rank),
            arrays.count,
        )
    return demand


def _replica_gains(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    """Exact remote reduction from adding every possible source-local copy."""

    gains = np.zeros((len(primary), num_ranks), dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        destinations = _destinations(source, topk, primary, replica_mask)
        unique = np.ones(topk.shape, dtype=bool)
        for left in range(topk.shape[1]):
            for right in range(left + 1, topk.shape[1]):
                same = destinations[:, left] == destinations[:, right]
                unique[same, left] = False
                unique[same, right] = False
        for column in range(topk.shape[1]):
            old = destinations[:, column]
            useful = (source != old) & unique[:, column]
            np.add.at(
                gains,
                (topk[useful, column], source[useful]),
                count[useful],
            )
    return gains


def _update_replica_gains(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    gains: np.ndarray,
    expert: int,
    target: int,
) -> None:
    """Update exact gains after moving one expert's target-local demand."""

    old_rank = primary[expert]
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        affected = (source == target) & np.any(topk == expert, axis=1)
        if not np.any(affected):
            continue
        selected_topk = topk[affected]
        destinations = _destinations(
            source[affected], selected_topk, primary, replica_mask
        )
        same_primary = destinations == old_rank
        exposes_one = np.count_nonzero(same_primary, axis=1) == 2
        counts = arrays.count[start:end][affected]
        for column in range(topk.shape[1]):
            exposed = (
                exposes_one
                & same_primary[:, column]
                & (selected_topk[:, column] != expert)
            )
            np.add.at(
                gains[:, target], selected_topk[exposed, column], counts[exposed]
            )
    gains[expert, target] = 0


def hot_expert_replication(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
    extra_copy_budget: int = 0,
    hot_experts: int = 16,
    candidate_ranks: int = 4,
    compute_imbalance_limit: float = 1.25,
    max_extra_per_rank: int = 1,
) -> ReplicaPlacement:
    """Greedily add source-local copies under memory and compute constraints."""

    if num_ranks < 1 or ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if rdma_cost < 1:
        raise ValueError("rdma_cost must be at least 1")
    if (
        extra_copy_budget < 0
        or hot_experts < 0
        or candidate_ranks < 1
        or max_extra_per_rank < 1
    ):
        raise ValueError("invalid replication parameters")
    if compute_imbalance_limit < 1:
        raise ValueError("compute_imbalance_limit must be at least 1")
    if any(not isinstance(rank, (int, np.integer)) for rank in placement.values()):
        raise ValueError("primary placement must contain one rank per expert")

    arrays = as_routed_arrays(tokens)
    replicas = _normalize_replicas(placement, num_ranks)
    primary, replica_mask = _replica_arrays(arrays, replicas, num_ranks)
    source_demand = _source_demand(arrays, len(primary), num_ranks)
    total_demand = source_demand.sum(axis=1)
    hot = sorted(replicas, key=lambda expert: (-int(total_demand[expert]), expert))[
        :hot_experts
    ]
    compute_load = np.zeros(num_ranks, dtype=np.int64)
    observed = primary >= 0
    np.add.at(compute_load, primary[observed], total_demand[observed])
    extra_by_rank = np.zeros(num_ranks, dtype=np.int64)
    average_load = float(compute_load.sum()) / num_ranks
    extra = 0
    gains = _replica_gains(arrays, primary, replica_mask, num_ranks)

    while extra < extra_copy_budget:
        best: tuple[tuple[int, ...], int, int, int] | None = None
        for expert in hot:
            targets = sorted(
                (
                    rank
                    for rank in range(num_ranks)
                    if not replica_mask[expert, rank]
                    and source_demand[expert, rank] > 0
                    and extra_by_rank[rank] < max_extra_per_rank
                ),
                key=lambda rank: (
                    -int(gains[expert, rank]),
                    -int(source_demand[expert, rank]),
                    rank,
                ),
            )[:candidate_ranks]
            for target in targets:
                gain = int(gains[expert, target])
                if gain <= 0:
                    continue
                moved = int(source_demand[expert, target])
                old_rank = int(primary[expert])
                next_load = compute_load.copy()
                next_load[old_rank] -= moved
                next_load[target] += moved
                next_max = int(next_load.max())
                if next_max > max(
                    int(compute_load.max()), average_load * compute_imbalance_limit
                ):
                    continue
                key = (-gain, next_max, -moved, expert, target)
                if best is None or key < best[0]:
                    best = (key, expert, target, moved)
        if best is None:
            break
        _, expert, target, moved = best
        old_rank = int(primary[expert])
        _update_replica_gains(
            arrays, primary, replica_mask, gains, expert, target
        )
        replicas[expert] += (target,)
        replica_mask[expert, target] = True
        compute_load[old_rank] -= moved
        compute_load[target] += moved
        extra_by_rank[target] += 1
        extra += 1

    metrics = _route_arrays(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    return ReplicaPlacement(replicas, metrics, extra)
