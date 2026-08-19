"""Constrained source-local expert replication for GRACE+."""

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
    routing_by_source: tuple[tuple[int, ...], ...]
    metrics: ReplicaMetrics
    extra_copies: int
    balance_copies: int = 0


def _replica_arrays(
    arrays: RoutedArrays,
    placement: Mapping[int, int | Sequence[int]],
    num_ranks: int,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray, np.ndarray]:
    replicas = {}
    for expert, value in placement.items():
        ranks = (value,) if isinstance(value, (int, np.integer)) else tuple(value)
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or any(not 0 <= rank < num_ranks for rank in ranks)
        ):
            raise ValueError("placement contains invalid replicas")
        replicas[int(expert)] = tuple(int(rank) for rank in ranks)
    missing = set(np.unique(arrays.topk_experts).tolist()) - set(replicas)
    if missing:
        raise ValueError(f"placement is missing experts: {sorted(missing)}")
    size = max(max(replicas, default=-1), int(arrays.topk_experts.max())) + 1
    primary = np.full(size, -1, dtype=np.intp)
    replica_mask = np.zeros((size, num_ranks), dtype=bool)
    for expert, ranks in replicas.items():
        primary[expert] = ranks[0]
        replica_mask[expert, ranks] = True
    return replicas, primary, replica_mask


def _destinations(
    source: np.ndarray,
    topk: np.ndarray,
    primary: np.ndarray,
    replica_mask: np.ndarray,
) -> np.ndarray:
    return np.where(replica_mask[topk, source[:, None]], source[:, None], primary[topk])


def _default_routing(
    primary: np.ndarray, replica_mask: np.ndarray, num_ranks: int
) -> np.ndarray:
    sources = np.arange(num_ranks)[:, None]
    return np.where(replica_mask.T, sources, primary[None, :])


def _route(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
    routing: np.ndarray | None = None,
) -> ReplicaMetrics:
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    compute = np.zeros(num_ranks, dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        destinations = (
            routing[source[:, None], topk]
            if routing is not None
            else _destinations(source, topk, primary, replica_mask)
        )
        for column in range(topk.shape[1]):
            np.add.at(compute, destinations[:, column], count)
        for rank in range(num_ranks):
            sent = np.any(destinations == rank, axis=1) & (source != rank)
            np.add.at(traffic[:, rank], source[sent], count[sent])
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
        compute_load=tuple(int(load) for load in compute),
    )


def evaluate_replicated_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int | Sequence[int]],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
) -> ReplicaMetrics:
    arrays = as_routed_arrays(tokens)
    _, primary, replica_mask = _replica_arrays(arrays, placement, num_ranks)
    return _route(
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


def _gains_and_traffic(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact remote removal for every source-local replica candidate."""

    gains = np.zeros((len(primary), num_ranks), dtype=np.int64)
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        destinations = _destinations(source, topk, primary, replica_mask)
        for rank in range(num_ranks):
            sent = np.any(destinations == rank, axis=1) & (source != rank)
            np.add.at(traffic[:, rank], source[sent], count[sent])
        unique_destination = np.ones(topk.shape, dtype=bool)
        for left in range(topk.shape[1]):
            for right in range(left + 1, topk.shape[1]):
                same = destinations[:, left] == destinations[:, right]
                unique_destination[same, left] = False
                unique_destination[same, right] = False
        for column in range(topk.shape[1]):
            old = destinations[:, column]
            useful = (source != old) & unique_destination[:, column]
            np.add.at(gains, (topk[useful, column], source[useful]), count[useful])
    return gains, traffic


def _key(
    traffic: np.ndarray,
    source: int,
    destination: int,
    gain: int,
    objective: str,
) -> tuple[int, int, int]:
    next_traffic = traffic.copy()
    next_traffic[source, destination] -= gain
    remote = int(next_traffic.sum())
    ingress = int(next_traffic.sum(axis=0).max())
    egress = int(next_traffic.sum(axis=1).max())
    pair = int(next_traffic.max())
    return (
        (max(ingress, egress), pair, remote)
        if objective == "ingress-egress"
        else (remote, max(ingress, egress), pair)
    )


def _update_gains(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    gains: np.ndarray,
    expert: int,
    target: int,
) -> None:
    """Update gains exposed when one old destination becomes less shared."""

    old_rank = primary[expert]
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        affected = (source == target) & np.any(topk == expert, axis=1)
        if not np.any(affected):
            continue
        selected = topk[affected]
        destinations = _destinations(source[affected], selected, primary, replica_mask)
        same_primary = destinations == old_rank
        exposes_one = np.count_nonzero(same_primary, axis=1) == 2
        counts = arrays.count[start:end][affected]
        for column in range(topk.shape[1]):
            exposed = (
                exposes_one & same_primary[:, column] & (selected[:, column] != expert)
            )
            np.add.at(gains[:, target], selected[exposed, column], counts[exposed])
    gains[expert, target] = 0


def replicate_hot_experts(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    objective: str = "ingress-egress",
    rdma_cost: float = 1.0,
    hot_experts: int = 16,
    candidate_ranks: int = 4,
    compute_imbalance_limit: float = 1.25,
    max_extra_per_rank: int = 0,
    max_replicas_per_expert: int = 2,
) -> ReplicaPlacement:
    """Greedily add the globally best feasible source-local replica."""

    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid replication objective")
    if num_ranks < 1 or ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if hot_experts < 0 or min(candidate_ranks, max_replicas_per_expert) < 1:
        raise ValueError("invalid replication limits")
    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    if compute_imbalance_limit < 1 or rdma_cost < 1:
        raise ValueError("invalid replication cost or compute limit")
    arrays = as_routed_arrays(tokens)
    replicas, primary, replica_mask = _replica_arrays(arrays, placement, num_ranks)
    source_demand = _source_demand(arrays, len(primary), num_ranks)
    total_demand = source_demand.sum(axis=1)
    compute = np.zeros(num_ranks, dtype=np.int64)
    observed = primary >= 0
    np.add.at(compute, primary[observed], total_demand[observed])
    average = float(compute.sum()) / num_ranks
    extra_by_rank = np.zeros(num_ranks, dtype=np.int64)
    gains, traffic = _gains_and_traffic(arrays, primary, replica_mask, num_ranks)
    current_key = _key(traffic, 0, 0, 0, objective)
    hot = np.argsort(-total_demand, kind="stable")[:hot_experts].tolist()
    added = 0
    while added < num_ranks * max_extra_per_rank:
        best = None
        for expert in hot:
            if (
                expert not in replicas
                or len(replicas[expert]) >= max_replicas_per_expert
            ):
                continue
            targets = sorted(
                (
                    rank
                    for rank in range(num_ranks)
                    if not replica_mask[expert, rank]
                    and gains[expert, rank] > 0
                    and extra_by_rank[rank] < max_extra_per_rank
                ),
                key=lambda rank: _key(
                    traffic,
                    rank,
                    int(primary[expert]),
                    int(gains[expert, rank]),
                    objective,
                )
                + (rank,),
            )[:candidate_ranks]
            for target in targets:
                gain = int(gains[expert, target])
                candidate_key = _key(
                    traffic, target, int(primary[expert]), gain, objective
                )
                if candidate_key >= current_key:
                    continue
                moved = int(source_demand[expert, target])
                next_compute = compute.copy()
                next_compute[primary[expert]] -= moved
                next_compute[target] += moved
                if next_compute.max() > max(
                    compute.max(), average * compute_imbalance_limit
                ):
                    continue
                candidate = (
                    candidate_key + (int(next_compute.max()), -moved, expert, target),
                    expert,
                    target,
                    moved,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        key, expert, target, moved = best
        destination = int(primary[expert])
        gain = int(gains[expert, target])
        _update_gains(arrays, primary, replica_mask, gains, expert, target)
        replicas[expert] += (target,)
        replica_mask[expert, target] = True
        traffic[target, destination] -= gain
        compute[destination] -= moved
        compute[target] += moved
        extra_by_rank[target] += 1
        current_key = key[:3]
        added += 1
    metrics = _route(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    routing = _default_routing(primary, replica_mask, num_ranks)
    return ReplicaPlacement(
        replicas,
        tuple(tuple(int(rank) for rank in row) for row in routing),
        metrics,
        added,
    )


def balance_replica_compute(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: ReplicaPlacement,
    *,
    num_ranks: int,
    ranks_per_node: int,
    objective: str = "ingress-egress",
    rdma_cost: float = 1.0,
    max_extra_per_rank: int = 0,
    max_replicas_per_expert: int = 3,
) -> ReplicaPlacement:
    """Add compute-driven replicas and statically reroute source-expert demand."""

    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid balance objective")
    if max_extra_per_rank < 0 or max_replicas_per_expert < 1:
        raise ValueError("invalid balance replication limits")
    arrays = as_routed_arrays(tokens)
    replicas, primary, replica_mask = _replica_arrays(
        arrays, placement.replicas_by_expert, num_ranks
    )
    routing = np.asarray(placement.routing_by_source, dtype=np.intp).copy()
    source_demand = _source_demand(arrays, len(primary), num_ranks)
    compute = np.zeros(num_ranks, dtype=np.int64)
    for source in range(num_ranks):
        np.add.at(compute, routing[source], source_demand[:, source])
    metrics = _route(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
        routing=routing,
    )
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for source in range(num_ranks):
        selected = arrays.source_rank == source
        destinations = routing[source, arrays.topk_experts[selected]]
        for rank in range(num_ranks):
            sent = np.any(destinations == rank, axis=1) & (source != rank)
            traffic[source, rank] = int(arrays.count[selected][sent].sum())
    balance_by_rank = np.zeros(num_ranks, dtype=np.int64)
    added = 0
    while added < num_ranks * max_extra_per_rank:
        current_compute_key = (int(compute.max()), int(np.square(compute).sum()))
        best = None
        for source in range(num_ranks):
            for expert in range(len(primary)):
                demand = int(source_demand[expert, source])
                old = int(routing[source, expert])
                if demand <= 0:
                    continue
                indexes = np.flatnonzero(
                    (arrays.source_rank == source)
                    & np.any(arrays.topk_experts == expert, axis=1)
                )
                old_destinations = routing[source, arrays.topk_experts[indexes]]
                for target in range(num_ranks):
                    if target == old:
                        continue
                    needs_copy = not replica_mask[expert, target]
                    if needs_copy and (
                        balance_by_rank[target] >= max_extra_per_rank
                        or len(replicas[expert]) >= max_replicas_per_expert
                    ):
                        continue
                    next_compute = compute.copy()
                    next_compute[old] -= demand
                    next_compute[target] += demand
                    compute_key = (
                        int(next_compute.max()),
                        int(np.square(next_compute).sum()),
                    )
                    if compute_key >= current_compute_key:
                        continue
                    old_removed = (old_destinations == old).sum(axis=1) == 1
                    target_added = ~np.any(old_destinations == target, axis=1)
                    old_delta = -int(
                        arrays.count[indexes][old_removed & (source != old)].sum()
                    )
                    target_delta = int(
                        arrays.count[indexes][target_added & (source != target)].sum()
                    )
                    next_traffic = traffic.copy()
                    next_traffic[source, old] += old_delta
                    next_traffic[source, target] += target_delta
                    remote = int(next_traffic.sum())
                    ingress = int(next_traffic.sum(axis=0).max())
                    egress = int(next_traffic.sum(axis=1).max())
                    pair = int(next_traffic.max())
                    communication_key = (
                        (max(ingress, egress), pair, remote)
                        if objective == "ingress-egress"
                        else (remote, max(ingress, egress), pair)
                    )
                    candidate = (
                        compute_key
                        + (int(target_added.any()),)
                        + communication_key
                        + (int(needs_copy), expert, source, target),
                        expert,
                        source,
                        old,
                        target,
                        demand,
                        old_delta,
                        target_delta,
                        needs_copy,
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate
        if best is None:
            break
        (
            _,
            expert,
            source,
            old,
            target,
            demand,
            old_delta,
            target_delta,
            needs_copy,
        ) = best
        if needs_copy:
            replicas[expert] += (target,)
            replica_mask[expert, target] = True
            balance_by_rank[target] += 1
            added += 1
        routing[source, expert] = target
        compute[old] -= demand
        compute[target] += demand
        traffic[source, old] += old_delta
        traffic[source, target] += target_delta
    metrics = _route(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
        routing=routing,
    )
    return ReplicaPlacement(
        replicas,
        tuple(tuple(int(rank) for rank in row) for row in routing),
        metrics,
        placement.extra_copies,
        added,
    )
