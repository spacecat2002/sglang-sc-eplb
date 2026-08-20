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
    quota_by_source: tuple[tuple[tuple[int, ...], ...], ...] | None = None


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


def _route_quota(
    arrays: RoutedArrays,
    quota: np.ndarray,
    source_demand: np.ndarray,
    *,
    ranks_per_node: int,
    rdma_cost: float,
) -> ReplicaMetrics:
    """Evaluate expected traffic for independently quota-routed Top-K experts."""

    num_ranks = quota.shape[2]
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.float64)
    compute = quota.sum(axis=(0, 1), dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        demand = source_demand[topk, source[:, None]]
        for target in range(num_ranks):
            probability = quota[source[:, None], topk, target] / demand
            sent = 1.0 - np.prod(1.0 - probability, axis=1)
            remote = source != target
            np.add.at(traffic[:, target], source[remote], count[remote] * sent[remote])
    source_node = np.arange(num_ranks)[:, None] // ranks_per_node
    target_node = np.arange(num_ranks)[None, :] // ranks_per_node
    costs = np.where(source_node == target_node, 1.0, rdma_cost)
    np.fill_diagonal(costs, 0.0)
    rounded = np.rint(traffic).astype(np.int64)
    return ReplicaMetrics(
        remote=int(rounded.sum()),
        weighted_remote=float((traffic * costs).sum()),
        max_pair_traffic=int(rounded.max()),
        max_ingress=int(rounded.sum(axis=0).max()),
        max_egress=int(rounded.sum(axis=1).max()),
        compute_load=tuple(int(load) for load in compute),
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
) -> ReplicaPlacement:
    """Greedily add the globally best feasible source-local replica."""

    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid replication objective")
    if num_ranks < 1 or ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if hot_experts < 0 or candidate_ranks < 1:
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
            if expert not in replicas:
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
) -> ReplicaPlacement:
    """Quickly balance source-expert demand with communication constraints."""

    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid balance objective")
    if max_extra_per_rank < 0:
        raise ValueError("invalid balance replication limits")
    arrays = as_routed_arrays(tokens)
    replicas, primary, replica_mask = _replica_arrays(
        arrays, placement.replicas_by_expert, num_ranks
    )
    routing = np.asarray(placement.routing_by_source, dtype=np.intp).copy()
    source_demand = _source_demand(arrays, len(primary), num_ranks)
    quota = np.zeros((num_ranks, len(primary), num_ranks), dtype=np.int64)
    for source in range(num_ranks):
        quota[source, np.arange(len(primary)), routing[source]] = source_demand[
            :, source
        ]
    compute = quota.sum(axis=(0, 1), dtype=np.int64)
    # This is the same compact input used by UltraEP's placement kernel. It is
    # a conservative communication proxy because Top-K bundle coalescing is
    # intentionally left to the single exact evaluation after solving.
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    for source in range(num_ranks):
        traffic[source] = quota[source].sum(axis=0)
        traffic[source, source] = 0
    balance_by_rank = np.zeros(num_ranks, dtype=np.int64)
    added = 0
    while True:
        current_compute_key = (int(compute.max()), int(np.square(compute).sum()))
        ingress_load = traffic.sum(axis=0)
        egress_load = traffic.sum(axis=1)
        remote_total = int(traffic.sum())
        max_egress_without_source = np.full(num_ranks, int(egress_load.max()))
        top_source = int(egress_load.argmax())
        hidden_egress = egress_load.copy()
        hidden_egress[top_source] = -1
        max_egress_without_source[top_source] = int(hidden_egress.max())
        row_max = traffic.max(axis=1)
        max_pair_without_source = np.full(num_ranks, int(row_max.max()))
        top_source = int(row_max.argmax())
        hidden_pair = row_max.copy()
        hidden_pair[top_source] = -1
        max_pair_without_source[top_source] = int(hidden_pair.max())
        best = None
        candidate_source, candidate_expert, candidate_old = np.nonzero(quota > 0)
        for source, expert, old in zip(
            candidate_source, candidate_expert, candidate_old
        ):
            available = int(quota[source, expert, old])
            targets = np.arange(num_ranks)
            needs_copy = ~replica_mask[expert]
            feasible = targets != old
            feasible &= ~needs_copy | (balance_by_rank < max_extra_per_rank)

            without_old = compute.copy()
            without_old[old] = -1
            unaffected = np.full(num_ranks, int(without_old.max()))
            top = int(without_old.argmax())
            without_old[top] = -1
            unaffected[top] = int(without_old.max())
            amount = np.minimum(
                available, np.maximum(0, (compute[old] - compute + 1) // 2)
            )
            feasible &= amount > 0
            old_compute = compute[old] - amount
            target_compute = compute + amount
            next_max = np.maximum(np.maximum(unaffected, target_compute), old_compute)
            next_squares = (
                current_compute_key[1]
                - int(compute[old] ** 2)
                + np.square(old_compute)
                - np.square(compute)
                + np.square(target_compute)
            )
            feasible &= (next_max < current_compute_key[0]) | (
                (next_max == current_compute_key[0])
                & (next_squares < current_compute_key[1])
            )

            old_delta = np.where(source != old, -amount, 0)
            target_delta = np.where(targets != source, amount, 0)
            remote = remote_total + old_delta + target_delta
            feasible &= (old_delta + target_delta <= 0) | (
                next_max < current_compute_key[0]
            )
            if not np.any(feasible):
                continue

            next_old_traffic = traffic[source, old] + old_delta
            next_target_traffic = traffic[source] + target_delta
            row_without_old = traffic[source].copy()
            row_without_old[old] = -1
            unaffected_pair = np.full(num_ranks, int(row_without_old.max()))
            top = int(row_without_old.argmax())
            row_without_old[top] = -1
            unaffected_pair[top] = int(row_without_old.max())
            max_pair = np.maximum(
                np.maximum(unaffected_pair, next_target_traffic),
                np.maximum(next_old_traffic, int(max_pair_without_source[source])),
            )

            ingress_without_old = ingress_load.copy()
            ingress_without_old[old] = -1
            unaffected_ingress = np.full(num_ranks, int(ingress_without_old.max()))
            top = int(ingress_without_old.argmax())
            ingress_without_old[top] = -1
            unaffected_ingress[top] = int(ingress_without_old.max())
            next_ingress = np.maximum(
                np.maximum(unaffected_ingress, ingress_load + target_delta),
                ingress_load[old] + old_delta,
            )
            next_source_egress = egress_load[source] + old_delta + target_delta
            next_egress = np.maximum(
                max_egress_without_source[source], next_source_egress
            )
            bottleneck = np.maximum(next_ingress, next_egress)
            adds_destination = (target_delta > 0) & (traffic[source] == 0)
            indexes = np.flatnonzero(feasible)
            communication = (
                (bottleneck, max_pair, remote)
                if objective == "ingress-egress"
                else (remote, bottleneck, max_pair)
            )
            order = np.lexsort(
                (
                    targets[indexes],
                    needs_copy[indexes],
                    communication[2][indexes],
                    communication[1][indexes],
                    communication[0][indexes],
                    adds_destination[indexes],
                    next_squares[indexes],
                    next_max[indexes],
                )
            )
            target = int(indexes[order[0]])
            candidate = (
                (
                    int(next_max[target]),
                    int(next_squares[target]),
                    int(adds_destination[target]),
                    int(communication[0][target]),
                    int(communication[1][target]),
                    int(communication[2][target]),
                    int(needs_copy[target]),
                    expert,
                    source,
                    target,
                ),
                expert,
                source,
                old,
                target,
                int(amount[target]),
                int(old_delta[target]),
                int(target_delta[target]),
                bool(needs_copy[target]),
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
            amount,
            old_delta,
            target_delta,
            needs_copy,
        ) = best
        if needs_copy:
            replicas[expert] += (target,)
            replica_mask[expert, target] = True
            balance_by_rank[target] += 1
            added += 1
            local_quota = quota[target, expert].copy()
            quota[target, expert] = 0
            quota[target, expert, target] = local_quota.sum()
            for old_rank, local_amount in enumerate(local_quota):
                if old_rank == target or local_amount == 0:
                    continue
                compute[old_rank] -= local_amount
                compute[target] += local_amount
                traffic[target, old_rank] -= local_amount
            continue
        quota[source, expert, old] -= amount
        quota[source, expert, target] += amount
        compute[old] -= amount
        compute[target] += amount
        traffic[source, old] += old_delta
        traffic[source, target] += target_delta
    routing = quota.argmax(axis=2)
    metrics = _route_quota(
        arrays,
        quota,
        source_demand,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    serialized_quota = tuple(
        tuple(
            (
                tuple(int(quota[source, expert, rank]) for rank in replicas[expert])
                if source_demand[expert, source]
                else tuple(
                    int(rank == routing[source, expert]) for rank in replicas[expert]
                )
            )
            for expert in range(len(primary))
        )
        for source in range(num_ranks)
    )
    return ReplicaPlacement(
        replicas,
        tuple(tuple(int(rank) for rank in row) for row in routing),
        metrics,
        placement.extra_copies,
        added,
        serialized_quota,
    )
