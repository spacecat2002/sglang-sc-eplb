"""Constrained source-local expert replication for GRACE+."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .expert_affinity_graph import RoutedArrays, RoutedToken, as_routed_arrays


_CHUNK_SIZE = 200_000


@dataclass(frozen=True)
class ReplicaMetrics:
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
class ReplicaPlacement:
    replicas_by_expert: dict[int, tuple[int, ...]]
    routing_by_source: tuple[tuple[int, ...], ...]
    metrics: ReplicaMetrics
    extra_copies: int
    balance_copies: int = 0
    quota_by_source: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    source_demand: np.ndarray | None = field(default=None, repr=False, compare=False)


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
    routing: np.ndarray | None = None,
    source_demand: np.ndarray | None = None,
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
            if source_demand is not None:
                np.add.at(source_demand, (topk[:, column], source), count)
        for rank in range(num_ranks):
            sent = np.any(destinations == rank, axis=1) & (source != rank)
            np.add.at(traffic[:, rank], source[sent], count[sent])
    return ReplicaMetrics(
        remote=int(traffic.sum()),
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
) -> ReplicaMetrics:
    arrays = as_routed_arrays(tokens)
    _, primary, replica_mask = _replica_arrays(arrays, placement, num_ranks)
    return _route(
        arrays,
        primary,
        replica_mask,
        num_ranks=num_ranks,
    )


def _route_quota(
    arrays: RoutedArrays,
    quota: np.ndarray,
    source_demand: np.ndarray,
    replicas: Mapping[int, Sequence[int]] | None = None,
) -> ReplicaMetrics:
    """Evaluate exact prefix-routed traffic without expanding token bundles."""

    num_ranks = quota.shape[2]
    if replicas is None:
        replicas = {
            expert: tuple(
                int(rank)
                for rank in np.flatnonzero(np.any(quota[:, expert, :] > 0, axis=0))
            )
            for expert in range(quota.shape[1])
        }
    if all(len(ranks) == 1 for ranks in replicas.values()):
        primary = np.asarray(
            [replicas[expert][0] for expert in range(quota.shape[1])], dtype=np.intp
        )
        replica_mask = np.zeros((len(primary), num_ranks), dtype=bool)
        replica_mask[np.arange(len(primary)), primary] = True
        return _route(arrays, primary, replica_mask, num_ranks=num_ranks)
    prefix = _quota_prefix(quota, replicas)
    num_experts = quota.shape[1]
    max_replicas = max(map(len, replicas.values()))
    ordered_ranks = np.full((num_ranks, num_experts, max_replicas), -1, dtype=np.intp)
    ordered_prefix = np.full_like(ordered_ranks, np.iinfo(np.int64).max, dtype=np.int64)
    replica_counts = np.zeros(num_experts, dtype=np.intp)
    for expert, ranks in replicas.items():
        replica_counts[expert] = len(ranks)
        for source in range(num_ranks):
            order = _source_replica_order(source, ranks)
            ordered_ranks[source, expert, : len(order)] = order
            ordered_prefix[source, expert, : len(order)] = prefix[source, expert, order]
    counters = np.zeros(num_ranks * num_experts, dtype=np.int64)
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    compute = quota.sum(axis=(0, 1), dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        keys = (
            source.astype(np.intp, copy=False)[:, None] * num_experts
            + topk.astype(np.intp, copy=False)
        ).ravel()
        weights = np.broadcast_to(count[:, None], topk.shape).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights, dtype=np.int64)
        before = cumulative - sorted_weights
        group_starts = np.flatnonzero(
            np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
        )
        group_ends = np.concatenate((group_starts[1:] - 1, [len(keys) - 1]))
        within = np.empty_like(before)
        within[order] = before - np.repeat(
            before[group_starts], np.diff(np.append(group_starts, len(keys)))
        )
        ordinals = (within + counters[keys]).reshape(topk.shape)
        counters[sorted_keys[group_starts]] += (
            cumulative[group_ends] - before[group_starts]
        )

        rank_order = ordered_ranks[source[:, None], topk]
        boundaries = ordered_prefix[source[:, None], topk]
        replica_index = np.sum(ordinals[:, :, None] >= boundaries, axis=2)
        destinations = np.take_along_axis(
            rank_order, replica_index[:, :, None], axis=2
        ).squeeze(2)
        for rank in range(num_ranks):
            sent = np.any(destinations == rank, axis=1) & (source != rank)
            np.add.at(traffic[:, rank], source[sent], count[sent])

        crossing = np.any(
            (boundaries > ordinals[:, :, None])
            & (boundaries < ordinals[:, :, None] + count[:, None, None]),
            axis=(1, 2),
        )
        for row in np.flatnonzero(crossing):
            source_rank = int(source[row])
            tokens = int(count[row])
            for target in set(destinations[row]):
                if target != source_rank:
                    traffic[source_rank, target] -= tokens
            offsets = {0, tokens}
            for column, expert in enumerate(topk[row]):
                ordinal = int(ordinals[row, column])
                for boundary in ordered_prefix[
                    source_rank, expert, : replica_counts[expert]
                ]:
                    offset = int(boundary) - ordinal
                    if 0 < offset < tokens:
                        offsets.add(offset)
            ordered = sorted(offsets)
            for left, right in zip(ordered, ordered[1:]):
                segment_destinations = []
                for column, expert in enumerate(topk[row]):
                    size = replica_counts[expert]
                    index = int(
                        np.searchsorted(
                            ordered_prefix[source_rank, expert, :size],
                            int(ordinals[row, column]) + left,
                            side="right",
                        )
                    )
                    segment_destinations.append(
                        int(ordered_ranks[source_rank, expert, index])
                    )
                segment = right - left
                for target in set(segment_destinations):
                    if target != source_rank:
                        traffic[source_rank, target] += segment
    return ReplicaMetrics(
        remote=int(traffic.sum()),
        max_pair_traffic=int(traffic.max(initial=0)),
        max_ingress=int(traffic.sum(axis=0).max(initial=0)),
        max_egress=int(traffic.sum(axis=1).max(initial=0)),
        compute_load=tuple(int(load) for load in compute),
    )


def _quota_prefix(
    quota: np.ndarray, replicas: Mapping[int, Sequence[int]]
) -> np.ndarray:
    """Build source/expert cumulative quotas with the local replica first."""

    prefix = np.zeros_like(quota)
    for expert, ranks in replicas.items():
        for source in range(quota.shape[0]):
            cumulative = 0
            for rank in _source_replica_order(source, ranks):
                cumulative += int(quota[source, expert, rank])
                prefix[source, expert, rank] = cumulative
    return prefix


def _source_replica_order(source: int, ranks: Sequence[int]) -> tuple[int, ...]:
    return (
        (source,) + tuple(rank for rank in ranks if rank != source)
        if source in ranks
        else tuple(ranks)
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


def replicate_source_top_experts(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    primary: Mapping[int, int],
    *,
    num_ranks: int,
    max_extra_per_rank: int,
    compute_imbalance_limit: float = 1.25,
) -> ReplicaPlacement:
    """Copy each source's top remote experts, then quota-route their compute."""

    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    if compute_imbalance_limit < 1 or not np.isfinite(compute_imbalance_limit):
        raise ValueError("invalid compute imbalance limit")
    arrays = as_routed_arrays(tokens)
    demand = _source_demand(arrays, len(primary), num_ranks)
    primary_rank = np.asarray([primary[expert] for expert in range(len(primary))])
    replicas = {expert: (rank,) for expert, rank in primary.items()}
    extra_copies = 0
    for rank in range(num_ranks):
        candidates = np.flatnonzero((demand[:, rank] > 0) & (primary_rank != rank))
        selected = sorted(
            candidates, key=lambda expert: (-demand[expert, rank], expert)
        )[:max_extra_per_rank]
        for expert in selected:
            replicas[int(expert)] += (rank,)
        extra_copies += len(selected)

    replicas, primary_rank, replica_mask = _replica_arrays(arrays, replicas, num_ranks)
    routing = _default_routing(primary_rank, replica_mask, num_ranks)
    instance_quota, compute = _greedy_instance_quotas(
        demand.sum(axis=1), replicas, num_ranks
    )
    quota = _source_quotas(demand, instance_quota, routing)
    average = demand.sum() / num_ranks
    capacity = max(int(compute.max()), int(np.ceil(average * compute_imbalance_limit)))
    changed = True
    while changed:
        changed = False
        for source in range(num_ranks):
            room = capacity - int(compute[source])
            local_experts = sorted(
                (expert for expert, ranks in replicas.items() if source in ranks),
                key=lambda expert: (-demand[expert, source], expert),
            )
            for expert in local_experts:
                for target in replicas[expert]:
                    if target == source or not room:
                        continue
                    moved = min(room, int(quota[source, expert, target]))
                    quota[source, expert, target] -= moved
                    quota[source, expert, source] += moved
                    compute[target] -= moved
                    compute[source] += moved
                    room -= moved
                    changed |= bool(moved)
    routing = np.where(quota.sum(axis=2) > 0, quota.argmax(axis=2), routing)
    serialized_quota = tuple(
        tuple(
            (
                tuple(int(quota[source, expert, rank]) for rank in replicas[expert])
                if demand[expert, source]
                else tuple(
                    int(rank == routing[source, expert]) for rank in replicas[expert]
                )
            )
            for expert in range(len(primary_rank))
        )
        for source in range(num_ranks)
    )
    return ReplicaPlacement(
        replicas,
        tuple(tuple(int(rank) for rank in row) for row in routing),
        _route_quota(arrays, quota, demand, replicas),
        extra_copies,
        quota_by_source=serialized_quota,
        source_demand=demand,
    )


def _waterfill(loads: np.ndarray, ranks: Sequence[int], demand: int) -> np.ndarray:
    """Split integer demand to minimize the maximum selected-rank load."""

    allocation = np.zeros(len(loads), dtype=np.int64)
    order = np.asarray(sorted(ranks, key=lambda rank: (loads[rank], rank)))
    level = int(loads[order[0]])
    remaining = demand
    for width in range(1, len(order)):
        next_level = int(loads[order[width]])
        needed = (next_level - level) * width
        if remaining < needed:
            quotient, remainder = divmod(remaining, width)
            allocation[order[:width]] = level - loads[order[:width]] + quotient
            allocation[order[:remainder]] += 1
            return allocation
        remaining -= needed
        level = next_level
    quotient, remainder = divmod(remaining, len(order))
    allocation[order] = level - loads[order] + quotient
    allocation[order[:remainder]] += 1
    return allocation


def _max_flow(capacity: list[dict[int, int]], source: int, sink: int) -> int:
    """Dinic max flow for the small expert-to-rank quota graph."""

    flow = 0
    while True:
        level = [-1] * len(capacity)
        level[source] = 0
        queue = [source]
        for node in queue:
            for target, remaining in capacity[node].items():
                if remaining and level[target] < 0:
                    level[target] = level[node] + 1
                    queue.append(target)
        if level[sink] < 0:
            return flow
        edge = [iter(row) for row in capacity]

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            for target in edge[node]:
                remaining = capacity[node][target]
                if remaining and level[target] == level[node] + 1:
                    sent = send(target, min(amount, remaining))
                    if sent:
                        capacity[node][target] -= sent
                        capacity[target][node] += sent
                        return sent
            return 0

        while sent := send(source, sum(capacity[source].values())):
            flow += sent


def _flow_quotas(
    demand: np.ndarray,
    replicas: Mapping[int, Sequence[int]],
    num_ranks: int,
    rank_capacity: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    num_experts = len(demand)
    source = num_experts + num_ranks
    sink = source + 1
    capacity = [dict() for _ in range(sink + 1)]

    def add(left: int, right: int, amount: int) -> None:
        capacity[left][right] = amount
        capacity[right][left] = 0

    total = int(demand.sum())
    for expert, amount in enumerate(demand):
        add(source, expert, int(amount))
        for rank in replicas[expert]:
            add(expert, num_experts + rank, int(amount))
    for rank in range(num_ranks):
        add(num_experts + rank, sink, rank_capacity)
    if _max_flow(capacity, source, sink) != total:
        return None
    quota = np.zeros((num_experts, num_ranks), dtype=np.int64)
    for expert in range(num_experts):
        for rank in replicas[expert]:
            quota[expert, rank] = capacity[num_experts + rank][expert]
    return quota, quota.sum(axis=0)


def _instance_quotas(
    demand: np.ndarray,
    replicas: Mapping[int, Sequence[int]],
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    low = (int(demand.sum()) + num_ranks - 1) // num_ranks
    high = int(demand.sum())
    best = None
    while low <= high:
        middle = (low + high) // 2
        result = _flow_quotas(demand, replicas, num_ranks, middle)
        if result is None:
            low = middle + 1
        else:
            best = result
            high = middle - 1
    if best is None:
        raise RuntimeError("replica graph cannot serve expert demand")
    return best


def _greedy_instance_quotas(
    demand: np.ndarray,
    replicas: Mapping[int, Sequence[int]],
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    quota = np.zeros((len(demand), num_ranks), dtype=np.int64)
    loads = np.zeros(num_ranks, dtype=np.int64)
    order = sorted(
        range(len(demand)),
        key=lambda expert: (len(replicas[expert]) > 1, -demand[expert], expert),
    )
    for expert in order:
        quota[expert] = _waterfill(loads, replicas[int(expert)], int(demand[expert]))
        loads += quota[expert]
    return quota, loads


def _source_quotas(
    source_demand: np.ndarray,
    instance_quota: np.ndarray,
    routing: np.ndarray,
) -> np.ndarray:
    num_experts, num_ranks = source_demand.shape
    quota = np.zeros((num_ranks, num_experts, num_ranks), dtype=np.int64)
    for expert in range(num_experts):
        capacity = instance_quota[expert].copy()
        remaining = source_demand[expert].copy()
        local = np.minimum(remaining, capacity)
        quota[np.arange(num_ranks), expert, np.arange(num_ranks)] = local
        remaining -= local
        capacity -= local
        for source in np.argsort(-remaining, kind="stable"):
            amount = int(remaining[source])
            while amount:
                targets = np.flatnonzero(capacity)
                target = min(
                    targets,
                    key=lambda rank: (
                        rank != routing[source, expert],
                        -capacity[rank],
                        rank,
                    ),
                )
                moved = min(amount, int(capacity[target]))
                quota[source, expert, target] += moved
                capacity[target] -= moved
                amount -= moved
    return quota


def _joint_quotas(
    source_demand: np.ndarray,
    replicas: Mapping[int, Sequence[int]],
    routing: np.ndarray,
) -> np.ndarray:
    """Jointly minimize peak rank load, then remote source-expert traffic."""

    num_experts, num_ranks = source_demand.shape
    expert_demand = source_demand.sum(axis=1)
    _, balanced_load = _instance_quotas(expert_demand, replicas, num_ranks)
    rank_capacity = int(balanced_load.max(initial=0))
    quota = np.zeros((num_ranks, num_experts, num_ranks), dtype=np.int64)
    fixed_load = np.zeros(num_ranks, dtype=np.int64)
    for expert, ranks in replicas.items():
        if len(ranks) == 1:
            target = ranks[0]
            quota[:, expert, target] = source_demand[expert]
            fixed_load[target] += expert_demand[expert]
    if np.any(fixed_load > rank_capacity):
        raise RuntimeError("fixed expert load exceeds balanced rank capacity")
    pairs = [
        (source, expert, int(source_demand[expert, source]))
        for expert in range(num_experts)
        if len(replicas[expert]) > 1
        for source in range(num_ranks)
        if source_demand[expert, source]
    ]
    if not pairs:
        return quota
    pair_offset = 1
    flexible = tuple(expert for expert, ranks in replicas.items() if len(ranks) > 1)
    expert_offset = pair_offset + len(pairs)
    expert_node = {
        expert: expert_offset + index for index, expert in enumerate(flexible)
    }
    rank_offset = expert_offset + len(flexible)
    sink = rank_offset + num_ranks
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int, cost: int) -> int:
        index = len(graph[left])
        graph[left].append([right, len(graph[right]), capacity, cost])
        graph[right].append([left, index, 0, -cost])
        return index

    local_edges = {}
    expert_rank_edges = {}
    total = sum(amount for _, _, amount in pairs)
    for pair_index, (source, expert, amount) in enumerate(pairs):
        node = pair_offset + pair_index
        add_edge(0, node, amount, 0)
        if source in replicas[expert]:
            local_edges[source, expert] = (
                node,
                add_edge(node, rank_offset + source, amount, 0),
            )
        add_edge(node, expert_node[expert], amount, 1)
    for expert in flexible:
        node = expert_node[expert]
        for rank in replicas[expert]:
            expert_rank_edges[expert, rank] = (
                node,
                add_edge(node, rank_offset + rank, int(expert_demand[expert]), 0),
            )
    for rank in range(num_ranks):
        add_edge(rank_offset + rank, sink, rank_capacity - int(fixed_load[rank]), 0)

    flow = 0
    while flow < total:
        distance = [None] * len(graph)
        previous = [None] * len(graph)
        distance[0] = 0
        queue = deque([0])
        queued = [False] * len(graph)
        queued[0] = True
        while queue:
            node = queue.popleft()
            queued[node] = False
            for edge_index, (target, _, capacity, cost) in enumerate(graph[node]):
                candidate = distance[node] + cost
                if capacity and (
                    distance[target] is None or candidate < distance[target]
                ):
                    distance[target] = candidate
                    previous[target] = (node, edge_index)
                    if not queued[target]:
                        queue.append(target)
                        queued[target] = True
        if previous[sink] is None:
            raise RuntimeError("replica graph cannot serve source-expert demand")
        moved = total - flow
        node = sink
        while node:
            parent, edge_index = previous[node]
            moved = min(moved, graph[parent][edge_index][2])
            node = parent
        node = sink
        while node:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge[2] -= moved
            graph[node][edge[1]][2] += moved
            node = parent
        flow += moved

    instance_quota = np.zeros((num_experts, num_ranks), dtype=np.int64)
    for (source, expert), (node, edge_index) in local_edges.items():
        edge = graph[node][edge_index]
        instance_quota[expert, source] += graph[edge[0]][edge[1]][2]
    for (expert, rank), (node, edge_index) in expert_rank_edges.items():
        edge = graph[node][edge_index]
        instance_quota[expert, rank] += graph[edge[0]][edge[1]][2]
    for expert, ranks in replicas.items():
        if len(ranks) == 1:
            instance_quota[expert, ranks[0]] = expert_demand[expert]
    return _source_quotas(source_demand, instance_quota, routing)


def _gains_and_traffic(
    arrays: RoutedArrays,
    primary: np.ndarray,
    replica_mask: np.ndarray,
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return replica gains, traffic, and source-expert demand in one scan."""

    gains = np.zeros((len(primary), num_ranks), dtype=np.int64)
    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    source_demand = np.zeros((len(primary), num_ranks), dtype=np.int64)
    for start in range(0, len(arrays), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(arrays))
        source = arrays.source_rank[start:end]
        topk = arrays.topk_experts[start:end]
        count = arrays.count[start:end]
        destinations = _destinations(source, topk, primary, replica_mask)
        for column in range(topk.shape[1]):
            np.add.at(source_demand, (topk[:, column], source), count)
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
    return gains, traffic, source_demand


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
    objective: str = "ingress-egress",
    hot_experts: int = 16,
    candidate_ranks: int = 4,
    compute_imbalance_limit: float = 1.25,
    max_extra_per_rank: int = 0,
) -> ReplicaPlacement:
    """Greedily add the globally best feasible source-local replica."""

    if objective not in {"remote", "ingress-egress"}:
        raise ValueError("invalid replication objective")
    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if hot_experts < 0 or candidate_ranks < 1:
        raise ValueError("invalid replication limits")
    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    if compute_imbalance_limit < 1:
        raise ValueError("invalid replication compute limit")
    arrays = as_routed_arrays(tokens)
    replicas, primary, replica_mask = _replica_arrays(arrays, placement, num_ranks)
    routing = _default_routing(primary, replica_mask, num_ranks)
    if max_extra_per_rank == 0:
        source_demand = np.zeros((len(primary), num_ranks), dtype=np.int64)
        metrics = _route(
            arrays,
            primary,
            replica_mask,
            num_ranks=num_ranks,
            source_demand=source_demand,
        )
        return ReplicaPlacement(
            replicas,
            tuple(tuple(int(rank) for rank in row) for row in routing),
            metrics,
            0,
            source_demand=source_demand,
        )
    gains, traffic, source_demand = _gains_and_traffic(
        arrays, primary, replica_mask, num_ranks
    )
    total_demand = source_demand.sum(axis=1)
    compute = np.zeros(num_ranks, dtype=np.int64)
    observed = primary >= 0
    np.add.at(compute, primary[observed], total_demand[observed])
    average = float(compute.sum()) / num_ranks
    extra_by_rank = np.zeros(num_ranks, dtype=np.int64)
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
    )
    routing = _default_routing(primary, replica_mask, num_ranks)
    return ReplicaPlacement(
        replicas,
        tuple(tuple(int(rank) for rank in row) for row in routing),
        metrics,
        added,
        source_demand=source_demand,
    )


def balance_replica_compute(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: ReplicaPlacement,
    *,
    num_ranks: int,
    max_extra_per_rank: int = 0,
) -> ReplicaPlacement:
    """Quickly balance source-expert demand with communication constraints."""

    if max_extra_per_rank < 0:
        raise ValueError("invalid balance replication limits")
    arrays = as_routed_arrays(tokens)
    replicas, primary, replica_mask = _replica_arrays(
        arrays, placement.replicas_by_expert, num_ranks
    )
    routing = np.asarray(placement.routing_by_source, dtype=np.intp).copy()
    source_demand = placement.source_demand
    if source_demand is None:
        source_demand = _source_demand(arrays, len(primary), num_ranks)
    expert_demand = source_demand.sum(axis=1)
    remote_demand = source_demand.copy()
    remote_demand[routing.T == np.arange(num_ranks)] = 0
    remote_sources = np.argsort(-remote_demand, axis=1, kind="stable")
    expert_order = np.argsort(-expert_demand, kind="stable")
    balance_by_rank = np.zeros(num_ranks, dtype=np.int64)
    added = 0
    while added < num_ranks * max_extra_per_rank:
        instance_quota, compute = _greedy_instance_quotas(
            expert_demand, replicas, num_ranks
        )
        current_key = (int(compute.max()), int(np.square(compute).sum()))
        best = None
        for expert in expert_order:
            if not expert_demand[expert]:
                continue
            base = compute - instance_quota[expert]
            # ponytail: one remote-priority target per expert; widen only if quality regresses.
            for target_value in remote_sources[expert]:
                target = int(target_value)
                if not remote_demand[expert, target]:
                    break
                if replica_mask[expert, target] or (
                    balance_by_rank[target] >= max_extra_per_rank
                ):
                    continue
                ranks = replicas[int(expert)] + (target,)
                allocation = _waterfill(base, ranks, int(expert_demand[expert]))
                next_compute = base + allocation
                compute_key = (
                    int(next_compute.max()),
                    int(np.square(next_compute).sum()),
                )
                if compute_key < current_key:
                    remote = int(
                        expert_demand[expert]
                        - sum(
                            min(
                                int(source_demand[expert, rank]),
                                int(allocation[rank]),
                            )
                            for rank in ranks
                        )
                    )
                    candidate = (
                        compute_key
                        + (
                            remote,
                            -int(source_demand[expert, target]),
                            expert,
                            target,
                        ),
                        int(expert),
                        target,
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                break
        if best is None:
            break
        _, expert, target = best
        replicas[expert] += (target,)
        replica_mask[expert, target] = True
        balance_by_rank[target] += 1
        added += 1
    quota = _joint_quotas(source_demand, replicas, routing)
    routing = np.where(quota.sum(axis=2) > 0, quota.argmax(axis=2), routing)
    metrics = _route_quota(
        arrays,
        quota,
        source_demand,
        replicas,
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
        source_demand,
    )
