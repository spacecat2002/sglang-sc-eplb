"""Constrained hot-expert replication for offline routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .cable_expert_placement import CableMetrics
from .expert_affinity_graph import RoutedArrays, RoutedToken, as_routed_arrays


@dataclass(frozen=True)
class ReplicaEvaluation:
    """Placement and communication metrics for a replicated expert layout."""

    replicas_by_expert: dict[int, tuple[int, ...]]
    experts_by_rank: dict[int, tuple[int, ...]]
    metrics: CableMetrics
    weighted_remote: float


def _route_arrays(
    arrays: RoutedArrays,
    replicas_by_expert: Mapping[int, Sequence[int]],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float,
) -> tuple[CableMetrics, float]:
    """Route every bundle with local-first, destination-reuse policy."""

    traffic = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    compute_load = np.zeros(num_ranks, dtype=np.int64)
    weighted_remote = 0.0
    for source, experts, count in zip(
        arrays.source_rank.tolist(),
        arrays.topk_experts.tolist(),
        arrays.count.tolist(),
    ):
        selected: list[int] = []
        selected_set: set[int] = set()
        for expert in experts:
            candidates = tuple(replicas_by_expert[int(expert)])
            if source in candidates:
                rank = source
            else:
                rank = min(
                    candidates,
                    key=lambda candidate: (
                        candidate not in selected_set,
                        compute_load[candidate],
                        candidate,
                    ),
                )
            selected.append(rank)
            selected_set.add(rank)
            compute_load[rank] += count
        for rank in selected_set:
            if rank == source:
                continue
            traffic[source, rank] += count
            weighted_remote += count * (
                1.0
                if rank // ranks_per_node == source // ranks_per_node
                else rdma_cost
            )

    egress = traffic.sum(axis=1)
    metrics = CableMetrics(
        remote=int(traffic.sum()),
        max_pair_traffic=int(traffic.max()),
        max_ingress=int(traffic.sum(axis=0).max()),
        max_egress=int(egress.max()),
        compute_load=tuple(int(load) for load in compute_load),
    )
    return metrics, weighted_remote


def evaluate_replicated_placement(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    replicas_by_expert: Mapping[int, Sequence[int]],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
) -> ReplicaEvaluation:
    """Evaluate a fixed replica map using deterministic locality-first routing."""

    if num_ranks < 1 or ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("invalid rank topology")
    if rdma_cost < 1:
        raise ValueError("rdma_cost must be at least 1")
    arrays = as_routed_arrays(tokens)
    normalized: dict[int, tuple[int, ...]] = {}
    for expert, ranks in replicas_by_expert.items():
        values = tuple(dict.fromkeys(int(rank) for rank in ranks))
        if not values or any(rank < 0 or rank >= num_ranks for rank in values):
            raise ValueError("replica map contains an invalid rank")
        normalized[int(expert)] = values
    missing = set(np.unique(arrays.topk_experts).tolist()) - set(normalized)
    if missing:
        raise ValueError(f"replica map is missing experts: {sorted(missing)}")
    metrics, weighted_remote = _route_arrays(
        arrays,
        normalized,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    experts_by_rank = {
        rank: tuple(
            expert
            for expert in sorted(normalized)
            for replica_rank in normalized[expert]
            if replica_rank == rank
        )
        for rank in range(num_ranks)
    }
    return ReplicaEvaluation(
        replicas_by_expert=normalized,
        experts_by_rank=experts_by_rank,
        metrics=metrics,
        weighted_remote=weighted_remote,
    )


def _source_demand(arrays: RoutedArrays, num_experts: int, num_ranks: int) -> np.ndarray:
    demand = np.zeros((num_experts, num_ranks), dtype=np.int64)
    for column in range(arrays.topk_experts.shape[1]):
        np.add.at(
            demand,
            (arrays.topk_experts[:, column], arrays.source_rank),
            arrays.count,
        )
    return demand


def hot_expert_replication(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
    extra_copies: int = 0,
    hot_experts: int = 16,
    max_replicas_per_expert: int = 2,
    compute_imbalance_limit: float = 1.25,
    min_remote_saving: int = 1,
) -> ReplicaEvaluation:
    """Greedily add source-local copies under memory and load constraints."""

    if extra_copies < 0 or hot_experts < 0 or max_replicas_per_expert < 1:
        raise ValueError("replication limits must be non-negative")
    if compute_imbalance_limit < 1 or min_remote_saving < 1:
        raise ValueError("invalid replication quality limits")
    arrays = as_routed_arrays(tokens)
    if not placement:
        raise ValueError("placement must not be empty")
    num_experts = max(int(arrays.topk_experts.max()), max(placement)) + 1
    source_demand = _source_demand(arrays, num_experts, num_ranks)
    replicas = {int(expert): (int(rank),) for expert, rank in placement.items()}
    current = evaluate_replicated_placement(
        arrays,
        replicas,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    if extra_copies == 0 or hot_experts == 0:
        return current

    observed = [expert for expert in np.argsort(-source_demand.sum(axis=1)) if expert in replicas]
    observed = observed[:hot_experts]
    for _ in range(extra_copies):
        average = sum(current.metrics.compute_load) / num_ranks
        load_limit = max(max(current.metrics.compute_load), average * compute_imbalance_limit)
        best: tuple[tuple[object, ...], int, int, ReplicaEvaluation] | None = None
        for expert_value in observed:
            expert = int(expert_value)
            existing = replicas[expert]
            if len(existing) >= max_replicas_per_expert:
                continue
            candidate_ranks = sorted(
                (
                    rank
                    for rank in range(num_ranks)
                    if rank not in existing and source_demand[expert, rank] > 0
                ),
                key=lambda rank: (-source_demand[expert, rank], rank),
            )
            for rank in candidate_ranks:
                candidate_map = dict(replicas)
                candidate_map[expert] = existing + (rank,)
                candidate = evaluate_replicated_placement(
                    arrays,
                    candidate_map,
                    num_ranks=num_ranks,
                    ranks_per_node=ranks_per_node,
                    rdma_cost=rdma_cost,
                )
                if max(candidate.metrics.compute_load) > load_limit:
                    continue
                saving = current.metrics.remote - candidate.metrics.remote
                if saving < min_remote_saving:
                    continue
                key = (
                    -saving,
                    candidate.metrics.max_ingress,
                    candidate.metrics.max_pair_traffic,
                    candidate.metrics.compute_imbalance,
                    expert,
                    rank,
                )
                if best is None or key < best[0]:
                    best = (key, expert, rank, candidate)
        if best is None:
            break
        _, expert, rank, current = best
        replicas[expert] = replicas[expert] + (rank,)
    return current
