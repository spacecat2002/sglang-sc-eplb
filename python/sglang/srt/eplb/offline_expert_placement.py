"""Offline primary-expert placement baselines for MoE routing traces."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping, Optional, Sequence

from .bundle_aware_replica_planner import RoutedToken
from .co_routing_graph_solver import CoRoutingGraph


@dataclass(frozen=True)
class OfflinePlacement:
    rank_by_expert: dict[int, int]
    experts_by_rank: dict[int, tuple[int, ...]]
    iterations: int = 0


def _placement(rank_by_expert: Mapping[int, int], num_ranks: int, iterations=0):
    rank_map = dict(rank_by_expert)
    return OfflinePlacement(
        rank_map,
        {
            rank: tuple(
                sorted(
                    expert
                    for expert, expert_rank in rank_map.items()
                    if expert_rank == rank
                )
            )
            for rank in range(num_ranks)
        },
        iterations,
    )


def _affinity(graph: CoRoutingGraph, expert: int, group: Sequence[int]) -> int:
    return sum(graph.adjacency.get(expert, {}).get(other, 0) for other in group)


def _spectral_groups(
    graph: CoRoutingGraph, experts: Sequence[int], num_groups: int
) -> list[list[int]]:
    """Deterministic normalized spectral clustering without a sklearn dependency."""

    if num_groups < 1 or len(experts) < num_groups:
        raise ValueError("spectral grouping requires at least one expert per group")
    if num_groups == 1:
        return [list(sorted(experts))]

    import numpy as np

    experts = tuple(sorted(experts))
    index = {expert: position for position, expert in enumerate(experts)}
    affinity = np.zeros((len(experts), len(experts)), dtype=np.float64)
    for (left, right), weight in graph.edges.items():
        if left in index and right in index:
            affinity[index[left], index[right]] = weight
            affinity[index[right], index[left]] = weight
    degree = affinity.sum(axis=1)
    scale = np.zeros_like(degree)
    nonzero = degree > 0
    scale[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalized = affinity * scale[:, None] * scale[None, :]
    _, vectors = np.linalg.eigh(normalized)
    embedding = vectors[:, -num_groups:]
    norms = np.linalg.norm(embedding, axis=1)
    embedding = embedding / np.maximum(norms[:, None], 1e-12)

    centers = [0]
    while len(centers) < num_groups:
        distance = np.min(
            np.stack(
                [
                    np.sum((embedding - embedding[center]) ** 2, axis=1)
                    for center in centers
                ]
            ),
            axis=0,
        )
        distance[centers] = -1
        centers.append(int(np.argmax(distance)))
    centers_array = embedding[centers].copy()
    labels = np.zeros(len(experts), dtype=np.int64)
    for _ in range(32):
        distances = np.sum(
            (embedding[:, None, :] - centers_array[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        for empty in set(range(num_groups)) - set(new_labels.tolist()):
            counts = np.bincount(new_labels, minlength=num_groups)
            donor = int(np.argmax(counts))
            donor_points = np.flatnonzero(new_labels == donor)
            moved = donor_points[
                np.argmax(
                    np.sum(
                        (embedding[donor_points] - centers_array[donor]) ** 2,
                        axis=1,
                    )
                )
            ]
            new_labels[moved] = empty
        new_centers = np.stack(
            [embedding[new_labels == group].mean(axis=0) for group in range(num_groups)]
        )
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels, centers_array = new_labels, new_centers

    return [
        [expert for expert, label in zip(experts, labels.tolist()) if label == group]
        for group in range(num_groups)
    ]


def _fill_minimum(graph: CoRoutingGraph, groups: list[list[int]], minimum: int) -> None:
    for target in range(len(groups)):
        while len(groups[target]) < minimum:
            candidates = []
            for donor, donor_experts in enumerate(groups):
                if len(donor_experts) <= minimum:
                    continue
                for expert in donor_experts:
                    loss = _affinity(
                        graph,
                        expert,
                        [other for other in donor_experts if other != expert],
                    )
                    gain = _affinity(graph, expert, groups[target])
                    candidates.append((loss - gain, expert, donor))
            if not candidates:
                raise ValueError("cannot satisfy minimum experts per group")
            _, expert, donor = min(candidates)
            groups[donor].remove(expert)
            groups[target].append(expert)


def _controlled_groups(
    graph: CoRoutingGraph,
    experts: Sequence[int],
    num_groups: int,
    nonuniform_ratio: float,
) -> list[list[int]]:
    groups = _spectral_groups(graph, experts, num_groups)
    ideal = len(experts) // num_groups
    delta = max(1, round(ideal * nonuniform_ratio))
    minimum = max(1, ideal - delta)
    maximum = ideal + delta
    overflow = []
    for group in groups:
        if len(group) <= maximum:
            continue
        ordered = sorted(
            group,
            key=lambda expert: (-_affinity(graph, expert, group), expert),
        )
        group[:] = ordered[:maximum]
        overflow.extend(ordered[maximum:])
    for expert in sorted(overflow):
        candidates = [
            group for group in range(num_groups) if len(groups[group]) < maximum
        ]
        target = min(
            candidates,
            key=lambda group: (
                -_affinity(graph, expert, groups[group]),
                len(groups[group]),
                group,
            ),
        )
        groups[target].append(expert)
    _fill_minimum(graph, groups, minimum)
    return groups


def grace_hierarchical_placement(
    graph: CoRoutingGraph,
    *,
    num_ranks: int,
    ranks_per_node: Optional[int] = None,
    nonuniform_ratio: float = 0.15,
) -> OfflinePlacement:
    """Reproduce GRACE-MoE's offline hierarchical expert grouping.

    Node groups use unconstrained spectral clustering. GPU groups within each
    node use Algorithm 2's controlled non-uniform bounds and affinity refill.
    """

    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if not 0 <= nonuniform_ratio < 1:
        raise ValueError("nonuniform_ratio must be in [0, 1)")
    ranks_per_node = ranks_per_node or num_ranks
    if ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if len(graph.experts) < num_ranks:
        raise ValueError("GRACE grouping requires at least one expert per rank")

    num_nodes = num_ranks // ranks_per_node
    node_groups = _spectral_groups(graph, graph.experts, num_nodes)
    _fill_minimum(graph, node_groups, ranks_per_node)
    rank_by_expert = {}
    for node, node_experts in enumerate(node_groups):
        gpu_groups = _controlled_groups(
            graph, node_experts, ranks_per_node, nonuniform_ratio
        )
        for local_rank, experts in enumerate(gpu_groups):
            rank = node * ranks_per_node + local_rank
            rank_by_expert.update({expert: rank for expert in experts})
    return _placement(rank_by_expert, num_ranks)


def evaluate_topology_hypergraph_objective(
    routed_tokens: Iterable[RoutedToken],
    rank_by_expert: Mapping[int, int],
    *,
    ranks_per_node: int,
    rdma_cost: float = 4.0,
) -> float:
    """Count source-independent GPU and node cuts for complete Top-K bundles."""

    if ranks_per_node < 1:
        raise ValueError("ranks_per_node must be positive")
    if rdma_cost < 1:
        raise ValueError("rdma_cost must be at least 1")
    total = 0.0
    for token in routed_tokens:
        ranks = {rank_by_expert[expert] for expert in token.topk_experts}
        nodes = {rank // ranks_per_node for rank in ranks}
        total += token.count * (len(ranks) - len(nodes) + rdma_cost * (len(nodes) - 1))
    return total


def refine_load_constrained_hypergraph_placement(
    routed_tokens: Iterable[RoutedToken],
    initial_rank_by_expert: Mapping[int, int],
    *,
    num_ranks: int,
    ranks_per_node: Optional[int] = None,
    rdma_cost: float = 4.0,
    max_load_ratio: float = 1.2,
    max_experts_per_rank: Optional[int] = None,
    min_experts_per_rank: int = 1,
    max_rounds: Optional[int] = 64,
) -> OfflinePlacement:
    """Refine full Top-K hyperedges under load and expert-memory constraints."""

    if num_ranks < 1 or max_load_ratio < 1:
        raise ValueError("num_ranks and max_load_ratio must be at least 1")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    ranks_per_node = ranks_per_node or num_ranks
    if ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")

    tokens = list(routed_tokens)
    rank_by_expert = dict(initial_rank_by_expert)
    if any(rank < 0 or rank >= num_ranks for rank in rank_by_expert.values()):
        raise ValueError("initial placement contains an invalid rank")
    observed = {expert for token in tokens for expert in token.topk_experts}
    if observed - rank_by_expert.keys():
        raise ValueError("initial placement is missing observed experts")
    expert_count = len(rank_by_expert)
    max_experts_per_rank = max_experts_per_rank or ceil(expert_count / num_ranks)
    if min_experts_per_rank < 0 or min_experts_per_rank > max_experts_per_rank:
        raise ValueError("invalid per-rank expert bounds")

    demand = {expert: 0 for expert in rank_by_expert}
    incidence: dict[int, set[int]] = {expert: set() for expert in rank_by_expert}
    for bundle, token in enumerate(tokens):
        for expert in token.topk_experts:
            demand[expert] += token.count
            incidence[expert].add(bundle)
    load_cap = max(
        max(demand.values(), default=0),
        sum(demand.values()) / num_ranks * max_load_ratio,
    )
    loads = [0] * num_ranks
    counts = [0] * num_ranks
    for expert, rank in rank_by_expert.items():
        loads[rank] += demand[expert]
        counts[rank] += 1

    def edge_cost(token: RoutedToken, overrides=()) -> float:
        changed = dict(overrides)
        ranks = {
            changed.get(expert, rank_by_expert[expert]) for expert in token.topk_experts
        }
        nodes = {rank // ranks_per_node for rank in ranks}
        return token.count * (len(ranks) - len(nodes) + rdma_cost * (len(nodes) - 1))

    def delta(affected: set[int], overrides) -> float:
        return sum(
            edge_cost(tokens[bundle], overrides) - edge_cost(tokens[bundle])
            for bundle in affected
        )

    def quality(objective: float, candidate_loads: Sequence[int]):
        violation = sum(max(0.0, load - load_cap) for load in candidate_loads)
        return violation, objective, sum(load * load for load in candidate_loads)

    objective = evaluate_topology_hypergraph_objective(
        tokens,
        rank_by_expert,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    rounds = 0
    while max_rounds is None or rounds < max_rounds:
        current = quality(objective, loads)
        best = None
        experts = sorted(rank_by_expert)

        for expert in experts:
            source = rank_by_expert[expert]
            if counts[source] <= min_experts_per_rank:
                continue
            for target in range(num_ranks):
                if target == source or counts[target] >= max_experts_per_rank:
                    continue
                candidate_loads = list(loads)
                candidate_loads[source] -= demand[expert]
                candidate_loads[target] += demand[expert]
                candidate_objective = objective + delta(
                    incidence[expert], ((expert, target),)
                )
                key = (
                    *quality(candidate_objective, candidate_loads),
                    0,
                    expert,
                    target,
                )
                if key[:3] < current and (best is None or key < best[0]):
                    best = (key, "move", expert, target, candidate_objective)

        for left_index, left in enumerate(experts):
            left_rank = rank_by_expert[left]
            for right in experts[left_index + 1 :]:
                right_rank = rank_by_expert[right]
                if left_rank == right_rank:
                    continue
                candidate_loads = list(loads)
                candidate_loads[left_rank] += demand[right] - demand[left]
                candidate_loads[right_rank] += demand[left] - demand[right]
                candidate_objective = objective + delta(
                    incidence[left] | incidence[right],
                    ((left, right_rank), (right, left_rank)),
                )
                key = (*quality(candidate_objective, candidate_loads), 1, left, right)
                if key[:3] < current and (best is None or key < best[0]):
                    best = (key, "swap", left, right, candidate_objective)

        if best is None:
            break
        _, action, left, right, objective = best
        if action == "move":
            source = rank_by_expert[left]
            rank_by_expert[left] = right
            loads[source] -= demand[left]
            loads[right] += demand[left]
            counts[source] -= 1
            counts[right] += 1
        else:
            left_rank = rank_by_expert[left]
            right_rank = rank_by_expert[right]
            rank_by_expert[left], rank_by_expert[right] = right_rank, left_rank
            loads[left_rank] += demand[right] - demand[left]
            loads[right_rank] += demand[left] - demand[right]
        rounds += 1

    return _placement(rank_by_expert, num_ranks, rounds)
