"""GRACE-MoE offline hierarchical expert grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .expert_affinity_graph import CoRoutingGraph


@dataclass(frozen=True)
class GracePlacement:
    rank_by_expert: dict[int, int]
    experts_by_rank: dict[int, tuple[int, ...]]


def _affinity(graph: CoRoutingGraph, expert: int, group: Sequence[int]) -> int:
    return sum(graph.adjacency.get(expert, {}).get(other, 0) for other in group)


def _spectral_groups(
    graph: CoRoutingGraph, experts: Sequence[int], num_groups: int
) -> list[list[int]]:
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
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1)[:, None], 1e-12)

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
    center_values = embedding[centers].copy()
    labels = np.zeros(len(experts), dtype=np.int64)
    for _ in range(32):
        distances = np.sum(
            (embedding[:, None, :] - center_values[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        for empty in set(range(num_groups)) - set(new_labels.tolist()):
            counts = np.bincount(new_labels, minlength=num_groups)
            donor = int(np.argmax(counts))
            points = np.flatnonzero(new_labels == donor)
            moved = points[
                np.argmax(
                    np.sum(
                        (embedding[points] - center_values[donor]) ** 2,
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
        labels, center_values = new_labels, new_centers
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
            group, key=lambda expert: (-_affinity(graph, expert, group), expert)
        )
        group[:] = ordered[:maximum]
        overflow.extend(ordered[maximum:])
    for expert in sorted(overflow):
        candidates = [
            index for index, group in enumerate(groups) if len(group) < maximum
        ]
        target = min(
            candidates,
            key=lambda index: (
                -_affinity(graph, expert, groups[index]),
                len(groups[index]),
                index,
            ),
        )
        groups[target].append(expert)
    _fill_minimum(graph, groups, minimum)
    return groups


def grace_hierarchical_placement(
    graph: CoRoutingGraph,
    *,
    num_ranks: int,
    ranks_per_node: int | None = None,
    nonuniform_ratio: float = 0.15,
) -> GracePlacement:
    """Apply GRACE node-level and controlled GPU-level spectral grouping."""

    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if not 0 <= nonuniform_ratio < 1:
        raise ValueError("nonuniform_ratio must be in [0, 1)")
    ranks_per_node = ranks_per_node or num_ranks
    if ranks_per_node < 1 or num_ranks % ranks_per_node:
        raise ValueError("ranks_per_node must divide num_ranks")
    if len(graph.experts) < num_ranks:
        raise ValueError("GRACE grouping requires at least one expert per rank")

    node_groups = _spectral_groups(graph, graph.experts, num_ranks // ranks_per_node)
    _fill_minimum(graph, node_groups, ranks_per_node)
    rank_by_expert = {}
    for node, node_experts in enumerate(node_groups):
        gpu_groups = _controlled_groups(
            graph, node_experts, ranks_per_node, nonuniform_ratio
        )
        for local_rank, experts in enumerate(gpu_groups):
            rank = node * ranks_per_node + local_rank
            rank_by_expert.update({expert: rank for expert in experts})
    return GracePlacement(
        rank_by_expert,
        {
            rank: tuple(
                sorted(
                    expert
                    for expert, expert_rank in rank_by_expert.items()
                    if expert_rank == rank
                )
            )
            for rank in range(num_ranks)
        },
    )
