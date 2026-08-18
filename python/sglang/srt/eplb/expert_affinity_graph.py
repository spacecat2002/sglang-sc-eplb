"""Expert co-activation affinity graph used by GRACE-MoE."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional

import numpy as np


Edge = tuple[int, int]


@dataclass(frozen=True)
class RoutedToken:
    """One token or an aggregate with the same source rank and Top-K experts."""

    source_rank: int
    topk_experts: tuple[int, ...]
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive")
        if not self.topk_experts:
            raise ValueError("topk_experts must not be empty")
        if self.source_rank < 0 or min(self.topk_experts) < 0:
            raise ValueError("source rank and expert ids must be non-negative")
        if len(set(self.topk_experts)) != len(self.topk_experts):
            raise ValueError("topk_experts must not contain duplicates")


@dataclass(frozen=True)
class RoutedArrays:
    """Compact routing bundles backed by integer NumPy arrays."""

    source_rank: np.ndarray
    topk_experts: np.ndarray
    count: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_rank)
        topk = np.asarray(self.topk_experts)
        count = np.asarray(self.count)
        if (
            source.ndim != 1
            or topk.ndim != 2
            or count.ndim != 1
            or len(source) != len(topk)
            or len(source) != len(count)
            or not len(source)
            or not topk.shape[1]
        ):
            raise ValueError("invalid routed array shapes")
        if any(value.dtype.kind not in "iu" for value in (source, topk, count)):
            raise ValueError("routed arrays must use integer dtypes")
        if source.min() < 0 or topk.min() < 0 or count.min() < 1:
            raise ValueError("routed arrays contain invalid values")
        for left in range(topk.shape[1]):
            for right in range(left + 1, topk.shape[1]):
                if np.any(topk[:, left] == topk[:, right]):
                    raise ValueError("topk_experts must not contain duplicates")
        object.__setattr__(self, "source_rank", source)
        object.__setattr__(self, "topk_experts", topk)
        object.__setattr__(self, "count", count)

    def __len__(self) -> int:
        return len(self.source_rank)


RoutingInput = Iterable[RoutedToken] | RoutedArrays


def as_routed_arrays(routed_tokens: RoutingInput) -> RoutedArrays:
    if isinstance(routed_tokens, RoutedArrays):
        return routed_tokens
    tokens = list(routed_tokens)
    if not tokens:
        raise ValueError("routed_tokens must not be empty")
    top_k = len(tokens[0].topk_experts)
    if any(len(token.topk_experts) != top_k for token in tokens):
        raise ValueError("all tokens must use the same Top-K")
    return RoutedArrays(
        np.fromiter(
            (token.source_rank for token in tokens), dtype=np.intp, count=len(tokens)
        ),
        np.asarray([token.topk_experts for token in tokens], dtype=np.intp),
        np.fromiter((token.count for token in tokens), dtype=np.int64, count=len(tokens)),
    )


def index_topk_experts(arrays: RoutedArrays, experts: Iterable[int]) -> np.ndarray:
    expert_values = np.asarray(tuple(experts), dtype=np.int64)
    if not len(expert_values):
        raise ValueError("experts must not be empty")
    if np.array_equal(expert_values, np.arange(len(expert_values))):
        if arrays.topk_experts.max() >= len(expert_values):
            raise ValueError("tokens contain an expert outside experts")
        return arrays.topk_experts
    positions = np.searchsorted(expert_values, arrays.topk_experts)
    if np.any(positions >= len(expert_values)) or np.any(
        expert_values[np.minimum(positions, len(expert_values) - 1)]
        != arrays.topk_experts
    ):
        raise ValueError("tokens contain an expert outside experts")
    return positions


@dataclass(frozen=True)
class CoRoutingGraph:
    experts: tuple[int, ...]
    demand: Mapping[int, int]
    edges: Mapping[Edge, int]
    adjacency: Mapping[int, Mapping[int, int]]

    @property
    def weighted_degree(self) -> dict[int, int]:
        return {
            expert: sum(self.adjacency.get(expert, {}).values())
            for expert in self.experts
        }


def build_co_routing_graph(
    routed_tokens: RoutingInput,
    *,
    experts: Optional[Iterable[int]] = None,
) -> CoRoutingGraph:
    """Build the weighted expert co-activation graph for GRACE."""

    if isinstance(routed_tokens, RoutedArrays):
        arrays = routed_tokens
        observed = set(experts or ()) | set(np.unique(arrays.topk_experts).tolist())
        ordered = tuple(sorted(observed))
        if not ordered:
            raise ValueError("routed_tokens or experts must not be empty")
        indexed = index_topk_experts(arrays, ordered)
        size = len(ordered)
        demand_values = np.zeros(size, dtype=np.int64)
        edge_values = np.zeros(size * size, dtype=np.int64)
        weights = arrays.count.astype(np.int64, copy=False)
        for column in range(indexed.shape[1]):
            np.add.at(demand_values, indexed[:, column], weights)
            for right_column in range(column + 1, indexed.shape[1]):
                left = indexed[:, column]
                right = indexed[:, right_column]
                low = np.minimum(left, right)
                high = np.maximum(left, right)
                np.add.at(edge_values, low * size + high, weights)
        demand = {
            expert: int(demand_values[index])
            for index, expert in enumerate(ordered)
        }
        edges = {
            (ordered[index // size], ordered[index % size]): int(weight)
            for index in np.flatnonzero(edge_values)
            if (weight := edge_values[index])
        }
    else:
        tokens = list(routed_tokens)
        if not tokens and experts is None:
            raise ValueError("routed_tokens or experts must not be empty")
        demand = defaultdict(int)
        edges = defaultdict(int)
        observed = set(experts or ())
        for token in tokens:
            observed.update(token.topk_experts)
            for expert in token.topk_experts:
                demand[expert] += token.count
            for left, right in combinations(token.topk_experts, 2):
                edges[tuple(sorted((left, right)))] += token.count
        ordered = tuple(sorted(observed))
    if not ordered:
        raise ValueError("routed_tokens or experts must not be empty")
    adjacency: dict[int, dict[int, int]] = {expert: {} for expert in ordered}
    for expert in ordered:
        demand.setdefault(expert, 0)
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(ordered, dict(demand), dict(edges), adjacency)


def evaluate_primary_remote(
    routed_tokens: RoutingInput, rank_by_expert: Mapping[int, int]
) -> int:
    """Count distinct remote destination ranks touched by each routed token."""

    if isinstance(routed_tokens, RoutedArrays):
        arrays = routed_tokens
        ranks = np.full(int(arrays.topk_experts.max()) + 1, -1, dtype=np.intp)
        for expert, rank in rank_by_expert.items():
            if expert < len(ranks):
                ranks[expert] = rank
        destinations = ranks[arrays.topk_experts]
        if np.any(destinations < 0):
            raise ValueError("rank_by_expert is missing a routed expert")
        return sum(
            int(
                arrays.count[
                    np.any(destinations == rank, axis=1)
                    & (arrays.source_rank != rank)
                ].sum(dtype=np.int64)
            )
            for rank in set(rank_by_expert.values())
        )

    return sum(
        token.count
        * len(
            {
                rank_by_expert[expert]
                for expert in token.topk_experts
                if rank_by_expert[expert] != token.source_rank
            }
        )
        for token in routed_tokens
    )


def _token_weighted_remote(
    token: RoutedToken,
    rank_by_expert: Mapping[int, int],
    ranks_per_node: int,
    rdma_cost: float,
) -> float:
    ranks = {
        rank_by_expert[expert]
        for expert in token.topk_experts
        if rank_by_expert[expert] != token.source_rank
    }
    return token.count * sum(
        1
        if rank // ranks_per_node == token.source_rank // ranks_per_node
        else rdma_cost
        for rank in ranks
    )


def evaluate_weighted_remote(
    routed_tokens: RoutingInput,
    rank_by_expert: Mapping[int, int],
    *,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
) -> float:
    """Count remote rank copies with an optional cross-node cost."""

    if ranks_per_node < 1 or rdma_cost < 1:
        raise ValueError("ranks_per_node and rdma_cost must be at least 1")
    if isinstance(routed_tokens, RoutedArrays):
        arrays = routed_tokens
        ranks = np.full(int(arrays.topk_experts.max()) + 1, -1, dtype=np.intp)
        for expert, rank in rank_by_expert.items():
            if expert < len(ranks):
                ranks[expert] = rank
        destinations = ranks[arrays.topk_experts]
        if np.any(destinations < 0):
            raise ValueError("rank_by_expert is missing a routed expert")
        total = 0.0
        for rank in set(rank_by_expert.values()):
            touched = np.any(destinations == rank, axis=1) & (
                arrays.source_rank != rank
            )
            costs = np.where(
                arrays.source_rank[touched] // ranks_per_node
                == rank // ranks_per_node,
                1.0,
                rdma_cost,
            )
            total += float((arrays.count[touched] * costs).sum())
        return total
    return sum(
        _token_weighted_remote(token, rank_by_expert, ranks_per_node, rdma_cost)
        for token in routed_tokens
    )
