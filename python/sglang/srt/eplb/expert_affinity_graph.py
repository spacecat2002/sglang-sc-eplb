"""Expert co-activation affinity graph used by GRACE-MoE."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional


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
    routed_tokens: Iterable[RoutedToken],
    *,
    experts: Optional[Iterable[int]] = None,
) -> CoRoutingGraph:
    """Build the weighted expert co-activation graph for GRACE."""

    tokens = list(routed_tokens)
    if not tokens and experts is None:
        raise ValueError("routed_tokens or experts must not be empty")
    demand: dict[int, int] = defaultdict(int)
    edges: dict[Edge, int] = defaultdict(int)
    observed = set(experts or ())
    for token in tokens:
        observed.update(token.topk_experts)
        for expert in token.topk_experts:
            demand[expert] += token.count
        for left, right in combinations(token.topk_experts, 2):
            edges[tuple(sorted((left, right)))] += token.count
    ordered = tuple(sorted(observed))
    adjacency: dict[int, dict[int, int]] = {expert: {} for expert in ordered}
    for expert in ordered:
        demand.setdefault(expert, 0)
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(ordered, dict(demand), dict(edges), adjacency)


def evaluate_primary_remote(
    routed_tokens: Iterable[RoutedToken], rank_by_expert: Mapping[int, int]
) -> int:
    """Count distinct remote destination ranks touched by each routed token."""

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
    routed_tokens: Iterable[RoutedToken],
    rank_by_expert: Mapping[int, int],
    *,
    ranks_per_node: int,
    rdma_cost: float = 1.0,
) -> float:
    """Count remote rank copies with an optional cross-node cost."""

    if ranks_per_node < 1 or rdma_cost < 1:
        raise ValueError("ranks_per_node and rdma_cost must be at least 1")
    return sum(
        _token_weighted_remote(token, rank_by_expert, ranks_per_node, rdma_cost)
        for token in routed_tokens
    )
