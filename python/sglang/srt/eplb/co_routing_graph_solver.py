"""Offline pairwise co-routing placement for MoE experts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence


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


@dataclass(frozen=True)
class GraphPlacement:
    rank_by_expert: dict[int, int]
    experts_by_rank: dict[int, tuple[int, ...]]
    initial_cut: int
    final_cut: int
    iterations: int


def build_co_routing_graph(
    routed_tokens: Iterable[RoutedToken],
    *,
    experts: Optional[Iterable[int]] = None,
) -> CoRoutingGraph:
    """Project Top-K bundles to weighted pairwise expert affinities."""

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
        for left, right in combinations(sorted(token.topk_experts), 2):
            edges[(left, right)] += token.count
    ordered = tuple(sorted(observed))
    adjacency: dict[int, dict[int, int]] = {expert: {} for expert in ordered}
    for expert in ordered:
        demand.setdefault(expert, 0)
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(ordered, dict(demand), dict(edges), adjacency)


def evaluate_pairwise_cut(
    graph: CoRoutingGraph, rank_by_expert: Mapping[int, int]
) -> int:
    return sum(
        weight
        for (left, right), weight in graph.edges.items()
        if rank_by_expert[left] != rank_by_expert[right]
    )


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
    swap: tuple[int, int, int, int] | None = None,
) -> float:
    ranks = set()
    for expert in token.topk_experts:
        rank = rank_by_expert[expert]
        if swap:
            left, right, left_rank, right_rank = swap
            if expert == left:
                rank = right_rank
            elif expert == right:
                rank = left_rank
        if rank != token.source_rank:
            ranks.add(rank)
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
    rdma_cost: float = 4.0,
) -> float:
    """Count remote rank copies, weighting cross-node copies by ``rdma_cost``."""

    if ranks_per_node < 1 or rdma_cost < 1:
        raise ValueError("ranks_per_node and rdma_cost must be at least 1")
    return sum(
        _token_weighted_remote(token, rank_by_expert, ranks_per_node, rdma_cost)
        for token in routed_tokens
    )


class CoRoutingGraphSolver:
    """Pairwise swaps, optionally reranked by exact weighted communication."""

    def __init__(
        self,
        *,
        num_ranks: int,
        slots_per_rank: int | Sequence[int],
        max_rounds: Optional[int] = 8,
        rerank_candidates: int = 32,
        max_compute_imbalance: Optional[float] = 1.2,
    ) -> None:
        if num_ranks < 1:
            raise ValueError("num_ranks must be positive")
        if max_rounds is not None and max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if rerank_candidates < 1:
            raise ValueError("rerank_candidates must be positive")
        if max_compute_imbalance is not None and max_compute_imbalance < 1:
            raise ValueError("max_compute_imbalance must be at least 1")
        capacities = (
            [slots_per_rank] * num_ranks
            if isinstance(slots_per_rank, int)
            else list(slots_per_rank)
        )
        if len(capacities) != num_ranks or any(value < 0 for value in capacities):
            raise ValueError("slots_per_rank must contain non-negative capacities")
        self.num_ranks = num_ranks
        self.capacities = capacities
        self.max_rounds = max_rounds
        self.rerank_candidates = rerank_candidates
        self.max_compute_imbalance = max_compute_imbalance

    def solve(
        self,
        graph: CoRoutingGraph,
        *,
        routed_tokens: Optional[Iterable[RoutedToken]] = None,
        ranks_per_node: Optional[int] = None,
        rdma_cost: float = 4.0,
    ) -> GraphPlacement:
        if len(graph.experts) > sum(self.capacities):
            raise ValueError("expert count exceeds total rank capacity")
        tokens = None if routed_tokens is None else list(routed_tokens)
        if tokens is not None:
            if ranks_per_node is None:
                ranks_per_node = self.num_ranks
            if not tokens:
                raise ValueError("routed_tokens must not be empty")
            if ranks_per_node < 1 or self.num_ranks % ranks_per_node:
                raise ValueError("ranks_per_node must divide num_ranks")
            if rdma_cost < 1:
                raise ValueError("rdma_cost must be at least 1")
            if any(token.source_rank >= self.num_ranks for token in tokens):
                raise ValueError("routed token source rank exceeds num_ranks")
        assignment = self._initial_assignment(graph)
        rank_by_expert = self._rank_map(assignment)
        initial_cut = evaluate_pairwise_cut(graph, rank_by_expert)
        rank_load = [
            sum(graph.demand[expert] for expert in experts) for experts in assignment
        ]
        token_costs = (
            [
                _token_weighted_remote(token, rank_by_expert, ranks_per_node, rdma_cost)
                for token in tokens
            ]
            if tokens is not None
            else None
        )
        token_indexes: dict[int, set[int]] = defaultdict(set)
        if tokens is not None:
            for index, token in enumerate(tokens):
                for expert in token.topk_experts:
                    token_indexes[expert].add(index)
        rounds = 0
        while self.max_rounds is None or rounds < self.max_rounds:
            move_gains = self._move_gains(graph, rank_by_expert)
            candidates = []
            for left_rank in range(self.num_ranks):
                for right_rank in range(left_rank + 1, self.num_ranks):
                    for left in assignment[left_rank]:
                        for right in assignment[right_rank]:
                            delta = (
                                move_gains[left][right_rank]
                                + move_gains[right][left_rank]
                                + 2 * graph.adjacency.get(left, {}).get(right, 0)
                            )
                            candidates.append(
                                (delta, left_rank, right_rank, left, right)
                            )
            if tokens is None:
                best = min((item for item in candidates if item[0] < 0), default=None)
                affected = None
            else:
                best = None
                affected = None
                best_key = None
                average_load = sum(rank_load) / self.num_ranks
                current_imbalance = max(rank_load) / average_load
                for candidate in nsmallest(self.rerank_candidates, candidates):
                    delta, left_rank, right_rank, left, right = candidate
                    next_load = list(rank_load)
                    next_load[left_rank] += graph.demand[right] - graph.demand[left]
                    next_load[right_rank] += graph.demand[left] - graph.demand[right]
                    if self.max_compute_imbalance is not None and (
                        max(next_load) / average_load
                        > max(current_imbalance, self.max_compute_imbalance)
                    ):
                        continue
                    indexes = token_indexes[left] | token_indexes[right]
                    swap = left, right, left_rank, right_rank
                    remote_delta = sum(
                        _token_weighted_remote(
                            tokens[index],
                            rank_by_expert,
                            ranks_per_node,
                            rdma_cost,
                            swap,
                        )
                        - token_costs[index]
                        for index in indexes
                    )
                    key = remote_delta, delta, left_rank, right_rank, left, right
                    if remote_delta < 0 and (best_key is None or key < best_key):
                        best_key = key
                        best = candidate
                        affected = indexes
            if best is None:
                break
            _, left_rank, right_rank, left, right = best
            assignment[left_rank].remove(left)
            assignment[left_rank].add(right)
            assignment[right_rank].remove(right)
            assignment[right_rank].add(left)
            rank_by_expert[left], rank_by_expert[right] = right_rank, left_rank
            rank_load[left_rank] += graph.demand[right] - graph.demand[left]
            rank_load[right_rank] += graph.demand[left] - graph.demand[right]
            if affected is not None:
                for index in affected:
                    token_costs[index] = _token_weighted_remote(
                        tokens[index], rank_by_expert, ranks_per_node, rdma_cost
                    )
            rounds += 1

        return GraphPlacement(
            rank_by_expert,
            {rank: tuple(sorted(experts)) for rank, experts in enumerate(assignment)},
            initial_cut,
            evaluate_pairwise_cut(graph, rank_by_expert),
            rounds,
        )

    def _initial_assignment(self, graph: CoRoutingGraph) -> list[set[int]]:
        assignment = [set() for _ in range(self.num_ranks)]
        loads = [0] * self.num_ranks
        degree = graph.weighted_degree
        order = sorted(
            graph.experts,
            key=lambda expert: (-degree[expert], -graph.demand[expert], expert),
        )
        for expert in order:
            candidates = [
                rank
                for rank in range(self.num_ranks)
                if len(assignment[rank]) < self.capacities[rank]
            ]
            if not candidates:
                raise ValueError("expert count exceeds total rank capacity")
            rank = min(
                candidates,
                key=lambda candidate: (
                    -sum(
                        graph.adjacency.get(expert, {}).get(other, 0)
                        for other in assignment[candidate]
                    ),
                    loads[candidate],
                    candidate,
                ),
            )
            assignment[rank].add(expert)
            loads[rank] += graph.demand[expert]
        return assignment

    def _move_gains(
        self, graph: CoRoutingGraph, rank_by_expert: Mapping[int, int]
    ) -> dict[int, list[int]]:
        result = {}
        for expert in graph.experts:
            weight_to_rank = [0] * self.num_ranks
            total = 0
            for neighbor, weight in graph.adjacency.get(expert, {}).items():
                weight_to_rank[rank_by_expert[neighbor]] += weight
                total += weight
            current = total - weight_to_rank[rank_by_expert[expert]]
            result[expert] = [total - weight - current for weight in weight_to_rank]
        return result

    @staticmethod
    def _rank_map(assignment: Sequence[set[int]]) -> dict[int, int]:
        return {
            expert: rank
            for rank, experts in enumerate(assignment)
            for expert in experts
        }
