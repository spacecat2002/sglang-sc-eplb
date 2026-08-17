"""Source-aware, capacity-constrained offline expert placement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .co_routing_graph_solver import (
    CoRoutingGraph,
    RoutedToken,
    _token_weighted_remote,
)


@dataclass(frozen=True)
class HypergraphPlacement:
    rank_by_expert: dict[int, int]
    experts_by_rank: dict[int, tuple[int, ...]]
    communication_cost: float
    compute_imbalance: float
    iterations: int
    restarts: int


class SourceAwareHypergraphSolver:
    """Optimize token-level communication with fixed expert slots.

    The solver uses a source-aware unary seed and exact bundle-cost swap
    refinement. It is a bounded heuristic; ``restarts`` and ``max_rounds``
    control the quality/runtime trade-off.
    """

    def __init__(
        self,
        *,
        num_ranks: int,
        slots_per_rank: int,
        max_rounds: int = 8,
        restarts: int = 2,
        candidates: int = 128,
        load_weight: float = 0.5,
        max_compute_imbalance: float = 1.2,
        seed: int = 0,
    ) -> None:
        if num_ranks < 1 or slots_per_rank < 1:
            raise ValueError("num_ranks and slots_per_rank must be positive")
        if max_rounds < 0 or restarts < 1 or candidates < 1:
            raise ValueError("max_rounds, restarts, and candidates are invalid")
        if load_weight < 0 or max_compute_imbalance < 1:
            raise ValueError("load_weight must be non-negative and imbalance >= 1")
        self.num_ranks = num_ranks
        self.slots_per_rank = slots_per_rank
        self.max_rounds = max_rounds
        self.restarts = restarts
        self.candidates = candidates
        self.load_weight = load_weight
        self.max_compute_imbalance = max_compute_imbalance
        self.seed = seed

    def solve(
        self,
        graph: CoRoutingGraph,
        tokens: Sequence[RoutedToken],
        *,
        ranks_per_node: int,
        rdma_cost: float = 1.0,
    ) -> HypergraphPlacement:
        if len(graph.experts) > self.num_ranks * self.slots_per_rank:
            raise ValueError("expert count exceeds total rank capacity")
        if not tokens:
            raise ValueError("tokens must not be empty")
        if ranks_per_node < 1 or self.num_ranks % ranks_per_node:
            raise ValueError("ranks_per_node must divide num_ranks")
        if any(token.source_rank >= self.num_ranks for token in tokens):
            raise ValueError("token source rank exceeds num_ranks")

        source_demand = self._source_demand(graph, tokens)
        token_indexes = self._token_indexes(tokens)
        rng = random.Random(self.seed)
        best = None
        total_iterations = 0
        for restart in range(self.restarts):
            assignment = self._initial_assignment(graph, source_demand, rng, restart)
            placement, iterations = self._refine(
                graph,
                tokens,
                assignment,
                token_indexes,
                ranks_per_node,
                rdma_cost,
                source_demand,
            )
            total_iterations += iterations
            score = self._score(placement[1], placement[2], self.load_weight)
            if best is None or score < best[0]:
                best = (score, placement)
        assert best is not None
        rank_by_expert, cost, loads = best[1]
        average = sum(loads) / self.num_ranks
        return HypergraphPlacement(
            rank_by_expert,
            {
                rank: tuple(
                    sorted(
                        expert
                        for expert, expert_rank in rank_by_expert.items()
                        if expert_rank == rank
                    )
                )
                for rank in range(self.num_ranks)
            },
            cost,
            max(loads, default=0) / average if average else 0.0,
            total_iterations,
            self.restarts,
        )

    @staticmethod
    def _source_demand(
        graph: CoRoutingGraph, tokens: Sequence[RoutedToken]
    ) -> dict[int, list[int]]:
        max_source = max(token.source_rank for token in tokens) + 1
        result = {expert: [0] * max_source for expert in graph.experts}
        for token in tokens:
            for expert in token.topk_experts:
                result[expert][token.source_rank] += token.count
        return result

    @staticmethod
    def _token_indexes(tokens: Sequence[RoutedToken]) -> dict[int, set[int]]:
        indexes: dict[int, set[int]] = {}
        for index, token in enumerate(tokens):
            for expert in token.topk_experts:
                indexes.setdefault(expert, set()).add(index)
        return indexes

    def _initial_assignment(
        self,
        graph: CoRoutingGraph,
        source_demand: Mapping[int, Sequence[int]],
        rng: random.Random,
        restart: int,
    ) -> list[set[int]]:
        assignment = [set() for _ in range(self.num_ranks)]
        loads = [0] * self.num_ranks
        degree = graph.weighted_degree
        order = sorted(
            graph.experts,
            key=lambda expert: (-graph.demand[expert], -degree[expert], expert),
        )
        if restart:
            # Alternate source-first and affinity-first seeds to explore
            # different local-search basins without nondeterminism.
            order.sort(
                key=lambda expert: (
                    -graph.demand[expert],
                    -degree[expert],
                    rng.random(),
                )
            )
        for expert in order:
            candidates = [
                rank
                for rank in range(self.num_ranks)
                if len(assignment[rank]) < self.slots_per_rank
            ]
            source = source_demand[expert]
            scores = {}
            for rank in candidates:
                local = sum(
                    graph.adjacency.get(expert, {}).get(other, 0)
                    for other in assignment[rank]
                )
                source_local = source[rank] if rank < len(source) else 0
                # Prefer source-local experts, then co-routed experts, while
                # keeping demand balanced as a deterministic tie-breaker.
                scores[rank] = (
                    (graph.demand[expert] - source_local, -local)
                    if restart % 2 == 0
                    else (-local, graph.demand[expert] - source_local)
                ) + (loads[rank], rank)
            rank = min(candidates, key=scores.__getitem__)
            assignment[rank].add(expert)
            loads[rank] += graph.demand[expert]
        return assignment

    def _refine(
        self,
        graph: CoRoutingGraph,
        tokens: Sequence[RoutedToken],
        assignment: list[set[int]],
        token_indexes: Mapping[int, set[int]],
        ranks_per_node: int,
        rdma_cost: float,
        source_demand: Mapping[int, Sequence[int]],
    ) -> tuple[tuple[dict[int, int], float, list[int]], int]:
        rank_by_expert = self._rank_map(assignment)
        token_costs = [
            _token_weighted_remote(token, rank_by_expert, ranks_per_node, rdma_cost)
            for token in tokens
        ]
        loads = [
            sum(graph.demand[expert] for expert in experts) for experts in assignment
        ]
        iterations = 0
        while iterations < self.max_rounds:
            average = sum(loads) / self.num_ranks
            current_cost = sum(token_costs)
            current_score = self._score(current_cost, loads, self.load_weight, average)
            candidates = []
            for left_rank in range(self.num_ranks):
                for right_rank in range(left_rank + 1, self.num_ranks):
                    for left in assignment[left_rank]:
                        for right in assignment[right_rank]:
                            approximate = self._unary_delta(
                                left,
                                right,
                                left_rank,
                                right_rank,
                                graph,
                                source_demand,
                            )
                            candidates.append(
                                (approximate, left_rank, right_rank, left, right)
                            )
            best = None
            for _, left_rank, right_rank, left, right in sorted(candidates)[
                : self.candidates
            ]:
                next_loads = list(loads)
                next_loads[left_rank] += graph.demand[right] - graph.demand[left]
                next_loads[right_rank] += graph.demand[left] - graph.demand[right]
                next_imbalance = max(next_loads) / average if average else 0.0
                if next_imbalance > max(
                    self.max_compute_imbalance, max(loads) / average
                ):
                    continue
                affected = token_indexes.get(left, set()) | token_indexes.get(
                    right, set()
                )
                swap = left, right, left_rank, right_rank
                delta = sum(
                    _token_weighted_remote(
                        tokens[index], rank_by_expert, ranks_per_node, rdma_cost, swap
                    )
                    - token_costs[index]
                    for index in affected
                )
                next_cost = current_cost + delta
                next_score = self._score(
                    next_cost, next_loads, self.load_weight, average
                )
                if best is None or next_score < best[0]:
                    best = (
                        next_score,
                        delta,
                        left_rank,
                        right_rank,
                        left,
                        right,
                        affected,
                        next_loads,
                    )
            if best is None or best[0] >= current_score:
                break
            _, delta, left_rank, right_rank, left, right, affected, loads = best
            assignment[left_rank].remove(left)
            assignment[left_rank].add(right)
            assignment[right_rank].remove(right)
            assignment[right_rank].add(left)
            rank_by_expert[left], rank_by_expert[right] = right_rank, left_rank
            for index in affected:
                token_costs[index] = _token_weighted_remote(
                    tokens[index], rank_by_expert, ranks_per_node, rdma_cost
                )
            iterations += 1
        return (rank_by_expert, sum(token_costs), loads), iterations

    @staticmethod
    def _score(
        communication_cost: float,
        loads: Sequence[int],
        load_weight: float,
        average: float | None = None,
    ) -> float:
        average = average or (sum(loads) / len(loads) if loads else 0.0)
        imbalance = max(loads, default=0) / average if average else 0.0
        return communication_cost + load_weight * communication_cost * max(
            0.0, imbalance - 1.0
        )

    @staticmethod
    def _unary_delta(
        left: int,
        right: int,
        left_rank: int,
        right_rank: int,
        graph: CoRoutingGraph,
        source_demand: Mapping[int, Sequence[int]],
    ) -> int:
        def cost(expert: int, rank: int) -> int:
            demand = graph.demand[expert]
            local = (
                source_demand[expert][rank] if rank < len(source_demand[expert]) else 0
            )
            return demand - local

        return (
            cost(left, right_rank)
            + cost(right, left_rank)
            - cost(left, left_rank)
            - cost(right, right_rank)
        )

    @staticmethod
    def _rank_map(assignment: Sequence[set[int]]) -> dict[int, int]:
        return {
            expert: rank
            for rank, experts in enumerate(assignment)
            for expert in experts
        }
