"""Offline co-routing graph placement for MoE experts.

The graph is an inexpensive approximation of the DeepEP communication
objective.  Each logical expert is a vertex and an edge counts how often two
experts occur in the same Top-K bundle.  A capacity-constrained partition
then groups frequently co-routed experts on the same rank.  The resulting
placement should always be validated by replaying the original Top-K bundles,
because a pairwise graph does not preserve the full hyperedge semantics.

This module is intentionally offline.  It does not allocate CUDA tensors or
run on the request path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .bundle_aware_replica_planner import RoutedToken


Edge = Tuple[int, int]


@dataclass(frozen=True)
class CoRoutingGraph:
    """Weighted pairwise projection of Top-K hyperedges."""

    experts: Tuple[int, ...]
    demand: Mapping[int, int]
    edges: Mapping[Edge, int]
    adjacency: Mapping[int, Mapping[int, int]]

    @property
    def total_edge_weight(self) -> int:
        return sum(self.edges.values())

    @property
    def weighted_degree(self) -> Dict[int, int]:
        return {
            expert: sum(self.adjacency.get(expert, {}).values())
            for expert in self.experts
        }


@dataclass(frozen=True)
class GraphPlacement:
    """A primary (one home rank per logical expert) graph placement."""

    rank_by_expert: Dict[int, int]
    experts_by_rank: Dict[int, Tuple[int, ...]]
    initial_cut: float
    final_cut: float
    iterations: int


@dataclass(frozen=True)
class HypergraphPlacement:
    """A placement refined against exact Top-K bundle remote-rank counts."""

    rank_by_expert: Dict[int, int]
    experts_by_rank: Dict[int, Tuple[int, ...]]
    initial_remote: int
    final_remote: int
    iterations: int
    balance_iterations: int = 0


def build_co_routing_graph(
    routed_tokens: Iterable[RoutedToken],
    *,
    experts: Optional[Iterable[int]] = None,
) -> CoRoutingGraph:
    """Build a weighted co-routing graph from aggregated Top-K bundles.

    For a bundle ``(e0, e1, ..., ek)`` with count ``c``, every unordered pair
    receives ``c`` additional edge weight.  The expert demand counts each
    selected expert once per token, so it can be used to seed a balanced
    partition.  Duplicate experts in a bundle are rejected by ``RoutedToken``.
    """

    tokens = list(routed_tokens)
    if not tokens and experts is None:
        raise ValueError("routed_tokens or experts must not be empty")
    demand: Dict[int, int] = defaultdict(int)
    edges: Dict[Edge, int] = defaultdict(int)
    observed: Set[int] = set(experts or ())
    for token in tokens:
        observed.update(token.topk_experts)
        for expert in token.topk_experts:
            demand[expert] += token.count
        for left, right in combinations(sorted(token.topk_experts), 2):
            edges[(left, right)] += token.count
    ordered_experts = tuple(sorted(observed))
    for expert in ordered_experts:
        demand.setdefault(expert, 0)
    adjacency: Dict[int, Dict[int, int]] = {expert: {} for expert in ordered_experts}
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(ordered_experts, dict(demand), dict(edges), adjacency)


def refine_hypergraph_placement(
    routed_tokens: Iterable[RoutedToken],
    initial_rank_by_expert: Mapping[int, int],
    *,
    num_ranks: int,
    max_rounds: Optional[int] = 8,
    balance_rounds: Optional[int] = 0,
) -> HypergraphPlacement:
    """Refine a placement using the exact distinct-remote-rank objective.

    Swapping two experts preserves the number of primary experts on every
    rank. Only bundles containing exactly one of the swapped experts can
    change their destination-rank cardinality.
    """

    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if balance_rounds is not None and balance_rounds < 0:
        raise ValueError("balance_rounds must be non-negative")
    tokens = list(routed_tokens)
    rank_by_expert = dict(initial_rank_by_expert)
    if any(rank < 0 or rank >= num_ranks for rank in rank_by_expert.values()):
        raise ValueError("initial placement contains an invalid rank")
    observed = {expert for token in tokens for expert in token.topk_experts}
    missing = observed - rank_by_expert.keys()
    if missing:
        raise ValueError(f"initial placement is missing experts: {sorted(missing)}")

    incidence: Dict[int, Set[int]] = {expert: set() for expert in rank_by_expert}
    for bundle_index, token in enumerate(tokens):
        for expert in token.topk_experts:
            incidence[expert].add(bundle_index)

    move_gains: Dict[int, list[int]] = {
        expert: [0] * num_ranks for expert in rank_by_expert
    }
    pair_correction: Dict[Edge, int] = defaultdict(int)

    def accumulate_bundle(bundle_index: int, sign: int) -> None:
        token = tokens[bundle_index]
        counts: Dict[int, int] = defaultdict(int)
        for expert in token.topk_experts:
            counts[rank_by_expert[expert]] += 1
        for expert in token.topk_experts:
            old_rank = rank_by_expert[expert]
            removed = old_rank != token.source_rank and counts[old_rank] == 1
            for target_rank in range(num_ranks):
                if target_rank == old_rank:
                    continue
                added = (
                    target_rank != token.source_rank and counts.get(target_rank, 0) == 0
                )
                move_gains[expert][target_rank] += (
                    sign * token.count * (int(added) - int(removed))
                )
        for left, right in combinations(sorted(token.topk_experts), 2):
            left_rank = rank_by_expert[left]
            right_rank = rank_by_expert[right]
            if left_rank == right_rank:
                continue
            independent_delta = 0
            if right_rank != token.source_rank and counts.get(right_rank, 0) == 0:
                independent_delta += 1
            if left_rank != token.source_rank and counts[left_rank] == 1:
                independent_delta -= 1
            if left_rank != token.source_rank and counts.get(left_rank, 0) == 0:
                independent_delta += 1
            if right_rank != token.source_rank and counts[right_rank] == 1:
                independent_delta -= 1
            pair_correction[(left, right)] -= sign * token.count * independent_delta

    for bundle_index in range(len(tokens)):
        accumulate_bundle(bundle_index, 1)

    initial_remote = evaluate_primary_remote(tokens, rank_by_expert)
    rounds = 0
    while max_rounds is None or rounds < max_rounds:
        best: Optional[Tuple[int, int, int]] = None
        experts = sorted(rank_by_expert)
        for left_index, left in enumerate(experts):
            left_rank = rank_by_expert[left]
            for right in experts[left_index + 1 :]:
                right_rank = rank_by_expert[right]
                if left_rank == right_rank:
                    continue
                delta = (
                    move_gains[left][right_rank]
                    + move_gains[right][left_rank]
                    + pair_correction.get((left, right), 0)
                )
                candidate = (delta, left, right)
                if delta < 0 and (best is None or candidate < best):
                    best = candidate
        if best is None:
            break
        _, left, right = best
        affected = incidence[left] | incidence[right]
        for bundle_index in affected:
            accumulate_bundle(bundle_index, -1)
        rank_by_expert[left], rank_by_expert[right] = (
            rank_by_expert[right],
            rank_by_expert[left],
        )
        for bundle_index in affected:
            accumulate_bundle(bundle_index, 1)
        rounds += 1

    balance_iterations = 0
    if balance_rounds is None or balance_rounds > 0:
        expert_demand: Dict[int, int] = defaultdict(int)
        for token in tokens:
            for expert in token.topk_experts:
                expert_demand[expert] += token.count
        rank_load = [0] * num_ranks
        for expert, rank in rank_by_expert.items():
            rank_load[rank] += expert_demand[expert]

        while balance_rounds is None or balance_iterations < balance_rounds:
            current_max = max(rank_load, default=0)
            max_without_pair = {
                (left_rank, right_rank): max(
                    (
                        load
                        for rank, load in enumerate(rank_load)
                        if rank not in {left_rank, right_rank}
                    ),
                    default=0,
                )
                for left_rank in range(num_ranks)
                for right_rank in range(left_rank + 1, num_ranks)
            }
            best_balance: Optional[Tuple[int, int, int, int]] = None
            experts = sorted(rank_by_expert)
            for left_index, left in enumerate(experts):
                left_rank = rank_by_expert[left]
                for right in experts[left_index + 1 :]:
                    right_rank = rank_by_expert[right]
                    if left_rank == right_rank:
                        continue
                    remote_delta = (
                        move_gains[left][right_rank]
                        + move_gains[right][left_rank]
                        + pair_correction.get((left, right), 0)
                    )
                    if remote_delta != 0:
                        continue
                    left_after = (
                        rank_load[left_rank]
                        - expert_demand[left]
                        + expert_demand[right]
                    )
                    right_after = (
                        rank_load[right_rank]
                        - expert_demand[right]
                        + expert_demand[left]
                    )
                    low_rank, high_rank = sorted((left_rank, right_rank))
                    new_max = max(
                        max_without_pair[(low_rank, high_rank)],
                        left_after,
                        right_after,
                    )
                    variance_delta = (
                        left_after * left_after
                        + right_after * right_after
                        - rank_load[left_rank] * rank_load[left_rank]
                        - rank_load[right_rank] * rank_load[right_rank]
                    )
                    if new_max > current_max or variance_delta >= 0:
                        continue
                    candidate = (new_max, variance_delta, left, right)
                    if best_balance is None or candidate < best_balance:
                        best_balance = candidate
            if best_balance is None:
                break
            _, _, left, right = best_balance
            affected = incidence[left] | incidence[right]
            for bundle_index in affected:
                accumulate_bundle(bundle_index, -1)
            left_rank = rank_by_expert[left]
            right_rank = rank_by_expert[right]
            rank_by_expert[left], rank_by_expert[right] = right_rank, left_rank
            rank_load[left_rank] += expert_demand[right] - expert_demand[left]
            rank_load[right_rank] += expert_demand[left] - expert_demand[right]
            for bundle_index in affected:
                accumulate_bundle(bundle_index, 1)
            balance_iterations += 1

    final_remote = evaluate_primary_remote(tokens, rank_by_expert)
    experts_by_rank = {
        rank: tuple(
            sorted(
                expert
                for expert, expert_rank in rank_by_expert.items()
                if expert_rank == rank
            )
        )
        for rank in range(num_ranks)
    }
    return HypergraphPlacement(
        rank_by_expert,
        experts_by_rank,
        initial_remote,
        final_remote,
        rounds,
        balance_iterations,
    )


class CoRoutingGraphSolver:
    """Capacity-constrained graph partition with deterministic swap refinement.

    ``slots_per_rank`` is the number of primary logical experts assigned to a
    rank.  Replica slots are deliberately outside this solver; after obtaining
    a primary placement, callers can pass it to ``BundleAwareReplicaPlanner``
    to add communication-aware copies.
    """

    def __init__(
        self,
        *,
        num_ranks: int,
        slots_per_rank: int | Sequence[int],
        max_rounds: Optional[int] = 8,
        balance_weight: float = 0.0,
    ):
        if num_ranks < 1:
            raise ValueError("num_ranks must be positive")
        if max_rounds is not None and max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if balance_weight < 0:
            raise ValueError("balance_weight must be non-negative")
        if isinstance(slots_per_rank, int):
            capacities = [slots_per_rank] * num_ranks
        else:
            capacities = list(slots_per_rank)
        if len(capacities) != num_ranks or any(capacity < 0 for capacity in capacities):
            raise ValueError("slots_per_rank must contain non-negative capacities")
        self.num_ranks = num_ranks
        self.capacities = capacities
        self.max_rounds = max_rounds
        self.balance_weight = balance_weight

    def solve(self, graph: CoRoutingGraph) -> GraphPlacement:
        """Return a graph placement and the pairwise cut before/after swaps."""

        if len(graph.experts) > sum(self.capacities):
            raise ValueError("expert count exceeds total rank capacity")
        assignment = self._initial_assignment(graph)
        rank_by_expert = self._assignment_rank_map(assignment)
        initial_cut = self._objective(graph, assignment, rank_by_expert)
        rank_load = [
            sum(graph.demand[expert] for expert in experts) for experts in assignment
        ]
        rounds = 0
        while self.max_rounds is None or rounds < self.max_rounds:
            # Compute all one-expert move gains once per round. A candidate
            # swap is then scored in O(1), instead of rescanning neighbors
            # and rank sets for every pair.
            move_gains = self._move_gains(graph, rank_by_expert)
            best: Optional[Tuple[float, int, int, int, int]] = None
            for left_rank in range(self.num_ranks):
                for right_rank in range(left_rank + 1, self.num_ranks):
                    for left_expert in assignment[left_rank]:
                        for right_expert in assignment[right_rank]:
                            edge_weight = graph.adjacency.get(left_expert, {}).get(
                                right_expert, 0
                            )
                            delta = (
                                move_gains[left_expert][right_rank]
                                + move_gains[right_expert][left_rank]
                                + 2 * edge_weight
                            )
                            if self.balance_weight:
                                delta += self._balance_delta(
                                    graph,
                                    rank_load,
                                    left_expert,
                                    right_expert,
                                    left_rank,
                                    right_rank,
                                )
                            candidate = (
                                delta,
                                left_rank,
                                right_rank,
                                left_expert,
                                right_expert,
                            )
                            if delta < 0 and (best is None or candidate < best):
                                best = candidate
            if best is None:
                break
            _, left_rank, right_rank, left_expert, right_expert = best
            assignment[left_rank].remove(left_expert)
            assignment[left_rank].add(right_expert)
            assignment[right_rank].remove(right_expert)
            assignment[right_rank].add(left_expert)
            rank_by_expert[left_expert] = right_rank
            rank_by_expert[right_expert] = left_rank
            rank_load[left_rank] += (
                graph.demand[right_expert] - graph.demand[left_expert]
            )
            rank_load[right_rank] += (
                graph.demand[left_expert] - graph.demand[right_expert]
            )
            rounds += 1

        experts_by_rank = {
            rank: tuple(sorted(experts)) for rank, experts in enumerate(assignment)
        }
        final_cut = self._objective(graph, assignment, rank_by_expert)
        return GraphPlacement(
            rank_by_expert, experts_by_rank, initial_cut, final_cut, rounds
        )

    def _initial_assignment(self, graph: CoRoutingGraph) -> List[Set[int]]:
        assignment = [set() for _ in range(self.num_ranks)]
        rank_load = [0] * self.num_ranks
        # High-degree vertices carry the graph structure; demand breaks ties.
        weighted_degree = graph.weighted_degree
        order = sorted(
            graph.experts,
            key=lambda expert: (
                -weighted_degree[expert],
                -graph.demand[expert],
                expert,
            ),
        )
        for expert in order:
            candidates = [
                rank
                for rank in range(self.num_ranks)
                if len(assignment[rank]) < self.capacities[rank]
            ]
            if not candidates:
                raise ValueError("expert count exceeds total rank capacity")

            def score(rank: int) -> Tuple[int, int, int]:
                connectivity = sum(
                    graph.adjacency.get(expert, {}).get(other, 0)
                    for other in assignment[rank]
                )
                # Prefer high connectivity, then less demand, then rank id.
                return (-connectivity, rank_load[rank], rank)

            rank = min(candidates, key=score)
            assignment[rank].add(expert)
            rank_load[rank] += graph.demand[expert]
        return assignment

    def _objective(
        self,
        graph: CoRoutingGraph,
        assignment: Sequence[Set[int]],
        rank_by_expert: Optional[Mapping[int, int]] = None,
    ) -> float:
        rank_by_expert = rank_by_expert or self._assignment_rank_map(assignment)
        cut = sum(
            weight
            for (left, right), weight in graph.edges.items()
            if rank_by_expert[left] != rank_by_expert[right]
        )
        if self.balance_weight == 0:
            return float(cut)
        loads = [
            sum(graph.demand[expert] for expert in experts) for experts in assignment
        ]
        average = sum(loads) / self.num_ranks
        imbalance = sum((load - average) ** 2 for load in loads) / max(average, 1.0)
        return cut + self.balance_weight * imbalance

    def _move_gains(
        self, graph: CoRoutingGraph, rank_by_expert: Mapping[int, int]
    ) -> Dict[int, List[float]]:
        gains: Dict[int, List[float]] = {}
        for expert in graph.experts:
            weight_to_rank = [0] * self.num_ranks
            total_weight = 0
            for neighbor, weight in graph.adjacency.get(expert, {}).items():
                weight_to_rank[rank_by_expert[neighbor]] += weight
                total_weight += weight
            home = rank_by_expert[expert]
            old_cut = total_weight - weight_to_rank[home]
            gains[expert] = [
                float(total_weight - weight_to_rank[rank] - old_cut)
                for rank in range(self.num_ranks)
            ]
        return gains

    def _balance_delta(
        self,
        graph: CoRoutingGraph,
        rank_load: Sequence[int],
        left_expert: int,
        right_expert: int,
        left_rank: int,
        right_rank: int,
    ) -> float:
        average = sum(rank_load) / self.num_ranks
        before = (rank_load[left_rank] - average) ** 2 + (
            rank_load[right_rank] - average
        ) ** 2
        left_after = (
            rank_load[left_rank]
            + graph.demand[right_expert]
            - graph.demand[left_expert]
        )
        right_after = (
            rank_load[right_rank]
            + graph.demand[left_expert]
            - graph.demand[right_expert]
        )
        after = (left_after - average) ** 2 + (right_after - average) ** 2
        return self.balance_weight * (after - before) / max(average, 1.0)

    @staticmethod
    def _assignment_rank_map(assignment: Sequence[Set[int]]) -> Dict[int, int]:
        return {
            expert: rank
            for rank, experts in enumerate(assignment)
            for expert in experts
        }

    @staticmethod
    def _rank_of(assignment: Sequence[Set[int]], expert: int) -> int:
        for rank, experts in enumerate(assignment):
            if expert in experts:
                return rank
        raise KeyError(f"expert {expert} is not assigned")


def evaluate_primary_remote(
    routed_tokens: Iterable[RoutedToken], rank_by_expert: Mapping[int, int]
) -> int:
    """Exact remote rank-copy count for a one-home-per-expert placement."""

    total = 0
    for token in routed_tokens:
        total += token.count * len(
            {
                rank_by_expert[expert]
                for expert in token.topk_experts
                if rank_by_expert[expert] != token.source_rank
            }
        )
    return total
