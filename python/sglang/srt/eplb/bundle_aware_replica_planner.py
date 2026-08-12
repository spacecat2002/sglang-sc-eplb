"""Bundle-aware replica planning for DeepEP normal-mode MoE dispatch.

DeepEP transmits one hidden state per ``(token, destination rank)`` rather
than one per selected expert.  A planner that scores experts independently
therefore overstates the communication benefit of copying one expert while a
second selected expert remains on the same remote rank.  This module models
the token's *set* of destination ranks and optimizes communication and expert
compute balance together.

This is an offline planner intended for replaying routed-expert traces before
integrating a policy in the runtime.  Logical expert ids are used throughout;
physical ids are a runtime implementation detail.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class RoutedToken:
    """A routed token or an aggregation of identical ``source, Top-K`` bundles."""

    source_rank: int
    topk_experts: Tuple[int, ...]
    count: int = 1

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("count must be positive")
        if not self.topk_experts:
            raise ValueError("topk_experts must not be empty")
        if len(set(self.topk_experts)) != len(self.topk_experts):
            raise ValueError("topk_experts must not contain duplicates")


@dataclass(frozen=True)
class ReplicaAction:
    """Copy the logical experts in ``experts`` to ``destination_rank``."""

    destination_rank: int
    experts: Tuple[int, ...]
    kind: str


@dataclass
class PlanMetrics:
    compute_load: List[int]
    send_load: List[float]
    recv_load: List[float]
    communication_load: List[float]
    weighted_communication: float
    unique_remote_rank_copies: int
    nvl_traffic: int
    rdma_traffic: int
    objective: float


@dataclass
class ReplicaPlan:
    replicas_by_rank: Dict[int, Set[int]]
    actions: List[ReplicaAction]
    baseline: PlanMetrics
    final: PlanMetrics


def _node_of(rank: int, ranks_per_node: Optional[int]) -> int:
    return rank if ranks_per_node is None else rank // ranks_per_node


def _link_cost(
    source_rank: int,
    destination_rank: int,
    ranks_per_node: Optional[int],
    rdma_cost: float,
) -> float:
    if _node_of(source_rank, ranks_per_node) == _node_of(
        destination_rank, ranks_per_node
    ):
        return 1.0
    return rdma_cost


def _copy_placement(placement: Mapping[int, Set[int]]) -> Dict[int, Set[int]]:
    return {rank: set(experts) for rank, experts in placement.items()}


class BundleAwareReplicaPlanner:
    """Greedily plan replicas against a joint compute/network critical path.

    ``baseline_rank_by_expert`` must contain exactly one permanent home rank
    for every logical expert observed in the trace.  Copies are added on top
    of these homes and never evict them.

    The implementation intentionally reroutes every bundle for each candidate
    action.  This is more expensive than a per-expert heuristic but captures
    the DeepEP rank-set deduplication that the planner exists to optimize.
    """

    def __init__(
        self,
        *,
        num_ranks: int,
        baseline_rank_by_expert: Mapping[int, int],
        replica_slots_per_rank: int | Sequence[int],
        ranks_per_node: Optional[int] = None,
        rdma_cost: float = 4.0,
        compute_weight: float = 1.0,
        communication_weight: float = 1.0,
        max_bundle_size: Optional[int] = None,
    ):
        if num_ranks < 1:
            raise ValueError("num_ranks must be positive")
        if rdma_cost < 1:
            raise ValueError("rdma_cost must be at least 1")
        if compute_weight < 0 or communication_weight < 0:
            raise ValueError("objective weights must be non-negative")
        self.num_ranks = num_ranks
        self.baseline_rank_by_expert = dict(baseline_rank_by_expert)
        if isinstance(replica_slots_per_rank, int):
            self.replica_slots_per_rank = [replica_slots_per_rank] * num_ranks
        else:
            self.replica_slots_per_rank = list(replica_slots_per_rank)
        if len(self.replica_slots_per_rank) != num_ranks:
            raise ValueError("replica_slots_per_rank must have num_ranks entries")
        if any(slots < 0 for slots in self.replica_slots_per_rank):
            raise ValueError("replica slots must be non-negative")
        self.ranks_per_node = ranks_per_node
        self.rdma_cost = rdma_cost
        self.compute_weight = compute_weight
        self.communication_weight = communication_weight
        self.max_bundle_size = max_bundle_size

    def plan(
        self, routed_tokens: Iterable[RoutedToken], max_actions: Optional[int] = None
    ) -> ReplicaPlan:
        tokens = list(routed_tokens)
        if not tokens:
            raise ValueError("routed_tokens must not be empty")
        self._validate_tokens(tokens)
        placement = self._baseline_placement()
        baseline = self._evaluate(tokens, placement)
        normalization = (
            max(sum(baseline.compute_load) / self.num_ranks, 1.0),
            max(sum(baseline.communication_load) / self.num_ranks, 1.0),
        )
        # Keep the normalizer fixed during greedy search.  Re-normalizing by
        # lower total traffic can hide a real communication improvement.
        baseline = self._evaluate(tokens, placement, normalization)
        current = baseline
        actions: List[ReplicaAction] = []

        while max_actions is None or len(actions) < max_actions:
            best_action: Optional[ReplicaAction] = None
            best_placement: Optional[Dict[int, Set[int]]] = None
            best_metrics: Optional[PlanMetrics] = None
            for action in self._candidates(tokens, placement):
                candidate_placement = self._apply_action(placement, action)
                if candidate_placement is None:
                    continue
                candidate_metrics = self._evaluate(
                    tokens, candidate_placement, normalization
                )
                if self._score(candidate_metrics) < self._score(current) and (
                    best_metrics is None
                    or self._score(candidate_metrics) < self._score(best_metrics)
                    or (
                        self._score(candidate_metrics) == self._score(best_metrics)
                        and len(action.experts) < len(best_action.experts)
                    )
                ):
                    best_action = action
                    best_placement = candidate_placement
                    best_metrics = candidate_metrics
            if best_action is None or best_placement is None or best_metrics is None:
                break
            actions.append(best_action)
            placement, current = best_placement, best_metrics

        return ReplicaPlan(placement, actions, baseline, current)

    def evaluate_baseline(self, routed_tokens: Iterable[RoutedToken]) -> PlanMetrics:
        """Replay home placement only, without considering replica actions."""
        tokens = list(routed_tokens)
        if not tokens:
            raise ValueError("routed_tokens must not be empty")
        self._validate_tokens(tokens)
        return self._evaluate(tokens, self._baseline_placement())

    def evaluate_placement(
        self,
        routed_tokens: Iterable[RoutedToken],
        replicas_by_rank: Mapping[int, Set[int]],
    ) -> PlanMetrics:
        """Replay an explicit placement, including selected replica actions."""
        tokens = list(routed_tokens)
        if not tokens:
            raise ValueError("routed_tokens must not be empty")
        self._validate_tokens(tokens)
        if set(replicas_by_rank) != set(range(self.num_ranks)):
            raise ValueError("replicas_by_rank must contain every rank")
        for expert, home_rank in self.baseline_rank_by_expert.items():
            if expert not in replicas_by_rank[home_rank]:
                raise ValueError(f"placement removed home expert {expert}")
        return self._evaluate(tokens, replicas_by_rank)

    def plan_fast(
        self,
        routed_tokens: Iterable[RoutedToken],
        *,
        max_candidates: int = 64,
    ) -> ReplicaPlan:
        """Choose one bounded-cost replica action for each destination rank.

        Unlike :meth:`plan`, this method never replays every bundle for every
        candidate.  It scans the route window once, retains only the hottest
        closure/singleton candidates, and scores their load deltas.  The result
        is deliberately approximate: it independently chooses at most one
        action for each destination rank, while ``plan`` remains the exact
        offline oracle.
        """
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        tokens = list(routed_tokens)
        if not tokens:
            raise ValueError("routed_tokens must not be empty")
        self._validate_tokens(tokens)
        placement = self._baseline_placement()
        baseline, closure_counts, local_demands = self._fast_baseline(tokens)
        normalization = (
            max(sum(baseline.compute_load) / self.num_ranks, 1.0),
            max(sum(baseline.communication_load) / self.num_ranks, 1.0),
        )
        baseline = self._with_objective(baseline, normalization)

        candidates_by_rank = self._fast_candidates_by_destination(
            closure_counts, local_demands, placement, max_candidates
        )
        proposals: List[Tuple[ReplicaAction, int, int]] = []
        for candidates in candidates_by_rank.values():
            best: Optional[
                Tuple[Tuple[float, float, float, int], ReplicaAction, int, int]
            ] = None
            for action, remote, closure_count in candidates.values():
                if self._apply_action(placement, action) is None:
                    continue
                candidate_metrics = self._fast_candidate_metrics(
                    baseline,
                    action,
                    remote,
                    closure_count,
                    local_demands,
                    normalization,
                )
                score = self._score(candidate_metrics)
                if score < self._score(baseline) and (best is None or score < best[0]):
                    best = (score, action, remote, closure_count)
            if best is not None:
                _, action, remote, closure_count = best
                proposals.append((action, remote, closure_count))

        final_placement = placement
        final_metrics = baseline
        actions: List[ReplicaAction] = []
        for action, remote, closure_count in sorted(
            proposals, key=lambda item: item[0].destination_rank
        ):
            next_placement = self._apply_action(final_placement, action)
            if next_placement is None:
                continue
            final_placement = next_placement
            final_metrics = self._fast_candidate_metrics(
                final_metrics,
                action,
                remote,
                closure_count,
                local_demands,
                normalization,
            )
            actions.append(action)
        return ReplicaPlan(final_placement, actions, baseline, final_metrics)

    def _fast_candidates_by_destination(
        self,
        closure_counts: Mapping[Tuple[int, int, Tuple[int, ...]], int],
        local_demands: Mapping[Tuple[int, int], int],
        placement: Mapping[int, Set[int]],
        max_candidates: int,
    ) -> Dict[int, Dict[Tuple[int, ...], Tuple[ReplicaAction, int, int]]]:
        """Keep a fixed hot-candidate budget separately for each rank."""
        closures_by_source: Dict[int, List[Tuple[Tuple[int, Tuple[int, ...]], int]]] = (
            defaultdict(list)
        )
        for (source, remote, experts), count in closure_counts.items():
            closures_by_source[source].append(((remote, experts), count))
        singles_by_source: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (source, expert), count in local_demands.items():
            singles_by_source[source].append((expert, count))

        closure_limit = max_candidates // 2
        result: Dict[int, Dict[Tuple[int, ...], Tuple[ReplicaAction, int, int]]] = {}
        for source in range(self.num_ranks):
            candidates: Dict[Tuple[int, ...], Tuple[ReplicaAction, int, int]] = {}
            for (remote, experts), count in heapq.nlargest(
                closure_limit,
                closures_by_source[source],
                key=lambda item: item[1]
                * _link_cost(source, item[0][0], self.ranks_per_node, self.rdma_cost),
            ):
                missing = tuple(
                    expert for expert in experts if expert not in placement[source]
                )
                if not missing or (
                    self.max_bundle_size is not None
                    and len(missing) > self.max_bundle_size
                ):
                    continue
                candidates[missing] = (
                    ReplicaAction(source, missing, "bundle-closure"),
                    remote,
                    count,
                )
            for expert, _ in heapq.nlargest(
                max_candidates - closure_limit,
                singles_by_source[source],
                key=lambda item: item[1],
            ):
                if expert not in placement[source]:
                    candidates.setdefault(
                        (expert,),
                        (ReplicaAction(source, (expert,), "single"), -1, 0),
                    )
            if candidates:
                result[source] = candidates
        return result

    @staticmethod
    def _score(metrics: PlanMetrics) -> Tuple[float, float, float, int]:
        """Order candidates by critical path, then aggregate traffic."""
        return (
            round(metrics.objective, 12),
            round(max(metrics.communication_load, default=0.0), 12),
            round(metrics.weighted_communication, 12),
            max(metrics.compute_load, default=0),
        )

    def _fast_baseline(
        self, tokens: Sequence[RoutedToken]
    ) -> Tuple[
        PlanMetrics,
        Dict[Tuple[int, int, Tuple[int, ...]], int],
        Dict[Tuple[int, int], int],
    ]:
        """Build home-placement loads and bounded fast-planner statistics."""
        compute = [0] * self.num_ranks
        send = [0.0] * self.num_ranks
        recv = [0.0] * self.num_ranks
        closure_counts: Dict[Tuple[int, int, Tuple[int, ...]], int] = defaultdict(int)
        local_demands: Dict[Tuple[int, int], int] = defaultdict(int)
        unique_remote_rank_copies = 0
        nvl_traffic = 0
        rdma_traffic = 0
        for token in tokens:
            grouped: Dict[int, List[int]] = defaultdict(list)
            for expert in token.topk_experts:
                home = self.baseline_rank_by_expert[expert]
                compute[home] += token.count
                local_demands[(token.source_rank, expert)] += token.count
                grouped[home].append(expert)
            for destination, experts in grouped.items():
                if destination == token.source_rank:
                    continue
                cost = _link_cost(
                    token.source_rank, destination, self.ranks_per_node, self.rdma_cost
                )
                send[token.source_rank] += token.count * cost
                recv[destination] += token.count * cost
                unique_remote_rank_copies += token.count
                closure_counts[
                    (token.source_rank, destination, tuple(sorted(experts)))
                ] += token.count
                if _node_of(token.source_rank, self.ranks_per_node) == _node_of(
                    destination, self.ranks_per_node
                ):
                    nvl_traffic += token.count
                else:
                    rdma_traffic += token.count
        comm = [max(outbound, inbound) for outbound, inbound in zip(send, recv)]
        return (
            PlanMetrics(
                compute,
                send,
                recv,
                comm,
                sum(send),
                unique_remote_rank_copies,
                nvl_traffic,
                rdma_traffic,
                0.0,
            ),
            closure_counts,
            local_demands,
        )

    def _fast_candidate_metrics(
        self,
        baseline: PlanMetrics,
        action: ReplicaAction,
        remote: int,
        closure_count: int,
        local_demands: Mapping[Tuple[int, int], int],
        normalization: Tuple[float, float],
    ) -> PlanMetrics:
        """Apply a local load delta without rerouting the entire window."""
        compute = list(baseline.compute_load)
        send = list(baseline.send_load)
        recv = list(baseline.recv_load)
        for expert in action.experts:
            demand = local_demands.get((action.destination_rank, expert), 0)
            home = self.baseline_rank_by_expert[expert]
            if home != action.destination_rank:
                compute[home] -= demand
                compute[action.destination_rank] += demand

        nvl_traffic = baseline.nvl_traffic
        rdma_traffic = baseline.rdma_traffic
        if closure_count:
            cost = _link_cost(
                action.destination_rank, remote, self.ranks_per_node, self.rdma_cost
            )
            delta = closure_count * cost
            send[action.destination_rank] -= delta
            recv[remote] -= delta
            if _node_of(action.destination_rank, self.ranks_per_node) == _node_of(
                remote, self.ranks_per_node
            ):
                nvl_traffic -= closure_count
            else:
                rdma_traffic -= closure_count
        comm = [max(outbound, inbound) for outbound, inbound in zip(send, recv)]
        metrics = PlanMetrics(
            compute,
            send,
            recv,
            comm,
            sum(send),
            baseline.unique_remote_rank_copies - closure_count,
            nvl_traffic,
            rdma_traffic,
            0.0,
        )
        return self._with_objective(metrics, normalization)

    def _with_objective(
        self, metrics: PlanMetrics, normalization: Tuple[float, float]
    ) -> PlanMetrics:
        avg_compute, avg_comm = normalization
        metrics.objective = max(
            self.compute_weight * load / avg_compute
            + self.communication_weight * traffic / avg_comm
            for load, traffic in zip(metrics.compute_load, metrics.communication_load)
        )
        return metrics

    def _validate_tokens(self, tokens: Sequence[RoutedToken]) -> None:
        for token in tokens:
            if not 0 <= token.source_rank < self.num_ranks:
                raise ValueError(f"invalid source rank {token.source_rank}")
            for expert in token.topk_experts:
                if expert not in self.baseline_rank_by_expert:
                    raise ValueError(f"expert {expert} has no baseline rank")

    def _baseline_placement(self) -> Dict[int, Set[int]]:
        placement = {rank: set() for rank in range(self.num_ranks)}
        for expert, rank in self.baseline_rank_by_expert.items():
            if not 0 <= rank < self.num_ranks:
                raise ValueError(f"invalid baseline rank {rank} for expert {expert}")
            placement[rank].add(expert)
        return placement

    def _candidates(
        self, tokens: Sequence[RoutedToken], placement: Mapping[int, Set[int]]
    ) -> Iterable[ReplicaAction]:
        """Yield compute-balancing single copies and communication-closing bundles."""
        seen: Set[Tuple[int, Tuple[int, ...], str]] = set()

        def add(destination: int, experts: Sequence[int], kind: str):
            missing = tuple(
                sorted(
                    expert
                    for expert in set(experts)
                    if expert not in placement[destination]
                )
            )
            if not missing:
                return
            if self.max_bundle_size is not None and len(missing) > self.max_bundle_size:
                return
            key = (destination, missing, kind)
            if key not in seen:
                seen.add(key)
                yield ReplicaAction(destination, missing, kind)

        # Single-expert copies permit compute offload even when they do not
        # immediately eliminate a destination rank from a bundle.
        observed_experts = sorted(
            {expert for token in tokens for expert in token.topk_experts}
        )
        for expert in observed_experts:
            for destination in range(self.num_ranks):
                yield from add(destination, [expert], "single")

        # A bundle-closure candidate copies every expert that the *current
        # routing decision* sends to one remote rank into the token's source
        # rank.  Looking at every rank that happens to host a replica would
        # over-expand the candidate and does not describe DeepEP's transfer.
        routed, _ = self._route_all(tokens, placement)
        for token, selected in routed:
            for remote_rank, experts in selected.items():
                if remote_rank != token.source_rank:
                    yield from add(token.source_rank, experts, "bundle-closure")

    def _apply_action(
        self, placement: Mapping[int, Set[int]], action: ReplicaAction
    ) -> Optional[Dict[int, Set[int]]]:
        missing = [
            expert
            for expert in action.experts
            if expert not in placement[action.destination_rank]
        ]
        used_slots = sum(
            1
            for expert in placement[action.destination_rank]
            if self.baseline_rank_by_expert[expert] != action.destination_rank
        )
        if (
            used_slots + len(missing)
            > self.replica_slots_per_rank[action.destination_rank]
        ):
            return None
        result = _copy_placement(placement)
        result[action.destination_rank].update(missing)
        return result

    def _route_bundle(
        self, token: RoutedToken, placement: Mapping[int, Set[int]], compute: List[int]
    ) -> Dict[int, List[int]]:
        """Jointly choose replicas, preferring existing destination ranks.

        This is deliberately token-joint: after choosing a rank for one expert,
        choosing that rank for another has zero additional DeepEP traffic.
        """
        candidates_by_expert = {
            expert: [rank for rank, experts in placement.items() if expert in experts]
            for expert in token.topk_experts
        }
        ordered_experts = sorted(
            token.topk_experts, key=lambda expert: len(candidates_by_expert[expert])
        )
        selected: Dict[int, List[int]] = defaultdict(list)
        for expert in ordered_experts:
            best_rank = min(
                candidates_by_expert[expert],
                key=lambda rank: (
                    self.compute_weight * compute[rank]
                    + (
                        0
                        if rank == token.source_rank or rank in selected
                        else self.communication_weight
                        * _link_cost(
                            token.source_rank,
                            rank,
                            self.ranks_per_node,
                            self.rdma_cost,
                        )
                    ),
                    0 if rank == token.source_rank else 1,
                    rank,
                ),
            )
            selected[best_rank].append(expert)
            compute[best_rank] += token.count
        return selected

    def _route_all(
        self, tokens: Sequence[RoutedToken], placement: Mapping[int, Set[int]]
    ) -> Tuple[List[Tuple[RoutedToken, Dict[int, List[int]]]], List[int]]:
        """Route high-volume bundles first so compute placement is stable."""
        compute = [0] * self.num_ranks
        routed: List[Optional[Dict[int, List[int]]]] = [None] * len(tokens)
        for index in sorted(range(len(tokens)), key=lambda i: (-tokens[i].count, i)):
            routed[index] = self._route_bundle(tokens[index], placement, compute)
        return (
            [(token, selected) for token, selected in zip(tokens, routed) if selected],
            compute,
        )

    def _evaluate(
        self,
        tokens: Sequence[RoutedToken],
        placement: Mapping[int, Set[int]],
        normalization: Optional[Tuple[float, float]] = None,
    ) -> PlanMetrics:
        send = [0.0] * self.num_ranks
        recv = [0.0] * self.num_ranks
        nvl_traffic = 0
        rdma_traffic = 0
        unique_remote_rank_copies = 0
        # Use the same volume-first ordering as closure-candidate generation.
        routed, compute = self._route_all(tokens, placement)
        for token, selected in routed:
            for destination in selected:
                if destination == token.source_rank:
                    continue
                count = token.count
                cost = _link_cost(
                    token.source_rank, destination, self.ranks_per_node, self.rdma_cost
                )
                send[token.source_rank] += count * cost
                recv[destination] += count * cost
                unique_remote_rank_copies += count
                if _node_of(token.source_rank, self.ranks_per_node) == _node_of(
                    destination, self.ranks_per_node
                ):
                    nvl_traffic += count
                else:
                    rdma_traffic += count
        comm = [max(outbound, inbound) for outbound, inbound in zip(send, recv)]
        avg_compute, avg_comm = normalization or (
            max(sum(compute) / self.num_ranks, 1.0),
            max(sum(comm) / self.num_ranks, 1.0),
        )
        objective = max(
            self.compute_weight * load / avg_compute
            + self.communication_weight * traffic / avg_comm
            for load, traffic in zip(compute, comm)
        )
        return PlanMetrics(
            compute,
            send,
            recv,
            comm,
            sum(send),
            unique_remote_rank_copies,
            nvl_traffic,
            rdma_traffic,
            objective,
        )
