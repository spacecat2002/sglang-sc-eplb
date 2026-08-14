"""CUDA fast path for offline communication-only MoE placement planning.

The large bundle-wise reductions and replica replay run on CUDA.  The host is
only used for the capacity-constrained initial assignment and for choosing
from the small set of aggregated replica candidates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import triton
import triton.language as tl

from .bundle_aware_replica_planner import ReplicaAction
from .co_routing_graph_solver import (
    CoRoutingGraph,
    GraphPlacement,
)


@triton.jit
def _demand_kernel(
    topk_ptr,
    count_ptr,
    demand_ptr,
    num_values,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_values
    experts = tl.load(topk_ptr + offsets, mask=mask, other=0)
    bundles = offsets // TOPK
    counts = tl.load(count_ptr + bundles, mask=mask, other=0).to(tl.int64)
    tl.atomic_add(demand_ptr + experts, counts, mask=mask)


@triton.jit
def _edge_kernel(
    topk_ptr,
    count_ptr,
    pair_left_ptr,
    pair_right_ptr,
    edges_ptr,
    num_pair_values,
    num_experts,
    TOPK: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_pair_values
    bundle = offsets // NUM_PAIRS
    pair = offsets % NUM_PAIRS
    left_slot = tl.load(pair_left_ptr + pair, mask=mask, other=0)
    right_slot = tl.load(pair_right_ptr + pair, mask=mask, other=0)
    left = tl.load(topk_ptr + bundle * TOPK + left_slot, mask=mask, other=0)
    right = tl.load(topk_ptr + bundle * TOPK + right_slot, mask=mask, other=0)
    count = tl.load(count_ptr + bundle, mask=mask, other=0).to(tl.int64)
    low = tl.minimum(left, right)
    high = tl.maximum(left, right)
    tl.atomic_add(edges_ptr + low * num_experts + high, count, mask=mask)


@triton.jit
def _hypergraph_swap_delta_kernel(
    topk_ptr,
    source_ptr,
    count_ptr,
    rank_by_expert_ptr,
    delta_ptr,
    num_values,
    num_experts,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_values
    candidate = offsets % num_experts
    incidence = offsets // num_experts
    bundle = incidence // TOPK
    slot = incidence % TOPK
    expert = tl.load(topk_ptr + bundle * TOPK + slot, mask=mask, other=0)
    expert_rank = tl.load(rank_by_expert_ptr + expert, mask=mask, other=-1)
    candidate_rank = tl.load(rank_by_expert_ptr + candidate, mask=mask, other=-1)
    source = tl.load(source_ptr + bundle, mask=mask, other=0)

    contains_candidate = tl.zeros((BLOCK,), dtype=tl.int1)
    old_rank_count = tl.zeros((BLOCK,), dtype=tl.int32)
    new_rank_count = tl.zeros((BLOCK,), dtype=tl.int32)
    for other_slot in tl.static_range(0, TOPK):
        other = tl.load(
            topk_ptr + bundle * TOPK + other_slot,
            mask=mask,
            other=0,
        )
        other_rank = tl.load(rank_by_expert_ptr + other, mask=mask, other=-1)
        contains_candidate |= other == candidate
        old_rank_count += (other_rank == expert_rank).to(tl.int32)
        new_rank_count += (other_rank == candidate_rank).to(tl.int32)

    valid = (
        mask
        & (candidate_rank >= 0)
        & (expert_rank != candidate_rank)
        & ~contains_candidate
    )
    removed = (expert_rank != source) & (old_rank_count == 1)
    added = (candidate_rank != source) & (new_rank_count == 0)
    weight = tl.load(count_ptr + bundle, mask=mask, other=0).to(tl.int64)
    delta = (added.to(tl.int64) - removed.to(tl.int64)) * weight
    low = tl.minimum(expert, candidate)
    high = tl.maximum(expert, candidate)
    tl.atomic_add(
        delta_ptr + low * num_experts + high,
        delta,
        mask=valid,
    )


@triton.jit
def _replica_replay_kernel(
    ordered_topk_ptr,
    source_ptr,
    placement_ptr,
    remote_ptr,
    num_ranks,
    rdma_cost,
    TOPK: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
    RANKS_PER_NODE: tl.constexpr,
):
    bundle = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_RANKS)
    rank_mask = ranks < num_ranks
    source = tl.load(source_ptr + bundle)
    selected = tl.zeros((BLOCK_RANKS,), dtype=tl.int1)
    for slot in tl.static_range(0, TOPK):
        expert = tl.load(ordered_topk_ptr + bundle * TOPK + slot)
        candidates = tl.load(
            placement_ptr + expert * num_ranks + ranks,
            mask=rank_mask,
            other=0,
        ).to(tl.int1)
        preferred = (ranks == source) | selected
        if RANKS_PER_NODE > 0:
            same_node = ranks // RANKS_PER_NODE == source // RANKS_PER_NODE
            link_cost = tl.where(same_node, 1.0, rdma_cost)
        else:
            link_cost = ranks.to(tl.float32) * 0.0 + rdma_cost
        communication = tl.where(preferred, 0.0, link_cost)
        # The large multipliers preserve the Python lexicographic order:
        # communication cost, source preference, then rank id.
        score = communication * 1048576.0
        score += tl.where(ranks == source, 0.0, 4096.0)
        score += ranks.to(tl.float32)
        score = tl.where(candidates & rank_mask, score, 1.0e30)
        best_rank = tl.argmin(score, axis=0)
        selected = selected | (ranks == best_rank)
    remote = tl.sum(selected & rank_mask & (ranks != source), axis=0)
    tl.store(remote_ptr + bundle, remote)


@dataclass(frozen=True)
class CudaCoRoutingGraph:
    """Dense CUDA representation used by the fast solver."""

    demand: torch.Tensor
    edge_weights: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.demand.numel()

    def to_cpu_graph(self) -> CoRoutingGraph:
        observed = torch.nonzero(self.demand, as_tuple=False).flatten()
        upper = torch.triu(self.edge_weights, diagonal=1)
        edge_indices = torch.nonzero(upper, as_tuple=False)
        observed_cpu = observed.cpu().tolist()
        demand_cpu = self.demand[observed].cpu().tolist()
        edge_indices_cpu = edge_indices.cpu().tolist()
        edge_values_cpu = upper[edge_indices[:, 0], edge_indices[:, 1]].cpu().tolist()
        demand = dict(zip(observed_cpu, demand_cpu))
        edges = {
            (left, right): weight
            for (left, right), weight in zip(edge_indices_cpu, edge_values_cpu)
        }
        adjacency = {expert: {} for expert in observed_cpu}
        for (left, right), weight in edges.items():
            adjacency[left][right] = weight
            adjacency[right][left] = weight
        return CoRoutingGraph(tuple(observed_cpu), demand, edges, adjacency)

    def cpu_summary(self) -> Tuple[Tuple[int, ...], Dict[int, int], int]:
        demand_cpu = self.demand.cpu().tolist()
        experts = tuple(
            expert for expert, demand in enumerate(demand_cpu) if demand > 0
        )
        demand = {expert: demand_cpu[expert] for expert in experts}
        edge_count = int(
            torch.count_nonzero(torch.triu(self.edge_weights, diagonal=1)).item()
        )
        return experts, demand, edge_count


@dataclass(frozen=True)
class CudaFastReplicaPlan:
    replicas_by_rank: Dict[int, Set[int]]
    actions: List[ReplicaAction]
    unique_remote_rank_copies: int
    solve_seconds: float
    replay_seconds: float


@dataclass(frozen=True)
class CudaHypergraphPlacement:
    rank_by_expert: Dict[int, int]
    experts_by_rank: Dict[int, Tuple[int, ...]]
    initial_remote: int
    final_remote: int
    iterations: int
    solve_seconds: float


def build_co_routing_graph_cuda(
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    *,
    num_experts: Optional[int] = None,
) -> CudaCoRoutingGraph:
    """Build demand and pairwise edge tensors using Triton reductions."""

    if not topk_experts.is_cuda or not count.is_cuda:
        raise ValueError("CUDA graph construction requires CUDA tensors")
    if topk_experts.ndim != 2 or count.shape != (topk_experts.shape[0],):
        raise ValueError("expected topk [bundles, K] and count [bundles]")
    if topk_experts.dtype != torch.int64 or count.dtype != torch.int64:
        raise ValueError("topk and count must use int64")
    num_experts = num_experts or int(topk_experts.max().item()) + 1
    demand = torch.zeros(num_experts, dtype=torch.int64, device=topk_experts.device)
    upper_edges = torch.zeros(
        (num_experts, num_experts), dtype=torch.int64, device=topk_experts.device
    )
    topk = topk_experts.shape[1]
    num_values = topk_experts.numel()
    block = 256
    _demand_kernel[(triton.cdiv(num_values, block),)](
        topk_experts,
        count,
        demand,
        num_values,
        TOPK=topk,
        BLOCK=block,
    )
    pair_slots = torch.triu_indices(
        topk, topk, offset=1, device=topk_experts.device, dtype=torch.int32
    )
    num_pairs = pair_slots.shape[1]
    num_pair_values = topk_experts.shape[0] * num_pairs
    if num_pairs:
        _edge_kernel[(triton.cdiv(num_pair_values, block),)](
            topk_experts,
            count,
            pair_slots[0],
            pair_slots[1],
            upper_edges,
            num_pair_values,
            num_experts,
            TOPK=topk,
            NUM_PAIRS=num_pairs,
            BLOCK=block,
        )
    edge_weights = upper_edges + upper_edges.T
    return CudaCoRoutingGraph(demand, edge_weights)


def _placement_objective(
    graph: CudaCoRoutingGraph,
    rank_by_expert: torch.Tensor,
    num_ranks: int,
    balance_weight: float,
) -> float:
    observed = graph.demand > 0
    different = rank_by_expert[:, None] != rank_by_expert[None, :]
    cut = torch.triu(graph.edge_weights * different, diagonal=1).sum().to(torch.float64)
    if balance_weight:
        loads = torch.zeros(num_ranks, dtype=torch.float64, device=graph.demand.device)
        loads.scatter_add_(
            0,
            rank_by_expert[observed],
            graph.demand[observed].to(torch.float64),
        )
        average = loads.sum() / num_ranks
        cut += balance_weight * (
            (loads - average).square().sum() / average.clamp_min(1)
        )
    return float(cut.item())


def solve_co_routing_graph_cuda(
    graph: CudaCoRoutingGraph,
    *,
    num_ranks: int,
    slots_per_rank: int | Sequence[int],
    max_rounds: Optional[int] = 8,
    balance_weight: float = 0.0,
) -> GraphPlacement:
    """Refine a capacity-constrained placement with GPU swap scoring."""

    if isinstance(slots_per_rank, int):
        capacities = [slots_per_rank] * num_ranks
    else:
        capacities = list(slots_per_rank)
    if len(capacities) != num_ranks or any(capacity < 0 for capacity in capacities):
        raise ValueError("slots_per_rank must contain non-negative capacities")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    edge_cpu = graph.edge_weights.cpu().numpy()
    demand_cpu = graph.demand.cpu().numpy()
    experts = [expert for expert, demand in enumerate(demand_cpu) if demand > 0]
    if len(experts) > sum(capacities):
        raise ValueError("expert count exceeds total rank capacity")
    degree = edge_cpu.sum(axis=1)
    order = sorted(
        experts,
        key=lambda expert: (-int(degree[expert]), -int(demand_cpu[expert]), expert),
    )
    assignment = [set() for _ in range(num_ranks)]
    rank_load = [0] * num_ranks
    for expert in order:
        candidates = [
            rank
            for rank in range(num_ranks)
            if len(assignment[rank]) < capacities[rank]
        ]
        rank = min(
            candidates,
            key=lambda candidate: (
                -sum(int(edge_cpu[expert, other]) for other in assignment[candidate]),
                rank_load[candidate],
                candidate,
            ),
        )
        assignment[rank].add(expert)
        rank_load[rank] += int(demand_cpu[expert])
    initial_rank_map = {
        expert: rank
        for rank, assigned_experts in enumerate(assignment)
        for expert in assigned_experts
    }
    rank_by_expert = torch.full(
        (graph.num_experts,), -1, dtype=torch.int64, device=graph.demand.device
    )
    expert_ids = torch.tensor(
        list(initial_rank_map), dtype=torch.int64, device=graph.demand.device
    )
    initial_ranks = torch.tensor(
        list(initial_rank_map.values()),
        dtype=torch.int64,
        device=graph.demand.device,
    )
    rank_by_expert[expert_ids] = initial_ranks
    initial_cut = _placement_objective(graph, rank_by_expert, num_ranks, balance_weight)
    observed = rank_by_expert >= 0
    observed_pair = observed[:, None] & observed[None, :]
    expert_index = torch.arange(graph.num_experts, device=graph.demand.device)
    rounds = 0
    while max_rounds is None or rounds < max_rounds:
        weight_to_rank = torch.zeros(
            (graph.num_experts, num_ranks),
            dtype=torch.int64,
            device=graph.demand.device,
        )
        destination = rank_by_expert.clamp_min(0)[None, :].expand(graph.num_experts, -1)
        weight_to_rank.scatter_add_(1, destination, graph.edge_weights)
        total_weight = graph.edge_weights.sum(dim=1)
        old_internal = weight_to_rank[expert_index, rank_by_expert.clamp_min(0)]
        old_cut = total_weight - old_internal
        move_gain = total_weight[:, None] - weight_to_rank - old_cut[:, None]
        left_to_right = move_gain.gather(
            1, rank_by_expert.clamp_min(0)[None, :].expand(graph.num_experts, -1)
        )
        swap_gain = left_to_right + left_to_right.T + 2 * graph.edge_weights
        if balance_weight:
            rank_load = torch.zeros(
                num_ranks, dtype=torch.float64, device=graph.demand.device
            )
            rank_load.scatter_add_(
                0,
                rank_by_expert[observed],
                graph.demand[observed].to(torch.float64),
            )
            average = rank_load.sum() / num_ranks
            left_rank = rank_by_expert.clamp_min(0)[:, None]
            right_rank = rank_by_expert.clamp_min(0)[None, :]
            left_load = rank_load[left_rank]
            right_load = rank_load[right_rank]
            demand_delta = graph.demand[None, :] - graph.demand[:, None]
            before = (left_load - average).square() + (right_load - average).square()
            after = (left_load + demand_delta - average).square() + (
                right_load - demand_delta - average
            ).square()
            swap_gain = swap_gain.to(torch.float64) + balance_weight * (
                after - before
            ) / average.clamp_min(1)
        valid = (
            observed_pair
            & (expert_index[:, None] < expert_index[None, :])
            & (rank_by_expert[:, None] != rank_by_expert[None, :])
        )
        infinity = torch.tensor(
            float("inf")
            if swap_gain.is_floating_point()
            else torch.iinfo(torch.int64).max,
            dtype=swap_gain.dtype,
            device=graph.demand.device,
        )
        candidates = torch.where(valid, swap_gain, infinity)
        best_gain_tensor = candidates.min()
        best_gain = float(best_gain_tensor.item())
        if best_gain >= 0:
            break
        left_rank = rank_by_expert[:, None].expand(graph.num_experts, -1)
        right_rank = rank_by_expert[None, :].expand(-1, graph.num_experts)
        rank_low = torch.minimum(left_rank, right_rank)
        rank_high = torch.maximum(left_rank, right_rank)
        expert_left = torch.where(
            left_rank < right_rank,
            expert_index[:, None],
            expert_index[None, :],
        )
        expert_right = torch.where(
            left_rank < right_rank,
            expert_index[None, :],
            expert_index[:, None],
        )
        tie_key = (
            (rank_low * num_ranks + rank_high) * graph.num_experts + expert_left
        ) * graph.num_experts + expert_right
        best_ties = valid & (swap_gain == best_gain_tensor)
        tie_limit = torch.iinfo(torch.int64).max
        selected_tie = torch.where(best_ties, tie_key, tie_limit)
        flat_index = int(torch.argmin(selected_tie).item())
        left = flat_index // graph.num_experts
        right = flat_index % graph.num_experts
        left_rank_value = rank_by_expert[left].clone()
        rank_by_expert[left] = rank_by_expert[right]
        rank_by_expert[right] = left_rank_value
        rounds += 1

    ranks_cpu = rank_by_expert[expert_ids].cpu().tolist()
    experts_cpu = expert_ids.cpu().tolist()
    rank_map = dict(zip(experts_cpu, ranks_cpu))
    experts_by_rank = {
        rank: tuple(sorted(expert for expert, home in rank_map.items() if home == rank))
        for rank in range(num_ranks)
    }
    return GraphPlacement(
        rank_map,
        experts_by_rank,
        initial_cut,
        _placement_objective(graph, rank_by_expert, num_ranks, balance_weight),
        rounds,
    )


def _evaluate_primary_remote_cuda(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    rank_by_expert: torch.Tensor,
) -> int:
    destinations = rank_by_expert[topk_experts].sort(dim=1).values
    is_new = torch.ones_like(destinations, dtype=torch.bool)
    is_new[:, 1:] = destinations[:, 1:] != destinations[:, :-1]
    remote = is_new & (destinations != source_rank[:, None])
    return int((remote.sum(dim=1).to(torch.int64) * count).sum().item())


def refine_hypergraph_placement_cuda(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    initial_rank_by_expert: Mapping[int, int],
    *,
    num_ranks: int,
    max_rounds: Optional[int] = 8,
) -> CudaHypergraphPlacement:
    """Refine placement against exact Top-K distinct-remote-rank traffic."""

    if not source_rank.is_cuda or not topk_experts.is_cuda or not count.is_cuda:
        raise ValueError("CUDA hypergraph refinement requires CUDA tensors")
    if source_rank.device != topk_experts.device or count.device != topk_experts.device:
        raise ValueError("all hypergraph tensors must use the same CUDA device")
    if (
        topk_experts.ndim != 2
        or source_rank.ndim != 1
        or source_rank.shape != count.shape
    ):
        raise ValueError("expected source/count [bundles] and topk [bundles, K]")
    if count.shape[0] != topk_experts.shape[0]:
        raise ValueError("bundle tensor lengths disagree")
    if any(
        tensor.dtype != torch.int64 for tensor in (source_rank, topk_experts, count)
    ):
        raise ValueError("source, topk, and count must use int64")
    if max_rounds is not None and max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if num_ranks < 1:
        raise ValueError("num_ranks must be positive")
    if not initial_rank_by_expert:
        raise ValueError("initial placement must not be empty")

    num_experts = max(max(initial_rank_by_expert), int(topk_experts.max().item())) + 1
    rank_by_expert = torch.full(
        (num_experts,), -1, dtype=torch.int64, device=topk_experts.device
    )
    expert_ids = torch.tensor(
        list(initial_rank_by_expert), dtype=torch.int64, device=topk_experts.device
    )
    initial_ranks = torch.tensor(
        list(initial_rank_by_expert.values()),
        dtype=torch.int64,
        device=topk_experts.device,
    )
    if bool(((initial_ranks < 0) | (initial_ranks >= num_ranks)).any().item()):
        raise ValueError("initial placement contains an invalid rank")
    rank_by_expert[expert_ids] = initial_ranks
    if bool((rank_by_expert[topk_experts] < 0).any().item()):
        raise ValueError("initial placement is missing an observed expert")

    initial_remote = _evaluate_primary_remote_cuda(
        source_rank, topk_experts, count, rank_by_expert
    )
    torch.cuda.current_stream(topk_experts.device).synchronize()
    solve_start = time.perf_counter()
    rounds = 0
    block = 256
    num_values = topk_experts.numel() * num_experts
    expert_index = torch.arange(num_experts, device=topk_experts.device)
    observed = rank_by_expert >= 0
    valid_pair = (
        observed[:, None]
        & observed[None, :]
        & (expert_index[:, None] < expert_index[None, :])
    )
    while max_rounds is None or rounds < max_rounds:
        deltas = torch.zeros(
            (num_experts, num_experts),
            dtype=torch.int64,
            device=topk_experts.device,
        )
        _hypergraph_swap_delta_kernel[(triton.cdiv(num_values, block),)](
            topk_experts,
            source_rank,
            count,
            rank_by_expert,
            deltas,
            num_values,
            num_experts,
            TOPK=topk_experts.shape[1],
            BLOCK=block,
        )
        cross_rank = rank_by_expert[:, None] != rank_by_expert[None, :]
        candidates = torch.where(
            valid_pair & cross_rank,
            deltas,
            torch.iinfo(torch.int64).max,
        )
        flat_index = int(torch.argmin(candidates).item())
        best_delta = int(candidates.flatten()[flat_index].item())
        if best_delta >= 0:
            break
        left = flat_index // num_experts
        right = flat_index % num_experts
        left_rank = rank_by_expert[left].clone()
        rank_by_expert[left] = rank_by_expert[right]
        rank_by_expert[right] = left_rank
        rounds += 1

    final_remote = _evaluate_primary_remote_cuda(
        source_rank, topk_experts, count, rank_by_expert
    )
    solve_seconds = time.perf_counter() - solve_start
    ranks_cpu = rank_by_expert[expert_ids].cpu().tolist()
    experts_cpu = expert_ids.cpu().tolist()
    rank_map = dict(zip(experts_cpu, ranks_cpu))
    experts_by_rank = {
        rank: tuple(sorted(expert for expert, home in rank_map.items() if home == rank))
        for rank in range(num_ranks)
    }
    return CudaHypergraphPlacement(
        rank_map,
        experts_by_rank,
        initial_remote,
        final_remote,
        rounds,
        solve_seconds,
    )


def _primary_traffic_matrix(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    rank_by_expert: torch.Tensor,
    num_ranks: int,
) -> torch.Tensor:
    destination = rank_by_expert[topk_experts].sort(dim=1).values
    is_new = torch.ones_like(destination, dtype=torch.bool)
    is_new[:, 1:] = destination[:, 1:] != destination[:, :-1]
    remote = is_new & (destination != source_rank[:, None])
    traffic = torch.zeros(
        num_ranks * num_ranks, dtype=torch.int64, device=topk_experts.device
    )
    pair_ids = source_rank[:, None] * num_ranks + destination
    traffic.scatter_add_(
        0,
        pair_ids[remote],
        count[:, None].expand_as(destination)[remote],
    )
    return traffic.reshape(num_ranks, num_ranks)


def _aggregate_closures(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    rank_by_expert: torch.Tensor,
    num_experts: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_chunks = []
    count_chunks = []
    topk = topk_experts.shape[1]
    earlier = torch.arange(topk, device=topk_experts.device)
    for start in range(0, count.numel(), chunk_size):
        stop = min(start + chunk_size, count.numel())
        experts = topk_experts[start:stop]
        sources = source_rank[start:stop]
        destinations = rank_by_expert[experts]
        same_destination = destinations[:, :, None] == destinations[:, None, :]
        has_earlier = (
            same_destination & (earlier[None, None, :] < earlier[None, :, None])
        ).any(dim=2)
        valid = (~has_earlier) & (destinations != sources[:, None])
        closures = torch.where(
            same_destination,
            experts[:, None, :],
            torch.full_like(experts[:, None, :], num_experts),
        )
        keys = torch.cat(
            (
                sources[:, None, None].expand(-1, topk, 1),
                destinations[:, :, None],
                closures,
            ),
            dim=2,
        )
        keys = keys[valid].to(torch.int32)
        weights = count[start:stop, None].expand(-1, topk)[valid]
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        totals = torch.zeros(
            unique.shape[0], dtype=torch.int64, device=topk_experts.device
        )
        totals.scatter_add_(0, inverse, weights)
        key_chunks.append(unique)
        count_chunks.append(totals)
    if not key_chunks:
        return (
            torch.empty((0, topk + 2), dtype=torch.int32, device=topk_experts.device),
            torch.empty(0, dtype=torch.int64, device=topk_experts.device),
        )
    all_keys = torch.cat(key_chunks)
    all_counts = torch.cat(count_chunks)
    unique, inverse = torch.unique(all_keys, dim=0, return_inverse=True)
    totals = torch.zeros(unique.shape[0], dtype=torch.int64, device=topk_experts.device)
    totals.scatter_add_(0, inverse, all_counts)
    return unique, totals


def evaluate_replica_remote_cuda(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    placement_mask: torch.Tensor,
    *,
    ranks_per_node: Optional[int] = None,
    rdma_cost: float = 4.0,
) -> int:
    """Replay replica-aware communication with one Triton program per bundle."""

    candidate_counts = placement_mask[topk_experts].sum(dim=2)
    order = torch.argsort(candidate_counts, dim=1, stable=True)
    ordered_topk = topk_experts.gather(1, order).contiguous()
    remote = torch.empty(count.numel(), dtype=torch.int32, device=topk_experts.device)
    num_ranks = placement_mask.shape[1]
    block_ranks = triton.next_power_of_2(num_ranks)
    _replica_replay_kernel[(count.numel(),)](
        ordered_topk,
        source_rank,
        placement_mask,
        remote,
        num_ranks,
        rdma_cost,
        TOPK=topk_experts.shape[1],
        BLOCK_RANKS=block_ranks,
        RANKS_PER_NODE=ranks_per_node or 0,
    )
    return int((remote.to(torch.int64) * count).sum().item())


def plan_communication_replicas_cuda(
    source_rank: torch.Tensor,
    topk_experts: torch.Tensor,
    count: torch.Tensor,
    rank_by_expert: Mapping[int, int],
    *,
    num_ranks: int,
    replica_slots_per_rank: int | Sequence[int],
    max_candidates: int = 32,
    max_bundle_size: Optional[int] = None,
    ranks_per_node: Optional[int] = None,
    rdma_cost: float = 4.0,
    chunk_size: int = 65536,
) -> CudaFastReplicaPlan:
    """Choose at most one communication-closing replica action per rank."""

    # Keep synchronization scoped to the stream used by the caller.  A
    # device-wide barrier would also wait for unrelated work and defeat the
    # layer-to-layer H2D prefetch in the offline driver.
    torch.cuda.current_stream(topk_experts.device).synchronize()
    solve_start = time.perf_counter()
    if isinstance(replica_slots_per_rank, int):
        capacities = [replica_slots_per_rank] * num_ranks
    else:
        capacities = list(replica_slots_per_rank)
    if len(capacities) != num_ranks:
        raise ValueError("replica slots must contain one value per rank")
    num_experts = max(rank_by_expert) + 1
    homes = torch.full(
        (num_experts,), -1, dtype=torch.int64, device=topk_experts.device
    )
    expert_ids = torch.tensor(
        list(rank_by_expert), dtype=torch.int64, device=topk_experts.device
    )
    home_ranks = torch.tensor(
        list(rank_by_expert.values()), dtype=torch.int64, device=topk_experts.device
    )
    homes[expert_ids] = home_ranks
    traffic = _primary_traffic_matrix(
        source_rank, topk_experts, count, homes, num_ranks
    )
    keys, closure_counts = _aggregate_closures(
        source_rank,
        topk_experts,
        count,
        homes,
        num_experts,
        chunk_size,
    )
    closure_sizes = (keys[:, 2:] != num_experts).sum(dim=1)
    ranks = torch.arange(num_ranks, device=topk_experts.device)
    if ranks_per_node is None:
        same_node = ranks[:, None] == ranks[None, :]
    else:
        same_node = ranks[:, None] // ranks_per_node == ranks[None, :] // ranks_per_node
    link_cost = torch.where(
        same_node,
        torch.ones((), dtype=torch.float64, device=topk_experts.device),
        torch.full((), rdma_cost, dtype=torch.float64, device=topk_experts.device),
    )
    weighted_traffic = traffic.to(torch.float64) * link_cost
    send = weighted_traffic.sum(dim=1)
    recv = weighted_traffic.sum(dim=0)
    baseline_max = torch.maximum(send, recv).max()
    baseline_total = send.sum()
    proposals = []
    for destination in range(num_ranks):
        valid = (keys[:, 0] == destination) & (closure_sizes <= capacities[destination])
        if max_bundle_size is not None:
            valid &= closure_sizes <= max_bundle_size
        candidate_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if not candidate_indices.numel():
            continue
        keep = min(max_candidates, candidate_indices.numel())
        candidate_remotes = keys[candidate_indices, 1].to(torch.int64)
        hot_score = (
            closure_counts[candidate_indices].to(torch.float64)
            * link_cost[destination, candidate_remotes]
        )
        hot = torch.topk(hot_score, keep).indices
        candidate_indices = candidate_indices[hot]
        candidate_counts = closure_counts[candidate_indices].to(torch.float64)
        remotes = keys[candidate_indices, 1].to(torch.int64)
        candidate_delta = candidate_counts * link_cost[destination, remotes]
        candidate_send = send[None, :].expand(keep, -1).clone()
        candidate_recv = recv[None, :].expand(keep, -1).clone()
        row = torch.arange(keep, device=topk_experts.device)
        candidate_send[row, destination] -= candidate_delta
        candidate_recv[row, remotes] -= candidate_delta
        max_load = torch.maximum(candidate_send, candidate_recv).max(dim=1).values
        total = baseline_total - candidate_delta
        improves = (max_load < baseline_max) | (
            (max_load == baseline_max) & (total < baseline_total)
        )
        if not bool(improves.any().item()):
            continue
        best_load = torch.where(
            improves, max_load, torch.full_like(max_load, float("inf"))
        ).min()
        load_ties = improves & (max_load == best_load)
        best_total = torch.where(
            load_ties, total, torch.full_like(total, float("inf"))
        ).min()
        score_ties = load_ties & (total == best_total)
        sizes = closure_sizes[candidate_indices]
        best_size = torch.where(
            score_ties,
            sizes,
            torch.full_like(sizes, torch.iinfo(sizes.dtype).max),
        ).min()
        finalist = torch.nonzero(
            score_ties & (sizes == best_size), as_tuple=False
        ).flatten()[0]
        best = candidate_indices[int(finalist.item())]
        proposals.append(best)

    placement_mask = torch.zeros(
        (num_experts, num_ranks), dtype=torch.bool, device=topk_experts.device
    )
    placement_mask[expert_ids, home_ranks] = True
    actions = []
    for candidate_index in proposals:
        key = keys[candidate_index].cpu().tolist()
        destination = key[0]
        experts = tuple(expert for expert in key[2:] if expert != num_experts)
        placement_mask[
            torch.tensor(experts, device=topk_experts.device), destination
        ] = True
        actions.append(ReplicaAction(destination, experts, "cuda-bundle-closure"))
    solve_seconds = time.perf_counter() - solve_start
    replay_start = time.perf_counter()
    remote = evaluate_replica_remote_cuda(
        source_rank,
        topk_experts,
        count,
        placement_mask,
        ranks_per_node=ranks_per_node,
        rdma_cost=rdma_cost,
    )
    torch.cuda.current_stream(topk_experts.device).synchronize()
    replay_seconds = time.perf_counter() - replay_start
    placement = {
        rank: set(torch.nonzero(placement_mask[:, rank]).flatten().cpu().tolist())
        for rank in range(num_ranks)
    }
    return CudaFastReplicaPlan(
        placement, actions, remote, solve_seconds, replay_seconds
    )
