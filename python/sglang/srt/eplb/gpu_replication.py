"""CUDA implementation of the source-aware replication simulation.

This module implements the GRACE+ source-demand objective rather than importing
UltraEP's placement solver. The large trace-dependent work is kept on CUDA;
the small ragged placement/quota serialization remains in the exact CPU path.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
import torch

from .expert_affinity_graph import RoutedArrays, RoutedToken, as_routed_arrays
from .grace_plus_replication import (
    ReplicaMetrics,
    ReplicaPlacement,
    balance_replica_compute,
    _replica_arrays,
)


_CUDA_CHUNK_SIZE = 262_144


def _record(timing: dict[str, float] | None, key: str, started: float) -> None:
    if timing is not None:
        torch.cuda.synchronize()
        timing[key] = timing.get(key, 0.0) + (
            time.perf_counter() - started
        ) * 1000.0


def _cuda_arrays(
    tokens: Sequence[RoutedToken] | RoutedArrays, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = as_routed_arrays(tokens)
    return (
        torch.as_tensor(arrays.source_rank, device=device, dtype=torch.int64),
        torch.as_tensor(arrays.topk_experts, device=device, dtype=torch.int64),
        torch.as_tensor(arrays.count, device=device, dtype=torch.int64),
    )


def _source_demand_cuda(
    source: torch.Tensor,
    topk: torch.Tensor,
    count: torch.Tensor,
    experts: int,
    ranks: int,
) -> torch.Tensor:
    keys = topk.reshape(-1) * ranks + source[:, None].expand_as(topk).reshape(-1)
    weights = count[:, None].expand_as(topk).reshape(-1)
    result = torch.zeros(experts * ranks, device=source.device, dtype=torch.int64)
    result.index_add_(0, keys, weights)
    return result.view(experts, ranks)


def _route_cuda(
    source: torch.Tensor,
    topk: torch.Tensor,
    count: torch.Tensor,
    primary: torch.Tensor,
    replica_mask: torch.Tensor,
    ranks: int,
) -> ReplicaMetrics:
    traffic = torch.zeros(ranks * ranks, device=source.device, dtype=torch.int64)
    compute = torch.zeros(ranks, device=source.device, dtype=torch.int64)
    for start in range(0, source.numel(), _CUDA_CHUNK_SIZE):
        end = min(start + _CUDA_CHUNK_SIZE, source.numel())
        src = source[start:end]
        ids = topk[start:end]
        cnt = count[start:end]
        destinations = primary[ids]
        local = replica_mask[ids].gather(
            2, src[:, None, None].expand(-1, ids.shape[1], 1)
        ).squeeze(2)
        destinations = torch.where(local, src[:, None], destinations)
        flat_count = cnt[:, None].expand_as(ids).reshape(-1)
        compute.index_add_(0, destinations.reshape(-1), flat_count)
        for target in range(ranks):
            sent = (destinations == target).any(dim=1) & (src != target)
            if sent.any():
                keys = src[sent] * ranks + target
                traffic.index_add_(0, keys, cnt[sent])
    traffic = traffic.view(ranks, ranks)
    return ReplicaMetrics(
        remote=int(traffic.sum().item()),
        max_pair_traffic=int(traffic.max().item()),
        max_ingress=int(traffic.sum(dim=0).max().item()),
        max_egress=int(traffic.sum(dim=1).max().item()),
        compute_load=tuple(int(x) for x in compute.cpu().tolist()),
    )


def evaluate_replicated_placement_cuda(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    placement: Mapping[int, int | Sequence[int]],
    *,
    num_ranks: int,
    device: str | torch.device = "cuda",
    timing: dict[str, float] | None = None,
) -> ReplicaMetrics:
    """Evaluate a placement with CUDA histogram and traffic operations."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda is unavailable")
    source, topk, count = _cuda_arrays(tokens, device)
    arrays = as_routed_arrays(tokens)
    _, primary_np, mask_np = _replica_arrays(arrays, placement, num_ranks)
    primary = torch.as_tensor(primary_np, device=device, dtype=torch.int64)
    replica_mask = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
    started = time.perf_counter()
    metrics = _route_cuda(source, topk, count, primary, replica_mask, num_ranks)
    _record(timing, "cuda_route_eval_ms", started)
    return metrics


def replicate_source_top_experts_cuda(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    primary: Mapping[int, int],
    *,
    num_ranks: int,
    max_extra_per_rank: int,
    compute_imbalance_limit: float | None = None,
    communication_budget_ratio: float | None = None,
    device: str | torch.device = "cuda",
    timing: dict[str, float] | None = None,
) -> ReplicaPlacement:
    """Run source-aware Top-N replication with CUDA trace processing.

    The replica choice matches the source-demand Top-N policy. If compute
    balancing is requested, the GPU-produced replica set is passed to the
    existing exact quota path; this preserves the plan while keeping all
    trace histogram and communication scoring on CUDA.
    """

    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda is unavailable")
    source, topk, count = _cuda_arrays(tokens, device)
    num_experts = len(primary)
    started = time.perf_counter()
    demand = _source_demand_cuda(
        source, topk, count, num_experts, num_ranks
    )
    primary_tensor = torch.as_tensor(
        [primary[e] for e in range(num_experts)], device=device, dtype=torch.int64
    )
    source_ids = torch.arange(num_ranks, device=device, dtype=torch.int64)[:, None]
    scores = demand.t().clone()
    scores = scores.masked_fill(primary_tensor[None, :] == source_ids, -1)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    selected = order[:, :max_extra_per_rank]
    valid = scores.gather(1, selected) > 0
    replica_mask = torch.zeros(
        (num_experts, num_ranks), device=device, dtype=torch.bool
    )
    replica_mask.scatter_(1, primary_tensor[:, None], True)
    for source_rank in range(num_ranks):
        ids = selected[source_rank][valid[source_rank]]
        if ids.numel():
            replica_mask[ids, source_rank] = True
    _record(timing, "communication_replication_ms", started)

    placement = {
        expert: tuple(
            int(rank)
            for rank in torch.nonzero(replica_mask[expert]).flatten().cpu().tolist()
        )
        for expert in range(num_experts)
    }
    metrics_started = time.perf_counter()
    metrics = _route_cuda(
        source, topk, count, primary_tensor, replica_mask, num_ranks
    )
    _record(timing, "quota_allocation_ms", metrics_started)
    copies = sum(max(0, len(ranks) - 1) for ranks in placement.values())
    routing = tuple(
        tuple(
            source_rank
            if bool(replica_mask[expert, source_rank].item())
            else int(primary[expert])
            for expert in range(num_experts)
        )
        for source_rank in range(num_ranks)
    )
    placement_result = ReplicaPlacement(
        placement,
        routing,
        metrics,
        copies,
        source_demand=demand.cpu().numpy(),
    )
    if compute_imbalance_limit is not None:
        quota_started = time.perf_counter()
        placement_result = balance_replica_compute(
            tokens,
            placement_result,
            num_ranks=num_ranks,
            max_extra_per_rank=0,
            communication_budget_ratio=communication_budget_ratio,
            timing=timing,
        )
        _record(timing, "compute_replication_ms", quota_started)
    elif communication_budget_ratio is not None:
        raise ValueError(
            "communication_budget_ratio requires --compute-imbalance-limit "
            "so that a quota can be solved"
        )
    return placement_result
