"""CUDA implementation of the source-aware replication simulation.

This module implements the GRACE+ source-demand objective rather than importing
UltraEP's placement solver. The large trace-dependent work is kept on CUDA;
the small ragged placement/quota serialization remains in the exact CPU path.
"""

from __future__ import annotations

import math
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from grace_cuda import _C

from .expert_affinity_graph import RoutedArrays, RoutedToken, as_routed_arrays
from .grace_plus_replication import (
    ReplicaMetrics,
    ReplicaPlacement,
    _replica_arrays,
)


def _record(timing: dict[str, float] | None, key: str, started: float) -> None:
    if timing is not None:
        torch.cuda.synchronize()
        timing[key] = timing.get(key, 0.0) + (time.perf_counter() - started) * 1000.0


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
    return _C.source_demand(source, topk, count, experts, ranks)


def _route_cuda(
    source: torch.Tensor,
    topk: torch.Tensor,
    count: torch.Tensor,
    primary: torch.Tensor,
    replica_mask: torch.Tensor,
    ranks: int,
) -> ReplicaMetrics:
    traffic, compute = _C.traffic(source, topk, count, primary, replica_mask, ranks)
    traffic = traffic.cpu().numpy()
    return ReplicaMetrics(
        remote=int(traffic.sum()),
        max_pair_traffic=int(traffic.max()),
        max_ingress=int(traffic.sum(axis=0).max()),
        max_egress=int(traffic.sum(axis=1).max()),
        compute_load=tuple(int(x) for x in compute.cpu().tolist()),
    )


def _quota_cuda(
    demand: torch.Tensor,
    replica_mask: torch.Tensor,
    primary: torch.Tensor,
    routing: torch.Tensor,
    compute_imbalance_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_demand = demand.sum(dim=1)
    expert_order = torch.argsort(expert_demand, descending=True, stable=True)
    flexible = replica_mask.sum(dim=1) > 1
    expert_order = expert_order[
        torch.argsort(flexible[expert_order].to(torch.int8), stable=True)
    ]
    source_order = torch.argsort(demand.t(), dim=1, descending=True, stable=True)
    return _C.solve_quota(
        demand,
        replica_mask,
        primary,
        routing,
        expert_order,
        source_order,
        compute_imbalance_limit,
    )


def _quota_route_cuda(
    source: torch.Tensor,
    topk: torch.Tensor,
    count: torch.Tensor,
    quota: torch.Tensor,
    replica_mask: torch.Tensor,
    primary: torch.Tensor,
    addition_order: torch.Tensor,
    ranks: int,
) -> ReplicaMetrics:
    experts = replica_mask.shape[0]
    keys = (source[:, None] * experts + topk).reshape(-1)
    weights = count[:, None].expand_as(topk).reshape(-1)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    sorted_weights = weights[order]
    before = sorted_weights.cumsum(0) - sorted_weights
    starts = torch.empty_like(sorted_keys, dtype=torch.bool)
    starts[0] = True
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_base = torch.cummax(torch.where(starts, before, 0), dim=0).values
    within = before - group_base
    ordinals = torch.empty_like(within)
    ordinals.scatter_(0, order, within)
    traffic, compute = _C.quota_traffic(
        source,
        topk,
        count,
        quota,
        replica_mask,
        primary,
        addition_order,
        ordinals.view_as(topk),
        ranks,
    )
    traffic = traffic.cpu().numpy()
    return ReplicaMetrics(
        remote=int(traffic.sum()),
        max_pair_traffic=int(traffic.max()),
        max_ingress=int(traffic.sum(axis=0).max()),
        max_egress=int(traffic.sum(axis=1).max()),
        compute_load=tuple(int(x) for x in compute.cpu().tolist()),
    )


def _communication_quota_cuda(
    demand: torch.Tensor,
    replica_mask: torch.Tensor,
    primary: torch.Tensor,
    routing: torch.Tensor,
    baseline: ReplicaMetrics,
    budget_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = demand.shape[1]
    rank_ids = torch.arange(ranks, device=demand.device, dtype=torch.int64)
    source_ids = rank_ids[:, None]
    budgets = tuple(
        math.ceil(value * budget_ratio)
        for value in (
            baseline.remote,
            baseline.max_pair_traffic,
            baseline.max_ingress,
            baseline.max_egress,
        )
    )
    candidates = []
    summaries = []
    rank_cost = None
    preferred = routing
    for _ in range(4):
        if rank_cost is not None:
            scores = rank_cost * ranks + rank_ids
            scores = torch.where(
                replica_mask[None, :, :],
                scores[None, None, :],
                torch.iinfo(torch.int64).max,
            )
            preferred = scores.argmin(dim=2)
            preferred = torch.where(replica_mask.t(), source_ids, preferred)
        quota, quota_routing = _quota_cuda(
            demand, replica_mask, primary, preferred, compute_imbalance_limit=-1.0
        )
        traffic = quota.sum(dim=1)
        traffic.fill_diagonal_(0)
        compute = quota.sum(dim=(0, 1))
        summaries.append(
            torch.stack(
                (
                    traffic.sum(),
                    traffic.max(),
                    traffic.sum(dim=0).max(),
                    traffic.sum(dim=1).max(),
                    compute.max(),
                    compute.square().sum(),
                )
            )
        )
        candidates.append((quota, quota_routing))
        rank_cost = traffic.sum(dim=0) + traffic.sum(dim=1)

    best = None
    for index, values in enumerate(torch.stack(summaries).cpu().tolist()):
        remote, pair, ingress, egress, max_compute, square_compute = values
        violation = max(
            remote / budgets[0] if budgets[0] else float(remote),
            pair / budgets[1] if budgets[1] else float(pair),
            ingress / budgets[2] if budgets[2] else float(ingress),
            egress / budgets[3] if budgets[3] else float(egress),
        )
        key = (violation, max_compute, square_compute, remote, pair)
        if best is None or key < best[0]:
            best = (key, index)
    assert best is not None
    return candidates[best[1]]


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
    max_compute_extra_per_rank: int = 0,
    compute_imbalance_limit: float | None = None,
    communication_budget_ratio: float | None = None,
    device: str | torch.device = "cuda",
    timing: dict[str, float] | None = None,
    materialize_quota: bool = True,
) -> ReplicaPlacement:
    """Run source-aware Top-N replication with CUDA trace processing.

    The replica choice matches the source-demand Top-N policy. If compute
    balancing is requested, the GPU-produced replica set is passed to the
    existing exact quota path; this preserves the plan while keeping all
    trace histogram and communication scoring on CUDA.
    """

    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    if max_compute_extra_per_rank < 0:
        raise ValueError("max_compute_extra_per_rank must be non-negative")
    if compute_imbalance_limit is not None and (
        compute_imbalance_limit < 1 or not math.isfinite(compute_imbalance_limit)
    ):
        raise ValueError("invalid compute imbalance limit")
    if communication_budget_ratio is not None and (
        compute_imbalance_limit is None
        or communication_budget_ratio < 0
        or not math.isfinite(communication_budget_ratio)
    ):
        raise ValueError(
            "communication budget requires a compute imbalance limit and a finite ratio"
        )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda is unavailable")
    started = time.perf_counter()
    source, topk, count = _cuda_arrays(tokens, device)
    num_experts = len(primary)
    demand = _source_demand_cuda(source, topk, count, num_experts, num_ranks)
    primary_np = np.asarray([primary[e] for e in range(num_experts)], dtype=np.int64)
    primary_tensor = torch.as_tensor(primary_np, device=device)
    replica_mask = _C.select_topn(demand, primary_tensor, max_extra_per_rank)
    routing_tensor = torch.where(
        replica_mask.t(),
        torch.arange(num_ranks, device=device, dtype=torch.int64)[:, None],
        primary_tensor[None, :],
    )
    _record(timing, "communication_replication_ms", started)

    quota = None
    balance_copies = None
    budget_baseline = None
    addition_order = torch.zeros_like(demand)
    if max_compute_extra_per_rank:
        if communication_budget_ratio is not None:
            metrics_started = time.perf_counter()
            initial_metrics = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            _record(timing, "communication_replication_ms", metrics_started)
            quota_started = time.perf_counter()
            initial_quota, routing_tensor = _communication_quota_cuda(
                demand,
                replica_mask,
                primary_tensor,
                routing_tensor,
                initial_metrics,
                communication_budget_ratio,
            )
            _record(timing, "quota_solve_ms", quota_started)
            allocation_started = time.perf_counter()
            budget_baseline = _quota_route_cuda(
                source,
                topk,
                count,
                initial_quota,
                replica_mask,
                primary_tensor,
                addition_order,
                num_ranks,
            )
            _record(timing, "quota_allocation_ms", allocation_started)
        elif compute_imbalance_limit is not None:
            quota_started = time.perf_counter()
            _, routing_tensor = _quota_cuda(
                demand,
                replica_mask,
                primary_tensor,
                routing_tensor,
                compute_imbalance_limit,
            )
            _record(timing, "quota_solve_ms", quota_started)
        compute_started = time.perf_counter()
        replica_mask, balance_copies, addition_order = _C.select_compute_replicas(
            demand, replica_mask, max_compute_extra_per_rank
        )
        _record(timing, "compute_replication_ms", compute_started)
        if communication_budget_ratio is not None:
            quota_started = time.perf_counter()
            _, routing_tensor = _quota_cuda(
                demand,
                replica_mask,
                primary_tensor,
                routing_tensor,
                compute_imbalance_limit=-1.0,
            )
            _record(timing, "quota_solve_ms", quota_started)

    needs_quota = compute_imbalance_limit is not None or max_compute_extra_per_rank > 0
    if not needs_quota:
        metrics_started = time.perf_counter()
        metrics = _route_cuda(
            source, topk, count, primary_tensor, replica_mask, num_ranks
        )
        _record(timing, "quota_allocation_ms", metrics_started)
    elif communication_budget_ratio is None:
        quota_started = time.perf_counter()
        quota, routing_tensor = _quota_cuda(
            demand,
            replica_mask,
            primary_tensor,
            routing_tensor,
            (-1.0 if max_compute_extra_per_rank else float(compute_imbalance_limit)),
        )
        _record(timing, "quota_solve_ms", quota_started)
        allocation_started = time.perf_counter()
        metrics = _quota_route_cuda(
            source,
            topk,
            count,
            quota,
            replica_mask,
            primary_tensor,
            addition_order,
            num_ranks,
        )
        _record(timing, "quota_allocation_ms", allocation_started)
    else:
        if budget_baseline is None:
            metrics_started = time.perf_counter()
            budget_baseline = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            _record(timing, "communication_replication_ms", metrics_started)
        quota_started = time.perf_counter()
        quota, routing_tensor = _communication_quota_cuda(
            demand,
            replica_mask,
            primary_tensor,
            routing_tensor,
            budget_baseline,
            communication_budget_ratio,
        )
        _record(timing, "quota_solve_ms", quota_started)
        allocation_started = time.perf_counter()
        metrics = _quota_route_cuda(
            source,
            topk,
            count,
            quota,
            replica_mask,
            primary_tensor,
            addition_order,
            num_ranks,
        )
        _record(timing, "quota_allocation_ms", allocation_started)

    replica_mask_np = replica_mask.cpu().numpy()
    demand_np = demand.cpu().numpy()
    routing_np = routing_tensor.cpu().numpy()
    balance_copy_count = int(balance_copies.item()) if balance_copies is not None else 0
    addition_order_np = addition_order.cpu().numpy()
    placement = {
        expert: (int(primary_np[expert]),)
        + tuple(
            int(rank)
            for rank in np.flatnonzero(replica_mask_np[expert])
            if rank != primary_np[expert] and not addition_order_np[expert, rank]
        )
        + tuple(
            int(rank)
            for rank in np.argsort(addition_order_np[expert], kind="stable")
            if addition_order_np[expert, rank]
        )
        for expert in range(num_experts)
    }
    routing = tuple(tuple(int(rank) for rank in row) for row in routing_np)
    serialized_quota = None
    if quota is not None and materialize_quota:
        quota_np = quota.cpu().numpy()
        serialized_quota = tuple(
            tuple(
                (
                    tuple(
                        int(quota_np[source_rank, expert, rank])
                        for rank in placement[expert]
                    )
                    if demand_np[expert, source_rank]
                    else tuple(
                        int(rank == routing_np[source_rank, expert])
                        for rank in placement[expert]
                    )
                )
                for expert in range(num_experts)
            )
            for source_rank in range(num_ranks)
        )
    placement_result = ReplicaPlacement(
        placement,
        routing,
        metrics,
        int(replica_mask_np.sum()) - num_experts - balance_copy_count,
        balance_copies=balance_copy_count,
        quota_by_source=serialized_quota,
        source_demand=demand_np,
    )
    return placement_result
