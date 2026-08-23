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
    if (
        isinstance(tokens, tuple)
        and len(tokens) == 3
        and all(isinstance(value, torch.Tensor) for value in tokens)
    ):
        return tuple(
            value.to(device=device, dtype=torch.int64, copy=False).contiguous()
            for value in tokens
        )  # type: ignore[return-value]
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
    expert_demand: torch.Tensor | None = None,
    demand_order: torch.Tensor | None = None,
    source_order: torch.Tensor | None = None,
    expert_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_demand is None:
        expert_demand = demand.sum(dim=1)
    if demand_order is None:
        demand_order = torch.argsort(expert_demand, descending=True, stable=True)
    if expert_order is None:
        flexible = replica_mask.sum(dim=1) > 1
        expert_order = demand_order[
            torch.argsort(flexible[demand_order].to(torch.int8), stable=True)
        ]
    if source_order is None:
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


def _bundle_ordinals_cuda(
    source: torch.Tensor, topk: torch.Tensor, count: torch.Tensor, experts: int
) -> torch.Tensor:
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
    return ordinals.view_as(topk)


def _quota_route_cuda(
    source: torch.Tensor,
    topk: torch.Tensor,
    count: torch.Tensor,
    quota: torch.Tensor,
    replica_mask: torch.Tensor,
    primary: torch.Tensor,
    addition_order: torch.Tensor,
    ranks: int,
    ordinals: torch.Tensor | None = None,
) -> ReplicaMetrics:
    experts = replica_mask.shape[0]
    if ordinals is None:
        ordinals = _bundle_ordinals_cuda(source, topk, count, experts)
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
    expert_demand: torch.Tensor | None = None,
    demand_order: torch.Tensor | None = None,
    source_order: torch.Tensor | None = None,
    expert_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = demand.shape[1]
    rank_ids = torch.arange(ranks, device=demand.device, dtype=torch.int64)
    source_ids = rank_ids[:, None]
    if demand_order is None:
        expert_demand = demand.sum(dim=1)
        demand_order = torch.argsort(expert_demand, descending=True, stable=True)
    if expert_order is None:
        flexible = replica_mask.sum(dim=1) > 1
        expert_order = demand_order[
            torch.argsort(flexible[demand_order].to(torch.int8), stable=True)
        ]
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
            demand,
            replica_mask,
            primary,
            preferred,
            compute_imbalance_limit=-1.0,
            expert_demand=expert_demand,
            demand_order=demand_order,
            source_order=source_order,
            expert_order=expert_order,
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

    summary = torch.stack(summaries)
    budget_tensor = torch.as_tensor(
        budgets, device=demand.device, dtype=torch.float64
    )
    values = summary[:, :4].to(torch.float64)
    violation = torch.where(budget_tensor > 0, values / budget_tensor, values)
    key = torch.stack(
        (
            violation.max(dim=1).values,
            summary[:, 4].to(torch.float64),
            summary[:, 5].to(torch.float64),
            summary[:, 0].to(torch.float64),
            summary[:, 1].to(torch.float64),
        ),
        dim=1,
    )
    remaining = torch.arange(4, device=demand.device)
    for column in range(key.shape[1]):
        column_values = key[remaining, column]
        remaining = remaining[column_values == column_values.min()]
        if remaining.numel() == 1:
            break
    return candidates[int(remaining[0].item())]


class GraceCudaRuntime:
    """GPU-resident state for repeated plans with fixed EP geometry."""

    def __init__(
        self, num_experts: int, num_ranks: int, device: str | torch.device = "cuda"
    ) -> None:
        self.num_experts = int(num_experts)
        self.num_ranks = int(num_ranks)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA runtime requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime requested but torch.cuda is unavailable")
        self.primary: torch.Tensor | None = None

    def _primary_tensor(self, primary: Mapping[int, int] | torch.Tensor) -> torch.Tensor:
        if isinstance(primary, torch.Tensor):
            value = primary.to(device=self.device, dtype=torch.int64, copy=False)
            if value.numel() != self.num_experts:
                raise ValueError("primary has the wrong number of experts")
            return value.contiguous()
        return torch.as_tensor(
            [primary[e] for e in range(self.num_experts)],
            device=self.device,
            dtype=torch.int64,
        )

    def source_topn(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        primary: Mapping[int, int] | torch.Tensor,
        max_extra_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build demand, replicas, and default routing without CPU round-trips."""
        source = source.to(
            device=self.device, dtype=torch.int64, copy=False
        ).contiguous()
        topk = topk.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        count = count.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        self.primary = self._primary_tensor(primary)
        return _C.fused_source_topn(
            source,
            topk,
            count,
            self.primary,
            self.num_experts,
            self.num_ranks,
            int(max_extra_per_rank),
        )

    def build_demand(
        self, source: torch.Tensor, topk: torch.Tensor, count: torch.Tensor
    ) -> torch.Tensor:
        source = source.to(
            device=self.device, dtype=torch.int64, copy=False
        ).contiguous()
        topk = topk.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        count = count.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        return _source_demand_cuda(
            source, topk, count, self.num_experts, self.num_ranks
        )

    def plan(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        primary: Mapping[int, int] | torch.Tensor,
        demand: torch.Tensor | None = None,
        **kwargs,
    ) -> ReplicaPlacement:
        """Run the full planner from GPU tensors; CPU copies happen only at return."""
        return replicate_source_top_experts_cuda(
            (source, topk, count),
            primary,
            num_ranks=self.num_ranks,
            device=self.device,
            runtime=self,
            demand_tensor=demand,
            **kwargs,
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
    max_compute_extra_per_rank: int = 0,
    compute_imbalance_limit: float | None = None,
    communication_budget_ratio: float | None = None,
    device: str | torch.device = "cuda",
    timing: dict[str, float] | None = None,
    materialize_quota: bool = True,
    runtime: GraceCudaRuntime | None = None,
    demand_tensor: torch.Tensor | None = None,
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
    num_experts = runtime.num_experts if runtime is not None else len(primary)
    if runtime is not None and runtime.num_ranks != num_ranks:
        raise ValueError("runtime rank count does not match num_ranks")
    if runtime is not None:
        primary_tensor = runtime._primary_tensor(primary)
        primary_np = None
    elif isinstance(primary, torch.Tensor):
        primary_tensor = primary.to(
            device=device, dtype=torch.int64, copy=False
        ).contiguous()
        primary_np = None
    else:
        primary_np = np.asarray(
            [primary[e] for e in range(num_experts)], dtype=np.int64
        )
        primary_tensor = torch.as_tensor(primary_np, device=device)
    if demand_tensor is None:
        demand, replica_mask, routing_tensor = _C.fused_source_topn(
            source,
            topk,
            count,
            primary_tensor,
            num_experts,
            num_ranks,
            max_extra_per_rank,
        )
    else:
        demand = demand_tensor.to(
            device=device, dtype=torch.int64, copy=False
        ).contiguous()
        if demand.shape != (num_experts, num_ranks):
            raise ValueError("demand has the wrong shape")
        replica_mask = _C.select_topn(demand, primary_tensor, max_extra_per_rank)
        routing_tensor = _C.default_routing(replica_mask, primary_tensor)
    _record(timing, "communication_replication_ms", started)

    quota = None
    balance_copies = None
    budget_baseline = None
    addition_order = torch.zeros_like(demand)
    needs_quota = compute_imbalance_limit is not None or max_compute_extra_per_rank > 0
    if needs_quota:
        expert_demand = demand.sum(dim=1)
        demand_order = torch.argsort(expert_demand, descending=True, stable=True)
        source_order = torch.argsort(demand.t(), dim=1, descending=True, stable=True)
    bundle_ordinals = (
        _bundle_ordinals_cuda(source, topk, count, num_experts) if needs_quota else None
    )
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
                expert_demand,
                demand_order,
                source_order,
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
                bundle_ordinals,
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
                expert_demand,
                demand_order,
                source_order,
            )
            _record(timing, "quota_solve_ms", quota_started)
        compute_started = time.perf_counter()
        replica_mask, balance_copies, addition_order = _C.select_compute_replicas(
            demand,
            replica_mask,
            expert_demand,
            demand_order,
            max_compute_extra_per_rank,
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
                expert_demand=expert_demand,
                demand_order=demand_order,
                source_order=source_order,
            )
            _record(timing, "quota_solve_ms", quota_started)
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
            expert_demand,
            demand_order,
            source_order,
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
            bundle_ordinals,
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
            expert_demand,
            demand_order,
            source_order,
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
            bundle_ordinals,
        )
        _record(timing, "quota_allocation_ms", allocation_started)

    replica_mask_np = replica_mask.cpu().numpy()
    demand_np = demand.cpu().numpy()
    routing_np = routing_tensor.cpu().numpy()
    if primary_np is None:
        primary_np = primary_tensor.cpu().numpy()
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
