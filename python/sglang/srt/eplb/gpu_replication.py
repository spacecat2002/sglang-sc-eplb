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


def _phase_start(timing: dict[str, float] | None):
    if timing is None:
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _record(timing: dict[str, float] | None, key: str, started: object) -> None:
    if timing is not None:
        if isinstance(started, torch.cuda.Event):
            finished = torch.cuda.Event(enable_timing=True)
            finished.record()
            finished.synchronize()
            elapsed_ms = started.elapsed_time(finished)
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        timing[key] = timing.get(key, 0.0) + elapsed_ms


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


def _within_communication_budget(
    metrics: ReplicaMetrics, baseline: ReplicaMetrics, ratio: float
) -> bool:
    budgets = tuple(
        math.ceil(value * ratio)
        for value in (
            baseline.remote,
            baseline.max_pair_traffic,
            baseline.max_ingress,
            baseline.max_egress,
        )
    )
    return all(
        value <= limit
        for value, limit in zip(
            (
                metrics.remote,
                metrics.max_pair_traffic,
                metrics.max_ingress,
                metrics.max_egress,
            ),
            budgets,
        )
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
    quota_out: torch.Tensor | None = None,
    routing_out: torch.Tensor | None = None,
    instance_out: torch.Tensor | None = None,
    loads_out: torch.Tensor | None = None,
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
    if (
        quota_out is None
        or routing_out is None
        or instance_out is None
        or loads_out is None
    ):
        return _C.solve_quota(
            demand,
            replica_mask,
            primary,
            routing,
            expert_order,
            source_order,
            compute_imbalance_limit,
        )
    _C.solve_quota_into(
        demand,
        replica_mask,
        primary,
        routing,
        expert_order,
        source_order,
        compute_imbalance_limit,
        quota_out,
        routing_out,
        instance_out,
        loads_out,
    )
    return quota_out, routing_out


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
    workspaces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
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
    for candidate_index in range(4):
        if rank_cost is not None:
            scores = rank_cost * ranks + rank_ids
            scores = torch.where(
                replica_mask[None, :, :],
                scores[None, None, :],
                torch.iinfo(torch.int64).max,
            )
            preferred = scores.argmin(dim=2)
            preferred = torch.where(replica_mask.t(), source_ids, preferred)
        workspace = workspaces[candidate_index] if workspaces is not None else None
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
            quota_out=workspace[0] if workspace else None,
            routing_out=workspace[1] if workspace else None,
            instance_out=workspace[2] if workspace else None,
            loads_out=workspace[3] if workspace else None,
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
    excess = (values - budget_tensor).clamp_min(0)
    key = torch.stack(
        (
            (excess > 0).sum(dim=1),
            excess.sum(dim=1),
            excess.max(dim=1).values,
            excess[:, 0],
            excess[:, 1],
            excess[:, 2],
            excess[:, 3],
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
        self.demand = torch.empty(
            (self.num_experts, self.num_ranks),
            device=self.device,
            dtype=torch.int64,
        )
        self.quota = torch.empty(
            (self.num_ranks, self.num_experts, self.num_ranks),
            device=self.device,
            dtype=torch.int64,
        )
        self.routing = torch.empty(
            (self.num_ranks, self.num_experts),
            device=self.device,
            dtype=torch.int64,
        )
        self.instance = torch.empty(
            (self.num_experts, self.num_ranks),
            device=self.device,
            dtype=torch.int64,
        )
        self.loads = torch.empty(
            (self.num_ranks,), device=self.device, dtype=torch.int64
        )
        self.candidate_workspaces = [
            (
                torch.empty_like(self.quota),
                torch.empty_like(self.routing),
                torch.empty_like(self.instance),
                torch.empty_like(self.loads),
            )
            for _ in range(4)
        ]
        self.replicas = torch.empty(
            (self.num_experts, self.num_ranks), device=self.device, dtype=torch.bool
        )
        self.compute_instance = torch.empty_like(self.replicas, dtype=torch.int64)
        self.compute_loads = torch.empty(
            (self.num_ranks,), device=self.device, dtype=torch.int64
        )
        self.compute_added_by_rank = torch.empty_like(self.compute_loads)
        self.compute_addition_order = torch.empty_like(self.compute_instance)
        self.compute_added = torch.empty((1,), device=self.device, dtype=torch.int64)

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
        _C.fused_source_topn_into(
            source,
            topk,
            count,
            self.primary,
            self.num_experts,
            self.num_ranks,
            int(max_extra_per_rank),
            self.demand,
            self.replicas,
            self.routing,
        )
        demand = self.demand
        return demand, self.replicas, self.routing

    def build_demand(
        self, source: torch.Tensor, topk: torch.Tensor, count: torch.Tensor
    ) -> torch.Tensor:
        source = source.to(
            device=self.device, dtype=torch.int64, copy=False
        ).contiguous()
        topk = topk.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        count = count.to(device=self.device, dtype=torch.int64, copy=False).contiguous()
        _C.source_demand_into(
            source, topk, count, self.num_experts, self.num_ranks, self.demand
        )
        return self.demand

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
    started = _phase_start(timing)
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
    started = _phase_start(timing)
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
        if runtime is None:
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
            _C.fused_source_topn_into(
                source,
                topk,
                count,
                primary_tensor,
                num_experts,
                num_ranks,
                max_extra_per_rank,
                runtime.demand,
                runtime.replicas,
                runtime.routing,
            )
            demand = runtime.demand
            replica_mask, routing_tensor = runtime.replicas, runtime.routing
    else:
        demand = demand_tensor.to(
            device=device, dtype=torch.int64, copy=False
        ).contiguous()
        if demand.shape != (num_experts, num_ranks):
            raise ValueError("demand has the wrong shape")
        if runtime is None:
            replica_mask = _C.select_topn(demand, primary_tensor, max_extra_per_rank)
            routing_tensor = _C.default_routing(replica_mask, primary_tensor)
        else:
            _C.select_topn_routing_into(
                demand,
                primary_tensor,
                max_extra_per_rank,
                runtime.replicas,
                runtime.routing,
            )
            replica_mask, routing_tensor = runtime.replicas, runtime.routing
    _record(timing, "communication_replication_ms", started)

    quota = None
    balance_copies = None
    budget_baseline = None
    if runtime is None:
        addition_order = torch.zeros_like(demand)
    else:
        addition_order = runtime.compute_addition_order
        addition_order.zero_()
    quota_out = runtime.quota if runtime is not None else None
    routing_out = runtime.routing if runtime is not None else None
    instance_out = runtime.instance if runtime is not None else None
    loads_out = runtime.loads if runtime is not None else None
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
            metrics_started = _phase_start(timing)
            budget_baseline = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            _record(timing, "communication_replication_ms", metrics_started)
        compute_started = _phase_start(timing)
        if runtime is None:
            replica_mask, balance_copies, addition_order = _C.select_compute_replicas(
                demand,
                replica_mask,
                expert_demand,
                demand_order,
                max_compute_extra_per_rank,
            )
        else:
            _C.select_compute_replicas_into(
                demand,
                replica_mask,
                expert_demand,
                demand_order,
                max_compute_extra_per_rank,
                runtime.compute_instance,
                runtime.compute_loads,
                runtime.compute_added_by_rank,
                runtime.compute_addition_order,
                runtime.compute_added,
            )
            balance_copies = runtime.compute_added
            addition_order = runtime.compute_addition_order
        _record(timing, "compute_replication_ms", compute_started)
        if runtime is None:
            routing_tensor = _C.default_routing(replica_mask, primary_tensor)
        else:
            _C.default_routing_into(replica_mask, primary_tensor, routing_tensor)
    if not needs_quota:
        metrics_started = _phase_start(timing)
        metrics = _route_cuda(
            source, topk, count, primary_tensor, replica_mask, num_ranks
        )
        _record(timing, "quota_allocation_ms", metrics_started)
    elif communication_budget_ratio is None:
        quota_started = _phase_start(timing)
        quota, routing_tensor = _quota_cuda(
            demand,
            replica_mask,
            primary_tensor,
            routing_tensor,
            (-1.0 if max_compute_extra_per_rank else float(compute_imbalance_limit)),
            expert_demand,
            demand_order,
            source_order,
            quota_out=quota_out,
            routing_out=routing_out,
            instance_out=instance_out,
            loads_out=loads_out,
        )
        _record(timing, "quota_solve_ms", quota_started)
        allocation_started = _phase_start(timing)
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
            metrics_started = _phase_start(timing)
            budget_baseline = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            _record(timing, "communication_replication_ms", metrics_started)
        quota_started = _phase_start(timing)
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
            workspaces=runtime.candidate_workspaces if runtime is not None else None,
        )
        _record(timing, "quota_solve_ms", quota_started)
        allocation_started = _phase_start(timing)
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

    if communication_budget_ratio is not None:
        if budget_baseline is None:
            budget_baseline = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
        if not _within_communication_budget(
            metrics, budget_baseline, communication_budget_ratio
        ):
            safe_metrics = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            if _within_communication_budget(
                safe_metrics, budget_baseline, communication_budget_ratio
            ):
                routing_tensor = torch.where(
                    replica_mask.t(),
                    torch.arange(num_ranks, device=demand.device)[:, None],
                    primary_tensor[None, :],
                )
                quota = torch.zeros(
                    (num_ranks, num_experts, num_ranks),
                    device=demand.device,
                    dtype=demand.dtype,
                )
                quota.scatter_(
                    2,
                    routing_tensor.unsqueeze(-1),
                    demand.t().unsqueeze(-1),
                )
                addition_order.zero_()
                metrics = safe_metrics

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
