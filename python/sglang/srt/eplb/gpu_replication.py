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


_PENDING_CUDA_TIMING = "_pending_cuda_timing"


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
            timing.setdefault(_PENDING_CUDA_TIMING, []).append(
                (key, started, finished)
            )
            return
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        timing[key] = timing.get(key, 0.0) + elapsed_ms


def _flush_cuda_timing(timing: dict[str, float] | None) -> None:
    if not timing:
        return
    pending = timing.pop(_PENDING_CUDA_TIMING, ())
    if not pending:
        return
    pending[-1][2].synchronize()
    for key, started, finished in pending:
        timing[key] = timing.get(key, 0.0) + started.elapsed_time(finished)


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


def _within_compute_limit(metrics: ReplicaMetrics, limit: float | None) -> bool:
    return limit is None or metrics.compute_imbalance <= limit + 1e-12


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
    sorted_keys, order = torch.sort(keys, stable=True)
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
    compute_imbalance_limit: float,
    expert_demand: torch.Tensor | None = None,
    demand_order: torch.Tensor | None = None,
    source_order: torch.Tensor | None = None,
    expert_order: torch.Tensor | None = None,
    workspaces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = demand.shape[1]
    if budget_ratio == 1.0:
        # Compute is the primary objective, so the first balanced solve is the
        # only candidate needed; exact traffic is checked after route eval.
        workspace = workspaces[0] if workspaces is not None else None
        return _quota_cuda(
            demand,
            replica_mask,
            primary,
            routing,
            compute_imbalance_limit,
            expert_demand=expert_demand,
            demand_order=demand_order,
            source_order=source_order,
            expert_order=expert_order,
            quota_out=workspace[0] if workspace else None,
            routing_out=workspace[1] if workspace else None,
            instance_out=workspace[2] if workspace else None,
            loads_out=workspace[3] if workspace else None,
        )
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
            compute_imbalance_limit=compute_imbalance_limit,
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
            summary[:, 4].to(torch.float64),
            summary[:, 5].to(torch.float64),
            (excess > 0).sum(dim=1),
            excess.sum(dim=1),
            excess.max(dim=1).values,
            excess[:, 0],
            excess[:, 1],
            excess[:, 2],
            excess[:, 3],
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
        self,
        num_experts: int,
        num_ranks: int,
        device: str | torch.device = "cuda",
        solver_sms: int = 1,
    ) -> None:
        self.num_experts = int(num_experts)
        self.num_ranks = int(num_ranks)
        self.solver_sms = int(solver_sms)
        if self.solver_sms < 1:
            raise ValueError("solver_sms must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA runtime requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime requested but torch.cuda is unavailable")
        self.primary = torch.empty(
            (self.num_experts,), device=self.device, dtype=torch.int64
        )
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
        self.replica_gains = torch.empty_like(self.replicas, dtype=torch.int64)
        rank_group_workspace = (
            self.num_ranks * (self.num_ranks + 1) * (self.num_experts + 1)
        )
        self.rank_group_values = torch.empty(
            rank_group_workspace, device=self.device, dtype=torch.int64
        )
        self.rank_group_choices = torch.empty_like(self.rank_group_values)
        self.compute_instance = torch.empty_like(self.replicas, dtype=torch.int64)
        self.compute_loads = torch.empty(
            (self.num_ranks,), device=self.device, dtype=torch.int64
        )
        self.compute_added_by_rank = torch.empty_like(self.compute_loads)
        self.compute_addition_order = torch.empty_like(self.compute_instance)
        self.compute_added = torch.empty((1,), device=self.device, dtype=torch.int64)
        self.affinity = torch.empty(
            (self.num_experts, self.num_experts),
            device=self.device,
            dtype=torch.int64,
        )
        self.affinity_degree = torch.empty_like(self.primary)
        self.affinity_score = torch.empty_like(self.primary)
        self.affinity_groups = torch.empty_like(self.primary)
        self.affinity_group_source = torch.empty(
            (self.num_ranks, self.num_ranks),
            device=self.device,
            dtype=torch.int64,
        )
        self.affinity_group_to_rank = torch.empty(
            (self.num_ranks,), device=self.device, dtype=torch.int64
        )
        self.affinity_eigenvalues = torch.empty_like(
            self.primary, dtype=torch.float64
        )
        self.affinity_dimensions = min(self.num_ranks, 4)
        self.affinity_embedding = torch.empty(
            (self.num_experts, self.affinity_dimensions),
            device=self.device,
            dtype=torch.float64,
        )
        self._use_affinity_subspace = self.num_experts <= 256
        if self._use_affinity_subspace:
            generator = torch.Generator(device=self.device).manual_seed(0)
            self.affinity_initial = torch.linalg.qr(
                torch.randn(
                    self.affinity_embedding.shape,
                    generator=generator,
                    device=self.device,
                    dtype=torch.float64,
                ),
                mode="reduced",
            ).Q.contiguous()
        else:
            self.affinity_float = torch.empty_like(
                self.affinity, dtype=torch.float64
            )
            self.affinity_scale = torch.empty_like(
                self.primary, dtype=torch.float64
            )
            self.affinity_eigenvectors = torch.empty_like(self.affinity_float)
        self.affinity_centers = torch.empty(
            (self.num_ranks, self.affinity_dimensions),
            device=self.device,
            dtype=torch.float64,
        )
        self.affinity_next_groups = torch.empty_like(self.primary)
        self.affinity_group_sizes = torch.empty_like(self.affinity_group_to_rank)
        self.affinity_overflow = torch.empty_like(self.primary)
        self.affinity_group_affinity = torch.empty(
            (self.num_experts, self.num_ranks),
            device=self.device,
            dtype=torch.int64,
        )
        self.affinity_swap_gains = torch.empty_like(self.affinity)
        self.affinity_allowed = torch.empty(
            (self.num_ranks, self.num_ranks), device=self.device, dtype=torch.bool
        )
        self.affinity_values = torch.empty_like(self.affinity_group_source)
        self.affinity_cost = torch.empty_like(self.affinity_group_source)
        self.affinity_hungarian_work = torch.empty(
            (6 * (self.num_ranks + 1),), device=self.device, dtype=torch.int64
        )

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
            self.replica_gains,
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

    def affinity_primary(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        timing: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Build a capacity-strict affinity placement without leaving CUDA."""
        if self.num_experts % self.num_ranks:
            raise ValueError("strict affinity placement requires experts divisible by ranks")
        source, topk, count = _cuda_arrays((source, topk, count), self.device)
        phase_started = _phase_start(timing)
        _C.affinity_histogram_into(
            source,
            topk,
            count,
            self.demand,
            self.affinity,
            self.affinity_degree,
            self.solver_sms,
        )
        _record(timing, "affinity_histogram_ms", phase_started)
        phase_started = _phase_start(timing)
        if self._use_affinity_subspace:
            _C.affinity_subspace_into(
                self.affinity,
                self.affinity_degree,
                self.affinity_initial,
                self.affinity_embedding,
                2,
            )
        else:
            _C.normalize_affinity_into(
                self.affinity,
                self.affinity_degree,
                self.affinity_scale,
                self.affinity_float,
            )
            torch.linalg.eigh(
                self.affinity_float,
                out=(self.affinity_eigenvalues, self.affinity_eigenvectors),
            )
            self.affinity_embedding.copy_(
                self.affinity_eigenvectors[:, -self.affinity_dimensions :]
            )
            torch.nn.functional.normalize(
                self.affinity_embedding,
                dim=1,
                eps=1e-12,
                out=self.affinity_embedding,
            )
        _record(timing, "affinity_embedding_ms", phase_started)
        spectral_started = _phase_start(timing)
        _C.spectral_groups_into(
            self.affinity_embedding,
            self.affinity,
            self.affinity_centers,
            self.affinity_eigenvalues,
            self.affinity_groups,
            self.affinity_next_groups,
            self.affinity_group_sizes,
            self.affinity_overflow,
            self.affinity_group_affinity,
            self.affinity_swap_gains,
        )
        _record(timing, "affinity_partition_ms", spectral_started)
        balance_started = _phase_start(timing)
        _C.balance_affinity_groups_into(
            self.demand,
            self.affinity,
            self.affinity_groups,
            self.affinity_overflow,
            self.affinity_group_sizes,
            self.affinity_group_affinity,
        )
        _record(timing, "affinity_balance_ms", balance_started)
        phase_started = _phase_start(timing)
        _C.group_source_into(
            source,
            topk,
            count,
            self.affinity_groups,
            self.affinity_group_source,
            self.solver_sms,
        )
        _C.congestion_hungarian_into(
            self.affinity_group_source,
            self.affinity_groups,
            self.affinity_allowed,
            self.affinity_values,
            self.affinity_cost,
            self.affinity_hungarian_work,
            self.affinity_group_to_rank,
            self.primary,
        )
        _record(timing, "affinity_hungarian_ms", phase_started)
        return self.primary

    def plan_unified(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        primary: Mapping[int, int] | torch.Tensor,
        demand: torch.Tensor | None = None,
        **kwargs,
    ) -> ReplicaPlacement:
        """Run the exact capacity-v2 solver through the unified entry point."""
        return replicate_source_top_experts_cuda(
            (source, topk, count),
            primary,
            num_ranks=self.num_ranks,
            device=self.device,
            runtime=self,
            demand_tensor=demand,
            compute_solver="unified",
            **kwargs,
        )

    def plan_grouped(
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

    def plan_pure(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        primary: Mapping[int, int] | torch.Tensor,
        demand: torch.Tensor | None = None,
        **kwargs,
    ) -> ReplicaPlacement:
        """Run pure Top-N replication without invoking affinity grouping."""
        return replicate_source_top_experts_cuda(
            (source, topk, count),
            primary,
            num_ranks=self.num_ranks,
            device=self.device,
            runtime=self,
            demand_tensor=demand,
            pure=True,
            **kwargs,
        )

    def plan_rank_group(
        self,
        source: torch.Tensor,
        topk: torch.Tensor,
        count: torch.Tensor,
        primary: Mapping[int, int] | torch.Tensor,
        demand: torch.Tensor | None = None,
        **kwargs,
    ) -> ReplicaPlacement:
        """Choose the better complete plan from rank-group and pure Top-N."""
        return replicate_rank_group_experts_cuda(
            (source, topk, count),
            primary,
            num_ranks=self.num_ranks,
            device=self.device,
            runtime=self,
            demand_tensor=demand,
            **kwargs,
        )

    plan = plan_grouped


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
    _flush_cuda_timing(timing)
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
    compute_solver: str | None = None,
    pure: bool = False,
    rank_group_replication: bool = False,
) -> ReplicaPlacement:
    """Run grouped or pure Top-N replication with CUDA trace processing.

    Replica gain counts only traffic whose remote destination disappears. With
    a resident runtime, compute replicas and their export quota are produced
    together by the CUDA capacity solver and consumed directly by quota routing.
    """

    if max_extra_per_rank < 0:
        raise ValueError("max_extra_per_rank must be non-negative")
    if max_compute_extra_per_rank < 0:
        raise ValueError("max_compute_extra_per_rank must be non-negative")
    if not pure and compute_solver not in {None, "legacy", "capacity-v2", "unified"}:
        raise ValueError("compute_solver must be 'legacy', 'capacity-v2', or 'unified'")
    if compute_solver == "unified":
        # Exact public alias: one implementation keeps output semantics identical.
        compute_solver = "capacity-v2"
    if rank_group_replication and not pure:
        raise ValueError("rank-group replication requires the pure planner")
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
        primary_np = (
            np.asarray([primary[e] for e in range(num_experts)], dtype=np.int64)
            if not isinstance(primary, torch.Tensor) and not materialize_quota
            else None
        )
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
            if not pure:
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
                demand = _C.source_demand(
                    source, topk, count, num_experts, num_ranks
                )
        elif pure:
            _C.source_demand_into(
                source, topk, count, num_experts, num_ranks, runtime.demand
            )
            demand = runtime.demand
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
                runtime.replica_gains,
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
        if not pure and runtime is None:
            gains = torch.empty_like(demand)
            replica_mask = torch.empty_like(demand, dtype=torch.bool)
            routing_tensor = torch.empty(
                (num_ranks, num_experts), device=device, dtype=torch.int64
            )
            _C.select_bundle_topn_routing_into(
                source,
                topk,
                count,
                primary_tensor,
                max_extra_per_rank,
                gains,
                replica_mask,
                routing_tensor,
                1,
            )
        elif not pure:
            _C.select_bundle_topn_routing_into(
                source,
                topk,
                count,
                primary_tensor,
                max_extra_per_rank,
                runtime.replica_gains,
                runtime.replicas,
                runtime.routing,
                runtime.solver_sms,
            )
            replica_mask, routing_tensor = runtime.replicas, runtime.routing
    if pure and rank_group_replication:
        workspace_size = num_ranks * (num_ranks + 1) * (num_experts + 1)
        if runtime is None:
            ordinals = torch.empty_like(demand)
            group_experts = torch.empty_like(demand)
            gains = torch.empty_like(demand)
            values = torch.empty(workspace_size, device=device, dtype=torch.int64)
            choices = torch.empty_like(values)
            replica_mask = torch.empty_like(demand, dtype=torch.bool)
            routing_tensor = torch.empty(
                (num_ranks, num_experts), device=device, dtype=torch.int64
            )
        else:
            ordinals = runtime.compute_instance
            group_experts = runtime.compute_addition_order
            gains = runtime.replica_gains
            values = runtime.rank_group_values
            choices = runtime.rank_group_choices
            replica_mask = runtime.replicas
            routing_tensor = runtime.routing
        _C.select_rank_group_topn_routing_into(
            source,
            topk,
            count,
            demand,
            primary_tensor,
            max_extra_per_rank,
            ordinals,
            group_experts,
            gains,
            values,
            choices,
            replica_mask,
            routing_tensor,
        )
    elif pure and runtime is None:
        replica_mask = _C.select_topn(demand, primary_tensor, max_extra_per_rank)
        routing_tensor = _C.default_routing(replica_mask, primary_tensor)
    elif pure:
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
    quota_compute_limit = (
        float(compute_imbalance_limit)
        if compute_imbalance_limit is not None
        else -1.0
    )
    fast_capacity = (
        runtime is not None
        and num_ranks > 16
        and not pure
        and compute_solver == "capacity-v2"
    )
    if runtime is None:
        addition_order = torch.zeros_like(demand)
    else:
        addition_order = runtime.compute_addition_order
        if not (
            max_compute_extra_per_rank
            and not pure
            and compute_solver == "capacity-v2"
        ):
            addition_order.zero_()
    quota_out = runtime.quota if runtime is not None else None
    routing_out = runtime.routing if runtime is not None else None
    instance_out = runtime.instance if runtime is not None else None
    loads_out = runtime.loads if runtime is not None else None
    needs_quota = compute_imbalance_limit is not None or max_compute_extra_per_rank > 0
    direct_export_quota = (
        runtime is not None
        and max_compute_extra_per_rank > 0
        and communication_budget_ratio in (None, 1.0)
    )
    if needs_quota and not direct_export_quota:
        expert_demand = demand.sum(dim=1)
        demand_order = torch.argsort(expert_demand, descending=True, stable=True)
        source_order = torch.argsort(demand.t(), dim=1, descending=True, stable=True)
        flexible = replica_mask.sum(dim=1) > 1
        expert_order = demand_order[
            torch.argsort(flexible[demand_order].to(torch.int8), stable=True)
        ]
    else:
        expert_demand = demand_order = source_order = expert_order = None
    bundle_ordinals = None
    if max_compute_extra_per_rank:
        if communication_budget_ratio is not None and not direct_export_quota:
            metrics_started = _phase_start(timing)
            budget_baseline = _route_cuda(
                source, topk, count, primary_tensor, replica_mask, num_ranks
            )
            _record(timing, "communication_replication_ms", metrics_started)
        compute_started = _phase_start(timing)
        if fast_capacity:
            _C.current_bundle_gains_fast_into(
                source,
                topk,
                count,
                primary_tensor,
                replica_mask,
                runtime.replica_gains,
                runtime.solver_sms,
            )
            _C.select_compute_replicas_fast_into(
                demand,
                runtime.replica_gains,
                replica_mask,
                primary_tensor,
                max_compute_extra_per_rank,
                quota_compute_limit,
                runtime.compute_instance,
                runtime.compute_loads,
                runtime.compute_added_by_rank,
                runtime.compute_addition_order,
                runtime.candidate_workspaces[0][2],
                runtime.quota,
                runtime.routing,
                runtime.compute_added,
            )
            balance_copies = runtime.compute_added
            addition_order = runtime.compute_addition_order
        elif not pure and compute_solver == "capacity-v2":
            if runtime is None:
                raise ValueError("capacity-v2 requires a resident CUDA runtime")
            _C.current_bundle_gains_into(
                source,
                topk,
                count,
                primary_tensor,
                replica_mask,
                runtime.replica_gains,
                runtime.candidate_workspaces[0][0],
                runtime.solver_sms,
            )
            _C.select_compute_replicas_v2_into(
                demand,
                runtime.replica_gains,
                runtime.candidate_workspaces[0][0],
                replica_mask,
                primary_tensor,
                max_compute_extra_per_rank,
                quota_compute_limit,
                runtime.compute_instance,
                runtime.compute_loads,
                runtime.compute_added_by_rank,
                runtime.compute_addition_order,
                runtime.quota,
                runtime.routing,
                runtime.compute_added,
            )
            balance_copies = runtime.compute_added
            addition_order = runtime.compute_addition_order
        elif runtime is None:
            replica_mask, balance_copies, addition_order = _C.select_compute_replicas(
                demand,
                replica_mask,
                primary_tensor,
                max_compute_extra_per_rank,
            )
        elif pure:
            _C.select_pure_compute_replicas_into(
                demand,
                replica_mask,
                primary_tensor,
                max_compute_extra_per_rank,
                runtime.compute_instance,
                runtime.compute_loads,
                runtime.compute_added_by_rank,
                runtime.compute_addition_order,
                runtime.quota,
                runtime.routing,
                runtime.compute_added,
            )
            balance_copies = runtime.compute_added
            addition_order = runtime.compute_addition_order
        else:
            _C.select_compute_replicas_into(
                demand,
                replica_mask,
                primary_tensor,
                max_compute_extra_per_rank,
                runtime.compute_instance,
                runtime.compute_loads,
                runtime.compute_added_by_rank,
                runtime.compute_addition_order,
                runtime.quota,
                runtime.routing,
                runtime.compute_added,
            )
            balance_copies = runtime.compute_added
            addition_order = runtime.compute_addition_order
        _record(timing, "compute_replication_ms", compute_started)
        if runtime is None:
            routing_tensor = _C.default_routing(replica_mask, primary_tensor)
        if not direct_export_quota:
            # Compute replicas change which experts are flexible.
            flexible = replica_mask.sum(dim=1) > 1
            expert_order = demand_order[
                torch.argsort(flexible[demand_order].to(torch.int8), stable=True)
            ]
    if not needs_quota:
        metrics = _route_cuda(
            source, topk, count, primary_tensor, replica_mask, num_ranks
        )
    elif direct_export_quota:
        quota_started = _phase_start(timing)
        quota = runtime.quota
        _record(timing, "quota_solve_ms", quota_started)
        bundle_ordinals = _bundle_ordinals_cuda(source, topk, count, num_experts)
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
    elif communication_budget_ratio is None:
        quota_started = _phase_start(timing)
        quota, routing_tensor = _quota_cuda(
            demand,
            replica_mask,
            primary_tensor,
            routing_tensor,
            quota_compute_limit,
            expert_demand,
            demand_order,
            source_order,
            expert_order=expert_order,
            quota_out=quota_out,
            routing_out=routing_out,
            instance_out=instance_out,
            loads_out=loads_out,
        )
        _record(timing, "quota_solve_ms", quota_started)
        bundle_ordinals = _bundle_ordinals_cuda(source, topk, count, num_experts)
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
            quota_compute_limit,
            expert_demand,
            demand_order,
            source_order,
            expert_order=expert_order,
            workspaces=runtime.candidate_workspaces if runtime is not None else None,
        )
        _record(timing, "quota_solve_ms", quota_started)
        bundle_ordinals = _bundle_ordinals_cuda(source, topk, count, num_experts)
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

    # Preserve the hard compute contract if v2 exhausts its replica slots.
    if (
        compute_solver == "capacity-v2"
        and compute_imbalance_limit is not None
        and not _within_compute_limit(metrics, compute_imbalance_limit)
    ):
        return replicate_source_top_experts_cuda(
            (source, topk, count),
            primary_tensor,
            num_ranks=num_ranks,
            max_extra_per_rank=max_extra_per_rank,
            max_compute_extra_per_rank=max_compute_extra_per_rank,
            compute_imbalance_limit=compute_imbalance_limit,
            communication_budget_ratio=communication_budget_ratio,
            device=device,
            timing=timing,
            materialize_quota=materialize_quota,
            runtime=runtime,
            demand_tensor=demand,
            compute_solver="legacy",
        )

    if communication_budget_ratio is not None and not _within_compute_limit(
        metrics, compute_imbalance_limit
    ):
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
    demand_np = demand.cpu().numpy() if materialize_quota else None
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
    _flush_cuda_timing(timing)
    return placement_result


def replicate_rank_group_experts_cuda(
    tokens: Sequence[RoutedToken] | RoutedArrays,
    primary: Mapping[int, int] | torch.Tensor,
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
    """Keep rank-group replication only when its final balanced plan wins."""
    kwargs = dict(
        num_ranks=num_ranks,
        max_extra_per_rank=max_extra_per_rank,
        max_compute_extra_per_rank=max_compute_extra_per_rank,
        compute_imbalance_limit=compute_imbalance_limit,
        communication_budget_ratio=communication_budget_ratio,
        device=device,
        timing=timing,
        materialize_quota=materialize_quota,
        runtime=runtime,
        demand_tensor=demand_tensor,
        pure=True,
    )
    topn = replicate_source_top_experts_cuda(tokens, primary, **kwargs)
    grouped = replicate_source_top_experts_cuda(
        tokens, primary, rank_group_replication=True, **kwargs
    )
    feasible = [
        result
        for result in (topn, grouped)
        if _within_compute_limit(result.metrics, compute_imbalance_limit)
    ]
    return min(
        feasible or [topn],
        key=lambda result: (
            result.metrics.remote,
            result.metrics.max_pair_traffic,
            result.metrics.max_ingress,
            result.metrics.max_egress,
            result.extra_copies,
        ),
    )
