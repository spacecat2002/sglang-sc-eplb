#!/usr/bin/env python3
"""Solve an offline Top-K bundle trace with a co-routing graph.

Input can be a compact ``.pt`` trace or JSON containing either a list of
bundle records or an object with a ``layers`` list. A record has this shape::

    {"source_rank": 0, "topk_experts": [3, 7, 9], "count": 128}

For the layered form, each layer may additionally contain ``gate``.

Example::

    python benchmark/solve_co_routing_graph.py \
        --input bundles.json --num-ranks 8 --slots-per-rank 32 \
        --device cuda --planner cuda-fast

Cross-dataset transfer replay::

    # Solve on dataset A.
    python benchmark/solve_co_routing_graph.py \
        --input dataset_a.pt --num-ranks 8 --placement-mode hybrid \
        --device cuda --planner cuda-fast \
        --save-placement dataset_a_placement.json

    # Solve dataset B once as the in-domain reference.
    python benchmark/solve_co_routing_graph.py \
        --input dataset_b.pt --num-ranks 8 --placement-mode hybrid \
        --device cuda --planner cuda-fast \
        --save-placement dataset_b_oracle.json

    # Freeze A's placement while replaying dataset B, without solving again.
    python benchmark/solve_co_routing_graph.py \
        --input dataset_b.pt --num-ranks 8 --device cuda \
        --load-placement dataset_a_placement.json \
        --oracle-placement dataset_b_oracle.json --replay-only
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import orjson

from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    RoutedToken,
)
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraph,
    CoRoutingGraphSolver,
    GraphPlacement,
    build_co_routing_graph,
    evaluate_primary_remote,
    refine_hypergraph_placement,
)
from sglang.srt.eplb.moe_bundle_trace import (
    compact_layer_from_records,
    load_compact_trace,
    save_compact_trace,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _raw_layers(raw: Any) -> list[tuple[str, list[Any]]]:
    if isinstance(raw, list):
        layers = [("layer0", raw)]
    elif isinstance(raw, dict) and isinstance(raw.get("layers"), list):
        layers = []
        for index, layer in enumerate(raw["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"layers[{index}] must be an object")
            layers.append(
                (str(layer.get("gate", f"layer{index}")), layer.get("bundles", []))
            )
    else:
        raise ValueError("input must be a bundle list or an object containing layers")

    result = []
    for name, entries in layers:
        if not isinstance(entries, list):
            raise ValueError(f"{name}.bundles must be a list")
        if entries:
            result.append((name, entries))
    if not result:
        raise ValueError("input contains no non-empty layers")
    return result


def _tokens_from_entries(name: str, entries: Sequence[Any]) -> list[RoutedToken]:
    tokens = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}.bundles[{index}] must be an object")
        try:
            source = int(entry["source_rank"])
            experts = tuple(sorted(int(expert) for expert in entry["topk_experts"]))
            count = int(entry.get("count", 1))
            tokens.append(RoutedToken(source, experts, count))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle in {name}: {entry!r}") from exc
    return tokens


@dataclass(frozen=True)
class _TensorTrace:
    source_rank: Any
    topk_experts: Any
    count: Any
    ready_event: Any = None
    host_buffers: tuple[Any, ...] = ()


class _PinnedTracePool:
    """Reuse two pinned host slots so H2D prefetch avoids allocator churn."""

    def __init__(self, layers: Sequence[tuple[str, Any]]) -> None:
        torch = _torch()
        max_bundles = 0
        topk_size = 0
        for _, layer in layers:
            source = layer["source_rank"]
            topk = layer["topk_experts"]
            count = layer["count"]
            max_bundles = max(max_bundles, int(source.shape[0]), int(count.shape[0]))
            topk_size = max(topk_size, int(topk.shape[1]))
        # Two slots are enough for one layer being consumed and one being
        # copied.  Use int64 once on the host so the CUDA path has no cast
        # kernels after the asynchronous copy.
        self._slots = [
            (
                torch.empty(max_bundles, dtype=torch.int64, pin_memory=True),
                torch.empty(
                    (max_bundles, topk_size), dtype=torch.int64, pin_memory=True
                ),
                torch.empty(max_bundles, dtype=torch.int64, pin_memory=True),
            )
            for _ in range(2)
        ]

    def stage(self, index: int, layer: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        source = layer["source_rank"]
        topk = layer["topk_experts"]
        count = layer["count"]
        slot = self._slots[index % len(self._slots)]
        source_dst = slot[0][: source.shape[0]]
        topk_dst = slot[1][: topk.shape[0], : topk.shape[1]]
        count_dst = slot[2][: count.shape[0]]
        source_dst.copy_(source)
        topk_dst.copy_(topk)
        count_dst.copy_(count)
        return source_dst, topk_dst, count_dst


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--device cuda requires PyTorch") from exc
    return torch


def _cuda_fast_planner_ops():
    try:
        from sglang.srt.eplb.cuda_fast_co_routing_planner import (
            build_co_routing_graph_cuda,
            plan_communication_replicas_cuda,
            refine_hypergraph_placement_cuda,
            solve_co_routing_graph_cuda,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--planner cuda-fast requires PyTorch, CUDA, and Triton"
        ) from exc
    return (
        build_co_routing_graph_cuda,
        solve_co_routing_graph_cuda,
        plan_communication_replicas_cuda,
        refine_hypergraph_placement_cuda,
    )


def _reshard_tensor_trace(
    source_tensor: Any,
    topk_tensor: Any,
    count_tensor: Any,
    source_ep: int,
    target_ep: int,
) -> _TensorTrace:
    torch = _torch()
    if source_ep == target_ep:
        return _TensorTrace(source_tensor, topk_tensor, count_tensor)
    if target_ep < source_ep or target_ep % source_ep:
        raise ValueError("target EP must be a multiple of source EP")
    fanout = target_ep // source_ep
    offsets = torch.arange(fanout, device=source_tensor.device)
    split_counts = count_tensor[:, None] // fanout + (
        offsets[None, :] < (count_tensor % fanout)[:, None]
    )
    keep = split_counts > 0
    return _TensorTrace(
        (source_tensor[:, None] * fanout + offsets[None, :]).expand_as(split_counts)[
            keep
        ],
        topk_tensor[:, None, :]
        .expand(-1, fanout, -1)[keep]
        .reshape(-1, topk_tensor.shape[1]),
        split_counts[keep],
    )


def _enqueue_cuda_tensors(
    source_cpu: Any,
    topk_cpu: Any,
    count_cpu: Any,
    *,
    device: str,
    source_ep: int,
    target_ep: int,
    copy_stream: Any = None,
) -> _TensorTrace:
    torch = _torch()
    source_host = source_cpu if source_cpu.is_pinned() else source_cpu.pin_memory()
    topk_host = topk_cpu if topk_cpu.is_pinned() else topk_cpu.pin_memory()
    count_host = count_cpu if count_cpu.is_pinned() else count_cpu.pin_memory()
    stream = copy_stream or torch.cuda.current_stream(device)
    with torch.cuda.device(torch.device(device)):
        with torch.cuda.stream(stream):
            source = source_host.to(device, non_blocking=True)
            topk = topk_host.to(device, non_blocking=True)
            count = count_host.to(device, non_blocking=True)
            if source.dtype != torch.int64:
                source = source.to(torch.int64)
            if topk.dtype != torch.int64:
                topk = topk.to(torch.int64)
            if count.dtype != torch.int64:
                count = count.to(torch.int64)
            # Compact traces are sorted at save time, so no Top-K sort is
            # needed on the hot path.
            trace = _reshard_tensor_trace(source, topk, count, source_ep, target_ep)
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
    return _TensorTrace(
        trace.source_rank,
        trace.topk_experts,
        trace.count,
        ready_event,
        (source_host, topk_host, count_host),
    )


def _wait_tensor_trace(trace: _TensorTrace) -> None:
    if trace.ready_event is not None:
        trace.ready_event.synchronize()


def _load_input(path: str) -> tuple[Any, list[tuple[str, Any]], bool]:
    input_path = Path(path)
    if input_path.suffix.lower() in {".pt", ".pth"}:
        compact = load_compact_trace(input_path)
        raw = {
            "num_ranks": compact["num_ranks"],
            "top_k": compact["top_k"],
            "format": compact["format"],
        }
        layers = [
            (str(layer.get("gate", f"layer{index}")), layer)
            for index, layer in enumerate(compact["layers"])
        ]
        return raw, layers, True
    raw = orjson.loads(input_path.read_bytes())
    return raw, _raw_layers(raw), False


def _load_placements(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    raw = orjson.loads(Path(path).read_bytes())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"placement file must contain a non-empty object: {path}")
    if isinstance(raw.get("layers"), list):
        converted = {}
        for index, layer in enumerate(raw["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"placement {path} has invalid layers[{index}]")
            gate = str(layer.get("gate", f"layer{index}"))
            rank_map = layer.get("rank_by_expert")
            if rank_map is None and isinstance(layer.get("experts_by_rank"), dict):
                rank_map = {
                    str(expert): int(rank)
                    for rank, experts in layer["experts_by_rank"].items()
                    for expert in experts
                }
            if not isinstance(rank_map, dict):
                raise ValueError(f"placement {path} has no rank map for {gate!r}")
            converted[gate] = rank_map
        raw = converted
    return raw


def _placement_for_layer(
    placements: Mapping[str, Any],
    *,
    path: str,
    gate: str,
    layer: int,
    observed_experts: Sequence[int],
    num_ranks: int,
) -> dict[int, int]:
    is_global = all(isinstance(value, int) for value in placements.values())
    candidate = placements if is_global else placements.get(gate)
    if candidate is None and not is_global:
        candidate = placements.get(str(layer), placements.get(f"layer{layer}"))
    if not isinstance(candidate, dict):
        raise ValueError(
            f"placement {path} has no mapping for gate {gate!r} (layer {layer})"
        )

    result = {}
    for expert in observed_experts:
        value = candidate.get(str(expert), candidate.get(expert))
        if not isinstance(value, int) or not 0 <= value < num_ranks:
            raise ValueError(
                f"placement {path} has no valid rank for expert {expert} "
                f"in gate {gate!r}"
            )
        result[expert] = value
    return result


def _graph_placement_from_rank_map(
    rank_by_expert: Mapping[int, int], num_ranks: int
) -> GraphPlacement:
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
    return GraphPlacement(dict(rank_by_expert), experts_by_rank, 0.0, 0.0, 0)


def _save_placements(path: str, result: Mapping[str, Any]) -> None:
    payload = {
        layer["gate"]: {
            str(expert): rank
            for expert, rank in sorted(layer["rank_by_expert"].items())
        }
        for layer in result["layers"]
    }
    Path(path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _tensorize_entries(
    name: str,
    entries: Sequence[Any],
    *,
    device: str,
    source_ep: int,
    target_ep: int,
) -> _TensorTrace:
    torch = _torch()
    sources = []
    topks = []
    counts = []
    topk_size = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}.bundles[{index}] must be an object")
        try:
            source = int(entry["source_rank"])
            experts = tuple(int(expert) for expert in entry["topk_experts"])
            count = int(entry.get("count", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle in {name}: {entry!r}") from exc
        if not experts:
            raise ValueError(f"invalid bundle in {name}: {entry!r}")
        if topk_size is None:
            topk_size = len(experts)
        elif len(experts) != topk_size:
            raise ValueError(
                f"--device cuda requires a fixed Top-K within {name}; "
                f"expected {topk_size}, got {len(experts)}"
            )
        sources.append(source)
        topks.append(experts)
        counts.append(count)

    source_cpu = torch.tensor(sources, dtype=torch.int64)
    topk_cpu = torch.tensor(topks, dtype=torch.int64)
    count_cpu = torch.tensor(counts, dtype=torch.int64)
    invalid = (
        (source_cpu < 0).any()
        | (source_cpu >= source_ep).any()
        | (count_cpu < 1).any()
        | (topk_cpu < 0).any()
    )
    topk_cpu = topk_cpu.sort(dim=1).values
    duplicates = (topk_cpu[:, 1:] == topk_cpu[:, :-1]).any()
    if bool(invalid.item()) or bool(duplicates.item()):
        raise ValueError(f"invalid source, count, or expert ids in {name}")
    if device.startswith("cuda"):
        trace = _enqueue_cuda_tensors(
            source_cpu,
            topk_cpu,
            count_cpu,
            device=device,
            source_ep=source_ep,
            target_ep=target_ep,
        )
        _wait_tensor_trace(trace)
        return trace
    source_tensor, topk_tensor, count_tensor = (
        source_cpu,
        topk_cpu,
        count_cpu,
    )
    return _reshard_tensor_trace(
        source_tensor, topk_tensor, count_tensor, source_ep, target_ep
    )


def _tensorize_compact_layer(
    name: str,
    layer: Mapping[str, Any],
    *,
    device: str,
    source_ep: int,
    target_ep: int,
    copy_stream: Any = None,
    host_pool: _PinnedTracePool | None = None,
    pool_index: int = 0,
) -> _TensorTrace:
    torch = _torch()
    source = layer.get("source_rank")
    topk = layer.get("topk_experts")
    count = layer.get("count")
    if not all(isinstance(value, torch.Tensor) for value in (source, topk, count)):
        raise ValueError(f"invalid compact tensors in {name}")
    if source.ndim != 1 or topk.ndim != 2 or count.ndim != 1:
        raise ValueError(f"invalid compact dimensions in {name}")
    if source.shape[0] != topk.shape[0] or count.shape[0] != topk.shape[0]:
        raise ValueError(f"compact tensor lengths disagree in {name}")
    if int(source.min().item()) < 0 or int(source.max().item()) >= source_ep:
        raise ValueError(f"invalid source ranks in compact layer {name}")
    if int(count.min().item()) < 1:
        raise ValueError(f"invalid bundle counts in compact layer {name}")
    if device.startswith("cuda"):
        if host_pool is not None:
            source, topk, count = host_pool.stage(pool_index, layer)
        return _enqueue_cuda_tensors(
            source,
            topk,
            count,
            device=device,
            source_ep=source_ep,
            target_ep=target_ep,
            copy_stream=copy_stream,
        )
    return _reshard_tensor_trace(
        source.to(torch.int64),
        topk.to(torch.int64),
        count.to(torch.int64),
        source_ep,
        target_ep,
    )


def _tokens_from_compact_layer(
    name: str, layer: Mapping[str, Any], source_ep: int, target_ep: int
) -> list[RoutedToken]:
    source = layer["source_rank"].tolist()
    topk = layer["topk_experts"].tolist()
    count = layer["count"].tolist()
    tokens = [
        RoutedToken(int(src), tuple(sorted(map(int, experts))), int(weight))
        for src, experts, weight in zip(source, topk, count)
    ]
    return _reshard_tokens(tokens, source_ep, target_ep)


def _materialize_layer_tokens(
    name: str,
    entries: Any,
    *,
    compact_input: bool,
    source_ep: int,
    target_ep: int,
) -> list[RoutedToken]:
    if compact_input:
        return _tokens_from_compact_layer(name, entries, source_ep, target_ep)
    return _reshard_tokens(
        _tokens_from_entries(name, entries),
        source_ep,
        target_ep,
    )


def _prefetched_layers(
    layers: Sequence[tuple[str, Any]],
    *,
    compact_input: bool,
    device: str,
    source_ep: int,
    target_ep: int,
) -> Iterable[tuple[str, Any, _TensorTrace | None, float]]:
    if not compact_input or not device.startswith("cuda"):
        for name, entries in layers:
            yield name, entries, None, 0.0
        return

    torch = _torch()
    copy_stream = torch.cuda.Stream(device=device)
    host_pool = _PinnedTracePool(layers)

    def submit(executor: ThreadPoolExecutor, index: int) -> Future[_TensorTrace]:
        name, layer = layers[index]
        return executor.submit(
            _tensorize_compact_layer,
            name,
            layer,
            device=device,
            source_ep=source_ep,
            target_ep=target_ep,
            copy_stream=copy_stream,
            host_pool=host_pool,
            pool_index=index,
        )

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="moe-trace-prefetch"
    ) as executor:
        future = submit(executor, 0)
        for index, (name, layer) in enumerate(layers):
            wait_start = time.perf_counter()
            trace = future.result()
            _wait_tensor_trace(trace)
            wait_seconds = time.perf_counter() - wait_start
            if index + 1 < len(layers):
                future = submit(executor, index + 1)
            yield name, layer, trace, wait_seconds


def _build_graph_tensor(trace: _TensorTrace, *, chunk_size: int) -> CoRoutingGraph:
    torch = _torch()
    max_expert = int(trace.topk_experts.max().item())
    num_experts = max_expert + 1
    demand = torch.zeros(
        num_experts, dtype=torch.int64, device=trace.topk_experts.device
    )
    edge_weights = torch.zeros(
        num_experts * num_experts,
        dtype=torch.int64,
        device=trace.topk_experts.device,
    )
    topk_size = trace.topk_experts.shape[1]
    pair_indices = torch.triu_indices(
        topk_size,
        topk_size,
        offset=1,
        device=trace.topk_experts.device,
    )
    for start in range(0, trace.count.numel(), chunk_size):
        stop = min(start + chunk_size, trace.count.numel())
        topk = trace.topk_experts[start:stop]
        count = trace.count[start:stop]
        demand.scatter_add_(
            0,
            topk.reshape(-1),
            count[:, None].expand_as(topk).reshape(-1),
        )
        left = topk[:, pair_indices[0]]
        right = topk[:, pair_indices[1]]
        pair_ids = torch.minimum(left, right) * num_experts + torch.maximum(left, right)
        edge_weights.scatter_add_(
            0,
            pair_ids.reshape(-1),
            count[:, None].expand_as(pair_ids).reshape(-1),
        )

    observed = torch.nonzero(demand, as_tuple=False).flatten()
    nonzero_edges = torch.nonzero(edge_weights, as_tuple=False).flatten()
    observed_cpu = observed.cpu().tolist()
    demand_cpu = demand[observed].cpu().tolist()
    edge_ids_cpu = nonzero_edges.cpu().tolist()
    edge_values_cpu = edge_weights[nonzero_edges].cpu().tolist()
    demand_map = dict(zip(observed_cpu, demand_cpu))
    edges = {
        (edge_id // num_experts, edge_id % num_experts): weight
        for edge_id, weight in zip(edge_ids_cpu, edge_values_cpu)
    }
    adjacency = {expert: {} for expert in observed_cpu}
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return CoRoutingGraph(tuple(observed_cpu), demand_map, edges, adjacency)


def _evaluate_primary_remote_tensor(
    trace: _TensorTrace,
    placements: Sequence[Mapping[int, int]],
    *,
    chunk_size: int,
) -> list[int]:
    torch = _torch()
    max_expert = int(trace.topk_experts.max().item())
    homes = torch.full(
        (len(placements), max_expert + 1),
        -1,
        dtype=torch.int64,
        device=trace.topk_experts.device,
    )
    for placement_index, placement in enumerate(placements):
        expert_ids = torch.tensor(
            list(placement), dtype=torch.int64, device=trace.topk_experts.device
        )
        rank_ids = torch.tensor(
            list(placement.values()),
            dtype=torch.int64,
            device=trace.topk_experts.device,
        )
        homes[placement_index, expert_ids] = rank_ids
    if bool((homes[:, trace.topk_experts.reshape(-1)] < 0).any().item()):
        raise ValueError("placement is missing an observed expert")

    totals = torch.zeros(
        len(placements), dtype=torch.int64, device=trace.topk_experts.device
    )
    for start in range(0, trace.count.numel(), chunk_size):
        stop = min(start + chunk_size, trace.count.numel())
        topk = trace.topk_experts[start:stop]
        destination = homes[:, topk].sort(dim=-1).values
        is_new = torch.ones_like(destination, dtype=torch.bool)
        is_new[:, :, 1:] = destination[:, :, 1:] != destination[:, :, :-1]
        is_remote = destination != trace.source_rank[start:stop][None, :, None]
        remote_per_bundle = (is_new & is_remote).sum(dim=-1)
        totals += (remote_per_bundle * trace.count[start:stop][None, :]).sum(dim=1)
    return totals.cpu().tolist()


def _baseline_homes(
    experts: Sequence[int], num_ranks: int, mode: str
) -> dict[int, int]:
    if mode == "round-robin":
        return {expert: expert % num_ranks for expert in experts}
    per_rank = (max(experts, default=-1) + 1 + num_ranks - 1) // num_ranks
    per_rank = max(per_rank, 1)
    return {expert: min(expert // per_rank, num_ranks - 1) for expert in experts}


def _format_table(rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _compute_loads(
    demand: Mapping[int, int],
    rank_by_expert: Mapping[int, int],
    num_ranks: int,
) -> list[int]:
    loads = [0] * num_ranks
    for expert, count in demand.items():
        loads[rank_by_expert[expert]] += count
    return loads


def _max_over_average(values: Sequence[int]) -> float:
    average = sum(values) / len(values) if values else 0.0
    return max(values, default=0) / average if average else 0.0


def _layer_number(gate: str, fallback: int) -> int:
    match = _LAYER_PATTERN.search(gate)
    return int(match.group(1)) if match else fallback


def _summarize_tokens(
    tokens: Sequence[RoutedToken],
) -> tuple[tuple[int, ...], dict[int, int]]:
    demand: dict[int, int] = {}
    for token in tokens:
        for expert in token.topk_experts:
            demand[expert] = demand.get(expert, 0) + token.count
    return tuple(sorted(demand)), demand


def _summarize_tensor_trace(
    trace: _TensorTrace, *, chunk_size: int
) -> tuple[tuple[int, ...], dict[int, int]]:
    torch = _torch()
    num_experts = int(trace.topk_experts.max().item()) + 1
    demand = torch.zeros(
        num_experts, dtype=torch.int64, device=trace.topk_experts.device
    )
    for start in range(0, trace.count.numel(), chunk_size):
        stop = min(start + chunk_size, trace.count.numel())
        topk = trace.topk_experts[start:stop]
        count = trace.count[start:stop]
        demand.scatter_add_(
            0,
            topk.reshape(-1),
            count[:, None].expand_as(topk).reshape(-1),
        )
    demand_cpu = demand.cpu().tolist()
    experts = tuple(expert for expert, value in enumerate(demand_cpu) if value > 0)
    return experts, {expert: demand_cpu[expert] for expert in experts}


def _balanced_initial_placement(
    experts: Sequence[int],
    demand: Mapping[int, int],
    *,
    num_ranks: int,
    slots_per_rank: int,
) -> GraphPlacement:
    if len(experts) > num_ranks * slots_per_rank:
        raise ValueError("expert count exceeds total rank capacity")
    assignments = [set() for _ in range(num_ranks)]
    loads = [0] * num_ranks
    for expert in sorted(experts, key=lambda value: (-demand[value], value)):
        candidates = [
            rank for rank in range(num_ranks) if len(assignments[rank]) < slots_per_rank
        ]
        rank = min(
            candidates, key=lambda value: (loads[value], len(assignments[value]), value)
        )
        assignments[rank].add(expert)
        loads[rank] += demand[expert]
    rank_by_expert = {
        expert: rank
        for rank, assigned_experts in enumerate(assignments)
        for expert in assigned_experts
    }
    experts_by_rank = {
        rank: tuple(sorted(assigned_experts))
        for rank, assigned_experts in enumerate(assignments)
    }
    return GraphPlacement(rank_by_expert, experts_by_rank, 0.0, 0.0, 0)


def _perturb_placement(
    rank_by_expert: Mapping[int, int], *, seed: int, swaps: int
) -> dict[int, int]:
    result = dict(rank_by_expert)
    experts = sorted(result)
    if len(experts) < 2 or swaps <= 0:
        return result
    rng = random.Random(seed)
    for _ in range(swaps):
        for _attempt in range(32):
            left, right = rng.sample(experts, 2)
            if result[left] != result[right]:
                result[left], result[right] = result[right], result[left]
                break
    return result


def _reshard_tokens(
    tokens: Sequence[RoutedToken], source_ep: int, target_ep: int
) -> list[RoutedToken]:
    """Deterministically split source-rank traffic across virtual target ranks."""

    if source_ep == target_ep:
        return list(tokens)
    if target_ep < source_ep or target_ep % source_ep:
        raise ValueError("target EP must be a multiple of source EP")
    fanout = target_ep // source_ep
    result: list[RoutedToken] = []
    for token in tokens:
        base, remainder = divmod(token.count, fanout)
        for offset in range(fanout):
            count = base + int(offset < remainder)
            if count:
                result.append(
                    RoutedToken(
                        token.source_rank * fanout + offset,
                        token.topk_experts,
                        count,
                    )
                )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.perf_counter()
    replay_only = getattr(args, "replay_only", False)
    load_placement = getattr(args, "load_placement", None)
    oracle_placement = getattr(args, "oracle_placement", None)
    use_tensor_backend = args.device != "cpu"
    if use_tensor_backend:
        torch = _torch()
        if not args.device.startswith("cuda"):
            raise ValueError("--device must be cpu, cuda, or cuda:<index>")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for --device {args.device}")
    if args.planner == "cuda-fast" and not use_tensor_backend and not replay_only:
        raise ValueError("--planner cuda-fast requires --device cuda")
    cuda_fast_ops = (
        _cuda_fast_planner_ops()
        if args.planner == "cuda-fast" and not replay_only
        else None
    )
    input_start = time.perf_counter()
    raw, layers, compact_input = _load_input(args.input)
    transfer_placements = _load_placements(load_placement)
    oracle_placements = _load_placements(oracle_placement)
    if args.compact_output:
        if compact_input:
            raise ValueError("--compact-output is only valid when --input is JSON")
        compact_layers = [
            compact_layer_from_records(name, entries) for name, entries in layers
        ]
        top_k = raw.get("top_k") if isinstance(raw, dict) else None
        if top_k is None:
            top_k = compact_layers[0]["topk_experts"].shape[1]
        source_ranks = raw.get("num_ranks") if isinstance(raw, dict) else None
        save_compact_trace(
            args.compact_output,
            num_ranks=source_ranks or args.source_ep or args.num_ranks,
            top_k=top_k,
            layers=compact_layers,
        )
        layers = [(layer["gate"], layer) for layer in compact_layers]
        compact_input = True
    input_load_seconds = time.perf_counter() - input_start
    trace_ep = args.source_ep
    if trace_ep is None and isinstance(raw, dict):
        trace_ep = raw.get("num_ranks")
    if trace_ep is None:
        trace_ep = args.num_ranks
    trace_ep = int(trace_ep)
    if trace_ep < 1:
        raise ValueError("source EP must be positive")
    round_limit = None if args.until_convergence else args.max_rounds
    pairwise_enabled = not replay_only and args.placement_mode in {"pairwise", "hybrid"}
    hypergraph_enabled = not replay_only and args.placement_mode in {
        "hypergraph",
        "hybrid",
    }
    hypergraph_limit = (
        None if args.hypergraph_until_convergence else args.hypergraph_rounds
    )
    hypergraph_balance_limit = (
        None if args.hypergraph_balance_rounds == -1 else args.hypergraph_balance_rounds
    )
    output_layers = []
    prepared_layers = _prefetched_layers(
        layers,
        compact_input=compact_input,
        device=args.device,
        source_ep=trace_ep,
        target_ep=args.num_ranks,
    )
    for layer_position, (
        name,
        entries,
        prefetched_trace,
        prefetch_wait_seconds,
    ) in enumerate(prepared_layers):
        layer_start = time.perf_counter()
        prepare_start = time.perf_counter()
        tensorize_start = time.perf_counter()
        tokens = None
        tensor_trace = None
        if prefetched_trace is not None:
            tensor_trace = prefetched_trace
            tensorize_seconds = prefetch_wait_seconds
        elif compact_input and use_tensor_backend:
            tensor_trace = _tensorize_compact_layer(
                name,
                entries,
                device=args.device,
                source_ep=trace_ep,
                target_ep=args.num_ranks,
            )
            _wait_tensor_trace(tensor_trace)
            tensorize_seconds = time.perf_counter() - tensorize_start
        elif compact_input:
            tokens = _tokens_from_compact_layer(name, entries, trace_ep, args.num_ranks)
            tensorize_seconds = time.perf_counter() - tensorize_start
        elif use_tensor_backend:
            tensor_trace = _tensorize_entries(
                name,
                entries,
                device=args.device,
                source_ep=trace_ep,
                target_ep=args.num_ranks,
            )
            tensorize_seconds = time.perf_counter() - tensorize_start
        else:
            tokens = _reshard_tokens(
                _tokens_from_entries(name, entries), trace_ep, args.num_ranks
            )
            tensorize_seconds = time.perf_counter() - tensorize_start
        graph_build_start = time.perf_counter()
        cuda_graph = None
        graph = None
        if pairwise_enabled:
            if tensor_trace is not None:
                if cuda_fast_ops is not None:
                    build_cuda_graph, _, _, _ = cuda_fast_ops
                    cuda_graph = build_cuda_graph(
                        tensor_trace.topk_experts, tensor_trace.count
                    )
                    graph_experts, graph_demand, graph_edge_count = (
                        cuda_graph.cpu_summary()
                    )
                else:
                    graph = _build_graph_tensor(
                        tensor_trace, chunk_size=args.tensor_chunk_size
                    )
                    graph_experts = graph.experts
                    graph_demand = graph.demand
                    graph_edge_count = len(graph.edges)
            else:
                graph = build_co_routing_graph(tokens)
                graph_experts = graph.experts
                graph_demand = graph.demand
                graph_edge_count = len(graph.edges)
        elif tensor_trace is not None:
            graph_experts, graph_demand = _summarize_tensor_trace(
                tensor_trace,
                chunk_size=args.tensor_chunk_size,
            )
            graph_edge_count = None
        else:
            graph_experts, graph_demand = _summarize_tokens(tokens)
            graph_edge_count = None
        graph_build_seconds = time.perf_counter() - graph_build_start
        prepare_seconds = time.perf_counter() - prepare_start
        capacity = args.slots_per_rank
        if capacity is None:
            capacity = (len(graph_experts) + args.num_ranks - 1) // args.num_ranks

        graph_solve_seconds = 0.0
        if replay_only:
            transfer_map = _placement_for_layer(
                transfer_placements,
                path=load_placement,
                gate=name,
                layer=_layer_number(name, layer_position),
                observed_experts=graph_experts,
                num_ranks=args.num_ranks,
            )
            placement = _graph_placement_from_rank_map(transfer_map, args.num_ranks)
            initial_cut = None
            final_cut = None
            graph_iterations = 0
        elif pairwise_enabled:
            solver = CoRoutingGraphSolver(
                num_ranks=args.num_ranks,
                slots_per_rank=capacity,
                max_rounds=round_limit,
                balance_weight=args.balance_weight,
            )
            graph_start = time.perf_counter()
            if cuda_graph is not None:
                _, solve_cuda_graph, _, _ = cuda_fast_ops
                placement = solve_cuda_graph(
                    cuda_graph,
                    num_ranks=args.num_ranks,
                    slots_per_rank=capacity,
                    max_rounds=round_limit,
                    balance_weight=args.balance_weight,
                )
            else:
                placement = solver.solve(graph)
            graph_solve_seconds = time.perf_counter() - graph_start
            initial_cut = placement.initial_cut
            final_cut = placement.final_cut
            graph_iterations = placement.iterations
        else:
            placement = _balanced_initial_placement(
                graph_experts,
                graph_demand,
                num_ranks=args.num_ranks,
                slots_per_rank=capacity,
            )
            initial_cut = None
            final_cut = None
            graph_iterations = 0

        baseline_homes = _baseline_homes(graph_experts, args.num_ranks, args.baseline)
        baseline_compute_load = _compute_loads(
            graph_demand, baseline_homes, args.num_ranks
        )
        seed_compute_load = _compute_loads(
            graph_demand, placement.rank_by_expert, args.num_ranks
        )
        oracle_map = None
        oracle_compute_load = None
        if replay_only and oracle_placements:
            oracle_map = _placement_for_layer(
                oracle_placements,
                path=oracle_placement,
                gate=name,
                layer=_layer_number(name, layer_position),
                observed_experts=graph_experts,
                num_ranks=args.num_ranks,
            )
            oracle_compute_load = _compute_loads(
                graph_demand, oracle_map, args.num_ranks
            )
        primary_replay_start = time.perf_counter()
        replay_placements = [baseline_homes, placement.rank_by_expert]
        if oracle_map is not None:
            replay_placements.append(oracle_map)
        if tensor_trace is not None:
            replay_values = _evaluate_primary_remote_tensor(
                tensor_trace,
                replay_placements,
                chunk_size=args.tensor_chunk_size,
            )
        else:
            replay_values = [
                evaluate_primary_remote(tokens, candidate)
                for candidate in replay_placements
            ]
        baseline_remote, seed_remote = replay_values[:2]
        oracle_remote = replay_values[2] if oracle_map is not None else None
        primary_replay_seconds = time.perf_counter() - primary_replay_start
        graph_remote = seed_remote if pairwise_enabled else None
        graph_compute_load = seed_compute_load if pairwise_enabled else None
        fixed_remote = seed_remote if replay_only else None
        fixed_compute_load = seed_compute_load if replay_only else None
        final_placement = placement
        hypergraph_remote = None
        hypergraph_iterations = 0
        hypergraph_balance_iterations = 0
        hypergraph_solve_seconds = 0.0
        if hypergraph_enabled:
            best_hypergraph_placement = None
            best_hypergraph_key = None
            if cuda_fast_ops is None and tokens is None:
                tokens = _materialize_layer_tokens(
                    name,
                    entries,
                    compact_input=compact_input,
                    source_ep=trace_ep,
                    target_ep=args.num_ranks,
                )
            for restart in range(args.hypergraph_restarts):
                initial_rank_map = (
                    placement.rank_by_expert
                    if restart == 0
                    else _perturb_placement(
                        placement.rank_by_expert,
                        seed=args.hypergraph_seed + 1009 * layer_position + restart,
                        swaps=args.hypergraph_kick_swaps,
                    )
                )
                if cuda_fast_ops is not None:
                    _, _, _, refine_cuda_hypergraph = cuda_fast_ops
                    candidate_placement = refine_cuda_hypergraph(
                        tensor_trace.source_rank,
                        tensor_trace.topk_experts,
                        tensor_trace.count,
                        initial_rank_map,
                        num_ranks=args.num_ranks,
                        max_rounds=hypergraph_limit,
                        balance_rounds=hypergraph_balance_limit,
                    )
                    hypergraph_solve_seconds += candidate_placement.solve_seconds
                else:
                    hypergraph_start = time.perf_counter()
                    candidate_placement = refine_hypergraph_placement(
                        tokens,
                        initial_rank_map,
                        num_ranks=args.num_ranks,
                        max_rounds=hypergraph_limit,
                        balance_rounds=hypergraph_balance_limit,
                    )
                    hypergraph_solve_seconds += time.perf_counter() - hypergraph_start
                if restart == 0 and candidate_placement.initial_remote != seed_remote:
                    raise RuntimeError(
                        "hypergraph refinement initial replay disagrees with seed replay"
                    )
                candidate_loads = _compute_loads(
                    graph_demand,
                    candidate_placement.rank_by_expert,
                    args.num_ranks,
                )
                candidate_key = (
                    candidate_placement.final_remote,
                    max(candidate_loads),
                    sum(load * load for load in candidate_loads),
                )
                if best_hypergraph_key is None or candidate_key < best_hypergraph_key:
                    best_hypergraph_placement = candidate_placement
                    best_hypergraph_key = candidate_key
            final_placement = best_hypergraph_placement
            hypergraph_remote = best_hypergraph_placement.final_remote
            hypergraph_iterations = best_hypergraph_placement.iterations
            hypergraph_balance_iterations = best_hypergraph_placement.balance_iterations
        hypergraph_compute_load = (
            _compute_loads(graph_demand, final_placement.rank_by_expert, args.num_ranks)
            if hypergraph_enabled
            else None
        )
        replica_remote = (
            hypergraph_remote if hypergraph_remote is not None else seed_remote
        )
        action_count = 0
        replica_action_details = []
        replica_materialize_seconds = 0.0
        replica_solve_seconds = 0.0
        replica_replay_seconds = 0.0
        if args.replica_slots_per_rank:
            if cuda_fast_ops is not None:
                _, _, plan_cuda_replicas, _ = cuda_fast_ops
                replica_plan = plan_cuda_replicas(
                    tensor_trace.source_rank,
                    tensor_trace.topk_experts,
                    tensor_trace.count,
                    final_placement.rank_by_expert,
                    num_ranks=args.num_ranks,
                    replica_slots_per_rank=args.replica_slots_per_rank,
                    max_candidates=args.fast_max_candidates,
                    max_bundle_size=args.max_bundle_size,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    chunk_size=min(args.tensor_chunk_size, 65536),
                )
                replica_remote = replica_plan.unique_remote_rank_copies
                action_count = len(replica_plan.actions)
                replica_action_details = [
                    {
                        "destination_rank": action.destination_rank,
                        "experts": list(action.experts),
                        "kind": action.kind,
                    }
                    for action in replica_plan.actions
                ]
                replica_solve_seconds = replica_plan.solve_seconds
                replica_replay_seconds = replica_plan.replay_seconds
            elif tokens is None:
                materialize_start = time.perf_counter()
                tokens = _materialize_layer_tokens(
                    name,
                    entries,
                    compact_input=compact_input,
                    source_ep=trace_ep,
                    target_ep=args.num_ranks,
                )
                replica_materialize_seconds = time.perf_counter() - materialize_start
            if cuda_fast_ops is None:
                replica_planner = BundleAwareReplicaPlanner(
                    num_ranks=args.num_ranks,
                    baseline_rank_by_expert=final_placement.rank_by_expert,
                    replica_slots_per_rank=args.replica_slots_per_rank,
                    ranks_per_node=args.ranks_per_node,
                    rdma_cost=args.rdma_cost,
                    compute_weight=0.0,
                    communication_weight=1.0,
                    max_bundle_size=args.max_bundle_size,
                )
                start = time.perf_counter()
                if args.planner == "fast":
                    replica_plan = replica_planner.plan_fast(
                        tokens, max_candidates=args.fast_max_candidates
                    )
                else:
                    replica_plan = replica_planner.plan(
                        tokens, max_actions=args.max_actions
                    )
                replica_solve_seconds = time.perf_counter() - start
                replay_start = time.perf_counter()
                replayed = replica_planner.evaluate_placement(
                    tokens, replica_plan.replicas_by_rank
                )
                replica_replay_seconds = time.perf_counter() - replay_start
                replica_remote = replayed.unique_remote_rank_copies
                action_count = len(replica_plan.actions)
                replica_action_details = [
                    {
                        "destination_rank": action.destination_rank,
                        "experts": list(action.experts),
                        "kind": action.kind,
                    }
                    for action in replica_plan.actions
                ]
        if tensor_trace is not None:
            token_count = int(tensor_trace.count.sum().item())
            bundle_count = tensor_trace.count.numel()
        else:
            token_count = sum(token.count for token in tokens)
            bundle_count = len(tokens)
        output_layers.append(
            {
                "gate": name,
                "layer": _layer_number(name, layer_position),
                "tokens": token_count,
                "bundles": bundle_count,
                "experts": len(graph_experts),
                "edges": graph_edge_count,
                "baseline_remote": baseline_remote,
                "transfer_remote": fixed_remote,
                "oracle_remote": oracle_remote,
                "graph_remote": graph_remote,
                "hypergraph_seed_remote": seed_remote if hypergraph_enabled else None,
                "hypergraph_remote": hypergraph_remote,
                "replica_remote": replica_remote,
                "baseline_compute_load": baseline_compute_load,
                "transfer_compute_load": fixed_compute_load,
                "oracle_compute_load": oracle_compute_load,
                "graph_compute_load": graph_compute_load,
                "hypergraph_compute_load": hypergraph_compute_load,
                "baseline_compute_imbalance": _max_over_average(baseline_compute_load),
                "transfer_compute_imbalance": (
                    _max_over_average(fixed_compute_load)
                    if fixed_compute_load is not None
                    else None
                ),
                "oracle_compute_imbalance": (
                    _max_over_average(oracle_compute_load)
                    if oracle_compute_load is not None
                    else None
                ),
                "graph_compute_imbalance": (
                    _max_over_average(graph_compute_load)
                    if graph_compute_load is not None
                    else None
                ),
                "hypergraph_compute_imbalance": (
                    _max_over_average(hypergraph_compute_load)
                    if hypergraph_compute_load is not None
                    else None
                ),
                "graph_delta": (graph_remote / baseline_remote - 1.0)
                if baseline_remote and graph_remote is not None
                else None,
                "hypergraph_delta": (hypergraph_remote / baseline_remote - 1.0)
                if baseline_remote and hypergraph_remote is not None
                else None,
                "replica_delta": (replica_remote / baseline_remote - 1.0)
                if baseline_remote
                else 0.0,
                "transfer_delta": (fixed_remote / baseline_remote - 1.0)
                if baseline_remote and fixed_remote is not None
                else None,
                "oracle_regret": (fixed_remote / oracle_remote - 1.0)
                if fixed_remote is not None and oracle_remote
                else None,
                "retained_gain": (
                    (baseline_remote - fixed_remote) / (baseline_remote - oracle_remote)
                )
                if fixed_remote is not None
                and oracle_remote is not None
                and baseline_remote > oracle_remote
                else None,
                "oracle_remote_gap": (fixed_remote - oracle_remote)
                if fixed_remote is not None and oracle_remote is not None
                else None,
                "initial_cut": initial_cut,
                "final_cut": final_cut,
                "iterations": graph_iterations,
                "prepare_seconds": prepare_seconds,
                "tensorize_seconds": tensorize_seconds,
                "graph_build_seconds": graph_build_seconds,
                "primary_replay_seconds": primary_replay_seconds,
                "layer_seconds": time.perf_counter() - layer_start,
                "graph_solve_seconds": graph_solve_seconds,
                "hypergraph_iterations": hypergraph_iterations,
                "hypergraph_balance_iterations": hypergraph_balance_iterations,
                "hypergraph_restarts": (
                    args.hypergraph_restarts if hypergraph_enabled else 0
                ),
                "hypergraph_solve_seconds": hypergraph_solve_seconds,
                "replica_actions": action_count,
                "replica_action_details": replica_action_details,
                "replica_materialize_seconds": replica_materialize_seconds,
                "replica_solve_seconds": replica_solve_seconds,
                "replica_replay_seconds": replica_replay_seconds,
                "experts_by_rank": {
                    str(rank): list(experts)
                    for rank, experts in final_placement.experts_by_rank.items()
                },
                "rank_by_expert": dict(final_placement.rank_by_expert),
            }
        )
    return {
        "source_ep": trace_ep,
        "num_ranks": args.num_ranks,
        "baseline": args.baseline,
        "planner": args.planner,
        "placement_mode": args.placement_mode,
        "replay_only": replay_only,
        "loaded_placement": load_placement,
        "oracle_placement": oracle_placement,
        "device": args.device,
        "round_limit": round_limit,
        "hypergraph_round_limit": (
            None if args.hypergraph_until_convergence else args.hypergraph_rounds
        ),
        "hypergraph_balance_round_limit": hypergraph_balance_limit,
        "input_format": "compact" if compact_input else "json",
        "input_load_seconds": input_load_seconds,
        "total_wall_seconds": time.perf_counter() - run_start,
        "layers": output_layers,
    }


def _optional_text(value: Any) -> str:
    return "-" if value is None else str(value)


def _optional_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


def _optional_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def _print_replay_result(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    rows = [
        [
            "layer",
            "tokens",
            "bundles",
            "baseline",
            "transfer",
            "oracle",
            "transfer_delta",
            "oracle_regret",
            "retained",
            "comp",
            "replay",
        ]
    ]
    for layer in result["layers"]:
        compute_values = [
            layer["baseline_compute_imbalance"],
            layer["transfer_compute_imbalance"],
        ]
        if layer["oracle_compute_imbalance"] is not None:
            compute_values.append(layer["oracle_compute_imbalance"])
        rows.append(
            [
                str(layer["layer"]),
                str(layer["tokens"]),
                str(layer["bundles"]),
                str(layer["baseline_remote"]),
                str(layer["transfer_remote"]),
                _optional_text(layer["oracle_remote"]),
                _optional_percent(layer["transfer_delta"]),
                _optional_percent(layer["oracle_regret"]),
                (
                    f"{layer['retained_gain']:.1%}"
                    if layer["retained_gain"] is not None
                    else "-"
                ),
                "->".join(f"{value:.2f}" for value in compute_values),
                f"{layer['primary_replay_seconds']:.3f}s",
            ]
        )

    baseline_total = sum(layer["baseline_remote"] for layer in result["layers"])
    transfer_total = sum(layer["transfer_remote"] for layer in result["layers"])
    oracle_available = all(
        layer["oracle_remote"] is not None for layer in result["layers"]
    )
    oracle_total = (
        sum(layer["oracle_remote"] for layer in result["layers"])
        if oracle_available
        else None
    )
    transfer_delta = transfer_total / baseline_total - 1.0 if baseline_total else 0.0
    oracle_regret = transfer_total / oracle_total - 1.0 if oracle_total else None
    retained_gain = (
        (baseline_total - transfer_total) / (baseline_total - oracle_total)
        if oracle_total is not None and baseline_total > oracle_total
        else None
    )

    print("Frozen placement transfer replay")
    print(
        f"device={result['device']} placement={result['loaded_placement']} "
        f"oracle={result['oracle_placement'] or '-'} "
        f"input={result['input_format']} "
        f"input_load={result['input_load_seconds']:.3f}s"
    )
    print(_format_table(rows))
    print(
        f"total baseline={baseline_total} transfer={transfer_total} "
        f"oracle={oracle_total if oracle_total is not None else '-'} "
        f"transfer_delta={transfer_delta:+.1%} "
        f"oracle_regret={_optional_percent(oracle_regret)} "
        "retained_gain="
        f"{f'{retained_gain:.1%}' if retained_gain is not None else '-'} "
        f"wall={result['total_wall_seconds']:.3f}s"
    )
    if getattr(args, "show_placement", False):
        for layer in result["layers"]:
            print(f"\n{layer['gate']} experts_by_rank")
            for rank, experts in layer["experts_by_rank"].items():
                print(f"  rank {rank}: {experts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="offline bundle JSON or compact .pt trace"
    )
    parser.add_argument(
        "--save-placement",
        default=None,
        help="save the solved per-gate primary placement as JSON",
    )
    parser.add_argument(
        "--load-placement",
        default=None,
        help="load a frozen per-gate primary placement for replay-only evaluation",
    )
    parser.add_argument(
        "--oracle-placement",
        default=None,
        help="optional target-dataset oracle placement for transfer regret metrics",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="evaluate --load-placement without running any placement solver",
    )
    parser.add_argument(
        "--compact-output",
        default=None,
        help=(
            "when input is JSON, also save a mmap-compatible compact .pt trace "
            "and use it for this run"
        ),
    )
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument(
        "--source-ep",
        type=int,
        default=None,
        help="source EP of the trace; defaults to trace JSON num_ranks",
    )
    parser.add_argument("--slots-per-rank", type=int, default=None)
    round_group = parser.add_mutually_exclusive_group()
    round_group.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="maximum swap rounds per layer (default: 8)",
    )
    round_group.add_argument(
        "--until-convergence",
        action="store_true",
        help="ignore the round limit and stop only when no improving swap remains",
    )
    hypergraph_group = parser.add_mutually_exclusive_group()
    hypergraph_group.add_argument(
        "--hypergraph-rounds",
        type=int,
        default=8,
        help=(
            "refine graph placement for this many rounds using exact Top-K "
            "remote-rank traffic (default: 8)"
        ),
    )
    hypergraph_group.add_argument(
        "--hypergraph-until-convergence",
        action="store_true",
        help="refine exact Top-K traffic until no improving swap remains",
    )
    parser.add_argument(
        "--hypergraph-balance-rounds",
        type=int,
        default=0,
        help=(
            "communication-neutral compute-balance swap rounds after hypergraph "
            "refinement; 0 disables and -1 runs to convergence (default: 0)"
        ),
    )
    parser.add_argument(
        "--hypergraph-restarts",
        type=int,
        default=1,
        help="number of exact-objective starts; best result is retained (default: 1)",
    )
    parser.add_argument(
        "--hypergraph-kick-swaps",
        type=int,
        default=2,
        help="cross-rank perturbation swaps before each additional start (default: 2)",
    )
    parser.add_argument("--hypergraph-seed", type=int, default=0)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "backend for graph statistics and primary replay: cpu, cuda, or "
            "cuda:<index> (default: cpu)"
        ),
    )
    parser.add_argument(
        "--tensor-chunk-size",
        type=int,
        default=262144,
        help="maximum bundles per CUDA graph/replay chunk (default: 262144)",
    )
    parser.add_argument(
        "--replica-slots-per-rank",
        type=int,
        default=0,
        help="optional additional copies after graph placement (default: 0)",
    )
    parser.add_argument("--ranks-per-node", type=int, default=None)
    parser.add_argument("--rdma-cost", type=float, default=4.0)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-bundle-size", type=int, default=None)
    parser.add_argument("--fast-max-candidates", type=int, default=32)
    parser.add_argument(
        "--planner", choices=["exact", "fast", "cuda-fast"], default="fast"
    )
    parser.add_argument(
        "--placement-mode",
        choices=["pairwise", "hypergraph", "hybrid"],
        default="pairwise",
        help=(
            "pairwise only, exact Top-K hypergraph only, or pairwise followed "
            "by hypergraph refinement (default: pairwise)"
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=["round-robin", "contiguous"],
        default="round-robin",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-placement",
        action="store_true",
        help="print expert ids assigned to every rank",
    )
    args = parser.parse_args()
    if args.num_ranks < 1:
        parser.error("--num-ranks must be positive")
    if args.source_ep is not None and args.source_ep < 1:
        parser.error("--source-ep must be positive")
    if args.slots_per_rank is not None and args.slots_per_rank < 1:
        parser.error("--slots-per-rank must be positive")
    if args.replica_slots_per_rank < 0:
        parser.error("--replica-slots-per-rank must be non-negative")
    if args.ranks_per_node is not None and args.ranks_per_node < 1:
        parser.error("--ranks-per-node must be positive")
    if args.rdma_cost < 1:
        parser.error("--rdma-cost must be at least 1")
    if args.fast_max_candidates < 1:
        parser.error("--fast-max-candidates must be positive")
    if args.tensor_chunk_size < 1:
        parser.error("--tensor-chunk-size must be positive")
    if args.max_rounds < 0:
        parser.error("--max-rounds must be non-negative")
    if args.hypergraph_rounds < 0:
        parser.error("--hypergraph-rounds must be non-negative")
    if args.hypergraph_balance_rounds < -1:
        parser.error("--hypergraph-balance-rounds must be -1 or non-negative")
    if args.hypergraph_restarts < 1:
        parser.error("--hypergraph-restarts must be positive")
    if args.hypergraph_kick_swaps < 0:
        parser.error("--hypergraph-kick-swaps must be non-negative")
    if args.replay_only and args.load_placement is None:
        parser.error("--replay-only requires --load-placement")
    if args.load_placement is not None and not args.replay_only:
        parser.error("--load-placement requires --replay-only")
    if args.oracle_placement is not None and not args.replay_only:
        parser.error("--oracle-placement requires --replay-only")
    if args.replay_only and args.replica_slots_per_rank:
        parser.error("--replay-only does not run replica planning")
    result = run(args)
    if args.save_placement:
        _save_placements(args.save_placement, result)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if args.replay_only:
        _print_replay_result(result, args)
        if args.save_placement:
            print(f"saved placement={args.save_placement}")
        return
    rows = [
        [
            "layer",
            "tokens",
            "bundles",
            "edges",
            "baseline",
            "graph",
            "hypergraph",
            "replica",
            "graph_delta",
            "hyper_delta",
            "replica_delta",
            "cut",
            "comp",
            "rounds",
            "hyper_rounds",
            "balance_rounds",
            "tensorize",
            "graph_build",
            "primary_replay",
            "layer_total",
            "actions",
            "graph_solve",
            "hyper_solve",
            "replica_input",
            "replica_solve",
            "replica_replay",
        ]
    ]
    for layer in result["layers"]:
        rows.append(
            [
                str(layer["layer"]),
                str(layer["tokens"]),
                str(layer["bundles"]),
                _optional_text(layer["edges"]),
                str(layer["baseline_remote"]),
                _optional_text(layer["graph_remote"]),
                _optional_text(layer["hypergraph_remote"]),
                str(layer["replica_remote"]),
                _optional_percent(layer["graph_delta"]),
                _optional_percent(layer["hypergraph_delta"]),
                f"{layer['replica_delta']:+.1%}",
                (
                    "-"
                    if layer["initial_cut"] is None
                    else f"{layer['initial_cut']}->{layer['final_cut']}"
                ),
                "->".join(
                    f"{value:.2f}"
                    for value in (
                        layer["baseline_compute_imbalance"],
                        layer["graph_compute_imbalance"],
                        layer["hypergraph_compute_imbalance"],
                    )
                    if value is not None
                ),
                str(layer["iterations"]),
                str(layer["hypergraph_iterations"]),
                str(layer["hypergraph_balance_iterations"]),
                f"{layer['tensorize_seconds']:.3f}s",
                f"{layer['graph_build_seconds']:.3f}s",
                f"{layer['primary_replay_seconds']:.3f}s",
                f"{layer['layer_seconds']:.3f}s",
                str(layer["replica_actions"]),
                f"{layer['graph_solve_seconds']:.3f}s",
                _optional_seconds(layer["hypergraph_solve_seconds"]),
                f"{layer['replica_materialize_seconds']:.3f}s",
                f"{layer['replica_solve_seconds']:.3f}s",
                f"{layer['replica_replay_seconds']:.3f}s",
            ]
        )
    print("Co-routing graph placement replay")
    round_mode = (
        "disabled"
        if result["placement_mode"] == "hypergraph"
        else (
            "until-convergence"
            if result["round_limit"] is None
            else f"max-{result['round_limit']}"
        )
    )
    hypergraph_limit = result["hypergraph_round_limit"]
    hypergraph_mode = (
        "disabled"
        if result["placement_mode"] == "pairwise"
        else (
            "until-convergence"
            if hypergraph_limit is None
            else f"max-{hypergraph_limit}"
        )
    )
    hypergraph_balance_limit = result["hypergraph_balance_round_limit"]
    hypergraph_balance_mode = (
        "disabled"
        if result["placement_mode"] == "pairwise" or hypergraph_balance_limit == 0
        else (
            "until-convergence"
            if hypergraph_balance_limit is None
            else f"max-{hypergraph_balance_limit}"
        )
    )
    print(
        f"device={result['device']} planner={result['planner']} "
        f"placement={result['placement_mode']} "
        f"rounds={round_mode} "
        f"hypergraph={hypergraph_mode} "
        f"hyper_balance={hypergraph_balance_mode} "
        f"input={result['input_format']} "
        f"input_load={result['input_load_seconds']:.3f}s"
    )
    print(_format_table(rows))
    print(
        "total tensorize="
        f"{sum(layer['tensorize_seconds'] for layer in result['layers']):.3f}s "
        "graph_build="
        f"{sum(layer['graph_build_seconds'] for layer in result['layers']):.3f}s "
        "primary_replay="
        f"{sum(layer['primary_replay_seconds'] for layer in result['layers']):.3f}s "
        "graph_solve="
        f"{sum(layer['graph_solve_seconds'] for layer in result['layers']):.3f}s "
        "hypergraph_solve="
        f"{sum(layer['hypergraph_solve_seconds'] for layer in result['layers']):.3f}s "
        "replica_input="
        f"{sum(layer['replica_materialize_seconds'] for layer in result['layers']):.3f}s "
        "replica_solve="
        f"{sum(layer['replica_solve_seconds'] for layer in result['layers']):.3f}s "
        "replica_replay="
        f"{sum(layer['replica_replay_seconds'] for layer in result['layers']):.3f}s "
        "wall="
        f"{result['total_wall_seconds']:.3f}s",
    )
    if args.show_placement:
        for layer in result["layers"]:
            print(f"\n{layer['gate']} experts_by_rank")
            for rank, experts in layer["experts_by_rank"].items():
                print(f"  rank {rank}: {experts}")
            if layer["replica_action_details"]:
                print(f"{layer['gate']} replica actions")
                for action in layer["replica_action_details"]:
                    print(
                        f"  {action['kind']}: experts={action['experts']} "
                        f"-> rank {action['destination_rank']}"
                    )
    if args.save_placement:
        print(f"saved placement={args.save_placement}")


if __name__ == "__main__":
    main()
