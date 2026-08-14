"""Compact tensor serialization for aggregated MoE routing bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch


COMPACT_TRACE_FORMAT = "sglang.moe_bundle_trace.v1"
_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _compact_signed_dtype(maximum: int) -> torch.dtype:
    if maximum <= torch.iinfo(torch.int16).max:
        return torch.int16
    if maximum <= torch.iinfo(torch.int32).max:
        return torch.int32
    return torch.int64


def compact_layer_from_bundles(
    gate: str,
    bundles: Mapping[Tuple[int, Tuple[int, ...]], int],
) -> dict[str, Any]:
    """Convert an aggregated bundle mapping to compact CPU tensors."""

    items = [
        ((int(source), tuple(sorted(int(expert) for expert in experts))), int(count))
        for (source, experts), count in bundles.items()
    ]
    items.sort()
    if not items:
        raise ValueError(f"cannot serialize empty bundle layer {gate}")
    topk = len(items[0][0][1])
    if topk < 1 or any(len(key[1]) != topk for key, _ in items):
        raise ValueError(f"all bundles in {gate} must have the same non-zero Top-K")
    max_source = max(key[0] for key, _ in items)
    max_expert = max(expert for (source, experts), _ in items for expert in experts)
    max_count = max(count for _, count in items)
    if (
        min(key[0] for key, _ in items) < 0
        or max_count < 1
        or any(
            len(set(experts)) != topk or min(experts, default=-1) < 0
            for (_, experts), _ in items
        )
    ):
        raise ValueError(f"invalid source rank or count in {gate}")
    source_rank = torch.tensor(
        [key[0] for key, _ in items], dtype=_compact_signed_dtype(max_source)
    )
    topk_experts = torch.tensor(
        [key[1] for key, _ in items], dtype=_compact_signed_dtype(max_expert)
    )
    count = torch.tensor(
        [value for _, value in items], dtype=_compact_signed_dtype(max_count)
    )
    return {
        "gate": gate,
        "source_rank": source_rank,
        "topk_experts": topk_experts,
        "count": count,
    }


def compact_layer_from_records(
    gate: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Convert JSON-style bundle records to a compact tensor layer."""

    sources = []
    topks = []
    counts = []
    top_k = None
    for index, record in enumerate(records):
        try:
            source = int(record["source_rank"])
            experts = tuple(sorted(int(value) for value in record["topk_experts"]))
            count = int(record.get("count", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bundle {gate}[{index}]: {record!r}") from exc
        if source < 0 or count < 1 or not experts or min(experts) < 0:
            raise ValueError(f"invalid bundle {gate}[{index}]: {record!r}")
        if top_k is None:
            top_k = len(experts)
        elif len(experts) != top_k:
            raise ValueError(f"inconsistent Top-K in bundle {gate}[{index}]")
        sources.append(source)
        topks.append(experts)
        counts.append(count)
    if not sources:
        raise ValueError(f"cannot serialize empty bundle layer {gate}")
    topk_tensor = torch.tensor(
        topks,
        dtype=_compact_signed_dtype(max(max(experts) for experts in topks)),
    )
    if bool((topk_tensor[:, 1:] == topk_tensor[:, :-1]).any().item()):
        raise ValueError(f"duplicate expert in bundle layer {gate}")
    return {
        "gate": gate,
        "source_rank": torch.tensor(sources, dtype=_compact_signed_dtype(max(sources))),
        "topk_experts": topk_tensor,
        "count": torch.tensor(counts, dtype=_compact_signed_dtype(max(counts))),
    }


def save_compact_trace(
    path: str | Path,
    *,
    num_ranks: int,
    top_k: int,
    layers: Iterable[Mapping[str, Any]],
) -> None:
    if num_ranks < 1 or top_k < 1:
        raise ValueError("num_ranks and top_k must be positive")
    trace = {
        "format": COMPACT_TRACE_FORMAT,
        "num_ranks": int(num_ranks),
        "top_k": int(top_k),
        "layers": list(layers),
    }
    if not trace["layers"]:
        raise ValueError("cannot serialize a trace without layers")
    for index, layer in enumerate(trace["layers"]):
        topk = layer.get("topk_experts")
        if not isinstance(topk, torch.Tensor) or topk.ndim != 2:
            raise ValueError(f"invalid compact layer {index}")
        if topk.shape[1] != top_k:
            raise ValueError(f"compact layer {index} Top-K disagrees with metadata")
    torch.save(trace, Path(path))


def load_compact_trace(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    trace = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(trace, dict) or trace.get("format") != COMPACT_TRACE_FORMAT:
        raise ValueError(f"{path} is not an {COMPACT_TRACE_FORMAT} trace")
    num_ranks = trace.get("num_ranks")
    top_k_metadata = trace.get("top_k")
    if not isinstance(num_ranks, int) or num_ranks < 1:
        raise ValueError("compact trace num_ranks must be a positive integer")
    if not isinstance(top_k_metadata, int) or top_k_metadata < 1:
        raise ValueError("compact trace top_k must be a positive integer")
    layers = trace.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("compact trace must contain non-empty layers")
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"compact trace layer {index} must be a mapping")
        source = layer.get("source_rank")
        topk = layer.get("topk_experts")
        count = layer.get("count")
        if not all(isinstance(value, torch.Tensor) for value in (source, topk, count)):
            raise ValueError(f"compact trace layer {index} is missing tensors")
        if (
            source.device.type != "cpu"
            or topk.device.type != "cpu"
            or count.device.type != "cpu"
        ):
            raise ValueError("compact trace tensors must load on CPU")
        if source.ndim != 1 or topk.ndim != 2 or count.ndim != 1:
            raise ValueError(f"invalid tensor dimensions in compact layer {index}")
        if source.shape[0] != topk.shape[0] or count.shape[0] != topk.shape[0]:
            raise ValueError(f"bundle counts disagree in compact layer {index}")
        if source.dtype not in _INTEGER_DTYPES or topk.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"compact trace layer {index} tensors must be integer")
        if count.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"compact trace layer {index} count must be integer")
        if topk.shape[1] != top_k_metadata:
            raise ValueError(
                f"compact trace layer {index} Top-K disagrees with metadata"
            )
        if source.numel() == 0:
            raise ValueError(f"compact trace layer {index} must not be empty")
        if int(source.min().item()) < 0 or int(source.max().item()) >= num_ranks:
            raise ValueError(f"compact trace layer {index} has invalid source ranks")
        if int(topk.min().item()) < 0 or int(count.min().item()) < 1:
            raise ValueError(f"compact trace layer {index} has invalid values")
        if topk.shape[1] > 1 and bool((topk[:, 1:] <= topk[:, :-1]).any().item()):
            raise ValueError(
                f"compact trace layer {index} experts must be sorted and unique"
            )
    return trace
