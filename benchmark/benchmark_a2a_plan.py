#!/usr/bin/env python3
"""Measure DeepEP A2A time before and after an offline expert plan."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


def _config_hidden_size(config: object) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError(f"model config has invalid hidden_size: {hidden_size!r}")
    return hidden_size


def _model_hidden_size(model: str, trust_remote_code: bool) -> int:
    from sglang.srt.utils.hf_transformers_utils import (
        get_config,
        get_hf_text_config,
    )

    config = get_hf_text_config(get_config(model, trust_remote_code))
    return _config_hidden_size(config)


def _layer(trace: Mapping[str, object], selector: str) -> Mapping[str, object]:
    layers = trace.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("trace contains no layers")
    if selector.isdigit():
        index = int(selector)
        if index >= len(layers):
            raise ValueError(f"layer index {index} is out of range")
        return layers[index]
    matches = [layer for layer in layers if layer.get("gate") == selector]
    if len(matches) != 1:
        raise ValueError(f"layer gate {selector!r} did not match exactly once")
    return matches[0]


def _normalize_plan(
    raw: Mapping[str, object], gate: str, num_experts: int, num_ranks: int
) -> dict[int, tuple[int, ...]]:
    values = raw.get(gate, raw)
    if isinstance(values, Mapping) and "replicas" in values:
        values = values["replicas"]
    if not isinstance(values, Mapping):
        raise ValueError("plan must contain an expert-to-rank mapping")
    result = {}
    for expert in range(num_experts):
        value = values.get(str(expert), values.get(expert))
        ranks = (value,) if isinstance(value, int) else tuple(value or ())
        if not ranks or any(
            not isinstance(rank, int) or not 0 <= rank < num_ranks for rank in ranks
        ):
            raise ValueError(f"plan has invalid or missing expert {expert}")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"plan has duplicate replicas for expert {expert}")
        result[expert] = ranks
    return result


def _normalize_routing(
    raw: Mapping[str, object], gate: str, num_experts: int, num_ranks: int
) -> list[list[int]] | None:
    values = raw.get(gate, raw)
    if not isinstance(values, Mapping) or "routing" not in values:
        return None
    routing = values["routing"]
    if (
        not isinstance(routing, list)
        or len(routing) != num_ranks
        or any(not isinstance(row, list) or len(row) != num_experts for row in routing)
        or any(
            not isinstance(rank, int) or not 0 <= rank < num_ranks
            for row in routing
            for rank in row
        )
    ):
        raise ValueError("plan contains invalid source-specific routing")
    return routing


def _normalize_quota(
    raw: Mapping[str, object],
    gate: str,
    plan: Mapping[int, Sequence[int]],
    num_experts: int,
    num_ranks: int,
) -> list[list[list[int]]] | None:
    values = raw.get(gate, raw)
    if not isinstance(values, Mapping) or "quota" not in values:
        return None
    quota = values["quota"]
    if quota is None:
        return None
    if (
        not isinstance(quota, list)
        or len(quota) != num_ranks
        or any(not isinstance(row, list) or len(row) != num_experts for row in quota)
        or any(
            not isinstance(values, list)
            or len(values) != len(plan[expert])
            or not all(isinstance(value, int) and value >= 0 for value in values)
            for row in quota
            for expert, values in enumerate(row)
        )
    ):
        raise ValueError("plan contains invalid quota routing")
    return quota


def _baseline(num_experts: int, num_ranks: int) -> dict[int, tuple[int, ...]]:
    return {
        expert: (min(expert * num_ranks // num_experts, num_ranks - 1),)
        for expert in range(num_experts)
    }


def _physical_maps(
    layouts: Mapping[str, Mapping[int, Sequence[int]]],
    num_experts: int,
    num_ranks: int,
    routes: Mapping[str, Sequence[Sequence[int]] | None] | None = None,
) -> tuple[int, dict[str, list[list[int]]]]:
    slots = max(
        sum(rank in ranks for ranks in layout.values())
        for layout in layouts.values()
        for rank in range(num_ranks)
    )
    maps = {}
    for name, layout in layouts.items():
        next_slot = [0] * num_ranks
        physical_by_expert = {}
        for expert in range(num_experts):
            physical = []
            for rank in layout[expert]:
                physical.append(rank * slots + next_slot[rank])
                next_slot[rank] += 1
            physical_by_expert[expert] = physical
        mapping = [[0] * num_experts for _ in range(num_ranks)]
        for source in range(num_ranks):
            for expert, ranks in layout.items():
                selected = routes and routes.get(name)
                target = selected[source][expert] if selected is not None else source
                replica = ranks.index(target) if target in ranks else 0
                mapping[source][expert] = physical_by_expert[expert][replica]
        maps[name] = mapping
    return slots, maps


def _physical_replicas(
    layout: Mapping[int, Sequence[int]], slots: int, num_ranks: int
) -> list[list[int]]:
    next_slot = [0] * num_ranks
    result = []
    for expert in range(len(layout)):
        physical = []
        for rank in layout[expert]:
            physical.append(rank * slots + next_slot[rank])
            next_slot[rank] += 1
        result.append(physical)
    return result


def _remote(
    layer: Mapping[str, object], mapping: Sequence[Sequence[int]], slots: int
) -> int:
    import torch

    mapping = torch.tensor(mapping, dtype=torch.int64)
    source = layer["source_rank"].long()
    topk = layer["topk_experts"].long()
    count = layer["count"].long()
    destinations = mapping[source[:, None], topk] // slots
    return sum(
        int(count[torch.any(destinations == rank, dim=1) & (source != rank)].sum())
        for rank in range(mapping.shape[0])
    )


def _quota_remote(
    layer: Mapping[str, object],
    physical_replicas: Sequence[Sequence[int]],
    quota: Sequence[Sequence[Sequence[int]]],
    slots: int,
    num_ranks: int,
) -> int:
    import numpy as np

    from sglang.srt.eplb.expert_affinity_graph import RoutedArrays
    from sglang.srt.eplb.grace_plus_replication import _route_quota, _source_demand

    arrays = RoutedArrays(
        layer["source_rank"].numpy(),
        layer["topk_experts"].numpy(),
        layer["count"].numpy(),
    )
    replicas = {
        expert: tuple(physical // slots for physical in values)
        for expert, values in enumerate(physical_replicas)
    }
    demand = _source_demand(arrays, len(replicas), num_ranks)
    dense_quota = np.zeros((num_ranks, len(replicas), num_ranks), dtype=np.int64)
    for source, row in enumerate(quota):
        for expert, values in enumerate(row):
            ranks = replicas[expert]
            order = sorted(
                range(len(ranks)), key=lambda index: (ranks[index] != source, index)
            )
            scaled = _scale_quota(
                [values[index] for index in order], int(demand[expert, source])
            )
            for index, value in zip(order, scaled):
                dense_quota[source, expert, ranks[index]] = value
    return _route_quota(arrays, dense_quota, demand, replicas).remote


def _load(args: argparse.Namespace):
    import torch

    trace = torch.load(
        Path(args.input), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(trace, Mapping) or not isinstance(trace.get("num_ranks"), int):
        raise ValueError("input is not a compact SGLang routing trace")
    num_ranks = trace["num_ranks"]
    if args.num_ranks is not None and args.num_ranks != num_ranks:
        raise ValueError("--num-ranks must match the trace")
    layer = _layer(trace, args.layer)
    gate = str(layer["gate"])
    num_experts = int(layer["topk_experts"].max()) + 1
    raw_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    plan = _normalize_plan(raw_plan, gate, num_experts, num_ranks)
    routing = _normalize_routing(raw_plan, gate, num_experts, num_ranks)
    quota = _normalize_quota(raw_plan, gate, plan, num_experts, num_ranks)
    if routing is not None:
        routing = [list(row) for row in routing]
        for source, row in enumerate(routing):
            for expert, target in enumerate(row):
                if target in plan[expert]:
                    continue
                if quota is None:
                    raise ValueError(
                        "routing selects a rank without that expert replica"
                    )
                row[expert] = source if source in plan[expert] else plan[expert][0]
    if quota is not None:
        quota = [[list(values) for values in row] for row in quota]
        for source, row in enumerate(quota):
            for expert, values in enumerate(row):
                if sum(values):
                    continue
                target = (
                    routing[source][expert]
                    if routing is not None
                    else source
                    if source in plan[expert]
                    else plan[expert][0]
                )
                values[plan[expert].index(target)] = 1
    layouts = {"baseline": _baseline(num_experts, num_ranks), "plan": plan}
    slots, maps = _physical_maps(
        layouts,
        num_experts,
        num_ranks,
        routes={"baseline": None, "plan": routing},
    )
    return (
        layer,
        gate,
        num_experts,
        num_ranks,
        slots,
        maps,
        quota,
        _physical_replicas(plan, slots, num_ranks),
    )


def _scale_quota(values: Sequence[int], amount: int) -> list[int]:
    total = sum(values)
    if total < 1:
        raise ValueError("quota must contain positive demand")
    numerators = [value * amount for value in values]
    result = [value // total for value in numerators]
    remainder = amount - sum(result)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(numerators[index] % total), index),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _quota_topk(
    logical_topk,
    physical_replicas,
    quota,
    source: int,
    slots: int,
    ordinals=None,
):
    """Apply the same source/expert prefix quota used by ``_route_quota``."""

    import torch

    result = torch.empty_like(logical_topk)
    for expert in torch.unique(logical_topk).tolist():
        selected = logical_topk == expert
        replicas = physical_replicas[expert]
        valid = replicas >= 0
        replicas = replicas[valid]
        local = replicas // slots == source
        order = torch.cat(
            (torch.nonzero(local).flatten(), torch.nonzero(~local).flatten())
        )
        replicas = replicas[order]
        values = quota[source, expert, : len(replicas)][order]
        if ordinals is None:
            amount = int(selected.sum())
            total = int(values.sum())
            if total < 1:
                raise ValueError("quota must contain positive demand")
            numerators = values * amount
            counts = torch.div(numerators, total, rounding_mode="floor")
            remainder = amount - int(counts.sum())
            if remainder:
                residual_order = torch.argsort(
                    numerators % total, descending=True, stable=True
                )
                counts[residual_order[:remainder]] += 1
            result[selected] = torch.repeat_interleave(replicas, counts)
            continue
        boundaries = torch.cumsum(values, dim=0)
        positions = ordinals[selected]
        replica_index = torch.searchsorted(boundaries, positions, right=True)
        result[selected] = replicas[replica_index]
    return result


def _sample(layer, rank: int, num_tokens: int, seed: int):
    import torch

    selected = layer["source_rank"] == rank
    if not bool(selected.any()):
        raise ValueError(f"trace layer contains no bundles for source rank {rank}")
    generator = torch.Generator().manual_seed(seed + rank)
    selected_indexes = torch.nonzero(selected).flatten()
    indexes = torch.multinomial(
        layer["count"][selected].double(),
        num_tokens,
        replacement=True,
        generator=generator,
    )
    bundle_indexes = selected_indexes[indexes]
    counts = layer["count"][bundle_indexes].long()
    generator = torch.Generator().manual_seed(seed + 7919 + rank)
    offsets = (
        torch.rand(counts.shape, generator=generator, dtype=torch.float64)
        * counts.double()
    ).long()
    bundle_topk = layer["topk_experts"][selected].long()
    prefix = torch.zeros_like(bundle_topk, dtype=torch.int64)
    running = torch.zeros(
        int(layer["topk_experts"].max().item()) + 1, dtype=torch.int64
    )
    for row, experts in enumerate(bundle_topk):
        for column, expert in enumerate(experts.tolist()):
            prefix[row, column] = running[expert]
            running[expert] += int(layer["count"][selected][row])
    ordinals = prefix[indexes] + offsets[:, None]
    return bundle_topk[indexes], ordinals


def _measure(
    buffer, x, topk, weights, num_experts, dispatch_config, combine_config, args
):
    import torch
    import torch.distributed as dist

    def step(events=None):
        if events:
            events[0].record()
        layout = buffer.get_dispatch_layout(topk, num_experts)
        if events:
            events[1].record()
        dispatch_args = dict(
            x=x,
            topk_idx=topk,
            topk_weights=weights,
            num_tokens_per_rank=layout[0],
            is_token_in_rank=layout[3],
            num_tokens_per_expert=layout[2],
            config=dispatch_config,
        )
        if layout[1] is not None:
            dispatch_args["num_tokens_per_rdma_rank"] = layout[1]
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
        if events:
            events[2].record()
        buffer.combine(x=recv_x, handle=handle, config=combine_config)
        if events:
            events[3].record()

    dist.barrier()
    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
        step(events)
        samples.append(events)
    torch.cuda.synchronize()
    local = torch.tensor(
        [
            statistics.mean(
                start.elapsed_time(layout) for start, layout, _, _ in samples
            ),
            statistics.mean(
                layout.elapsed_time(dispatch) for _, layout, dispatch, _ in samples
            ),
            statistics.mean(
                dispatch.elapsed_time(combine) for _, _, dispatch, combine in samples
            ),
            statistics.mean(
                start.elapsed_time(combine) for start, _, _, combine in samples
            ),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return local.cpu().tolist()


def _worker(local_rank: int, args: argparse.Namespace) -> None:
    import deep_ep
    import torch
    import torch.distributed as dist

    layer, gate, num_experts, num_ranks, slots, maps, quota, replicas = _load(args)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=local_rank,
        world_size=num_ranks,
    )
    group = dist.new_group(list(range(num_ranks)))
    physical_experts = slots * num_ranks
    if args.num_sms:
        deep_ep.Buffer.set_num_sms(args.num_sms)
    dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine_config = deep_ep.Buffer.get_combine_config(num_ranks)
    hidden_bytes = args.hidden * 2
    nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks),
    )
    buffer = deep_ep.Buffer(
        group,
        nvl_bytes,
        0,
        low_latency_mode=False,
        num_qps_per_rank=deep_ep.Buffer.num_sms,
    )
    maps = {
        name: torch.tensor(mapping, dtype=torch.int64, device="cuda")
        for name, mapping in maps.items()
    }
    max_replicas = max(map(len, replicas))
    quota = (
        torch.tensor(
            [
                [values + [0] * (max_replicas - len(values)) for values in row]
                for row in quota
            ],
            dtype=torch.int64,
            device="cuda",
        )
        if quota
        else None
    )
    replicas = torch.tensor(
        [row + [-1] * (max_replicas - len(row)) for row in replicas],
        dtype=torch.int64,
        device="cuda",
    )
    logical_topk, ordinals = _sample(
        layer, local_rank, args.tokens_per_rank, args.seed
    )
    logical_topk = logical_topk.cuda()
    ordinals = ordinals.cuda()
    x = torch.randn(
        (args.tokens_per_rank, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.ones(logical_topk.shape, dtype=torch.float32, device="cuda")
    results = {name: [] for name in maps}
    remote = {}
    for repeat in range(args.repeats):
        order = tuple(maps) if repeat % 2 == 0 else tuple(reversed(maps))
        for name in order:
            topk = (
                _quota_topk(
                    logical_topk,
                    replicas,
                    quota,
                    local_rank,
                    slots,
                    ordinals,
                )
                if name == "plan" and quota is not None
                else maps[name][local_rank, logical_topk]
            )
            destinations = topk // slots
            local_remote = torch.zeros((), dtype=torch.int64, device="cuda")
            for rank in range(num_ranks):
                if rank != local_rank:
                    local_remote += torch.any(destinations == rank, dim=1).sum()
            dist.all_reduce(local_remote)
            remote[name] = int(local_remote)
            results[name].append(
                _measure(
                    buffer,
                    x,
                    topk,
                    weights,
                    physical_experts,
                    dispatch_config,
                    combine_config,
                    args,
                )
            )
    if local_rank == 0:
        summary = {
            name: [statistics.median(sample[i] for sample in samples) for i in range(4)]
            for name, samples in results.items()
        }
        print(
            "method    remote  layout-us  dispatch-us  combine-us  total-us  A2A-MiB",
            flush=True,
        )
        for name, times in summary.items():
            mib = remote[name] * args.hidden * 2 * 2 / (1024**2)
            print(
                f"{name:<9} {remote[name]:>7}  "
                f"{times[0] * 1000:>9.2f}  {times[1] * 1000:>11.2f}  "
                f"{times[2] * 1000:>10.2f}  {times[3] * 1000:>8.2f}  {mib:>7.2f}",
                flush=True,
            )
        print(
            f"[result] gate={gate} slots/rank={slots} "
            f"num-sms={deep_ep.Buffer.num_sms} "
            f"communication-speedup={summary['baseline'][3] / summary['plan'][3]:.3f}x",
            flush=True,
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="compact routing trace (.pt)")
    parser.add_argument("--plan", required=True, help="saved placement JSON")
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        required=True,
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--layer", default="0", help="layer index or exact gate name")
    parser.add_argument("--num-ranks", type=int)
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num-sms",
        type=int,
        default=0,
        help="DeepEP communication SMs; 0 uses the DeepEP default",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8361)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.hidden = _model_hidden_size(args.model, args.trust_remote_code)
    except Exception as error:
        parser.error(f"could not read model hidden_size: {error}")
    if (
        min(
            args.hidden,
            args.tokens_per_rank,
            args.iterations,
            args.repeats,
        )
        < 1
        or min(args.warmups, args.num_sms) < 0
    ):
        parser.error("invalid benchmark parameters")
    layer, gate, num_experts, num_ranks, slots, maps, quota, replicas = _load(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "gate": gate,
                    "num_ranks": num_ranks,
                    "num_experts": num_experts,
                    "physical_slots_per_rank": slots,
                    "remote": {
                        name: (
                            _quota_remote(layer, replicas, quota, slots, num_ranks)
                            if name == "plan" and quota is not None
                            else _remote(layer, mapping, slots)
                        )
                        for name, mapping in maps.items()
                    },
                },
                indent=2,
            )
        )
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < num_ranks:
        parser.error(f"benchmark requires {num_ranks} local CUDA devices")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=num_ranks)


if __name__ == "__main__":
    main()
