#!/usr/bin/env python3
"""Measure DeepEP A2A time before and after an offline expert plan."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


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


def _baseline(num_experts: int, num_ranks: int) -> dict[int, tuple[int, ...]]:
    return {
        expert: (min(expert * num_ranks // num_experts, num_ranks - 1),)
        for expert in range(num_experts)
    }


def _physical_maps(
    layouts: Mapping[str, Mapping[int, Sequence[int]]],
    num_experts: int,
    num_ranks: int,
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
                replica = ranks.index(source) if source in ranks else 0
                mapping[source][expert] = physical_by_expert[expert][replica]
        maps[name] = mapping
    return slots, maps


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
        int(
            count[
                torch.any(destinations == rank, dim=1) & (source != rank)
            ].sum()
        )
        for rank in range(mapping.shape[0])
    )


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
    layouts = {"baseline": _baseline(num_experts, num_ranks), "plan": plan}
    slots, maps = _physical_maps(layouts, num_experts, num_ranks)
    return layer, gate, num_experts, num_ranks, slots, maps


def _sample(layer, rank: int, num_tokens: int, seed: int):
    import torch

    selected = layer["source_rank"] == rank
    if not bool(selected.any()):
        raise ValueError(f"trace layer contains no bundles for source rank {rank}")
    generator = torch.Generator().manual_seed(seed + rank)
    indexes = torch.multinomial(
        layer["count"][selected].double(),
        num_tokens,
        replacement=True,
        generator=generator,
    )
    return layer["topk_experts"][selected][indexes].long()


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
                dispatch.elapsed_time(combine)
                for _, _, dispatch, combine in samples
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

    layer, gate, num_experts, num_ranks, slots, maps = _load(args)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=local_rank,
        world_size=num_ranks,
    )
    group = dist.new_group(list(range(num_ranks)))
    physical_experts = slots * num_ranks
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
        num_qps_per_rank=args.num_qps_per_rank or deep_ep.Buffer.num_sms,
    )
    maps = {
        name: torch.tensor(mapping, dtype=torch.int64, device="cuda")
        for name, mapping in maps.items()
    }
    logical_topk = _sample(
        layer, local_rank, args.tokens_per_rank, args.seed
    ).cuda()
    x = torch.randn(
        (args.tokens_per_rank, args.hidden), dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.ones(
        logical_topk.shape, dtype=torch.float32, device="cuda"
    )
    results = {name: [] for name in maps}
    remote = {}
    for repeat in range(args.repeats):
        order = tuple(maps) if repeat % 2 == 0 else tuple(reversed(maps))
        for name in order:
            topk = maps[name][local_rank, logical_topk]
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
            f"qps/rank={args.num_qps_per_rank or deep_ep.Buffer.num_sms} "
            f"communication-speedup={summary['baseline'][3] / summary['plan'][3]:.3f}x",
            flush=True,
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="compact routing trace (.pt)")
    parser.add_argument("--plan", required=True, help="saved placement JSON")
    parser.add_argument("--layer", default="0", help="layer index or exact gate name")
    parser.add_argument("--num-ranks", type=int)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num-qps-per-rank",
        type=int,
        default=0,
        help="DeepEP QPs per rank; 0 uses SGLang's Buffer.num_sms default",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8361)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(
        args.hidden,
        args.tokens_per_rank,
        args.iterations,
        args.repeats,
    ) < 1 or min(args.warmups, args.num_qps_per_rank) < 0:
        parser.error("invalid benchmark parameters")
    layer, gate, num_experts, num_ranks, slots, maps = _load(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "gate": gate,
                    "num_ranks": num_ranks,
                    "num_experts": num_experts,
                    "physical_slots_per_rank": slots,
                    "remote": {
                        name: _remote(layer, mapping, slots)
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
