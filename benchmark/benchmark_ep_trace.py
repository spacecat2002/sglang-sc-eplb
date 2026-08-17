#!/usr/bin/env python3
"""Collect MoE routing traces for Pairwise and GRACE-MoE placement.

The script runs a Hugging Face MoE model, captures each router's real Top-K
experts and aggregates identical ``(source_rank, Top-K)`` bundles for the
Pairwise and GRACE-MoE placement comparison.

It is an offline what-if tool.  ``--num-ranks`` and ``--source-rank-mode``
simulate how input tokens are sharded across EP ranks; they do not turn a
single-GPU Transformers forward pass into a distributed DeepEP measurement.

Example:
  python benchmark/benchmark_ep_trace.py \
      --model Qwen/Qwen3-30B-A3B --dataset sharegpt \
      --num-samples 128 --max-length 1024 --batch-size 2 \
      --num-ranks 8 --ranks-per-node 8 \
      --bundles-output /tmp/moe_bundles.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch

from bench_next_layer_gate_topm import (
    _batches,
    _discover_gates,
    _first_tensor,
    _flatten_activation,
    _infer_top_k,
    _load_model,
    _read_texts,
)
from sglang.srt.eplb.co_routing_graph_solver import RoutedToken
from sglang.srt.eplb.moe_bundle_trace import (
    compact_layer_from_bundles,
    save_compact_trace,
)


def _home_rank(expert: int, num_experts: int, num_ranks: int, mode: str) -> int:
    if mode == "round-robin":
        return expert % num_ranks
    experts_per_rank = (num_experts + num_ranks - 1) // num_ranks
    return min(expert // experts_per_rank, num_ranks - 1)


def _load_home_placements(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("--home-placement-json must contain a JSON object")
    return raw


def _homes_for_gate(
    gate_name: str,
    observed_experts: Sequence[int],
    args: argparse.Namespace,
    home_placements: Mapping[str, Any],
    num_ranks: int,
) -> Dict[int, int]:
    """Resolve either a global map or a per-gate map from a JSON file."""
    candidate: Any = home_placements.get(gate_name, home_placements)
    if candidate and not all(isinstance(value, int) for value in candidate.values()):
        raise ValueError(
            "--home-placement-json must be {expert: rank} or "
            "{gate_name: {expert: rank}}"
        )
    homes = {
        expert: _home_rank(
            expert, max(observed_experts) + 1, num_ranks, args.home_placement
        )
        for expert in observed_experts
    }
    for expert in observed_experts:
        value = candidate.get(str(expert), candidate.get(expert)) if candidate else None
        if value is not None:
            if not isinstance(value, int) or not 0 <= value < num_ranks:
                raise ValueError(
                    f"invalid home rank {value!r} for {gate_name} expert {expert}"
                )
            homes[expert] = value
    return homes


def _source_ranks(
    attention_mask: torch.Tensor, num_ranks: int, mode: str, start: int
) -> torch.Tensor:
    """Return one simulated source EP rank per non-padding token."""
    batch_size, sequence_length = attention_mask.shape
    if mode == "prompt-round-robin":
        ranks = (
            torch.arange(batch_size, device=attention_mask.device) + start
        ) % num_ranks
        flattened = ranks[:, None].expand(-1, sequence_length).reshape(-1)
    else:
        # Assign only valid tokens in the global stream. Padding positions
        # must not consume a rank slot when batches contain variable lengths.
        valid_count = int(attention_mask.sum().item())
        return (
            torch.arange(valid_count, device=attention_mask.device) + start
        ) % num_ranks
    return flattened[attention_mask.reshape(-1).bool()]


def _node_of(rank: int, ranks_per_node: int | None) -> int:
    return rank if ranks_per_node is None else rank // ranks_per_node


def _link_cost(
    source_rank: int, destination_rank: int, args: argparse.Namespace
) -> float:
    if _node_of(source_rank, args.ranks_per_node) == _node_of(
        destination_rank, args.ranks_per_node
    ):
        return 1.0
    return args.rdma_cost


def _route_metrics(
    routed_tokens: Sequence[RoutedToken],
    homes: Mapping[int, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compute rank-set traffic directly, without constructing a planner."""
    compute = [0] * args.num_ranks
    send = [0.0] * args.num_ranks
    recv = [0.0] * args.num_ranks
    remote_copies = 0
    nvl_traffic = 0
    rdma_traffic = 0
    for token in routed_tokens:
        for expert in token.topk_experts:
            compute[homes[expert]] += token.count
        destinations = {homes[expert] for expert in token.topk_experts}
        remote_destinations = {
            destination
            for destination in destinations
            if destination != token.source_rank
        }
        remote_copies += token.count * len(remote_destinations)
        for destination in remote_destinations:
            cost = _link_cost(token.source_rank, destination, args)
            send[token.source_rank] += token.count * cost
            recv[destination] += token.count * cost
            if _node_of(token.source_rank, args.ranks_per_node) == _node_of(
                destination, args.ranks_per_node
            ):
                nvl_traffic += token.count
            else:
                rdma_traffic += token.count
    communication = [max(outbound, inbound) for outbound, inbound in zip(send, recv)]
    return {
        "compute_load": compute,
        "send_load": send,
        "recv_load": recv,
        "communication_load": communication,
        "weighted_communication": sum(send),
        "unique_remote_rank_copies": remote_copies,
        "nvl_traffic": nvl_traffic,
        "rdma_traffic": rdma_traffic,
        "compute_imbalance": _max_over_avg(compute),
        "communication_imbalance": _max_over_avg(communication),
    }


def _max_over_avg(values: Sequence[float]) -> float:
    average = sum(values) / max(len(values), 1)
    return max(values, default=0.0) / average if average else 0.0


def _format_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows
    )
    return "\n".join(lines)


def _layer_result(
    gate_name: str,
    bundles: Mapping[Tuple[int, Tuple[int, ...]], int],
    args: argparse.Namespace,
    home_placements: Mapping[str, Any],
) -> Dict[str, Any]:
    routed_tokens = [
        RoutedToken(source_rank, experts, count)
        for (source_rank, experts), count in sorted(bundles.items())
    ]
    observed_experts = sorted(
        {expert for token in routed_tokens for expert in token.topk_experts}
    )
    homes = _homes_for_gate(
        gate_name, observed_experts, args, home_placements, args.num_ranks
    )
    metrics_start = time.perf_counter()
    metrics = _route_metrics(routed_tokens, homes, args)
    metrics_seconds = time.perf_counter() - metrics_start
    return {
        "gate": gate_name,
        "tokens": sum(token.count for token in routed_tokens),
        "bundles": len(routed_tokens),
        "solve_seconds": metrics_seconds,
        "metrics_seconds": metrics_seconds,
        "homes": homes,
        "metrics": metrics,
    }


def _format_result(result: Mapping[str, Any]) -> str:
    lines = [
        "MoE routing trace for Pairwise/GRACE placement (DeepEP rank-set dedup)",
        (
            f"model={result['model']}  dataset={result['dataset']}  prompts={result['num_prompts']}  "
            f"K={result['top_k']}  simulated_ep={result['num_ranks']}  "
            f"source={result['source_rank_mode']}"
        ),
        "remote: token-to-remote-rank copies; comp/comm: max-to-average rank load.",
    ]
    rows = []
    for layer in result["layers"]:
        metrics = layer["metrics"]
        rows.append(
            [
                layer["gate"],
                str(layer["tokens"]),
                str(layer["bundles"]),
                str(metrics["unique_remote_rank_copies"]),
                f"{metrics['weighted_communication']:.0f}",
                f"{metrics['compute_imbalance']:.2f}",
                f"{metrics['communication_imbalance']:.2f}",
            ]
        )
    lines.append(
        _format_table(
            [
                "gate",
                "tokens",
                "bundles",
                "remote_copies",
                "weighted",
                "comp",
                "comm",
            ],
            rows,
        )
    )
    return "\n".join(lines)


@torch.inference_mode()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer = _load_model(args.model, args.revision, args.dtype, args.device)
    if args.top_k is None:
        args.top_k = _infer_top_k(model)
    gates = _discover_gates(model, args.gate_pattern)
    print(
        f"[setup] found {len(gates)} router gates; Top-K={args.top_k}; "
        f"simulated_ep={args.num_ranks}",
        flush=True,
    )
    captured: Dict[str, torch.Tensor] = {}
    capture_enabled = [True]
    handles = []
    for name, gate in gates:

        def post_hook(_module, _inputs, output, gate_name=name):
            if not capture_enabled[0]:
                return
            tensor = _first_tensor(output)
            if tensor is not None and tensor.ndim >= 2:
                captured[gate_name] = _flatten_activation(tensor).detach()

        handles.append(gate.register_forward_hook(post_hook))

    print("[data] loading prompts", flush=True)
    texts = _read_texts(args)
    if not texts:
        raise ValueError("the selected dataset did not contain any usable prompts")
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(
        f"[data] ready: prompts={len(texts)} batches={total_batches} "
        f"batch_size={args.batch_size} max_length={args.max_length}",
        flush=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bundles_by_gate: Dict[str, Counter[Tuple[int, Tuple[int, ...]]]] = {
        name: Counter() for name, _ in gates
    }
    home_placements = _load_home_placements(args.home_placement_json)
    shard_offset = 0
    captured_tokens = 0
    try:
        for batch_index, text_batch in enumerate(
            _batches(texts, args.batch_size), start=1
        ):
            captured.clear()
            batch_start = time.perf_counter()
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            valid_token_count = int(encoded["attention_mask"].sum().item())
            source_ranks = _source_ranks(
                encoded["attention_mask"],
                args.num_ranks,
                args.source_rank_mode,
                shard_offset,
            )
            shard_offset += (
                valid_token_count
                if args.source_rank_mode == "token-round-robin"
                else len(text_batch)
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            for name, logits in captured.items():
                padded_token_count = encoded["attention_mask"].numel()
                valid_token_count_for_gate = source_ranks.numel()
                if logits.shape[0] == padded_token_count:
                    # Many HF MoE routers run on the padded [B, S, H] tensor
                    # and emit logits for padding rows as well.
                    valid_mask = encoded["attention_mask"].reshape(-1).bool()
                    logits = logits[valid_mask.to(logits.device)]
                if logits.shape[0] != valid_token_count_for_gate:
                    raise RuntimeError(
                        f"{name} emitted {logits.shape[0]} usable router rows; "
                        f"expected {valid_token_count_for_gate} non-padding rows "
                        f"or {padded_token_count} padded rows"
                    )
                topk = logits.float().topk(args.top_k, dim=-1).indices.cpu().tolist()
                for source_rank, experts in zip(source_ranks.cpu().tolist(), topk):
                    bundles_by_gate[name][(source_rank, tuple(sorted(experts)))] += 1
            captured_tokens += valid_token_count
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % args.log_interval == 0
            ):
                bundles_per_layer = [
                    len(bundles) for bundles in bundles_by_gate.values()
                ]
                bundle_count = sum(bundles_per_layer)
                print(
                    f"[capture] batch={batch_index}/{total_batches} "
                    f"tokens={valid_token_count} total_tokens={captured_tokens} "
                    f"layer_bundles=sum={bundle_count} "
                    f"range={min(bundles_per_layer, default=0)}-"
                    f"{max(bundles_per_layer, default=0)} "
                    f"elapsed={time.perf_counter() - batch_start:.1f}s",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    nonempty_layers = [
        (name, bundles) for name, bundles in bundles_by_gate.items() if bundles
    ]
    if args.bundles_output:
        output_path = Path(args.bundles_output)
        if output_path.suffix.lower() in {".pt", ".pth"}:
            compact_layers = [
                compact_layer_from_bundles(name, bundles)
                for name, bundles in nonempty_layers
            ]
            save_compact_trace(
                output_path,
                num_ranks=args.num_ranks,
                top_k=args.top_k,
                layers=compact_layers,
            )
            output_kind = "compact tensor"
        else:
            trace = {
                "num_ranks": args.num_ranks,
                "top_k": args.top_k,
                "layers": [
                    {
                        "gate": name,
                        "bundles": [
                            {
                                "source_rank": source_rank,
                                "topk_experts": list(experts),
                                "count": count,
                            }
                            for (source_rank, experts), count in sorted(bundles.items())
                        ],
                    }
                    for name, bundles in nonempty_layers
                ],
            }
            output_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
            output_kind = "JSON"
        print(
            f"[data] saved {output_kind} Top-K bundles to {args.bundles_output}",
            flush=True,
        )
    print(f"[metrics] replaying {len(nonempty_layers)} router layers", flush=True)
    layers = []
    for layer_index, (name, bundles) in enumerate(nonempty_layers, start=1):
        plan_start = time.perf_counter()
        print(
            f"[metrics] layer={layer_index}/{len(nonempty_layers)} gate={name} "
            f"bundles={len(bundles)}",
            flush=True,
        )
        layer = _layer_result(name, bundles, args, home_placements)
        layers.append(layer)
        print(
            f"[metrics] layer={layer_index}/{len(nonempty_layers)} "
            f"remote={layer['metrics']['unique_remote_rank_copies']} "
            f"comp={layer['metrics']['compute_imbalance']:.2f} "
            f"comm={layer['metrics']['communication_imbalance']:.2f} "
            f"elapsed={time.perf_counter() - plan_start:.3f}s",
            flush=True,
        )
    total_solve_seconds = sum(layer["solve_seconds"] for layer in layers)
    print(f"[metrics] total replay time={total_solve_seconds:.3f}s", flush=True)
    return {
        "model": args.model,
        "dataset": args.dataset or "text",
        "dataset_path": args.dataset_path,
        "num_prompts": len(texts),
        "top_k": args.top_k,
        "num_ranks": args.num_ranks,
        "source_rank_mode": args.source_rank_mode,
        "total_replay_seconds": total_solve_seconds,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face model id or local path"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None, help="one prompt per line")
    parser.add_argument(
        "--dataset", default=None, help="dataset alias or Hugging Face dataset repo"
    )
    parser.add_argument(
        "--dataset-path", default=None, help="local .json or .jsonl file"
    )
    parser.add_argument(
        "--dataset-name", default=None, help="deprecated alias for --dataset"
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--prompt-fields", nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="print capture progress every N batches (default: 1)",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-pattern", default=r"(?:^|\.)(?:gate|router)$")
    parser.add_argument(
        "--num-ranks", type=int, required=True, help="simulated EP world size"
    )
    parser.add_argument("--ranks-per-node", type=int, default=None)
    parser.add_argument("--rdma-cost", type=float, default=4.0)
    parser.add_argument(
        "--home-placement", choices=["round-robin", "contiguous"], default="round-robin"
    )
    parser.add_argument(
        "--home-placement-json",
        default=None,
        help="optional {expert: rank} or {gate_name: {expert: rank}} JSON placement",
    )
    parser.add_argument(
        "--source-rank-mode",
        choices=["token-round-robin", "prompt-round-robin"],
        default="prompt-round-robin",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output", default=None, help="write complete JSON results to this path"
    )
    parser.add_argument(
        "--bundles-output",
        default=None,
        help=(
            "save aggregated bundles for graph replay; .pt/.pth writes the fast "
            "compact tensor format, other suffixes write JSON"
        ),
    )
    args = parser.parse_args()
    if args.num_ranks < 1:
        parser.error("--num-ranks must be positive")
    if args.log_interval < 1:
        parser.error("--log-interval must be positive")
    result = run(args)
    output = json.dumps(result, indent=2)
    print(output if args.json else _format_result(result))
    if args.output:
        Path(args.output).write_text(output + "\n")


if __name__ == "__main__":
    main()
