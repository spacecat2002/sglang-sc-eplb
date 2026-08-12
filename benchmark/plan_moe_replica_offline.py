#!/usr/bin/env python3
"""Replay real MoE routes from offline inference and plan expert replicas.

The script runs a Hugging Face MoE model, captures each router's real Top-K
experts, aggregates identical ``(source_rank, Top-K)`` bundles, and replays
them through :class:`BundleAwareReplicaPlanner`.  It models DeepEP normal
mode: a token is transferred once for every *destination rank*, even if
multiple selected experts live on that rank.

It is an offline what-if tool.  ``--num-ranks`` and ``--source-rank-mode``
simulate how input tokens are sharded across EP ranks; they do not turn a
single-GPU Transformers forward pass into a distributed DeepEP measurement.

Example:
  python benchmark/plan_moe_replica_offline.py \
      --model Qwen/Qwen3-30B-A3B --dataset sharegpt \
      --num-samples 128 --max-length 1024 --batch-size 2 \
      --num-ranks 8 --ranks-per-node 8 --replica-slots-per-rank 8 \
      --max-actions 32 --compute-weight 1 --communication-weight 1
"""

from __future__ import annotations

import argparse
import json
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
from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    PlanMetrics,
    RoutedToken,
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
            expert, max(observed_experts) + 1, args.num_ranks, args.home_placement
        )
        for expert in observed_experts
    }
    for expert in observed_experts:
        value = candidate.get(str(expert), candidate.get(expert)) if candidate else None
        if value is not None:
            if not isinstance(value, int) or not 0 <= value < args.num_ranks:
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
        flattened = (
            torch.arange(batch_size * sequence_length, device=attention_mask.device)
            + start
        ) % num_ranks
    return flattened[attention_mask.reshape(-1).bool()]


def _metrics_dict(metrics: PlanMetrics) -> Dict[str, Any]:
    return {
        "compute_load": metrics.compute_load,
        "send_load": metrics.send_load,
        "recv_load": metrics.recv_load,
        "communication_load": metrics.communication_load,
        "weighted_communication": metrics.weighted_communication,
        "unique_remote_rank_copies": metrics.unique_remote_rank_copies,
        "nvl_traffic": metrics.nvl_traffic,
        "rdma_traffic": metrics.rdma_traffic,
        "objective": metrics.objective,
    }


def _max_over_avg(values: Sequence[float]) -> float:
    average = sum(values) / max(len(values), 1)
    return max(values, default=0.0) / average if average else 0.0


def _percent_change(before: float, after: float) -> str:
    if before == 0:
        return "n/a" if after else "0.0%"
    return f"{(after / before - 1.0):+.1%}"


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
    homes = _homes_for_gate(gate_name, observed_experts, args, home_placements)
    planner = BundleAwareReplicaPlanner(
        num_ranks=args.num_ranks,
        baseline_rank_by_expert=homes,
        replica_slots_per_rank=args.replica_slots_per_rank,
        ranks_per_node=args.ranks_per_node,
        rdma_cost=args.rdma_cost,
        compute_weight=args.compute_weight,
        communication_weight=args.communication_weight,
        max_bundle_size=args.max_bundle_size,
    )
    plan = planner.plan(routed_tokens, max_actions=args.max_actions)
    return {
        "gate": gate_name,
        "tokens": sum(token.count for token in routed_tokens),
        "bundles": len(routed_tokens),
        "homes": homes,
        "actions": [
            {
                "destination_rank": action.destination_rank,
                "experts": list(action.experts),
                "kind": action.kind,
            }
            for action in plan.actions
        ],
        "baseline": _metrics_dict(plan.baseline),
        "planned": _metrics_dict(plan.final),
        "replicas_by_rank": {
            str(rank): sorted(experts)
            for rank, experts in plan.replicas_by_rank.items()
        },
    }


def _format_result(result: Mapping[str, Any], show_actions: bool) -> str:
    lines = [
        "MoE replica planner replay (DeepEP normal-mode rank-set dedup)",
        (
            f"model={result['model']}  dataset={result['dataset']}  prompts={result['num_prompts']}  "
            f"K={result['top_k']}  simulated_ep={result['num_ranks']}  "
            f"source={result['source_rank_mode']}"
        ),
        "remote: token-to-remote-rank copies; comp/comm: max-to-average rank load.",
    ]
    rows = []
    for layer in result["layers"]:
        baseline = layer["baseline"]
        planned = layer["planned"]
        rows.append(
            [
                layer["gate"],
                str(layer["tokens"]),
                str(layer["bundles"]),
                str(len(layer["actions"])),
                f"{baseline['unique_remote_rank_copies']}->{planned['unique_remote_rank_copies']} "
                f"({_percent_change(baseline['unique_remote_rank_copies'], planned['unique_remote_rank_copies'])})",
                f"{baseline['weighted_communication']:.0f}->{planned['weighted_communication']:.0f}",
                f"{_max_over_avg(baseline['compute_load']):.2f}->{_max_over_avg(planned['compute_load']):.2f}",
                f"{_max_over_avg(baseline['communication_load']):.2f}->{_max_over_avg(planned['communication_load']):.2f}",
                f"{baseline['objective']:.3f}->{planned['objective']:.3f}",
            ]
        )
    lines.append(
        _format_table(
            [
                "gate",
                "tokens",
                "bundles",
                "copies",
                "remote",
                "weighted",
                "comp",
                "comm",
                "objective",
            ],
            rows,
        )
    )
    if show_actions:
        for layer in result["layers"]:
            if not layer["actions"]:
                continue
            lines.append(f"\n{layer['gate']} replica actions")
            lines.extend(
                f"  {action['kind']}: experts={action['experts']} -> rank {action['destination_rank']}"
                for action in layer["actions"]
            )
    return "\n".join(lines)


@torch.inference_mode()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer = _load_model(args.model, args.revision, args.dtype, args.device)
    if args.top_k is None:
        args.top_k = _infer_top_k(model)
    gates = _discover_gates(model, args.gate_pattern)
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

    texts = _read_texts(args)
    if not texts:
        raise ValueError("the selected dataset did not contain any usable prompts")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bundles_by_gate: Dict[str, Counter[Tuple[int, Tuple[int, ...]]]] = {
        name: Counter() for name, _ in gates
    }
    home_placements = _load_home_placements(args.home_placement_json)
    shard_offset = 0
    try:
        for text_batch in _batches(texts, args.batch_size):
            captured.clear()
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            source_ranks = _source_ranks(
                encoded["attention_mask"],
                args.num_ranks,
                args.source_rank_mode,
                shard_offset,
            )
            shard_offset += (
                encoded["attention_mask"].numel()
                if args.source_rank_mode == "token-round-robin"
                else len(text_batch)
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
            for name, logits in captured.items():
                if logits.shape[0] != source_ranks.numel():
                    raise RuntimeError(
                        f"{name} emitted {logits.shape[0]} router rows for "
                        f"{source_ranks.numel()} non-padding tokens"
                    )
                topk = logits.float().topk(args.top_k, dim=-1).indices.cpu().tolist()
                for source_rank, experts in zip(source_ranks.cpu().tolist(), topk):
                    bundles_by_gate[name][(source_rank, tuple(sorted(experts)))] += 1
    finally:
        for handle in handles:
            handle.remove()

    layers = [
        _layer_result(name, bundles, args, home_placements)
        for name, bundles in bundles_by_gate.items()
        if bundles
    ]
    return {
        "model": args.model,
        "dataset": args.dataset or "text",
        "dataset_path": args.dataset_path,
        "num_prompts": len(texts),
        "top_k": args.top_k,
        "num_ranks": args.num_ranks,
        "source_rank_mode": args.source_rank_mode,
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
    parser.add_argument("--replica-slots-per-rank", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-bundle-size", type=int, default=None)
    parser.add_argument("--compute-weight", type=float, default=1.0)
    parser.add_argument("--communication-weight", type=float, default=1.0)
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
    parser.add_argument("--show-actions", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output", default=None, help="write complete JSON results to this path"
    )
    args = parser.parse_args()
    if args.num_ranks < 1:
        parser.error("--num-ranks must be positive")
    result = run(args)
    output = json.dumps(result, indent=2)
    print(output if args.json else _format_result(result, args.show_actions))
    if args.output:
        Path(args.output).write_text(output + "\n")


if __name__ == "__main__":
    main()
