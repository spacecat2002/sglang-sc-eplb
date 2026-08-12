#!/usr/bin/env python3
"""Train residual next-layer MoE gate predictors for Top-M expert prefetching.

Each predictor learns from the previous MoE gate input ``h[l]`` while the
teacher is the real next gate output ``gate[l + 1](h[l + 1])``:

    predicted_logits = frozen_gate[l + 1](h[l]) + low_rank_residual(h[l])

The base model and every original router remain frozen.  The checkpoint only
contains the small residual predictors, so it can be loaded independently by
a future SGLang integration.

Example:
  python benchmark/train_next_layer_gate_predictor.py \
      --model Qwen/Qwen3-30B-A3B \
      --dataset sharegpt --dataset-path /data/ShareGPT.json \
      --num-samples 2000 --max-length 1024 --batch-size 1 \
      --top-m 16 --rank 64 --epochs 3 \
      --checkpoint-out next_gate_predictor.pt
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from bench_next_layer_gate_topm import (
    _batches,
    _discover_gates,
    _first_tensor,
    _flatten_activation,
    _infer_top_k,
    _load_model,
    _read_texts,
)


class LowRankGateResidual(nn.Module):
    """RMS-normalized low-rank residual, initialized to preserve the baseline."""

    def __init__(self, hidden_size: int, num_experts: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.up(F.silu(self.down(x)))


@dataclass
class PairBatch:
    index: int
    previous: torch.Tensor
    teacher_logits: torch.Tensor


def _valid_pair_batches(
    gates: Sequence[Tuple[str, nn.Module]],
    activations: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
    valid_tokens: torch.Tensor,
    pair_indices: Optional[set[int]],
) -> Iterable[PairBatch]:
    for index, ((name0, _), (name1, _)) in enumerate(zip(gates, gates[1:])):
        if pair_indices is not None and index not in pair_indices:
            continue
        if name0 not in activations or name1 not in logits:
            continue
        previous = activations[name0]
        teacher_logits = logits[name1]
        token_count = min(previous.shape[0], teacher_logits.shape[0])
        previous = previous[:token_count]
        teacher_logits = teacher_logits[:token_count]
        if valid_tokens.numel() == token_count:
            previous = previous[valid_tokens]
            teacher_logits = teacher_logits[valid_tokens]
        if previous.numel():
            yield PairBatch(index, previous, teacher_logits)


def _capture_hooks(
    gates: Sequence[Tuple[str, nn.Module]],
    activations: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
    enabled: List[bool],
) -> List[Any]:
    handles = []
    for name, gate in gates:

        def pre_hook(_module, inputs, gate_name=name):
            if not enabled[0]:
                return
            tensor = _first_tensor(inputs)
            if tensor is not None:
                activations[gate_name] = _flatten_activation(tensor).detach()

        def post_hook(_module, _inputs, output, gate_name=name):
            if not enabled[0]:
                return
            tensor = _first_tensor(output)
            if tensor is not None and tensor.ndim >= 2:
                logits[gate_name] = _flatten_activation(tensor).detach().float()

        handles.extend(
            [
                gate.register_forward_pre_hook(pre_hook),
                gate.register_forward_hook(post_hook),
            ]
        )
    return handles


def _build_predictors(
    pair_batches: Iterable[PairBatch], args: argparse.Namespace
) -> nn.ModuleDict:
    predictors = nn.ModuleDict()
    for pair in pair_batches:
        predictors[str(pair.index)] = LowRankGateResidual(
            pair.previous.shape[-1],
            pair.teacher_logits.shape[-1],
            args.rank,
            args.rms_norm_eps,
        ).to(args.device)
    if not predictors:
        raise RuntimeError("no usable adjacent gate pairs were captured")
    return predictors


def _losses(
    predicted_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    top_m: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    num_experts = predicted_logits.shape[-1]
    teacher_topk = teacher_logits.topk(top_k, dim=-1).indices
    positive_logits = predicted_logits.gather(1, teacher_topk)

    # The (M+1)-th score is the boundary an actual Top-K expert must cross.
    if top_m < num_experts:
        boundary = predicted_logits.topk(top_m + 1, dim=-1).values[:, -1].detach()
        rank_loss = F.softplus(boundary.unsqueeze(-1) + args.margin - positive_logits).mean()
    else:
        rank_loss = predicted_logits.new_zeros(())

    temperature = args.temperature
    distill_loss = F.kl_div(
        F.log_softmax(predicted_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2

    target_load = torch.zeros_like(predicted_logits)
    target_load.scatter_(1, teacher_topk, 1.0 / top_k)
    load_loss = F.mse_loss(
        F.softmax(predicted_logits / temperature, dim=-1).mean(dim=0),
        target_load.mean(dim=0),
    )
    loss = (
        args.rank_loss_weight * rank_loss
        + args.distill_loss_weight * distill_loss
        + args.load_loss_weight * load_loss
    )
    return loss, {
        "rank": rank_loss.detach().item(),
        "distill": distill_loss.detach().item(),
        "load": load_loss.detach().item(),
    }


def _predict_logits(
    gate: nn.Module, predictor: nn.Module, previous: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        base_logits = _first_tensor(gate(previous))
    if base_logits is None:
        raise RuntimeError("could not extract logits from a gate output")
    return _flatten_activation(base_logits).float() + predictor(previous)


def _split_texts(texts: List[str], validation_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    validation_count = int(len(texts) * validation_ratio)
    if validation_ratio > 0 and len(texts) > 1:
        validation_count = max(validation_count, 1)
    validation_indices = set(indices[:validation_count])
    return (
        [text for index, text in enumerate(texts) if index not in validation_indices],
        [text for index, text in enumerate(texts) if index in validation_indices],
    )


def _run_teacher(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    args: argparse.Namespace,
    activations: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
    capture_enabled: List[bool],
) -> torch.Tensor:
    activations.clear()
    logits.clear()
    capture_enabled[0] = True
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.no_grad():
        model(**encoded, use_cache=False)
    capture_enabled[0] = False
    return encoded["attention_mask"].reshape(-1).bool()


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    tokenizer: Any,
    gates: Sequence[Tuple[str, nn.Module]],
    predictors: nn.ModuleDict,
    texts: List[str],
    top_k: int,
    args: argparse.Namespace,
    activations: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
    capture_enabled: List[bool],
    pair_indices: Optional[set[int]],
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for text_batch in _batches(texts, args.batch_size):
        valid_tokens = _run_teacher(
            model, tokenizer, text_batch, args, activations, logits, capture_enabled
        )
        for pair in _valid_pair_batches(gates, activations, logits, valid_tokens, pair_indices):
            if str(pair.index) not in predictors:
                continue
            predicted = _predict_logits(gates[pair.index + 1][1], predictors[str(pair.index)], pair.previous)
            actual = pair.teacher_logits.topk(top_k, dim=-1).indices
            candidate = predicted.topk(min(args.top_m, predicted.shape[-1]), dim=-1).indices
            matched = (actual.unsqueeze(-1) == candidate.unsqueeze(-2)).any(dim=-1)
            entry = stats.setdefault(pair.index, {"tokens": 0.0, "hits": 0.0, "full": 0.0})
            entry["tokens"] += actual.shape[0]
            entry["hits"] += matched.sum().item()
            entry["full"] += matched.all(dim=-1).sum().item()
    for entry in stats.values():
        tokens = max(entry["tokens"], 1.0)
        entry["recall"] = entry["hits"] / (tokens * top_k)
        entry["coverage"] = entry["full"] / tokens
    return stats


def _print_evaluation(
    stats: Dict[int, Dict[str, float]], gates: Sequence[Tuple[str, nn.Module]], top_m: int
) -> None:
    print(f"Validation Top-{top_m} coverage")
    print("pair       tokens  recall  full")
    print("---------  ------  ------  ------")
    for index, entry in stats.items():
        print(
            f"L{index}->L{index + 1:<3}  {int(entry['tokens']):6d}  "
            f"{entry['recall']:6.2%}  {entry['coverage']:6.2%}"
        )


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    model, tokenizer = _load_model(args.model, args.revision, args.dtype, args.device)
    print("[setup] freezing base-model parameters", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    top_k = args.top_k or _infer_top_k(model)
    gates = _discover_gates(model, args.gate_pattern)
    pair_indices = set(args.train_pairs) if args.train_pairs else None
    if pair_indices is not None and (min(pair_indices) < 0 or max(pair_indices) >= len(gates) - 1):
        raise ValueError(f"--train-pairs must be between 0 and {len(gates) - 2}")
    print("[data] loading prompts", flush=True)
    texts = _read_texts(args)
    train_texts, validation_texts = _split_texts(texts, args.validation_ratio, args.seed)
    if not train_texts:
        raise ValueError("no training prompts remain after the validation split")
    print(
        f"[setup] gates={len(gates)} pairs={len(gates) - 1} "
        f"train_prompts={len(train_texts)} validation_prompts={len(validation_texts)} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    activations: Dict[str, torch.Tensor] = {}
    logits: Dict[str, torch.Tensor] = {}
    capture_enabled = [True]
    handles = _capture_hooks(gates, activations, logits, capture_enabled)
    predictors: Optional[nn.ModuleDict] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            epoch_losses = {"loss": 0.0, "rank": 0.0, "distill": 0.0, "load": 0.0}
            updates = 0
            total_batches = (len(train_texts) + args.batch_size - 1) // args.batch_size
            print(f"[train] epoch {epoch}/{args.epochs} ({total_batches} batches)", flush=True)
            for batch_index, text_batch in enumerate(_batches(train_texts, args.batch_size), start=1):
                valid_tokens = _run_teacher(
                    model, tokenizer, text_batch, args, activations, logits, capture_enabled
                )
                pairs = list(
                    _valid_pair_batches(gates, activations, logits, valid_tokens, pair_indices)
                )
                if predictors is None:
                    predictors = _build_predictors(pairs, args)
                    optimizer = torch.optim.AdamW(
                        predictors.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                    )
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                batch_loss: Optional[torch.Tensor] = None
                batch_terms = {"rank": 0.0, "distill": 0.0, "load": 0.0}
                for pair in pairs:
                    predictor = predictors.get_submodule(str(pair.index))
                    predicted = _predict_logits(gates[pair.index + 1][1], predictor, pair.previous)
                    loss, terms = _losses(predicted, pair.teacher_logits, top_k, args.top_m, args)
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                    for name, value in terms.items():
                        batch_terms[name] += value
                if batch_loss is None:
                    continue
                batch_loss = batch_loss / len(pairs)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(predictors.parameters(), args.max_grad_norm)
                optimizer.step()
                epoch_losses["loss"] += batch_loss.detach().item()
                for name, value in batch_terms.items():
                    epoch_losses[name] += value / len(pairs)
                updates += 1
                if batch_index == 1 or batch_index % args.log_interval == 0:
                    print(
                        f"[train] epoch={epoch} batch={batch_index}/{total_batches} "
                        f"loss={batch_loss.detach().item():.5f} "
                        f"elapsed={time.perf_counter() - epoch_start:.1f}s",
                        flush=True,
                    )
            if not updates:
                raise RuntimeError("no predictor updates were made")
            print(
                f"epoch={epoch} loss={epoch_losses['loss'] / updates:.5f} "
                f"rank={epoch_losses['rank'] / updates:.5f} "
                f"kl={epoch_losses['distill'] / updates:.5f} "
                f"load={epoch_losses['load'] / updates:.5f} "
                f"elapsed={time.perf_counter() - epoch_start:.1f}s",
                flush=True,
            )
            if validation_texts and predictors is not None:
                _print_evaluation(
                    _evaluate(
                        model,
                        tokenizer,
                        gates,
                        predictors,
                        validation_texts,
                        top_k,
                        args,
                        activations,
                        logits,
                        capture_enabled,
                        pair_indices,
                    ),
                    gates,
                    args.top_m,
                )
    finally:
        for handle in handles:
            handle.remove()

    if predictors is None:
        raise RuntimeError("predictors were not initialized")
    checkpoint = {
        "format_version": 1,
        "model": args.model,
        "gate_pattern": args.gate_pattern,
        "top_k": top_k,
        "top_m": args.top_m,
        "rank": args.rank,
        "pair_names": {
            index: (gates[int(index)][0], gates[int(index) + 1][0])
            for index in predictors.keys()
        },
        "predictors": predictors.state_dict(),
    }
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint_out)
    print(f"Saved {len(predictors)} predictors to {args.checkpoint_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--dataset", default=None, help="dataset alias or any Hugging Face dataset repo")
    parser.add_argument("--dataset-path", default=None, help="local .json/.jsonl file")
    parser.add_argument("--dataset-name", default=None, help="deprecated alias override")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--prompt-fields", nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-pattern", default=r"(?:^|\.)(?:gate|router)$")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-m", type=int, required=True, help="deployment prefetch candidate size M")
    parser.add_argument("--train-pairs", type=int, nargs="+", default=None, help="e.g. 0 1 trains L0->L1 and L1->L2")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="emit training progress every N batches",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss-weight", type=float, default=0.25)
    parser.add_argument("--load-loss-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint-out", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
