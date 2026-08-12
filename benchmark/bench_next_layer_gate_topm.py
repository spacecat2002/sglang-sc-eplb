#!/usr/bin/env python3
"""Measure next-layer MoE gate prediction from the previous layer activation.

For every adjacent pair of MoE gates, the script compares:

    actual:  gate[i + 1](x[i + 1])
    predict: gate[i + 1](x[i])

where ``x[i]`` is the activation that was actually passed to gate ``i``.
This isolates the error caused by using the previous layer activation as a
lookahead signal.  The gate weights are not changed and the true router is
still used as the reference.

Examples:
  python benchmark/bench_next_layer_gate_topm.py \
      --model Qwen/Qwen3-30B-A3B \
      --dataset sharegpt --dataset-path ShareGPT.json --num-samples 128 \
      --top-k 8 --top-m 8 12 16

  python benchmark/bench_next_layer_gate_topm.py \
      --model Qwen/Qwen3-30B-A3B --dataset humaneval --num-samples 164

  python benchmark/bench_next_layer_gate_topm.py \
      --model Qwen/Qwen3-30B-A3B --dataset allenai/c4 \
      --dataset-config en --dataset-split validation --prompt-fields text

The script uses Hugging Face Transformers directly, so it is intended for a
single-GPU/offline profiling run.  Use a small model first to validate module
names and memory requirements.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _natural_key(name: str) -> Tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name))


def _flatten_activation(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        raise TypeError(f"expected a tensor with at least 2 dimensions, got {type(x)}")
    return x.reshape(-1, x.shape[-1])


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for key in ("router_logits", "logits", "output"):
            if key in value:
                tensor = _first_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _read_texts(args: argparse.Namespace) -> List[str]:
    if not args.dataset and not args.dataset_path:
        if args.text_file:
            texts = [line.rstrip("\n") for line in Path(args.text_file).read_text().splitlines()]
            texts = _sample_texts(texts, args.num_samples, args.seed)
            if texts:
                return texts
        if args.text:
            return [args.text]
        return [
            "Mixture of experts models route each token to a small number of experts."
        ]

    dataset_name, default_split, default_fields = _dataset_defaults(args.dataset)
    if args.dataset_name:
        dataset_name = args.dataset_name
    records = _load_records(args, dataset_name, default_split)
    fields = args.prompt_fields or default_fields
    if args.dataset == "sharegpt":
        return _sample_texts(
            [_sharegpt_prompt(record) for record in records], args.num_samples, args.seed
        )
    return _sample_texts(
        [_record_prompt(record, fields) for record in records], args.num_samples, args.seed
    )


def _dataset_defaults(dataset: Optional[str]) -> Tuple[Optional[str], str, List[str]]:
    aliases = {
        "sharegpt": ("anon8231489123/ShareGPT_Vicuna_unfiltered", "train", []),
        "humaneval": ("openai/openai_humaneval", "test", ["prompt"]),
        "summary": ("EdinburghNLP/xsum", "test", ["document"]),
        "summarization": ("EdinburghNLP/xsum", "test", ["document"]),
    }
    if dataset in aliases:
        return aliases[dataset]
    return dataset, "train", ["text"]


def _load_records(
    args: argparse.Namespace, default_dataset: Optional[str], default_split: str
) -> List[Dict[str, Any]]:
    """Load local JSON/JSONL first; otherwise use the Hugging Face datasets hub."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for --dataset: pip install datasets") from exc

    if args.dataset_path:
        suffix = Path(args.dataset_path).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError("--dataset-path must point to a .json or .jsonl file")
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    else:
        if not default_dataset:
            raise ValueError("pass --dataset for a Hugging Face dataset or --dataset-path for local JSON")
        dataset = load_dataset(
            default_dataset,
            args.dataset_config,
            split=args.dataset_split or default_split,
            revision=args.dataset_revision,
        )
    return [dict(record) for record in dataset]


def _record_prompt(record: Dict[str, Any], fields: List[str]) -> str:
    values = []
    for field in fields:
        value: Any = record
        for key in field.split("."):
            if not isinstance(value, dict) or key not in value:
                raise KeyError(
                    f"field {field!r} was not found; available top-level fields: "
                    f"{', '.join(sorted(record))}"
                )
            value = value[key]
        if value is not None:
            values.append(str(value))
    return "\n".join(values)


def _sharegpt_prompt(record: Dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    turns = []
    for turn in conversations:
        role = str(turn.get("from", turn.get("role", "user"))).lower()
        content = turn.get("value", turn.get("content", ""))
        if not content:
            continue
        if role in {"human", "user"}:
            turns.append(f"User: {content}")
        elif role in {"gpt", "assistant"}:
            turns.append(f"Assistant: {content}")
    return "\n".join(turns)


def _sample_texts(texts: List[str], num_samples: int, seed: int) -> List[str]:
    texts = [text for text in texts if text.strip()]
    if num_samples <= 0 or num_samples >= len(texts):
        return texts
    indices = random.Random(seed).sample(range(len(texts)), num_samples)
    return [texts[index] for index in indices]


def _topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(logits.reshape(-1, logits.shape[-1]), k=k, dim=-1).indices


def _metric_counts(actual: torch.Tensor, predicted: torch.Tensor) -> Tuple[int, int]:
    matched = (actual.unsqueeze(-1) == predicted.unsqueeze(-2)).any(dim=-1)
    return matched.sum().item(), matched.all(dim=-1).sum().item()


def _batches(items: List[str], batch_size: int):
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _layer_label(name: str) -> str:
    """Prefer the transformer layer index, while retaining nonstandard gate names."""
    match = re.search(r"(?:layers|h)\.(\d+)\.", name)
    return f"L{match.group(1)}" if match else name


def _format_terminal_result(result: Dict[str, Any]) -> str:
    """Render the per-layer metrics without requiring a JSON viewer."""
    lines = [
        "Next-layer gate Top-M prediction",
        (
            f"model={result['model']}  dataset={result['dataset']}  "
            f"prompts={result['num_prompts']}  K={result['top_k']}"
        ),
        "R: Top-K expert recall; Full: tokens whose entire real Top-K is in Top-M",
    ]
    headers = ["layers", "tokens"]
    for top_m in result["top_m"]:
        headers.extend([f"M={top_m} R", "Full"])
    rows = []
    totals = {top_m: {"recall": 0.0, "full": 0.0} for top_m in result["top_m"]}
    total_tokens = 0
    for layer in result["layers"]:
        metric_by_m = {metric["top_m"]: metric for metric in layer["metrics"]}
        row = [
            f"{_layer_label(layer['from_gate'])}->{_layer_label(layer['to_gate'])}",
            str(layer["tokens"]),
        ]
        for top_m in result["top_m"]:
            metric = metric_by_m[top_m]
            row.extend(
                [
                    f"{metric['token_recall']:.2%}",
                    f"{metric['token_full_topk_coverage']:.2%}",
                ]
            )
            totals[top_m]["recall"] += metric["token_recall"] * layer["tokens"]
            totals[top_m]["full"] += metric["token_full_topk_coverage"] * layer["tokens"]
        total_tokens += layer["tokens"]
        rows.append(row)

    summary = ["weighted avg", str(total_tokens)]
    for top_m in result["top_m"]:
        summary.extend(
            [
                f"{totals[top_m]['recall'] / max(total_tokens, 1):.2%}",
                f"{totals[top_m]['full'] / max(total_tokens, 1):.2%}",
            ]
        )
    rows.append(summary)
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _load_model(model_name: str, revision: Optional[str], dtype: str, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return model, tokenizer


def _infer_top_k(model: torch.nn.Module) -> int:
    config = model.config
    for key in ("num_experts_per_tok", "num_experts_per_token", "top_k"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer router top-k; pass --top-k explicitly")


def _discover_gates(model: torch.nn.Module, pattern: str) -> List[Tuple[str, torch.nn.Module]]:
    regex = re.compile(pattern)
    gates = [(name, module) for name, module in model.named_modules() if regex.search(name)]
    gates.sort(key=lambda item: _natural_key(item[0]))
    if len(gates) < 2:
        raise RuntimeError(
            f"found {len(gates)} gate modules with --gate-pattern {pattern!r}; "
            "try --gate-pattern to match the model's router modules"
        )
    return gates


@torch.inference_mode()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer = _load_model(args.model, args.revision, args.dtype, args.device)
    if args.top_k is None:
        args.top_k = _infer_top_k(model)
    if args.top_m is None:
        args.top_m = [args.top_k, int(args.top_k * 1.5), args.top_k * 2]
    gates = _discover_gates(model, args.gate_pattern)
    activations: Dict[str, torch.Tensor] = {}
    logits: Dict[str, torch.Tensor] = {}
    top_k = args.top_k
    top_ms = sorted(set(args.top_m))
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    capture_enabled = [True]

    handles = []
    for name, gate in gates:
        def pre_hook(_module, inputs, gate_name=name):
            if not capture_enabled[0]:
                return
            tensor = _first_tensor(inputs)
            if tensor is not None:
                activations[gate_name] = _flatten_activation(tensor).detach()

        def post_hook(_module, _inputs, output, gate_name=name):
            if not capture_enabled[0]:
                return
            tensor = _first_tensor(output)
            if tensor is not None and tensor.ndim >= 2:
                logits[gate_name] = _flatten_activation(tensor).detach().float()

        handles.extend([gate.register_forward_pre_hook(pre_hook), gate.register_forward_hook(post_hook)])

    texts = _read_texts(args)
    if not texts:
        raise ValueError("the selected dataset did not contain any usable prompts")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for text_batch in _batches(texts, args.batch_size):
            capture_enabled[0] = True
            activations.clear()
            logits.clear()
            encoded = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
            capture_enabled[0] = False
            valid_tokens = encoded["attention_mask"].reshape(-1).bool()
            for (name0, _), (name1, gate1) in zip(gates, gates[1:]):
                if name0 not in activations or name1 not in logits:
                    continue
                previous = activations[name0]
                actual_logits = logits[name1]
                token_count = min(previous.shape[0], actual_logits.shape[0])
                previous = previous[:token_count]
                actual_logits = actual_logits[:token_count]
                if valid_tokens.numel() == token_count:
                    previous = previous[valid_tokens]
                    actual_logits = actual_logits[valid_tokens]
                actual_topk = _topk(actual_logits, top_k)
                predicted_logits = _first_tensor(gate1(previous))
                if predicted_logits is None:
                    raise RuntimeError(f"could not extract logits from gate {name1}")
                predicted_logits = _flatten_activation(predicted_logits).float()
                key = (name0, name1)
                layer_stat = stats.setdefault(
                    key,
                    {
                        "from_gate": name0,
                        "to_gate": name1,
                        "tokens": 0,
                        "metrics": {m: {"hits": 0, "full": 0} for m in top_ms},
                    },
                )
                layer_stat["tokens"] += actual_topk.shape[0]
                for m in top_ms:
                    if m < top_k or m > predicted_logits.shape[-1]:
                        raise ValueError(f"top-m={m} must satisfy top-k <= M <= num_experts")
                    hits, full = _metric_counts(actual_topk, _topk(predicted_logits, m))
                    layer_stat["metrics"][m]["hits"] += hits
                    layer_stat["metrics"][m]["full"] += full
    finally:
        for handle in handles:
            handle.remove()

    layer_results = []
    for layer_stat in stats.values():
        token_count = layer_stat["tokens"]
        metrics = []
        for m in top_ms:
            counts = layer_stat["metrics"][m]
            metrics.append(
                {
                    "top_m": m,
                    "token_recall": counts["hits"] / max(token_count * top_k, 1),
                    "token_full_topk_coverage": counts["full"] / max(token_count, 1),
                }
            )
        layer_results.append(
            {
                "from_gate": layer_stat["from_gate"],
                "to_gate": layer_stat["to_gate"],
                "tokens": token_count,
                "metrics": metrics,
            }
        )
    result = {
        "model": args.model,
        "dataset": args.dataset or "text",
        "dataset_path": args.dataset_path,
        "num_prompts": len(texts),
        "seed": args.seed,
        "top_k": top_k,
        "top_m": top_ms,
        "gates": [name for name, _ in gates],
        "layers": layer_results,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None, help="one prompt per line")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "ShareGPT/HumanEval/summary alias or any Hugging Face dataset repo; "
            "takes precedence over --text and --text-file"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="local .json/.jsonl dataset file; required to use a local fixed copy",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="deprecated alias for --dataset; used only when --dataset is an alias",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--prompt-fields",
        nargs="+",
        default=None,
        help="one or more fields to concatenate into each prompt, supports dot paths",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="fixed number of sampled prompts; 0 means use every prompt",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=None, help="router K; defaults to model config")
    parser.add_argument(
        "--top-m",
        type=int,
        nargs="+",
        default=None,
        help="candidate M values; defaults to K, 1.5K, and 2K",
    )
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-pattern", default=r"(?:^|\.)(?:gate|router)$")
    parser.add_argument("--output", default=None, help="write complete JSON results to this path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the terminal table")
    args = parser.parse_args()
    result = run(args)
    output = json.dumps(result, indent=2)
    print(output if args.json else _format_terminal_result(result))
    if args.output:
        Path(args.output).write_text(output + "\n")


if __name__ == "__main__":
    main()
