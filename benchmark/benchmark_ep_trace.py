#!/usr/bin/env python3
"""Capture a real distributed SGLang MoE routing trace.

The script launches ``sglang.Engine`` with real TP/EP workers, records each
worker's router output and EP rank, then writes a compact trace for
``compare_pairwise_grace.py``.

Example:

    PYTHONPATH=python python benchmark/benchmark_ep_trace.py \
        --model Qwen/Qwen3-30B-A3B \
        --tp-size 8 --ep-size 8 --moe-a2a-backend deepep \
        --dataset sharegpt --num-samples 128 --batch-size 8 \
        --output /tmp/qwen3_ep8_trace.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from bench_next_layer_gate_topm import _batches, _read_texts
from sglang.srt.eplb.moe_bundle_trace import (
    compact_layers_from_recorder_packs,
    save_compact_trace,
)


def _engine_args(args: argparse.Namespace) -> dict[str, Any]:
    extra = json.loads(args.engine_args_json)
    if not isinstance(extra, dict):
        raise ValueError("--engine-args-json must contain a JSON object")
    result = {
        **extra,
        "model_path": args.model,
        "tp_size": args.tp_size,
        "ep_size": args.ep_size,
        "moe_a2a_backend": args.moe_a2a_backend,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "expert_distribution_recorder_mode": "per_token",
        "enable_eplb": False,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "log_level": args.log_level,
    }
    if args.quantization:
        result["quantization"] = args.quantization
    if args.mem_fraction_static is not None:
        result["mem_fraction_static"] = args.mem_fraction_static
    return result


def _record_files(directory: Path) -> set[Path]:
    return {
        path.resolve() for path in directory.glob("expert_distribution_recorder_*.pt")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sglang as sgl
    from sglang.srt.environ import envs

    texts = _read_texts(args)
    if not texts:
        raise ValueError("no non-empty prompts were loaded")
    record_dir = Path(args.record_dir).resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    previous_files = _record_files(record_dir)
    envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(str(record_dir))

    engine = None
    recording = False
    try:
        engine = sgl.Engine(**_engine_args(args))
        engine.start_expert_distribution_record()
        recording = True
        batches = list(_batches(texts, args.batch_size))
        for index, prompts in enumerate(batches, start=1):
            engine.generate(
                prompt=prompts,
                sampling_params={
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                },
            )
            print(f"[capture] batch {index}/{len(batches)}", flush=True)
        engine.stop_expert_distribution_record()
        recording = False
        engine.dump_expert_distribution_record()
    finally:
        if engine is not None:
            if recording:
                engine.stop_expert_distribution_record()
            engine.shutdown()

    paths = sorted(_record_files(record_dir) - previous_files)
    if not paths:
        raise RuntimeError(f"SGLang wrote no recorder dumps to {record_dir}")
    packs = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    top_k, layers = compact_layers_from_recorder_packs(packs, num_ranks=args.ep_size)
    save_compact_trace(
        args.output,
        num_ranks=args.ep_size,
        top_k=top_k,
        layers=layers,
    )
    return {
        "output": str(Path(args.output).resolve()),
        "model": args.model,
        "prompts": len(texts),
        "ep_size": args.ep_size,
        "top_k": top_k,
        "layers": len(layers),
        "recorder_files": [str(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "--model-path", dest="model", required=True)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument(
        "--moe-a2a-backend",
        choices=["deepep", "flashinfer", "mooncake", "nixl", "mori", "megamoe"],
        default="deepep",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization")
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--engine-args-json", default="{}")

    parser.add_argument("--text")
    parser.add_argument("--text-file", help="one prompt per line")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--prompt-fields", nargs="+")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--record-dir",
        default="/tmp/sglang_ep_trace",
        help="temporary SGLang per-rank recorder directory",
    )
    parser.add_argument("--output", "--bundles-output", dest="output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.tp_size = args.tp_size or args.ep_size
    if args.tp_size < 1 or args.ep_size < 1:
        parser.error("--tp-size and --ep-size must be positive")
    if args.batch_size < 1 or args.num_samples < 1:
        parser.error("--batch-size and --num-samples must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if Path(args.output).suffix not in {".pt", ".pth"}:
        parser.error("--output must be a .pt or .pth file")

    result = run(args)
    output = json.dumps(result, indent=2)
    print(output if args.json else f"[trace] {result}")


if __name__ == "__main__":
    main()
