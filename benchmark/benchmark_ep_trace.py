#!/usr/bin/env python3
"""Capture a real distributed SGLang MoE routing trace.

The script launches ``sglang.Engine`` with attention DP and MoE EP, then writes
the returned routed experts and actual DP source ranks as a compact trace for
``compare_grace.py``.

Example:

    PYTHONPATH=python python benchmark/benchmark_ep_trace.py \
        --model Qwen/Qwen3-30B-A3B \
        --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
        --moe-a2a-backend deepep \
        --dataset sharegpt --num-samples 128 --batch-size 8 \
        --show-stage-timing \
        --output /tmp/qwen3_ep8_trace.pt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from bench_next_layer_gate_topm import _batches, _read_texts
from sglang.srt.eplb.moe_bundle_trace import (
    compact_layer_from_bundles,
    save_compact_trace,
)
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)


def _engine_args(args: argparse.Namespace) -> dict[str, Any]:
    extra = json.loads(args.engine_args_json)
    if not isinstance(extra, dict):
        raise ValueError("--engine-args-json must contain a JSON object")
    result = {
        **extra,
        "model_path": args.model,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "ep_size": args.ep_size,
        "enable_dp_attention": args.enable_dp_attention,
        "moe_a2a_backend": args.moe_a2a_backend,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enable_return_routed_experts": True,
        "enable_eplb": False,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "log_level": args.log_level,
    }
    if args.quantization:
        result["quantization"] = args.quantization
    if args.mem_fraction_static is not None:
        result["mem_fraction_static"] = args.mem_fraction_static
    if args.moe_a2a_backend == "hybridep":
        result.setdefault("moe_runner_backend", "deep_gemm")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sglang as sgl

    texts = _read_texts(args)
    if not texts:
        raise ValueError("no non-empty prompts were loaded")

    engine = None
    bundles_by_layer = None
    top_k = None
    num_layers = None
    observed_ranks = set()
    try:
        print(
            f"[engine] launching tp={args.tp_size} dp={args.dp_size} "
            f"ep={args.ep_size} backend={args.moe_a2a_backend}",
            flush=True,
        )
        engine = sgl.Engine(**_engine_args(args))
        if args.show_stage_timing:
            Path(os.environ["SGLANG_MOE_STAGE_TIMING_READY_FILE"]).touch()
        config = engine.server_args.get_model_config().hf_text_config
        num_layers = int(config.num_hidden_layers)
        top_k = int(config.num_experts_per_tok)
        bundles_by_layer = [Counter() for _ in range(num_layers)]
        print(f"[engine] ready, layers={num_layers} top_k={top_k}", flush=True)

        batches = list(_batches(texts, args.batch_size))
        for index, prompts in enumerate(batches, start=1):
            responses = engine.generate(
                prompt=prompts,
                sampling_params={
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                },
                return_routed_experts=True,
            )
            if isinstance(responses, dict):
                responses = [responses]
            for response in responses:
                meta = response["meta_info"]
                if "dp_rank" not in meta or "routed_experts" not in meta:
                    raise RuntimeError(
                        "SGLang response is missing dp_rank or routed_experts"
                    )
                source_rank = int(meta["dp_rank"])
                if not 0 <= source_rank < args.dp_size:
                    raise ValueError(f"invalid source DP rank {source_rank}")
                routed = extract_routed_experts_from_meta_info(response)
                width = num_layers * top_k
                if routed.size == 0 or routed.size % width:
                    raise ValueError(
                        f"invalid routed expert result with {routed.size} values"
                    )
                routed = routed.reshape(-1, num_layers, top_k)
                observed_ranks.add(source_rank)
                for layer_index in range(num_layers):
                    bundles_by_layer[layer_index].update(
                        (source_rank, tuple(sorted(map(int, experts))))
                        for experts in routed[:, layer_index]
                    )
            print(f"[capture] batch {index}/{len(batches)}", flush=True)
    finally:
        if engine is not None:
            engine.shutdown()

    layers = [
        compact_layer_from_bundles(f"model.layers.{index}.gate", bundles)
        for index, bundles in enumerate(bundles_by_layer)
    ]
    save_compact_trace(
        args.output,
        num_ranks=args.dp_size,
        top_k=top_k,
        layers=layers,
    )
    return {
        "output": str(Path(args.output).resolve()),
        "model": args.model,
        "prompts": len(texts),
        "dp_size": args.dp_size,
        "ep_size": args.ep_size,
        "top_k": top_k,
        "layers": num_layers,
        "observed_source_ranks": sorted(observed_ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "--model-path", dest="model", required=True)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument(
        "--enable-dp-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--moe-a2a-backend",
        choices=[
            "none",
            "deepep",
            "hybridep",
            "flashinfer",
            "mooncake",
            "nixl",
            "mori",
            "megamoe",
        ],
        default="none",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization")
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--engine-args-json", default="{}")
    parser.add_argument(
        "--show-stage-timing",
        action="store_true",
        help="report MoE dispatch/combine communication versus expert compute time",
    )
    parser.add_argument("--stage-timing-interval", type=int, help=argparse.SUPPRESS)

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

    parser.add_argument("--output", "--bundles-output", dest="output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    args.tp_size = args.tp_size or args.ep_size
    args.dp_size = args.dp_size or args.ep_size
    if min(args.tp_size, args.dp_size, args.ep_size) < 1:
        parser.error("--tp-size, --dp-size and --ep-size must be positive")
    if not args.enable_dp_attention or not (
        args.tp_size == args.dp_size == args.ep_size
    ):
        parser.error("trace capture requires --enable-dp-attention and tp=dp=ep")
    if args.batch_size < 1 or args.num_samples < 1:
        parser.error("--batch-size and --num-samples must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if Path(args.output).suffix not in {".pt", ".pth"}:
        parser.error("--output must be a .pt or .pth file")

    timing_ready_file = None
    if args.show_stage_timing:
        fd, timing_ready_file = tempfile.mkstemp(prefix="sglang-moe-timing-")
        os.close(fd)
        os.unlink(timing_ready_file)
        os.environ["SGLANG_MOE_STAGE_TIMING"] = "1"
        os.environ["SGLANG_MOE_STAGE_TIMING_READY_FILE"] = timing_ready_file

    try:
        result = run(args)
    finally:
        if timing_ready_file is not None:
            Path(timing_ready_file).unlink(missing_ok=True)
    output = json.dumps(result, indent=2)
    print(output if args.json else f"[trace] {result}")


if __name__ == "__main__":
    main()
