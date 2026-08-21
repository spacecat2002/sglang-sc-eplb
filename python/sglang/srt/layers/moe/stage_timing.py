import os
from collections import deque
from typing import Optional

import torch

from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.utils import get_bool_env_var


def _format_moe_stage_timing(
    batch: int,
    layers: int,
    dispatch_ms: float,
    compute_ms: float,
    combine_ms: float,
    rank: int,
    is_prefill: bool = True,
) -> str:
    mode = "prefill" if is_prefill else "decode"
    communication_ms = dispatch_ms + combine_ms
    total_ms = communication_ms + compute_ms
    communication_pct = 100 * communication_ms / total_ms if total_ms else 0.0
    compute_pct = 100 * compute_ms / total_ms if total_ms else 0.0
    return (
        f"[MoE stage timing][rank={rank}][{mode}={batch}] layers={layers} "
        f"communication_ms={communication_ms:.3f} compute_ms={compute_ms:.3f} "
        f"communication_pct={communication_pct:.2f}% compute_pct={compute_pct:.2f}% "
        f"dispatch_ms={dispatch_ms:.3f} combine_ms={combine_ms:.3f}"
    )


class _MoEStageTimer:
    def __init__(self):
        self.ready_file = os.environ.get("SGLANG_MOE_STAGE_TIMING_READY_FILE")
        self.ready = self.ready_file is None
        self.layer_ids = set()
        self.batch = 0
        self.is_prefill = True
        # ponytail: one in-flight sample; use per-subbatch state to time TBO.
        self.events = []
        self.pending = deque()

    def register_layer(self, layer_id: int):
        self.layer_ids.add(layer_id)

    def _enabled(self):
        if not self.ready and os.path.exists(self.ready_file):
            self.ready = True
        return self.ready

    @staticmethod
    def _record_event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def pre_dispatch(self, _layer_id, *_args):
        if not self._enabled():
            self.events = []
            return
        self.is_prefill = get_is_extend_in_batch()
        self.events = [self._record_event()]

    def post_dispatch(self, _layer_id, *_args):
        if self.events:
            self.events.append(self._record_event())

    def begin_compute(self, _layer_id):
        if self.events:
            self.events.append(self._record_event())

    def end_compute(self, _layer_id):
        if self.events:
            self.events.append(self._record_event())

    def pre_combine(self, _layer_id, *_args):
        if self.events:
            self.events.append(self._record_event())

    def post_combine(self, layer_id, *_args):
        if not self.events:
            return
        self.events.append(self._record_event())
        if len(self.events) != 6:
            self.events = []
            return
        self.pending.append(tuple(self.events))
        self.events = []
        if layer_id == max(self.layer_ids):
            self.pending[-1][-1].synchronize()
            self._report_batch()

    def _report_batch(self):
        dispatch_ms = compute_ms = combine_ms = 0.0
        layers = len(self.pending)
        while self.pending:
            events = self.pending.popleft()
            dispatch_ms += events[0].elapsed_time(events[1])
            compute_ms += events[2].elapsed_time(events[3])
            combine_ms += events[4].elapsed_time(events[5])
        self.batch += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(
            _format_moe_stage_timing(
                self.batch,
                layers,
                dispatch_ms,
                compute_ms,
                combine_ms,
                rank,
                self.is_prefill,
            ),
            flush=True,
        )


_moe_stage_timer = None


def _get_moe_stage_timer() -> Optional[_MoEStageTimer]:
    global _moe_stage_timer
    if not get_bool_env_var("SGLANG_MOE_STAGE_TIMING"):
        return None
    if _moe_stage_timer is None:
        _moe_stage_timer = _MoEStageTimer()
    return _moe_stage_timer
