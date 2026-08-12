"""Optional per-rank expert-load snapshots for inference benchmarking."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ExpertLoadLogger:
    _instance: Optional[ExpertLoadLogger] = None

    @classmethod
    def get(cls) -> ExpertLoadLogger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.enabled = os.getenv("SGLANG_ENABLE_EXPERT_LOAD_LOGGING", "0") == "1"
        self.prefill_only = os.getenv("SGLANG_EXPERT_LOAD_PREFILL_ONLY", "1") == "1"
        self.max_steps = int(os.getenv("SGLANG_EXPERT_LOAD_MAX_STEPS", "100000"))
        self.save_path = Path(
            os.getenv("SGLANG_EXPERT_LOAD_LOG_PATH", "./expert_loads")
        )
        self._initialized = False
        self._current_is_prefill = True
        self._record_current_forward = False
        self._prefill_idx = 0
        self._decode_idx = 0
        self._records: dict[tuple[bool, int, int], np.ndarray] = {}
        self._lock = threading.Lock()

        if self.enabled:
            self.save_path.mkdir(parents=True, exist_ok=True)
            atexit.register(self.save)

    def init_metadata(
        self,
        *,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_rank: int,
    ) -> None:
        if self._initialized or not self.enabled:
            return
        self.num_local_experts = num_local_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.global_rank = global_rank
        self._initialized = True
        logger.info(
            "ExpertLoadLogger initialized: global_rank=%d, ep_rank=%d/%d, "
            "num_local_experts=%d, prefill_only=%s, save_path=%s",
            global_rank,
            ep_rank,
            ep_size,
            num_local_experts,
            self.prefill_only,
            self.save_path,
        )

    @contextmanager
    def forward_context(self, *, is_prefill: bool, is_idle: bool):
        active = self.enabled and not is_idle and (is_prefill or not self.prefill_only)
        self._current_is_prefill = is_prefill
        self._record_current_forward = active
        try:
            yield
        finally:
            if active and self._initialized:
                if is_prefill:
                    self._prefill_idx += 1
                else:
                    self._decode_idx += 1
            self._record_current_forward = False

    def record(self, *, layer_id: int, counts: Sequence[int]) -> None:
        if not self._record_current_forward or not self._initialized:
            return
        step = self._prefill_idx if self._current_is_prefill else self._decode_idx
        if step >= self.max_steps:
            return
        values = np.asarray(counts, dtype=np.int32).copy()
        with self._lock:
            self._records[(self._current_is_prefill, step, layer_id)] = values

    def save(self) -> None:
        if not self.enabled or not self._initialized:
            return
        with self._lock:
            records = dict(self._records)

        max_layer_id = max((key[2] for key in records), default=-1)
        num_layers = max_layer_id + 1
        prefill = np.zeros(
            (
                min(self._prefill_idx, self.max_steps),
                num_layers,
                self.num_local_experts,
            ),
            dtype=np.int32,
        )
        decode = None
        if not self.prefill_only:
            decode = np.zeros(
                (
                    min(self._decode_idx, self.max_steps),
                    num_layers,
                    self.num_local_experts,
                ),
                dtype=np.int32,
            )

        for (is_prefill, step, layer_id), values in records.items():
            target = prefill if is_prefill else decode
            if target is not None and step < target.shape[0]:
                target[step, layer_id, : values.shape[0]] = values

        self.save_path.mkdir(parents=True, exist_ok=True)
        arrays = {"prefill_loads": prefill}
        if decode is not None:
            arrays["decode_loads"] = decode
        np.savez_compressed(
            self.save_path / f"rank_{self.global_rank}_expert_loads.npz", **arrays
        )

        metadata = {
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
            "global_rank": self.global_rank,
            "num_local_experts": self.num_local_experts,
            "num_layers": num_layers,
            "num_prefill_steps": prefill.shape[0],
            "num_decode_steps": 0 if decode is None else decode.shape[0],
            "prefill_only": self.prefill_only,
        }
        with (self.save_path / f"rank_{self.global_rank}_metadata.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(metadata, output, indent=2)

        logger.info(
            "ExpertLoadLogger saved rank %d data to %s",
            self.global_rank,
            self.save_path,
        )


def expert_load_forward_context(forward_batch):
    logger_instance = ExpertLoadLogger.get()
    return logger_instance.forward_context(
        is_prefill=not forward_batch.forward_mode.is_decode(),
        is_idle=forward_batch.forward_mode.is_idle(),
    )
