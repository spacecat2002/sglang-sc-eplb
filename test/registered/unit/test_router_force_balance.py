import contextlib
import io
import unittest

import torch
from sglang.srt.layers.moe.ep_moe.layer import _build_force_balanced_topk_ids
from sglang.srt.layers.moe.stage_timing import (
    _format_moe_stage_timing,
    _MoEStageTimer,
)


class TestRouterForceBalance(unittest.TestCase):
    def test_rank_first_balancing(self):
        template = torch.zeros((4, 2), dtype=torch.int32)
        actual = _build_force_balanced_topk_ids(
            template,
            num_experts=8,
            ep_size=4,
            global_token_start=0,
        )
        expected = torch.tensor([[0, 2], [4, 6], [1, 3], [5, 7]], dtype=torch.int32)
        torch.testing.assert_close(actual, expected)

    def test_global_offset_and_padding(self):
        template = torch.tensor([[0, 0], [-1, -1]], dtype=torch.int64)
        actual = _build_force_balanced_topk_ids(
            template,
            num_experts=8,
            ep_size=4,
            global_token_start=2,
        )
        expected = torch.tensor([[1, 3], [-1, -1]], dtype=torch.int64)
        torch.testing.assert_close(actual, expected)

    def test_stage_timing_ratio(self):
        report = _format_moe_stage_timing(3, 4, 10.0, 30.0, 10.0, rank=2)
        self.assertIn("[rank=2][prefill=3] layers=4", report)
        self.assertIn("communication_pct=40.00% compute_pct=60.00%", report)

        decode_report = _format_moe_stage_timing(
            3, 4, 10.0, 30.0, 10.0, rank=2, is_prefill=False
        )
        self.assertIn("[rank=2][decode=3] layers=4", decode_report)

    def test_stage_timing_resets_each_prefill(self):
        class Event:
            def __init__(self, timestamp):
                self.timestamp = timestamp

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        timer = _MoEStageTimer()
        timer.pending.append(tuple(Event(t) for t in (0, 1, 1, 4, 4, 6)))
        with contextlib.redirect_stdout(io.StringIO()):
            timer._report_batch()

        timer.pending.append(tuple(Event(t) for t in (0, 2, 2, 3, 3, 7)))
        timer.is_prefill = False
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timer._report_batch()

        self.assertIn("[decode=2] layers=1", output.getvalue())
        self.assertIn("communication_ms=6.000 compute_ms=1.000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
