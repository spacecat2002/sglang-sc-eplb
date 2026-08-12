import unittest

import torch

from sglang.srt.layers.moe.ep_moe.layer import _build_force_balanced_topk_ids


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


if __name__ == "__main__":
    unittest.main()
