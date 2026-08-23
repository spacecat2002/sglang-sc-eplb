import hashlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_loader.loader import _MoeCpuWeightStore


class TestMoeCpuWeightStore(unittest.TestCase):
    def _capture_for_rank(self, local_rank):
        store = _MoeCpuWeightStore.__new__(_MoeCpuWeightStore)
        store.local_rank = local_rank
        store.writer_ranks = (0, 2)
        store.num_experts = 2
        store.fd = 1
        store.offset = 0
        store.signature = hashlib.sha1()
        store.metadata = {} if local_rank == 0 else None
        weights = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(2)),
            ("model.layers.0.mlp.experts.1.gate_proj.weight", torch.ones(3)),
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(4)),
        ]
        with patch.object(store, "_write_all") as write:
            self.assertEqual(list(store.capture(weights)), weights)
        return store, write.call_args_list

    def test_two_numa_writers_split_complete_experts(self):
        root, root_writes = self._capture_for_rank(0)
        peer, peer_writes = self._capture_for_rank(2)

        self.assertEqual([call.args[2] for call in root_writes], [0])
        self.assertEqual([call.args[2] for call in peer_writes], [8])
        self.assertEqual(root.offset, 20)
        self.assertEqual(peer.offset, 20)
        self.assertEqual(len(root.metadata), 2)

    def test_fused_expert_tensor_is_split_on_expert_dimension(self):
        weight = torch.arange(12).view(2, 6)
        offsets = []
        for local_rank in (0, 2):
            store = _MoeCpuWeightStore.__new__(_MoeCpuWeightStore)
            store.local_rank = local_rank
            store.writer_ranks = (0, 2)
            store.num_experts = 2
            store.fd = 1
            store.offset = 0
            store.signature = hashlib.sha1()
            store.metadata = {} if local_rank == 0 else None
            with patch.object(store, "_write_all") as write:
                list(store.capture([("model.layers.0.mlp.experts.w13", weight)]))
            offsets.append([call.args[2] for call in write.call_args_list])

        self.assertEqual(offsets, [[0], [weight[0].nbytes]])


if __name__ == "__main__":
    unittest.main()
