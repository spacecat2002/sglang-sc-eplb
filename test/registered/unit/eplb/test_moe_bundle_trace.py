import torch

from sglang.srt.eplb.moe_bundle_trace import (
    COMPACT_TRACE_FORMAT,
    compact_layer_from_bundles,
    compact_layers_from_recorder_packs,
    compact_layer_from_records,
    load_compact_trace,
    save_compact_trace,
)


def test_compact_trace_round_trip_uses_small_integer_types(tmp_path):
    layer = compact_layer_from_bundles(
        "model.layers.0.mlp.gate",
        {
            (0, (1, 3)): 7,
            (1, (2, 4)): 11,
        },
    )
    path = tmp_path / "trace.pt"
    save_compact_trace(path, num_ranks=2, top_k=2, layers=[layer])

    trace = load_compact_trace(path)
    assert trace["format"] == COMPACT_TRACE_FORMAT
    assert trace["num_ranks"] == 2
    assert trace["layers"][0]["source_rank"].dtype == torch.int16
    assert trace["layers"][0]["topk_experts"].dtype == torch.int16
    assert trace["layers"][0]["count"].dtype == torch.int16
    assert torch.equal(trace["layers"][0]["topk_experts"], layer["topk_experts"])


def test_json_records_are_canonicalized_for_compact_trace():
    layer = compact_layer_from_records(
        "gate",
        [
            {"source_rank": 1, "topk_experts": [4, 2], "count": 3},
            {"source_rank": 0, "topk_experts": [3, 1], "count": 2},
        ],
    )
    assert layer["source_rank"].tolist() == [1, 0]
    assert layer["topk_experts"].tolist() == [[2, 4], [1, 3]]
    assert layer["count"].tolist() == [3, 2]


def test_sglang_recorder_packs_use_real_rank_and_logical_experts():
    expert_map = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    packs = [
        {
            "rank": 0,
            "last_physical_to_logical_map": expert_map,
            "records": [
                {
                    "rank": 0,
                    "topk_ids_of_layer": torch.tensor(
                        [
                            [[0, 1, -1], [1, 0, -1]],
                            [[0, 2, -1], [0, 2, -1]],
                        ]
                    ),
                }
            ],
        },
        {
            "rank": 1,
            "last_physical_to_logical_map": expert_map,
            "records": [
                {
                    "rank": 1,
                    "topk_ids_of_layer": torch.tensor(
                        [
                            [[2, 3, -1]],
                            [[1, 3, -1]],
                        ]
                    ),
                }
            ],
        },
    ]

    top_k, layers = compact_layers_from_recorder_packs(packs, num_ranks=2)

    assert top_k == 2
    assert layers[0]["source_rank"].tolist() == [0, 1]
    assert layers[0]["topk_experts"].tolist() == [[0, 2], [1, 3]]
    assert layers[0]["count"].tolist() == [2, 1]
    assert layers[1]["topk_experts"].tolist() == [[0, 1], [2, 3]]
