import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.expert_affinity_graph import RoutedArrays
from sglang.srt.eplb.grace_plus_replication import _route_quota


_PATH = Path(__file__).parents[4] / "benchmark" / "benchmark_a2a_plan.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_a2a_plan", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_model_hidden_size():
    assert _MODULE._config_hidden_size(SimpleNamespace(hidden_size=4096)) == 4096

    with pytest.raises(ValueError, match="invalid hidden_size"):
        _MODULE._config_hidden_size(SimpleNamespace())


def test_physical_maps_pad_layouts_and_select_source_local_replicas():
    layouts = {
        "baseline": {0: (0,), 1: (0,), 2: (1,), 3: (1,)},
        "plan": {0: (1, 0), 1: (0,), 2: (1,), 3: (1,)},
    }

    slots, maps = _MODULE._physical_maps(layouts, num_experts=4, num_ranks=2)

    assert slots == 3
    assert maps["plan"][0][0] // slots == 0
    assert maps["plan"][1][0] // slots == 1
    assert [value // slots for value in maps["baseline"][0]] == [0, 0, 1, 1]


def test_physical_maps_use_source_specific_routing():
    layouts = {
        "plan": {0: (0, 1), 1: (0,), 2: (1,), 3: (1,)},
    }
    routing = [[1, 0, 1, 1], [0, 0, 1, 1]]

    slots, maps = _MODULE._physical_maps(
        layouts,
        num_experts=4,
        num_ranks=2,
        routes={"plan": routing},
    )

    assert maps["plan"][0][0] // slots == 1
    assert maps["plan"][1][0] // slots == 0


def test_normalize_quota_matches_replica_order():
    raw = {
        "gate": {
            "replicas": {"0": [0, 1], "1": [1]},
            "quota": [[[6, 4], [1]], [[0, 10], [1]]],
        }
    }
    plan = {0: (0, 1), 1: (1,)}

    assert _MODULE._normalize_quota(raw, "gate", plan, 2, 2) == raw["gate"]["quota"]


def test_normalize_quota_accepts_disabled_and_unobserved_routes():
    plan = {0: (0, 1)}

    assert _MODULE._normalize_quota({"quota": None}, "gate", plan, 1, 2) is None
    assert _MODULE._normalize_quota(
        {"quota": [[[1, 0]], [[0, 0]]]}, "gate", plan, 1, 2
    ) == [[[1, 0]], [[0, 0]]]


def test_scale_quota_preserves_total_with_stable_remainders():
    assert _MODULE._scale_quota([3, 1], 6) == [5, 1]


def test_quota_topk_uses_local_first_deterministic_prefix():
    logical_topk = torch.tensor([[0, 1], [2, 0], [1, 0], [0, 2]])
    replicas = torch.tensor([[4, 0], [5, -1], [2, -1]])
    quota = torch.tensor([[[1, 3], [4, 0], [4, 0]]])

    routed = _MODULE._quota_topk(logical_topk, replicas, quota, source=0, slots=4)

    assert routed.tolist() == [[0, 5], [2, 0], [5, 0], [4, 2]]

    planner = _route_quota(
        RoutedArrays(
            source_rank=torch.zeros(len(logical_topk), dtype=torch.int64).numpy(),
            topk_experts=logical_topk.numpy(),
            count=torch.ones(len(logical_topk), dtype=torch.int64).numpy(),
        ),
        quota=torch.tensor(
            [[[3, 1], [0, 2], [2, 0]], [[0, 0], [0, 0], [0, 0]]]
        ).numpy(),
        source_demand=torch.tensor([[4, 0], [2, 0], [2, 0]]).numpy(),
        replicas={0: (1, 0), 1: (1,), 2: (0,)},
    )
    benchmark_remote = sum(torch.any(routed // 4 == rank, dim=1).sum() for rank in (1,))
    assert planner.remote == int(benchmark_remote)
