import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[4] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from solve_robust_co_routing_graph import (  # noqa: E402
    _LayerTrace,
    _quality,
    _updated_weights,
    _weighted_cpu_trace,
)
from sglang.srt.eplb.bundle_aware_replica_planner import RoutedToken  # noqa: E402


def test_weighted_trace_equalizes_different_dataset_sizes():
    large = _LayerTrace(
        "large",
        "large.json",
        [RoutedToken(0, (0, 1), 100)],
        False,
        (0, 1),
        {0: 100, 1: 100},
        100,
        1,
    )
    small = _LayerTrace(
        "small",
        "small.json",
        [RoutedToken(0, (0, 1), 10)],
        False,
        (0, 1),
        {0: 10, 1: 10},
        10,
        1,
    )

    merged = _weighted_cpu_trace(
        [large, small],
        weights=[0.5, 0.5],
        normalizers=[100, 10],
        precision=1024,
    )

    assert merged[0].count == merged[1].count


def test_reweighting_increases_weight_of_worst_dataset():
    weights = _updated_weights(
        [0.5, 0.5],
        [{"normalized_remote": 0.5}, {"normalized_remote": 1.0}],
        rate=1.0,
    )

    assert sum(weights) == 1.0
    assert weights[1] > weights[0]


def test_quality_prioritizes_compute_cap_before_communication():
    communication_better_but_over_cap = [
        {"normalized_remote": 0.4, "compute_inflation": 1.3},
        {"normalized_remote": 0.4, "compute_inflation": 1.0},
    ]
    communication_worse_but_feasible = [
        {"normalized_remote": 0.6, "compute_inflation": 1.1},
        {"normalized_remote": 0.6, "compute_inflation": 1.0},
    ]

    over_cap_key, _ = _quality(
        communication_better_but_over_cap,
        worst_weight=1.0,
        max_compute_inflation=1.2,
    )
    feasible_key, _ = _quality(
        communication_worse_but_feasible,
        worst_weight=1.0,
        max_compute_inflation=1.2,
    )

    assert feasible_key < over_cap_key
