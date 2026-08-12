"""CPU tests for bundle-aware DeepEP replica replay."""

from sglang.srt.eplb.bundle_aware_replica_planner import (
    BundleAwareReplicaPlanner,
    RoutedToken,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _planner(**kwargs):
    defaults = dict(
        num_ranks=2,
        baseline_rank_by_expert={0: 1, 1: 1},
        replica_slots_per_rank=2,
        compute_weight=0.0,
        communication_weight=1.0,
    )
    defaults.update(kwargs)
    return BundleAwareReplicaPlanner(**defaults)


def test_one_expert_copy_does_not_remove_deepep_transfer():
    planner = _planner()
    tokens = [RoutedToken(0, (0, 1), count=100)]
    placement = planner._baseline_placement()
    copied = planner._apply_action(
        placement,
        next(
            action
            for action in planner._candidates(tokens, placement)
            if action.experts == (0,)
        ),
    )

    assert copied is not None
    assert planner._evaluate(tokens, placement).unique_remote_rank_copies == 100
    assert planner._evaluate(tokens, copied).unique_remote_rank_copies == 100


def test_bundle_closure_removes_deepep_transfer_and_is_planned():
    plan = _planner().plan([RoutedToken(0, (0, 1), count=100)])

    assert plan.baseline.unique_remote_rank_copies == 100
    assert plan.final.unique_remote_rank_copies == 0
    assert len(plan.actions) == 1  # One collective copy, not two misleading copies.
    assert plan.actions[0].kind == "bundle-closure"
    assert plan.actions[0].experts == (0, 1)


def test_high_compute_weight_can_choose_single_helper_copy():
    planner = BundleAwareReplicaPlanner(
        num_ranks=2,
        baseline_rank_by_expert={0: 0, 1: 0},
        replica_slots_per_rank=1,
        compute_weight=100.0,
        communication_weight=0.01,
    )
    # e1 creates a hot home rank; e0 is then worth serving remotely despite
    # adding one transfer.  This exercises the joint, not comm-first, score.
    plan = planner.plan(
        [RoutedToken(0, (1,), count=100), RoutedToken(0, (0,), count=10)],
        max_actions=1,
    )

    assert plan.actions[0].kind == "single"
    assert plan.actions[0].experts == (0,)
    assert plan.final.compute_load == [100, 10]
    assert plan.final.unique_remote_rank_copies == 10
