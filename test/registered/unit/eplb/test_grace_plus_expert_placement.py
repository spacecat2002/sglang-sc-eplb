from unittest.mock import patch

import numpy as np

from sglang.srt.eplb.expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    build_co_routing_graph,
    evaluate_primary_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_expert_placement
from sglang.srt.eplb.grace_plus_expert_placement import (
    grace_plus_expert_placement,
)
from sglang.srt.eplb.grace_plus_replication import (
    ReplicaPlacement,
    _instance_quotas,
    _joint_quotas,
    _quota_prefix,
    _route_quota,
    balance_replica_compute,
    evaluate_replicated_placement,
    replicate_hot_experts,
)


def test_grace_plus_accepts_arrays_and_tokens():
    tokens = [RoutedToken(0, (0, 1), 7), RoutedToken(1, (1, 2), 5)]
    arrays = RoutedArrays(
        np.array([0, 1], dtype=np.int16),
        np.array([[0, 1], [1, 2]], dtype=np.int16),
        np.array([7, 5], dtype=np.int16),
    )
    graph = build_co_routing_graph(tokens, experts=range(3))
    grace = grace_expert_placement(graph, num_ranks=2)
    kwargs = dict(
        experts=range(3),
        num_ranks=2,
        initial_placement=grace.rank_by_expert,
        capacity_ratio=0.5,
        objective="remote",
        refine_rounds=1,
        swap_rounds=0,
    )
    assert (
        grace_plus_expert_placement(tokens, **kwargs)["rank_by_expert"]
        == grace_plus_expert_placement(arrays, **kwargs)["rank_by_expert"]
    )


def test_ingress_egress_objective_is_supported():
    tokens = [
        RoutedToken(0, (0, 1), 20),
        RoutedToken(0, (2, 3), 20),
        RoutedToken(1, (0, 2), 3),
        RoutedToken(1, (1, 3), 3),
    ]
    result = grace_plus_expert_placement(
        tokens,
        experts=range(4),
        num_ranks=2,
        capacity_ratio=0,
        initial_placement={0: 0, 1: 0, 2: 1, 3: 1},
        align_groups=True,
        objective="ingress-egress",
        refine_rounds=1,
        swap_rounds=1,
    )
    assert result["metrics"].max_ingress >= 0


def test_remote_objective_keeps_grace_remote_non_increasing():
    tokens = [RoutedToken(0, (0, 1), 10), RoutedToken(1, (2, 3), 10)]
    grace = grace_expert_placement(
        build_co_routing_graph(tokens, experts=range(4)), num_ranks=2
    )
    refined = grace_plus_expert_placement(
        tokens,
        experts=range(4),
        num_ranks=2,
        capacity_ratio=0,
        initial_placement=grace.rank_by_expert,
        objective="remote",
        refine_rounds=2,
        swap_rounds=1,
    )
    assert evaluate_primary_remote(
        tokens, refined["rank_by_expert"]
    ) <= evaluate_primary_remote(tokens, grace.rank_by_expert)


def test_default_communication_refinement_does_not_worsen_compute_peak():
    tokens = [
        RoutedToken(1, (0,), 50),
        RoutedToken(0, (2,), 50),
        RoutedToken(0, (1,), 1),
        RoutedToken(1, (3,), 1),
    ]
    initial = {0: 0, 1: 0, 2: 1, 3: 1}

    result = grace_plus_expert_placement(
        tokens,
        experts=range(4),
        num_ranks=2,
        capacity_ratio=0.5,
        initial_placement=initial,
        objective="remote",
        refine_rounds=2,
        swap_rounds=0,
    )

    initial_peak = max(
        evaluate_replicated_placement(tokens, initial, num_ranks=2).compute_load
    )
    assert max(result["metrics"].compute_load) <= initial_peak


def test_replication_uses_source_local_routing_and_respects_slots():
    tokens = [
        RoutedToken(0, (0,), 20),
        RoutedToken(1, (1,), 15),
    ]
    result = replicate_hot_experts(
        tokens,
        {0: 1, 1: 0},
        num_ranks=2,
        objective="remote",
        max_extra_per_rank=1,
        compute_imbalance_limit=2,
    )

    assert result.extra_copies == 2
    assert result.replicas_by_expert == {0: (1, 0), 1: (0, 1)}
    assert result.metrics.remote == 0

    congestion = replicate_hot_experts(
        tokens,
        {0: 1, 1: 0},
        num_ranks=2,
        objective="ingress-egress",
        max_extra_per_rank=1,
        hot_experts=1,
        compute_imbalance_limit=2,
    )
    assert congestion.extra_copies == 1
    assert max(congestion.metrics.max_ingress, congestion.metrics.max_egress) == 15


def test_compute_balancing_adds_replica_and_reroutes_static_demand():
    tokens = [
        RoutedToken(0, (0,), 20),
        RoutedToken(1, (0,), 20),
        RoutedToken(0, (1,), 5),
    ]
    communication = replicate_hot_experts(
        tokens,
        {0: 0, 1: 1},
        num_ranks=2,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=1,
    )

    assert balanced.balance_copies == 1
    assert balanced.replicas_by_expert[0] == (0, 1)
    assert balanced.routing_by_source[1][0] == 1
    assert max(balanced.metrics.compute_load) < max(communication.metrics.compute_load)
    assert balanced.metrics.remote < communication.metrics.remote


def test_replication_caches_source_demand_without_an_extra_scan():
    communication = replicate_hot_experts(
        [
            RoutedToken(0, (0,), 7),
            RoutedToken(1, (0,), 5),
            RoutedToken(1, (1,), 5),
        ],
        {0: 0, 1: 1},
        num_ranks=2,
        max_extra_per_rank=0,
    )

    assert communication.source_demand.tolist() == [[7, 5], [0, 5]]
    with patch(
        "sglang.srt.eplb.grace_plus_replication._source_demand",
        side_effect=AssertionError("source demand scanned twice"),
    ):
        balance_replica_compute(
            [
                RoutedToken(0, (0,), 7),
                RoutedToken(1, (0,), 5),
                RoutedToken(1, (1,), 5),
            ],
            communication,
            num_ranks=2,
        )


def test_compute_balancing_prefers_the_largest_remote_source():
    tokens = [
        RoutedToken(0, (0,), 100),
        RoutedToken(1, (0,), 30),
        RoutedToken(2, (0,), 10),
        RoutedToken(1, (1,), 60),
        RoutedToken(2, (2,), 60),
    ]
    communication = replicate_hot_experts(
        tokens,
        {0: 0, 1: 1, 2: 2},
        num_ranks=3,
        max_extra_per_rank=0,
    )

    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=3,
        max_extra_per_rank=1,
    )

    assert balanced.replicas_by_expert[0][:2] == (0, 1)


def test_compute_balancing_rejects_remote_for_variance_only():
    tokens = [
        RoutedToken(0, (0,), 10),
        RoutedToken(1, (1,), 8),
        RoutedToken(2, (2,), 2),
    ]
    communication = replicate_hot_experts(
        tokens,
        {0: 0, 1: 1, 2: 2},
        num_ranks=3,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=3,
        max_extra_per_rank=1,
    )

    assert balanced.balance_copies == 0
    assert balanced.metrics == communication.metrics


def test_compute_balancing_zero_budget_still_generates_quota():
    tokens = [RoutedToken(0, (0,), 10), RoutedToken(1, (0,), 10)]
    communication = replicate_hot_experts(
        tokens,
        {0: 0},
        num_ranks=2,
        max_extra_per_rank=0,
    )

    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=0,
    )

    assert balanced.balance_copies == 0
    assert balanced.quota_by_source == (((10,),), ((10,),))


def test_compute_balancing_uses_quota_routing():
    tokens = [RoutedToken(source, (0,), 20) for source in range(4)]
    replicas = {0: (0, 1)}
    communication = ReplicaPlacement(
        replicas,
        ((0,), (1,), (0,), (0,)),
        evaluate_replicated_placement(tokens, replicas, num_ranks=4),
        extra_copies=1,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=4,
        max_extra_per_rank=1,
    )

    assert balanced.replicas_by_expert[0] == (0, 1, 2, 3)
    assert balanced.metrics.compute_load == (20, 20, 20, 20)
    assert balanced.quota_by_source is not None


def test_quota_splits_one_source_expert_demand():
    tokens = [RoutedToken(0, (0,), 100), RoutedToken(1, (0,), 20)]
    communication = replicate_hot_experts(
        tokens,
        {0: 0},
        num_ranks=2,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=1,
    )

    assert balanced.metrics.compute_load == (60, 60)
    assert balanced.quota_by_source[0][0] == (60, 40)
    assert balanced.quota_by_source[1][0] == (0, 20)


def test_unobserved_source_expert_keeps_valid_routing():
    tokens = [RoutedToken(0, (0,), 10)]
    communication = replicate_hot_experts(
        tokens,
        {0: 1},
        num_ranks=2,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=0,
    )

    assert balanced.routing_by_source[1][0] == 1
    assert balanced.quota_by_source[1][0] == (1,)


def test_instance_quota_is_globally_balanced():
    _, loads = _instance_quotas(
        np.array([14, 7, 13, 15]),
        {0: (0, 1), 1: (0,), 2: (2,), 3: (0, 2)},
        3,
    )

    assert loads.tolist() == [18, 13, 18]


def test_joint_quota_chooses_local_among_compute_optimal_solutions():
    source_demand = np.array([[10, 0], [0, 10]])
    replicas = {0: (1, 0), 1: (0, 1)}
    routing = np.array([[0, 0], [1, 1]])

    quota = _joint_quotas(source_demand, replicas, routing)

    assert quota.sum(axis=(0, 1)).tolist() == [10, 10]
    assert quota[0, 0].tolist() == [10, 0]
    assert quota[1, 1].tolist() == [0, 10]


def test_joint_quota_shares_rank_capacity_across_experts():
    source_demand = np.array([[10, 0], [10, 0]])
    replicas = {0: (0, 1), 1: (0, 1)}
    routing = np.zeros((2, 2), dtype=np.int64)

    quota = _joint_quotas(source_demand, replicas, routing)

    assert quota.sum(axis=(0, 1)).tolist() == [10, 10]
    assert quota[:, :, 0].sum() == 10
    assert quota[:, :, 1].sum() == 10


def test_quota_prefix_routes_local_before_remote_and_matches_counts():
    arrays = RoutedArrays(
        np.array([0]),
        np.array([[0, 1]]),
        np.array([10]),
    )
    quota = np.zeros((2, 2, 2), dtype=np.int64)
    quota[0, 0] = [6, 4]
    quota[0, 1] = [4, 6]
    replicas = {0: (0, 1), 1: (0, 1)}

    prefix = _quota_prefix(quota, replicas)
    metrics = _route_quota(arrays, quota, np.array([[10, 0], [10, 0]]), replicas)

    assert prefix[0, 0].tolist() == [6, 10]
    assert prefix[0, 1].tolist() == [4, 10]
    assert metrics.remote == 6
    assert metrics.compute_load == (10, 10)


def test_quota_prefix_puts_local_replica_before_primary():
    quota = np.zeros((2, 1, 2), dtype=np.int64)
    quota[1, 0] = [4, 6]

    prefix = _quota_prefix(quota, {0: (0, 1)})

    assert prefix[1, 0].tolist() == [10, 6]
