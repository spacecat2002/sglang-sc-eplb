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
    _capacity_export_replicas,
    _instance_quotas,
    _greedy_instance_quotas,
    _joint_quotas,
    _quota_prefix,
    _rebalance_quota_compute,
    _route_quota,
    _source_quotas,
    balance_replica_compute,
    evaluate_replicated_placement,
    replicate_hot_experts,
    replicate_source_top_experts,
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


def test_source_top_experts_copies_at_most_the_per_rank_limit():
    tokens = [
        RoutedToken(0, (2,), 10),
        RoutedToken(0, (3,), 7),
        RoutedToken(0, (1,), 20),
        RoutedToken(1, (0,), 9),
    ]

    result = replicate_source_top_experts(
        tokens,
        {0: 0, 1: 0, 2: 1, 3: 1},
        num_ranks=2,
        max_extra_per_rank=1,
    )

    assert result.replicas_by_expert == {
        0: (0, 1),
        1: (0,),
        2: (1, 0),
        3: (1,),
    }
    assert result.extra_copies == 2
    assert result.quota_by_source is None
    assert result.metrics == evaluate_replicated_placement(
        tokens, result.replicas_by_expert, num_ranks=2
    )


def test_source_top_experts_quota_limits_local_compute():
    result = replicate_source_top_experts(
        [
            RoutedToken(0, (0,), 20),
            RoutedToken(1, (0,), 80),
            RoutedToken(1, (1,), 50),
        ],
        {0: 0, 1: 1},
        num_ranks=2,
        max_extra_per_rank=1,
        compute_imbalance_limit=1,
    )

    assert result.replicas_by_expert == {0: (0, 1), 1: (1,)}
    assert result.metrics.compute_load == (75, 75)
    assert result.metrics.remote == 55


def test_source_top_experts_runs_compute_solver_in_one_call():
    tokens = [
        RoutedToken(0, (0,), 100),
        RoutedToken(1, (1,), 10),
        RoutedToken(0, (2,), 40),
    ]
    primary = {0: 0, 1: 1, 2: 1}
    communication = replicate_source_top_experts(
        tokens, primary, num_ranks=2, max_extra_per_rank=0
    )
    separate = balance_replica_compute(
        tokens, communication, num_ranks=2, max_extra_per_rank=1
    )
    fused = replicate_source_top_experts(
        tokens,
        primary,
        num_ranks=2,
        max_extra_per_rank=0,
        max_compute_extra_per_rank=1,
        compute_imbalance_limit=1.0,
    )

    assert fused == separate


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


def test_compute_replica_is_retained_when_budget_is_infeasible():
    tokens = [
        RoutedToken(0, (0,), 100),
        RoutedToken(1, (1,), 10),
        RoutedToken(0, (2,), 10),
    ]
    communication = replicate_hot_experts(
        tokens,
        {0: 0, 1: 1, 2: 1},
        num_ranks=2,
        max_extra_per_rank=0,
    )

    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=1,
        communication_budget_ratio=1.0,
    )

    assert balanced.balance_copies == 1
    assert max(balanced.metrics.compute_load) < max(communication.metrics.compute_load)
    assert balanced.replicas_by_expert[0] == (0, 1)
    assert balanced.metrics.remote > communication.metrics.remote


def test_compute_replica_is_added_when_budget_allows_compute_balance():
    tokens = [
        RoutedToken(0, (0,), 100),
        RoutedToken(1, (1,), 10),
        RoutedToken(0, (2,), 40),
    ]
    communication = replicate_hot_experts(
        tokens,
        {0: 0, 1: 1, 2: 1},
        num_ranks=2,
        max_extra_per_rank=0,
    )

    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        max_extra_per_rank=1,
        communication_budget_ratio=3.0,
    )

    assert balanced.balance_copies == 1
    assert balanced.replicas_by_expert[0] == (0, 1)
    assert max(balanced.metrics.compute_load) < max(communication.metrics.compute_load)
    budget = 3 * communication.metrics.remote
    assert balanced.metrics.remote <= budget


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


def test_compute_balancing_prioritizes_compute_over_zero_remote():
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

    assert balanced.balance_copies == 2
    assert balanced.metrics.compute_load == (7, 7, 6)
    assert balanced.metrics.remote > communication.metrics.remote


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


def test_greedy_quota_places_fixed_load_before_replicated_experts():
    _, loads = _greedy_instance_quotas(
        np.array([40, 5]),
        {0: (0, 1), 1: (1,)},
        2,
    )

    assert loads.tolist() == [23, 22]


def test_compute_rebalance_moves_quota_between_experts_to_meet_capacity():
    replicas = {0: (2,), 1: (0, 2), 2: (1, 2)}
    quota = np.zeros((3, 3, 3), dtype=np.int64)
    quota[0, 0, 2] = 4
    quota[0, 1, 0] = 9
    quota[0, 1, 2] = 3
    quota[0, 2, 1] = 9
    quota[0, 2, 2] = 5
    loads = quota.sum(axis=(0, 1), dtype=np.int64)

    _rebalance_quota_compute(quota, replicas, loads, capacity=10)

    assert loads.tolist() == [10, 10, 10]
    assert np.array_equal(
        quota.sum(axis=2),
        np.array([[4, 12, 14], [0, 0, 0], [0, 0, 0]]),
    )


def test_compute_selector_reuses_existing_augmenting_capacity():
    replicas = {0: (2,), 1: (0, 2), 2: (1, 2)}
    selected, added = _capacity_export_replicas(
        np.array([[4, 0, 0], [12, 0, 0], [14, 0, 0]]),
        replicas.copy(),
        num_ranks=3,
        max_extra_per_rank=1,
    )

    assert selected == replicas
    assert added == 0


def test_compute_selector_copies_locally_instead_of_balancing_remotely():
    replicas = {0: (0, 1), 1: (1, 0)}
    source_demand = np.array([[0, 10, 0], [10, 0, 10]])

    selected, added = _capacity_export_replicas(
        source_demand,
        replicas.copy(),
        num_ranks=3,
        max_extra_per_rank=1,
    )

    assert selected[1] == (1, 0, 2)
    assert added == 1


def test_compute_rebalance_uses_an_augmenting_path():
    quota = np.zeros((3, 2, 3), dtype=np.int64)
    quota[0, 0, 0] = 12
    quota[0, 1, 1] = 10
    loads = quota.sum(axis=(0, 1), dtype=np.int64)

    _rebalance_quota_compute(quota, {0: (0, 1), 1: (1, 2)}, loads, capacity=8)

    assert loads.tolist() == [8, 8, 6]
    assert np.array_equal(
        quota.sum(axis=2),
        np.array([[12, 10], [0, 0], [0, 0]]),
    )


def test_compute_rebalance_skips_an_unreachable_overloaded_rank():
    replicas = {0: (0, 1, 2), 1: (0, 2), 2: (0,)}
    source_demand = np.array([[15, 0, 0], [7, 0, 0], [16, 0, 0]])
    instance, loads = _greedy_instance_quotas(
        source_demand.sum(axis=1), replicas, 3
    )
    quota = _source_quotas(
        source_demand, instance, np.zeros((3, 3), dtype=np.int64)
    )

    assert loads.tolist() == [16, 8, 14]
    _rebalance_quota_compute(quota, replicas, loads, capacity=13)

    # Rank 0 is fixed above capacity, but rank 2 can still export to rank 1.
    assert loads.tolist() == [16, 9, 13]


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
        np.array([0, 0, 0]),
        np.array([[0, 1], [0, 1], [0, 1]]),
        np.array([4, 4, 2]),
    )
    quota = np.zeros((2, 2, 2), dtype=np.int64)
    quota[0, 0] = [6, 4]
    quota[0, 1] = [4, 6]
    replicas = {0: (0, 1), 1: (0, 1)}

    prefix = _quota_prefix(quota, replicas)
    with patch("sglang.srt.eplb.grace_plus_replication._CHUNK_SIZE", 1):
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
