import numpy as np

from sglang.srt.eplb.expert_affinity_graph import (
    RoutedArrays,
    RoutedToken,
    build_co_routing_graph,
    evaluate_primary_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_hierarchical_placement
from sglang.srt.eplb.grace_plus_expert_placement import (
    grace_plus_expert_placement,
)
from sglang.srt.eplb.grace_plus_replication import (
    ReplicaPlacement,
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
    grace = grace_hierarchical_placement(graph, num_ranks=2)
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
    grace = grace_hierarchical_placement(
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


def test_replication_uses_source_local_routing_and_respects_slots():
    tokens = [
        RoutedToken(0, (0,), 20),
        RoutedToken(1, (1,), 15),
    ]
    result = replicate_hot_experts(
        tokens,
        {0: 1, 1: 0},
        num_ranks=2,
        ranks_per_node=2,
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
        ranks_per_node=2,
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
        ranks_per_node=2,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        ranks_per_node=2,
        objective="remote",
        max_extra_per_rank=1,
    )

    assert balanced.balance_copies == 1
    assert balanced.replicas_by_expert[0] == (0, 1)
    assert balanced.routing_by_source[1][0] == 1
    assert max(balanced.metrics.compute_load) < max(communication.metrics.compute_load)
    assert balanced.metrics.remote == communication.metrics.remote - 20


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
        ranks_per_node=3,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=3,
        ranks_per_node=3,
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
        ranks_per_node=2,
        max_extra_per_rank=0,
    )

    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        ranks_per_node=2,
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
        evaluate_replicated_placement(tokens, replicas, num_ranks=4, ranks_per_node=4),
        extra_copies=1,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=4,
        ranks_per_node=4,
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
        ranks_per_node=2,
        max_extra_per_rank=0,
    )
    balanced = balance_replica_compute(
        tokens,
        communication,
        num_ranks=2,
        ranks_per_node=2,
        max_extra_per_rank=1,
    )

    assert balanced.metrics.compute_load == (60, 60)
    assert balanced.quota_by_source[0][0] == (60, 40)
    assert balanced.quota_by_source[1][0] == (0, 20)
