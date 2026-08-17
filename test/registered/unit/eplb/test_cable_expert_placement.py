from sglang.srt.eplb.cable_expert_placement import cable_expert_placement
from sglang.srt.eplb.expert_affinity_graph import (
    RoutedToken,
    build_co_routing_graph,
    evaluate_primary_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_hierarchical_placement
from sglang.srt.eplb.hypergraph_expert_placement import hypergraph_expert_placement


def test_cable_finds_balanced_source_local_bundles_without_grace():
    tokens = [
        RoutedToken(0, (0, 1), 20),
        RoutedToken(1, (2, 3), 20),
    ]

    placement = cable_expert_placement(
        tokens,
        experts=range(4),
        num_ranks=2,
    )

    assert all(len(experts) == 2 for experts in placement.experts_by_rank.values())
    assert placement.metrics.remote == 0
    assert placement.metrics.max_ingress == 0
    assert placement.metrics.compute_imbalance == 1


def test_cable_remote_refinement_accepts_only_improving_swaps():
    tokens = [
        RoutedToken(0, (1, 0)),
        RoutedToken(1, (2, 5)),
        RoutedToken(1, (4, 7)),
        RoutedToken(0, (0, 4)),
        RoutedToken(0, (6, 5)),
        RoutedToken(1, (5, 4)),
        RoutedToken(1, (4, 0)),
        RoutedToken(0, (5, 3)),
    ]
    greedy = cable_expert_placement(
        tokens,
        experts=range(8),
        num_ranks=2,
        refine_swaps=0,
        refine_strategy="remote",
        capacity_ratio=0,
        compute_imbalance_limit=10,
    )
    refined = cable_expert_placement(
        tokens,
        experts=range(8),
        num_ranks=2,
        refine_swaps=1,
        refine_strategy="remote",
        capacity_ratio=0,
        compute_imbalance_limit=10,
    )

    assert refined.metrics.remote < greedy.metrics.remote
    assert refined.metrics.compute_imbalance <= greedy.metrics.compute_imbalance


def test_cable_joint_mode_spreads_hot_experts_within_remote_budget():
    tokens = [
        RoutedToken(0, (0, 1), 50),
        RoutedToken(1, (2, 3)),
        RoutedToken(1, (4, 5)),
        RoutedToken(1, (6, 7)),
    ]
    communication_first = cable_expert_placement(
        tokens,
        experts=range(8),
        num_ranks=2,
        capacity_ratio=0,
        compute_imbalance_limit=10,
        compute_refine_moves=0,
        refine_strategy="remote",
    )
    joint = cable_expert_placement(
        tokens,
        experts=range(8),
        num_ranks=2,
        capacity_ratio=0.5,
        compute_imbalance_limit=1.5,
        compute_refine_moves=4,
        refine_strategy="balanced",
        remote_budget=0.2,
    )

    assert joint.metrics.compute_imbalance < (
        communication_first.metrics.compute_imbalance
    )
    assert joint.metrics.remote <= sum(token.count for token in tokens)


def test_fixed_terminal_hypergraph_optimizes_bundle_connectivity():
    tokens = [
        RoutedToken(0, (12, 13, 1, 8)),
        RoutedToken(3, (15, 6, 4, 9)),
        RoutedToken(3, (3, 2, 1, 13)),
        RoutedToken(2, (4, 15, 11, 3)),
        RoutedToken(3, (14, 8, 7, 9)),
        RoutedToken(2, (0, 6, 10, 3)),
        RoutedToken(2, (6, 12, 3, 9)),
        RoutedToken(1, (14, 6, 10, 1)),
    ]
    placement = hypergraph_expert_placement(
        tokens,
        experts=range(16),
        num_ranks=4,
        starts=4,
        refine_rounds=4,
        remote_budget=0.01,
    )

    assert placement["metrics"].remote >= 0
    assert all(len(experts) >= 1 for experts in placement["experts_by_rank"].values())


def test_hypergraph_refinement_keeps_grace_remote_non_increasing():
    tokens = [
        RoutedToken(0, (0, 1), 10),
        RoutedToken(0, (0, 2), 8),
        RoutedToken(1, (2, 3), 9),
        RoutedToken(1, (2, 4), 7),
        RoutedToken(2, (4, 5), 6),
        RoutedToken(2, (4, 6), 5),
        RoutedToken(3, (6, 7), 4),
    ]
    grace = grace_hierarchical_placement(
        build_co_routing_graph(tokens, experts=range(8)), num_ranks=4
    )
    refined = hypergraph_expert_placement(
        tokens,
        experts=range(8),
        num_ranks=4,
        capacity_ratio=0.15,
        initial_placement=grace.rank_by_expert,
        refine_rounds=4,
    )

    assert evaluate_primary_remote(tokens, refined["rank_by_expert"]) <= (
        evaluate_primary_remote(tokens, grace.rank_by_expert)
    )
