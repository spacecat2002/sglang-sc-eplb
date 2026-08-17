from sglang.srt.eplb.cable_expert_placement import cable_expert_placement
from sglang.srt.eplb.expert_affinity_graph import RoutedToken


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
        tokens, experts=range(8), num_ranks=2, refine_swaps=0
    )
    refined = cable_expert_placement(
        tokens, experts=range(8), num_ranks=2, refine_swaps=1
    )

    assert refined.metrics.remote < greedy.metrics.remote
    assert refined.metrics.compute_imbalance <= greedy.metrics.compute_imbalance
