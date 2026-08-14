from sglang.srt.eplb.bundle_aware_replica_planner import RoutedToken
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_primary_remote,
)


def test_build_graph_counts_bundle_pairs_and_demand():
    graph = build_co_routing_graph(
        [
            RoutedToken(0, (0, 1, 2), count=3),
            RoutedToken(1, (0, 1), count=2),
        ]
    )
    assert graph.experts == (0, 1, 2)
    assert graph.demand == {0: 5, 1: 5, 2: 3}
    assert graph.edges == {(0, 1): 5, (0, 2): 3, (1, 2): 3}


def test_solver_preserves_capacity_and_reduces_graph_cut():
    tokens = [
        RoutedToken(0, (0, 1), count=10),
        RoutedToken(1, (2, 3), count=10),
    ]
    graph = build_co_routing_graph(tokens)
    placement = CoRoutingGraphSolver(
        num_ranks=2, slots_per_rank=2, max_rounds=4
    ).solve(graph)
    assert all(len(experts) <= 2 for experts in placement.experts_by_rank.values())
    assert sorted(placement.rank_by_expert) == [0, 1, 2, 3]
    assert placement.final_cut <= placement.initial_cut


def test_primary_remote_replays_distinct_destination_ranks():
    tokens = [RoutedToken(0, (0, 1, 2), count=4)]
    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 1}) == 1 * 4
    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 2}) == 2 * 4

