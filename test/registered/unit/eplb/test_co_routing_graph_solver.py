from sglang.srt.eplb.bundle_aware_replica_planner import RoutedToken
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_primary_remote,
    refine_hypergraph_placement,
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
    placement = CoRoutingGraphSolver(num_ranks=2, slots_per_rank=2, max_rounds=4).solve(
        graph
    )
    assert all(len(experts) <= 2 for experts in placement.experts_by_rank.values())
    assert sorted(placement.rank_by_expert) == [0, 1, 2, 3]
    assert placement.final_cut <= placement.initial_cut


def test_solver_can_run_until_no_improving_swap_remains():
    graph = build_co_routing_graph(
        [
            RoutedToken(0, (0, 1), count=10),
            RoutedToken(0, (0, 2), count=8),
            RoutedToken(1, (1, 3), count=7),
            RoutedToken(1, (2, 3), count=5),
        ]
    )
    initial = CoRoutingGraphSolver(num_ranks=2, slots_per_rank=2, max_rounds=0).solve(
        graph
    )
    converged = CoRoutingGraphSolver(
        num_ranks=2, slots_per_rank=2, max_rounds=None
    ).solve(graph)

    assert converged.final_cut <= initial.final_cut
    assert converged.iterations >= initial.iterations


def test_primary_remote_replays_distinct_destination_ranks():
    tokens = [RoutedToken(0, (0, 1, 2), count=4)]
    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 1}) == 1 * 4
    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 2}) == 2 * 4


def test_hypergraph_refinement_optimizes_exact_bundle_remote():
    tokens = [
        RoutedToken(0, (0, 1), count=10),
        RoutedToken(1, (2, 3), count=10),
    ]
    initial = {0: 0, 1: 1, 2: 0, 3: 1}

    placement = refine_hypergraph_placement(
        tokens,
        initial,
        num_ranks=2,
        max_rounds=None,
    )

    assert placement.initial_remote == 20
    assert placement.final_remote == 0
    assert placement.iterations == 1
    assert evaluate_primary_remote(tokens, placement.rank_by_expert) == 0


def test_hypergraph_refinement_can_improve_converged_pairwise_graph():
    tokens = [
        RoutedToken(1, (0, 1, 3), 3),
        RoutedToken(2, (0, 2, 5), 17),
        RoutedToken(0, (0, 3, 5), 14),
        RoutedToken(0, (0, 1, 3), 2),
        RoutedToken(2, (0, 1, 5), 19),
        RoutedToken(2, (0, 1, 3), 2),
        RoutedToken(2, (1, 2, 3), 5),
        RoutedToken(2, (0, 2, 4), 18),
        RoutedToken(2, (0, 1, 5), 12),
        RoutedToken(0, (0, 4, 5), 7),
        RoutedToken(1, (3, 4, 5), 11),
        RoutedToken(1, (2, 3, 4), 10),
    ]
    graph = build_co_routing_graph(tokens)
    graph_placement = CoRoutingGraphSolver(
        num_ranks=3,
        slots_per_rank=2,
        max_rounds=None,
    ).solve(graph)
    graph_remote = evaluate_primary_remote(tokens, graph_placement.rank_by_expert)

    hypergraph_placement = refine_hypergraph_placement(
        tokens,
        graph_placement.rank_by_expert,
        num_ranks=3,
        max_rounds=None,
    )

    assert graph_remote == 164
    assert hypergraph_placement.final_remote == 145


def test_hypergraph_compute_balance_preserves_exact_remote():
    tokens = [
        RoutedToken(2, (0,), 100),
        RoutedToken(2, (1,), 90),
        RoutedToken(2, (2,), 10),
        RoutedToken(2, (3,), 1),
    ]
    initial = {0: 0, 1: 0, 2: 1, 3: 1}

    placement = refine_hypergraph_placement(
        tokens,
        initial,
        num_ranks=3,
        max_rounds=0,
        balance_rounds=None,
    )

    assert placement.initial_remote == 201
    assert placement.final_remote == 201
    assert placement.balance_iterations == 1
    assert placement.experts_by_rank == {0: (1, 2), 1: (0, 3), 2: ()}
