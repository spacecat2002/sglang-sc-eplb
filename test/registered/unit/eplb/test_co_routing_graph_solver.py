from sglang.srt.eplb.bundle_aware_replica_planner import RoutedToken
from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    build_co_routing_graph,
    evaluate_destination_rank_copies,
    evaluate_primary_remote,
    refine_hypergraph_placement,
)
from sglang.srt.eplb.offline_expert_placement import (
    evaluate_topology_hypergraph_objective,
    grace_hierarchical_placement,
    refine_load_constrained_hypergraph_placement,
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


def test_destination_rank_copies_include_source_rank():
    tokens = [RoutedToken(0, (0, 1, 2), count=4)]

    assert evaluate_destination_rank_copies(tokens, {0: 0, 1: 1, 2: 1}) == 8


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


def test_source_agnostic_hypergraph_optimizes_destination_cardinality():
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
        objective="source-agnostic",
    )

    assert placement.objective == "source-agnostic"
    assert placement.initial_remote == 20
    assert placement.final_remote == 0
    assert placement.initial_objective == 40
    assert placement.final_objective == 20
    assert placement.iterations == 1


def test_source_agnostic_hypergraph_is_source_permutation_invariant():
    first = [
        RoutedToken(0, (0, 1), count=10),
        RoutedToken(1, (2, 3), count=10),
    ]
    permuted = [
        RoutedToken(1, (0, 1), count=10),
        RoutedToken(0, (2, 3), count=10),
    ]
    initial = {0: 0, 1: 1, 2: 0, 3: 1}

    expected = refine_hypergraph_placement(
        first,
        initial,
        num_ranks=2,
        max_rounds=None,
        objective="source-agnostic",
    )
    actual = refine_hypergraph_placement(
        permuted,
        initial,
        num_ranks=2,
        max_rounds=None,
        objective="source-agnostic",
    )

    assert actual.rank_by_expert == expected.rank_by_expert
    assert actual.initial_objective == expected.initial_objective
    assert actual.final_objective == expected.final_objective
    assert actual.iterations == expected.iterations


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


def test_grace_grouping_respects_controlled_nonuniform_bounds():
    tokens = [
        RoutedToken(0, (0, 1), 20),
        RoutedToken(0, (2, 3), 18),
        RoutedToken(2, (4, 5), 16),
        RoutedToken(2, (6, 7), 14),
    ]
    placement = grace_hierarchical_placement(
        build_co_routing_graph(tokens),
        num_ranks=4,
        ranks_per_node=2,
        nonuniform_ratio=0.15,
    )

    counts = [len(experts) for experts in placement.experts_by_rank.values()]
    assert sorted(placement.rank_by_expert) == list(range(8))
    for node in range(2):
        node_counts = counts[node * 2 : node * 2 + 2]
        ideal = sum(node_counts) // 2
        delta = max(1, round(ideal * 0.15))
        assert min(node_counts) >= max(1, ideal - delta)
        assert max(node_counts) <= ideal + delta


def test_topology_hypergraph_refinement_reduces_node_and_gpu_cuts():
    tokens = [
        RoutedToken(0, (0, 1), 10),
        RoutedToken(2, (2, 3), 10),
    ]
    initial = {0: 0, 1: 2, 2: 1, 3: 3}
    before = evaluate_topology_hypergraph_objective(
        tokens, initial, ranks_per_node=2, rdma_cost=4
    )

    placement = refine_load_constrained_hypergraph_placement(
        tokens,
        initial,
        num_ranks=4,
        ranks_per_node=2,
        rdma_cost=4,
        max_load_ratio=1,
        min_experts_per_rank=1,
        max_experts_per_rank=1,
        max_rounds=None,
    )
    after = evaluate_topology_hypergraph_objective(
        tokens, placement.rank_by_expert, ranks_per_node=2, rdma_cost=4
    )

    assert before == 80
    assert after == 20


def test_topology_hypergraph_refinement_enforces_compute_cap_first():
    tokens = [
        RoutedToken(0, (0,), 100),
        RoutedToken(0, (1,), 90),
        RoutedToken(1, (2,), 10),
        RoutedToken(1, (3,), 1),
        RoutedToken(0, (0, 1), 20),
        RoutedToken(1, (2, 3), 20),
    ]
    initial = {0: 0, 1: 0, 2: 1, 3: 1}
    placement = refine_load_constrained_hypergraph_placement(
        tokens,
        initial,
        num_ranks=2,
        max_load_ratio=1.2,
        min_experts_per_rank=2,
        max_experts_per_rank=2,
        max_rounds=None,
    )
    demand = {expert: 0 for expert in initial}
    for token in tokens:
        for expert in token.topk_experts:
            demand[expert] += token.count
    loads = [0, 0]
    for expert, rank in placement.rank_by_expert.items():
        loads[rank] += demand[expert]

    assert max(loads) / (sum(loads) / 2) <= 1.2
