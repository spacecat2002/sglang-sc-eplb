from sglang.srt.eplb.co_routing_graph_solver import (
    CoRoutingGraphSolver,
    RoutedToken,
    build_co_routing_graph,
    evaluate_pairwise_cut,
    evaluate_primary_remote,
    evaluate_weighted_remote,
)
from sglang.srt.eplb.grace_expert_placement import grace_hierarchical_placement


def test_pairwise_graph_and_solver():
    tokens = [
        RoutedToken(0, (0, 1), 10),
        RoutedToken(1, (2, 3), 10),
    ]
    graph = build_co_routing_graph(tokens)
    placement = CoRoutingGraphSolver(
        num_ranks=2, slots_per_rank=2, max_rounds=None
    ).solve(graph)

    assert graph.demand == {0: 10, 1: 10, 2: 10, 3: 10}
    assert graph.edges == {(0, 1): 10, (2, 3): 10}
    assert all(len(experts) == 2 for experts in placement.experts_by_rank.values())
    assert placement.final_cut <= placement.initial_cut
    assert evaluate_pairwise_cut(graph, placement.rank_by_expert) == 0


def test_remote_replay_counts_distinct_destination_ranks():
    tokens = [RoutedToken(0, (0, 1, 2), 4)]

    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 1}) == 4
    assert evaluate_primary_remote(tokens, {0: 0, 1: 1, 2: 2}) == 8
    assert (
        evaluate_weighted_remote(
            tokens, {0: 0, 1: 1, 2: 2}, ranks_per_node=2, rdma_cost=4
        )
        == 20
    )


def test_pairwise_reranks_swaps_by_real_communication():
    tokens = [
        RoutedToken(1, (4, 5, 6), 2),
        RoutedToken(3, (3, 5, 7), 10),
        RoutedToken(1, (2, 7), 3),
        RoutedToken(3, (0, 1, 5), 14),
        RoutedToken(3, (3, 6), 5),
    ]
    graph = build_co_routing_graph(tokens)
    initial = CoRoutingGraphSolver(num_ranks=4, slots_per_rank=2, max_rounds=0).solve(
        graph
    )
    reranked = CoRoutingGraphSolver(
        num_ranks=4,
        slots_per_rank=2,
        max_rounds=8,
        rerank_candidates=32,
        max_compute_imbalance=1.5,
    ).solve(graph, routed_tokens=tokens, ranks_per_node=2, rdma_cost=4)

    loads = [
        sum(graph.demand[expert] for expert in reranked.experts_by_rank[rank])
        for rank in range(4)
    ]
    assert (
        evaluate_weighted_remote(
            tokens, initial.rank_by_expert, ranks_per_node=2, rdma_cost=4
        )
        == 189
    )
    assert (
        evaluate_weighted_remote(
            tokens, reranked.rank_by_expert, ranks_per_node=2, rdma_cost=4
        )
        == 91
    )
    assert reranked.final_cut == initial.final_cut
    assert max(loads) / (sum(loads) / 4) <= 1.5


def test_grace_respects_hierarchical_nonuniform_bounds():
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
