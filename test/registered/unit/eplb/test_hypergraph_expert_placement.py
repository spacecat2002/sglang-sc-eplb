from sglang.srt.eplb.co_routing_graph_solver import RoutedToken, build_co_routing_graph
from sglang.srt.eplb.hypergraph_expert_placement import SourceAwareHypergraphSolver


def test_source_aware_hypergraph_keeps_exact_capacity_and_local_bundles():
    tokens = [
        RoutedToken(0, (0, 1), 20),
        RoutedToken(1, (2, 3), 20),
    ]
    graph = build_co_routing_graph(tokens)
    placement = SourceAwareHypergraphSolver(
        num_ranks=2,
        slots_per_rank=2,
        max_rounds=8,
        restarts=2,
        candidates=16,
        seed=7,
    ).solve(graph, tokens, ranks_per_node=2, rdma_cost=1)

    assert all(len(experts) == 2 for experts in placement.experts_by_rank.values())
    assert placement.communication_cost == 0
    assert placement.compute_imbalance == 1
