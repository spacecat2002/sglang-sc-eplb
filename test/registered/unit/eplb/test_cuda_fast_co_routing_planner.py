import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from sglang.srt.eplb.bundle_aware_replica_planner import (  # noqa: E402
    BundleAwareReplicaPlanner,
    ReplicaAction,
    RoutedToken,
)
from sglang.srt.eplb.co_routing_graph_solver import (  # noqa: E402
    CoRoutingGraphSolver,
    build_co_routing_graph,
    refine_hypergraph_placement,
)
from sglang.srt.eplb.cuda_fast_co_routing_planner import (  # noqa: E402
    build_co_routing_graph_cuda,
    evaluate_replica_remote_cuda,
    plan_communication_replicas_cuda,
    refine_hypergraph_placement_cuda,
    solve_co_routing_graph_cuda,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _trace():
    tokens = [
        RoutedToken(0, (0, 1, 2), count=7),
        RoutedToken(1, (1, 2, 3), count=5),
        RoutedToken(0, (0, 2, 3), count=3),
    ]
    topk = torch.tensor(
        [token.topk_experts for token in tokens], device="cuda", dtype=torch.int64
    )
    count = torch.tensor(
        [token.count for token in tokens], device="cuda", dtype=torch.int64
    )
    source = torch.tensor(
        [token.source_rank for token in tokens], device="cuda", dtype=torch.int64
    )
    return tokens, source, topk, count


def test_cuda_graph_and_placement_match_cpu_solver():
    tokens, _, topk, count = _trace()
    cpu_graph = build_co_routing_graph(tokens)
    cuda_graph = build_co_routing_graph_cuda(topk, count)

    assert cuda_graph.to_cpu_graph() == cpu_graph
    cpu_placement = CoRoutingGraphSolver(
        num_ranks=2, slots_per_rank=2, max_rounds=4
    ).solve(cpu_graph)
    cuda_placement = solve_co_routing_graph_cuda(
        cuda_graph,
        num_ranks=2,
        slots_per_rank=2,
        max_rounds=4,
    )
    assert cuda_placement == cpu_placement


def test_cuda_replica_replay_matches_communication_only_cpu_replay():
    tokens, source, topk, count = _trace()
    homes = {0: 0, 1: 0, 2: 1, 3: 1}
    replicas = {0: {0, 1, 2}, 1: {1, 2, 3}}
    planner = BundleAwareReplicaPlanner(
        num_ranks=2,
        baseline_rank_by_expert=homes,
        replica_slots_per_rank=2,
        compute_weight=0.0,
        communication_weight=1.0,
    )
    expected = planner.evaluate_placement(tokens, replicas)
    placement_mask = torch.zeros((4, 2), dtype=torch.bool, device="cuda")
    for rank, experts in replicas.items():
        placement_mask[list(experts), rank] = True

    actual = evaluate_replica_remote_cuda(
        source, topk, count, placement_mask, ranks_per_node=2
    )
    assert actual == expected.unique_remote_rank_copies


def test_cuda_fast_planner_selects_communication_closing_replica():
    source = torch.tensor([0], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64)
    count = torch.tensor([100], device="cuda", dtype=torch.int64)

    plan = plan_communication_replicas_cuda(
        source,
        topk,
        count,
        {0: 0, 1: 0, 2: 1, 3: 1},
        num_ranks=2,
        replica_slots_per_rank=1,
        max_candidates=4,
        ranks_per_node=2,
    )
    assert plan.actions == [ReplicaAction(0, (2,), "cuda-bundle-closure")]
    assert plan.unique_remote_rank_copies == 0


def test_cuda_hypergraph_refinement_uses_exact_bundle_objective():
    source = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([10, 10], device="cuda", dtype=torch.int64)

    placement = refine_hypergraph_placement_cuda(
        source,
        topk,
        count,
        {0: 0, 1: 1, 2: 0, 3: 1},
        num_ranks=2,
        max_rounds=None,
    )

    assert placement.initial_remote == 20
    assert placement.final_remote == 0
    assert placement.iterations == 1


def test_cuda_incremental_hypergraph_matches_cpu_across_rounds():
    tokens = [
        RoutedToken(1, (0, 1, 3), 3),
        RoutedToken(2, (0, 2, 5), 17),
        RoutedToken(0, (0, 3, 5), 14),
        RoutedToken(2, (0, 1, 5), 19),
        RoutedToken(2, (1, 2, 3), 5),
        RoutedToken(2, (0, 2, 4), 18),
        RoutedToken(0, (0, 4, 5), 7),
        RoutedToken(1, (3, 4, 5), 11),
    ]
    source = torch.tensor(
        [token.source_rank for token in tokens], device="cuda", dtype=torch.int64
    )
    topk = torch.tensor(
        [token.topk_experts for token in tokens], device="cuda", dtype=torch.int64
    )
    count = torch.tensor(
        [token.count for token in tokens], device="cuda", dtype=torch.int64
    )
    initial = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    expected = refine_hypergraph_placement(
        tokens,
        initial,
        num_ranks=3,
        max_rounds=None,
    )
    actual = refine_hypergraph_placement_cuda(
        source,
        topk,
        count,
        initial,
        num_ranks=3,
        max_rounds=None,
    )

    assert actual.rank_by_expert == expected.rank_by_expert
    assert actual.final_remote == expected.final_remote
    assert actual.iterations == expected.iterations


def test_cuda_source_agnostic_hypergraph_matches_cpu():
    tokens = [
        RoutedToken(0, (0, 1), count=10),
        RoutedToken(1, (2, 3), count=10),
    ]
    source = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([10, 10], device="cuda", dtype=torch.int64)
    initial = {0: 0, 1: 1, 2: 0, 3: 1}

    expected = refine_hypergraph_placement(
        tokens,
        initial,
        num_ranks=2,
        max_rounds=None,
        objective="source-agnostic",
    )
    actual = refine_hypergraph_placement_cuda(
        source,
        topk,
        count,
        initial,
        num_ranks=2,
        max_rounds=None,
        objective="source-agnostic",
    )

    assert actual.rank_by_expert == expected.rank_by_expert
    assert actual.initial_remote == 20
    assert actual.final_remote == 0
    assert actual.initial_objective == expected.initial_objective
    assert actual.final_objective == expected.final_objective
    assert actual.iterations == expected.iterations


def test_cuda_source_agnostic_hypergraph_ignores_source_rank():
    topk = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([10, 10], device="cuda", dtype=torch.int64)
    initial = {0: 0, 1: 1, 2: 0, 3: 1}

    placements = [
        refine_hypergraph_placement_cuda(
            torch.tensor(sources, device="cuda", dtype=torch.int64),
            topk,
            count,
            initial,
            num_ranks=2,
            max_rounds=None,
            objective="source-agnostic",
        )
        for sources in ([0, 1], [1, 0])
    ]

    assert placements[0].rank_by_expert == placements[1].rank_by_expert
    assert placements[0].final_objective == placements[1].final_objective
    assert placements[0].iterations == placements[1].iterations


def test_cuda_hypergraph_compute_balance_preserves_remote():
    source = torch.tensor([2, 2, 2, 2], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0], [1], [2], [3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([100, 90, 10, 1], device="cuda", dtype=torch.int64)

    placement = refine_hypergraph_placement_cuda(
        source,
        topk,
        count,
        {0: 0, 1: 0, 2: 1, 3: 1},
        num_ranks=3,
        max_rounds=0,
        balance_rounds=None,
    )

    assert placement.initial_remote == 201
    assert placement.final_remote == 201
    assert placement.balance_iterations == 1
    assert placement.experts_by_rank == {0: (1, 2), 1: (0, 3), 2: ()}
