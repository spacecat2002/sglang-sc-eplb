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
)
from sglang.srt.eplb.cuda_fast_co_routing_planner import (  # noqa: E402
    build_co_routing_graph_cuda,
    evaluate_replica_remote_cuda,
    plan_communication_replicas_cuda,
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
