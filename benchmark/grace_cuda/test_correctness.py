import torch

from grace_cuda import _C


def test_kernels() -> None:
    source = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int64)
    demand = _C.source_demand(source, topk, count, 4, 2)
    assert demand.cpu().tolist() == [[2, 0], [2, 3], [5, 3], [5, 0]]
    demand_into = torch.empty_like(demand)
    _C.source_demand_into(source, topk, count, 4, 2, demand_into)
    assert torch.equal(demand_into, demand)

    affinity = torch.empty((4, 4), device="cuda", dtype=torch.int64)
    affinity_degree = torch.empty(4, device="cuda", dtype=torch.int64)
    affinity_score = torch.empty_like(affinity_degree)
    affinity_groups = torch.empty_like(affinity_degree)
    group_source = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    group_to_rank = torch.empty(2, device="cuda", dtype=torch.int64)
    affinity_primary = torch.empty_like(affinity_degree)
    affinity_demand = torch.empty((4, 2), device="cuda", dtype=torch.int64)
    affinity_source = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64)
    affinity_topk = torch.tensor(
        [[0, 2], [0, 2], [1, 3], [1, 3]], device="cuda", dtype=torch.int64
    )
    affinity_count = torch.tensor([5, 5, 7, 7], device="cuda", dtype=torch.int64)
    _C.affinity_primary_into(
        affinity_source,
        affinity_topk,
        affinity_count,
        affinity_demand,
        affinity,
        affinity_degree,
        affinity_score,
        affinity_groups,
        group_source,
        group_to_rank,
        affinity_primary,
    )
    assert affinity[0, 2].item() == 10
    assert affinity[1, 3].item() == 14
    assert affinity_groups[0].item() == affinity_groups[2].item()
    assert affinity_groups[1].item() == affinity_groups[3].item()
    assert torch.bincount(affinity_groups, minlength=2).cpu().tolist() == [2, 2]
    assert sorted(group_to_rank.cpu().tolist()) == [0, 1]
    assert affinity_primary.cpu().tolist() == [0, 1, 0, 1]
    affinity_replicas = torch.nn.functional.one_hot(
        affinity_primary, num_classes=2
    ).bool()
    sequential_primary = torch.tensor([0, 0, 1, 1], device="cuda")
    sequential_replicas = torch.nn.functional.one_hot(
        sequential_primary, num_classes=2
    ).bool()
    affinity_traffic, _ = _C.traffic(
        affinity_source,
        affinity_topk,
        affinity_count,
        affinity_primary,
        affinity_replicas,
        2,
    )
    sequential_traffic, _ = _C.traffic(
        affinity_source,
        affinity_topk,
        affinity_count,
        sequential_primary,
        sequential_replicas,
        2,
    )
    assert affinity_traffic.sum().item() == 0
    assert affinity_traffic.sum().item() < sequential_traffic.sum().item()

    degree = affinity.sum(dim=1).to(torch.float64)
    scale = degree.sqrt().reciprocal()
    scale.masked_fill_(degree == 0, 0)
    normalized = affinity.to(torch.float64) * scale[:, None] * scale[None, :]
    _, eigenvectors = torch.linalg.eigh(normalized)
    embedding = torch.nn.functional.normalize(eigenvectors[:, -2:], dim=1)
    centers = torch.empty((2, 2), device="cuda", dtype=torch.float64)
    strict_groups = torch.empty(4, device="cuda", dtype=torch.int64)
    next_groups = torch.empty_like(strict_groups)
    group_sizes = torch.empty(2, device="cuda", dtype=torch.int64)
    overflow = torch.empty_like(strict_groups)
    _C.spectral_groups_into(
        embedding.contiguous(),
        affinity,
        centers,
        torch.empty(4, device="cuda", dtype=torch.float64),
        strict_groups,
        next_groups,
        group_sizes,
        overflow,
    )
    assert torch.bincount(strict_groups, minlength=2).cpu().tolist() == [2, 2]
    strict_group_source = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    _C.group_source_into(
        affinity_source,
        affinity_topk,
        affinity_count,
        strict_groups,
        strict_group_source,
    )
    strict_primary = torch.empty_like(strict_groups)
    _C.congestion_hungarian_into(
        strict_group_source,
        strict_groups,
        torch.empty((2, 2), device="cuda", dtype=torch.bool),
        torch.empty((2, 2), device="cuda", dtype=torch.int64),
        torch.empty((2, 2), device="cuda", dtype=torch.int64),
        torch.empty(18, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        strict_primary,
    )
    strict_replicas = torch.nn.functional.one_hot(
        strict_primary, num_classes=2
    ).bool()
    strict_traffic, _ = _C.traffic(
        affinity_source,
        affinity_topk,
        affinity_count,
        strict_primary,
        strict_replicas,
        2,
    )
    assert strict_traffic.sum().item() == 0

    move_primary = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64)
    move_source = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int64)
    move_topk = torch.tensor([[0], [1], [2], [3]], device="cuda", dtype=torch.int64)
    move_count = torch.tensor([20, 1, 10, 10], device="cuda", dtype=torch.int64)
    move_demand = _C.source_demand(move_source, move_topk, move_count, 4, 2)
    _C.refine_congestion_into(
        move_source,
        move_topk,
        move_count,
        move_demand,
        move_primary,
        1,
        3,
        2.0,
        4,
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty(5, device="cuda", dtype=torch.int64),
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        torch.empty((2, 2), device="cuda", dtype=torch.int64),
        torch.empty((4, 2, 5), device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
    )
    assert move_primary.cpu().tolist() == [1, 0, 0, 0]

    primary = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64)
    replicas = _C.select_topn(demand, primary, 0)
    assert replicas.cpu().tolist() == [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ]

    fused_demand, fused_replicas, fused_routing = _C.fused_source_topn(
        source, topk, count, primary, 4, 2, 0
    )
    assert torch.equal(fused_demand, demand)
    assert torch.equal(fused_replicas, replicas)
    assert fused_routing.cpu().tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    into_replicas = torch.empty_like(replicas)
    into_routing = torch.empty_like(fused_routing)
    _C.select_topn_into(demand, primary, 0, into_replicas)
    _C.default_routing_into(into_replicas, primary, into_routing)
    assert torch.equal(into_replicas, replicas)
    assert torch.equal(into_routing, fused_routing)
    fused_into_replicas = torch.empty_like(replicas)
    fused_into_routing = torch.empty_like(fused_routing)
    _C.select_topn_routing_into(
        demand, primary, 0, fused_into_replicas, fused_into_routing
    )
    assert torch.equal(fused_into_replicas, replicas)
    assert torch.equal(fused_into_routing, fused_routing)
    fused_all_demand = torch.empty_like(demand)
    fused_all_replicas = torch.empty_like(replicas)
    fused_all_routing = torch.empty_like(fused_routing)
    _C.fused_source_topn_into(
        source,
        topk,
        count,
        primary,
        4,
        2,
        0,
        fused_all_demand,
        fused_all_replicas,
        fused_all_routing,
    )
    assert torch.equal(fused_all_demand, demand)
    assert torch.equal(fused_all_replicas, replicas)
    assert torch.equal(fused_all_routing, fused_routing)

    traffic, compute = _C.traffic(source, topk, count, primary, replicas, 2)
    assert traffic.cpu().tolist() == [[0, 5], [3, 0]]
    assert compute.cpu().tolist() == [7, 13]

    replicas = _C.select_topn(demand, primary, 1)
    routing = torch.where(
        replicas.t(),
        torch.arange(2, device="cuda", dtype=torch.int64)[:, None],
        primary[None, :],
    )
    expert_total = demand.sum(dim=1)
    expert_order = torch.argsort(expert_total, descending=True, stable=True)
    flexible = replicas.sum(dim=1) > 1
    expert_order = expert_order[
        torch.argsort(flexible[expert_order].to(torch.int8), stable=True)
    ]
    source_order = torch.argsort(demand.t(), dim=1, descending=True, stable=True)
    quota, routing = _C.solve_quota(
        demand, replicas, primary, routing, expert_order, source_order, 1.25
    )
    assert quota.cpu().tolist() == [
        [[2, 0], [2, 0], [5, 0], [0, 5]],
        [[0, 0], [0, 3], [0, 3], [0, 0]],
    ]
    assert routing.cpu().tolist() == [[0, 0, 0, 1], [0, 1, 1, 1]]
    quota_into = torch.empty_like(quota)
    routing_into = torch.empty_like(routing)
    instance_into = torch.empty_like(demand)
    loads_into = torch.empty((2,), device="cuda", dtype=torch.int64)
    _C.solve_quota_into(
        demand,
        replicas,
        primary,
        routing,
        expert_order,
        source_order,
        1.25,
        quota_into,
        routing_into,
        instance_into,
        loads_into,
    )
    assert torch.equal(quota_into, quota)
    assert torch.equal(routing_into, routing)

    ordinals = torch.zeros_like(topk)
    traffic, compute = _C.quota_traffic(
        source,
        topk,
        count,
        quota,
        replicas,
        primary,
        torch.zeros_like(demand),
        ordinals,
        2,
    )
    assert traffic.cpu().tolist() == [[0, 5], [0, 0]]
    assert compute.cpu().tolist() == [9, 11]

    bundle_source = torch.tensor([0, 0], device="cuda", dtype=torch.int64)
    bundle_topk = torch.tensor([[0], [0]], device="cuda", dtype=torch.int64)
    bundle_count = torch.tensor([3, 4], device="cuda", dtype=torch.int64)
    bundle_quota = torch.tensor([[[3, 4]], [[0, 0]]], device="cuda", dtype=torch.int64)
    bundle_replicas = torch.tensor([[True, True]], device="cuda")
    bundle_primary = torch.tensor([0], device="cuda", dtype=torch.int64)
    bundle_ordinals = torch.tensor([[0], [3]], device="cuda", dtype=torch.int64)
    traffic, compute = _C.quota_traffic(
        bundle_source,
        bundle_topk,
        bundle_count,
        bundle_quota,
        bundle_replicas,
        bundle_primary,
        torch.zeros((1, 2), device="cuda", dtype=torch.int64),
        bundle_ordinals,
        2,
    )
    assert traffic.cpu().tolist() == [[0, 4], [0, 0]]
    assert compute.cpu().tolist() == [3, 4]

    crossing_source = torch.tensor([0], device="cuda", dtype=torch.int64)
    crossing_topk = torch.tensor([[0]], device="cuda", dtype=torch.int64)
    crossing_count = torch.tensor([5], device="cuda", dtype=torch.int64)
    crossing_ordinals = torch.zeros_like(crossing_topk)
    traffic, compute = _C.quota_traffic(
        crossing_source,
        crossing_topk,
        crossing_count,
        bundle_quota,
        bundle_replicas,
        bundle_primary,
        torch.zeros((1, 2), device="cuda", dtype=torch.int64),
        crossing_ordinals,
        2,
    )
    assert traffic.cpu().tolist() == [[0, 2], [0, 0]]
    assert compute.cpu().tolist() == [3, 2]

    unbalanced = torch.tensor([[10, 0], [10, 0]], device="cuda", dtype=torch.int64)
    initial = torch.tensor([[True, False], [True, False]], device="cuda")
    balanced, added, addition_order = _C.select_compute_replicas(
        unbalanced, initial, torch.tensor([0, 0], device="cuda"), 1
    )
    assert balanced.cpu().tolist() == [[True, True], [True, False]]
    assert added.item() == 1
    assert addition_order.cpu().tolist() == [[0, 1], [0, 0]]
    into_replicas = initial.clone()
    into_instance = torch.empty_like(unbalanced)
    into_loads = torch.empty((2,), device="cuda", dtype=torch.int64)
    into_added_by_rank = torch.empty_like(into_loads)
    into_order = torch.empty_like(unbalanced)
    into_quota = torch.empty((2, 2, 2), device="cuda", dtype=torch.int64)
    into_routing = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    into_added = torch.empty((1,), device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_into(
        unbalanced,
        into_replicas,
        torch.tensor([0, 0], device="cuda"),
        1,
        into_instance,
        into_loads,
        into_added_by_rank,
        into_order,
        into_quota,
        into_routing,
        into_added,
    )
    assert torch.equal(into_replicas, balanced)
    assert torch.equal(into_order, addition_order)
    assert into_added.item() == added.item()
    assert into_quota.sum(dim=(0, 1)).cpu().tolist() == [10, 10]
    assert torch.equal(into_quota.sum(dim=2), unbalanced.t())

    # A requested 1.0x limit must actually localize quota from an overloaded
    # rank when the replica set can serve the other rank.
    limit_demand = torch.tensor(
        [[6, 13, 0], [16, 7, 14], [15, 17, 7], [11, 7, 7], [14, 9, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    limit_replicas = torch.tensor(
        [
            [False, False, True],
            [True, False, False],
            [True, True, True],
            [False, True, True],
            [False, True, True],
        ],
        device="cuda",
        dtype=torch.bool,
    )
    limit_primary = torch.tensor([2, 0, 0, 1, 1], device="cuda", dtype=torch.int64)
    limit_routing = torch.tensor(
        [[2, 0, 0, 1, 1], [2, 0, 1, 1, 1], [2, 0, 2, 2, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    limit_expert_order = torch.tensor(
        [1, 0, 2, 3, 4], device="cuda", dtype=torch.int64
    )
    limit_source_order = torch.tensor(
        [[1, 2, 4, 3, 0], [2, 0, 4, 1, 3], [1, 2, 3, 0, 4]],
        device="cuda",
        dtype=torch.int64,
    )
    limit_quota, _ = _C.solve_quota(
        limit_demand,
        limit_replicas,
        limit_primary,
        limit_routing,
        limit_expert_order,
        limit_source_order,
        1.0,
    )
    limit_compute = limit_quota.sum(dim=(0, 1))
    assert limit_compute.cpu().tolist() == [48, 47, 48]

    # Greedy per-expert waterfill yields [11, 9, 10], but the replica graph
    # admits the exact [10, 10, 10] capacity through cross-expert rebalance.
    augment_demand = torch.tensor(
        [[4, 0, 0], [12, 0, 0], [14, 0, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    augment_replicas = torch.tensor(
        [[False, False, True], [True, False, True], [False, True, True]],
        device="cuda",
    )
    augment_primary = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)
    augment_routing = torch.tensor(
        [[2, 0, 1], [2, 0, 1], [2, 2, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    augment_order = torch.tensor([0, 2, 1], device="cuda", dtype=torch.int64)
    augment_source_order = torch.tensor(
        [[2, 1, 0], [0, 1, 2], [0, 1, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    augment_quota, _ = _C.solve_quota(
        augment_demand,
        augment_replicas,
        augment_primary,
        augment_routing,
        augment_order,
        augment_source_order,
        1.0,
    )
    assert augment_quota.sum(dim=(0, 1)).cpu().tolist() == [10, 10, 10]
    selected, added, _ = _C.select_compute_replicas(
        augment_demand,
        augment_replicas.clone(),
        augment_primary,
        1,
    )
    assert torch.equal(selected, augment_replicas)
    assert added.item() == 0

    # At the same optimal threshold, prefer a new source-local copy over
    # exporting the source's load to an already-present remote replica.
    local_demand = torch.tensor(
        [
            [0, 0, 0, 10],
            [10, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 5],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    local_replicas = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ],
        device="cuda",
    )
    local_selected, local_added, _ = _C.select_compute_replicas(
        local_demand,
        local_replicas,
        torch.tensor([0, 0, 1, 2, 3], device="cuda"),
        1,
    )
    assert local_selected[0].cpu().tolist() == [True, True, False, True]
    assert local_added.item() == 1

    # Ranks 1 and 2 have equal compute slack, but rank 1 already has remote
    # ingress. The communication pass sends expert 0 to rank 2 first.
    ingress_demand = torch.tensor(
        [[10, 0, 0], [5, 0, 0], [0, 0, 5]],
        device="cuda",
        dtype=torch.int64,
    )
    ingress_replicas = torch.eye(3, device="cuda", dtype=torch.bool)
    _, ingress_added, ingress_order = _C.select_compute_replicas(
        ingress_demand,
        ingress_replicas,
        torch.arange(3, device="cuda", dtype=torch.int64),
        1,
    )
    assert ingress_added.item() == 2
    assert ingress_order[0].cpu().tolist() == [0, 2, 1]

    # Direct exports get stuck at [8, 8, 4]. Graph augmentation adds the two
    # edges needed for the multi-hop [7, 7, 6] plan.
    chain_demand = torch.diag(
        torch.tensor([10, 8, 2], device="cuda", dtype=torch.int64)
    )
    chain_replicas = torch.eye(3, device="cuda", dtype=torch.bool)
    chain_primary = torch.arange(3, device="cuda", dtype=torch.int64)
    chain_instance = torch.empty_like(chain_demand)
    chain_loads = torch.empty(3, device="cuda", dtype=torch.int64)
    chain_slots = torch.empty_like(chain_loads)
    chain_order = torch.empty_like(chain_demand)
    chain_quota = torch.empty((3, 3, 3), device="cuda", dtype=torch.int64)
    chain_routing = torch.empty_like(chain_demand)
    chain_added = torch.empty(1, device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_into(
        chain_demand,
        chain_replicas,
        chain_primary,
        1,
        chain_instance,
        chain_loads,
        chain_slots,
        chain_order,
        chain_quota,
        chain_routing,
        chain_added,
    )
    assert chain_added.item() == 2
    assert chain_quota.sum(dim=(0, 1)).cpu().tolist() == [7, 7, 6]

    # The first overloaded rank is fixed above capacity. The solver must still
    # rebalance a later overloaded rank that has a feasible export path.
    blocked_demand = torch.tensor(
        [[15, 0, 0], [7, 0, 0], [16, 0, 0]], device="cuda", dtype=torch.int64
    )
    blocked_replicas = torch.tensor(
        [[True, True, True], [True, False, True], [True, False, False]],
        device="cuda",
    )
    blocked_primary = torch.zeros(3, device="cuda", dtype=torch.int64)
    blocked_routing = torch.zeros((3, 3), device="cuda", dtype=torch.int64)
    blocked_expert_order = torch.tensor([2, 0, 1], device="cuda")
    blocked_source_order = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]], device="cuda"
    )
    blocked_quota, _ = _C.solve_quota(
        blocked_demand,
        blocked_replicas,
        blocked_primary,
        blocked_routing,
        blocked_expert_order,
        blocked_source_order,
        1.0,
    )
    assert blocked_quota.sum(dim=(0, 1)).cpu().tolist() == [16, 9, 13]


if __name__ == "__main__":
    test_kernels()
    print("grace_cuda kernels: OK")
