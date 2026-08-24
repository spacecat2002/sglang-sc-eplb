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
    expert_demand = unbalanced.sum(dim=1)
    expert_order = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    balanced, added, addition_order = _C.select_compute_replicas(
        unbalanced, initial, expert_demand, expert_order, 1
    )
    assert balanced.cpu().tolist() == [[True, True], [True, False]]
    assert added.item() == 1
    assert addition_order.cpu().tolist() == [[0, 1], [0, 0]]
    into_replicas = initial.clone()
    into_instance = torch.empty_like(unbalanced)
    into_loads = torch.empty((2,), device="cuda", dtype=torch.int64)
    into_added_by_rank = torch.empty_like(into_loads)
    into_order = torch.empty_like(unbalanced)
    into_added = torch.empty((1,), device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_into(
        unbalanced,
        into_replicas,
        expert_demand,
        expert_order,
        1,
        into_instance,
        into_loads,
        into_added_by_rank,
        into_order,
        into_added,
    )
    assert torch.equal(into_replicas, balanced)
    assert torch.equal(into_order, addition_order)
    assert into_added.item() == added.item()

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


if __name__ == "__main__":
    test_kernels()
    print("grace_cuda kernels: OK")
