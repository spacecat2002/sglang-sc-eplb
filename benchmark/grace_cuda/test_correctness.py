import torch

from grace_cuda import _C


def test_kernels() -> None:
    source = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)
    topk = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda", dtype=torch.int64)
    count = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int64)
    demand = _C.source_demand(source, topk, count, 4, 2)
    assert demand.cpu().tolist() == [[2, 0], [2, 3], [5, 3], [5, 0]]

    primary = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64)
    replicas = _C.select_topn(demand, primary, 0)
    assert replicas.cpu().tolist() == [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ]

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

    unbalanced = torch.tensor([[10, 0], [10, 0]], device="cuda", dtype=torch.int64)
    initial = torch.tensor([[True, False], [True, False]], device="cuda")
    balanced, added, addition_order = _C.select_compute_replicas(unbalanced, initial, 1)
    assert balanced.cpu().tolist() == [[True, True], [True, False]]
    assert added.item() == 1
    assert addition_order.cpu().tolist() == [[0, 1], [0, 0]]


if __name__ == "__main__":
    test_kernels()
    print("grace_cuda kernels: OK")
