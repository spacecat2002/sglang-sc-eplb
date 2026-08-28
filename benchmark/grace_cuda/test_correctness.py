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
    generator = torch.Generator(device="cuda").manual_seed(0)
    initial = torch.linalg.qr(
        torch.randn(
            (4, 4), generator=generator, device="cuda", dtype=torch.float64
        ),
        mode="reduced",
    ).Q.contiguous()
    subspace = torch.empty_like(initial)
    _C.affinity_subspace_into(
        affinity, affinity_degree, initial, subspace, 2
    )
    assert torch.isfinite(subspace).all()
    assert torch.allclose(
        torch.linalg.vector_norm(subspace, dim=1),
        torch.ones(4, device="cuda", dtype=torch.float64),
    )
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
        torch.empty((4, 2), device="cuda", dtype=torch.int64),
        torch.empty((4, 4), device="cuda", dtype=torch.int64),
    )
    assert torch.bincount(strict_groups, minlength=2).cpu().tolist() == [2, 2]
    strict_group_source = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    _C.group_source_into(
        affinity_source,
        affinity_topk,
        affinity_count,
        strict_groups,
        strict_group_source,
        1,
    )
    # Fixed-K demand/gain paths must remain equivalent to the generic kernels.
    for k in (1, 2, 4, 8, 16, 10):
        test_tokens = 257
        test_ranks = 4
        test_experts = 16
        test_source = torch.arange(test_tokens, device="cuda") % test_ranks
        test_topk = (
            torch.arange(test_tokens * k, device="cuda")
            .view(test_tokens, k)
            .remainder(test_experts)
            .to(torch.int64)
        )
        test_count = (torch.arange(test_tokens, device="cuda") % 7 + 1).to(
            torch.int64
        )
        test_primary = torch.arange(test_experts, device="cuda") % test_ranks
        generic_demand = _C.source_demand(
            test_source, test_topk, test_count, test_experts, test_ranks
        )
        fused_demand, _, _ = _C.fused_source_topn(
            test_source,
            test_topk,
            test_count,
            test_primary,
            test_experts,
            test_ranks,
            2,
        )
        assert torch.equal(fused_demand, generic_demand)

        replicas = torch.nn.functional.one_hot(
            test_primary, num_classes=test_ranks
        ).bool()
        replicas[:, 1] |= test_primary != 1
        fast_gain = torch.empty_like(generic_demand)
        generic_gain = torch.empty_like(generic_demand)
        generic_cover = torch.empty(
            (test_ranks, test_experts, test_ranks),
            device="cuda",
            dtype=torch.int64,
        )
        _C.current_bundle_gains_into(
            test_source,
            test_topk,
            test_count,
            test_primary,
            replicas,
            generic_gain,
            generic_cover,
            1,
        )
        _C.current_bundle_gains_fast_into(
            test_source,
            test_topk,
            test_count,
            test_primary,
            replicas,
            fast_gain,
            1,
        )
        assert torch.equal(fast_gain, generic_gain)

    # The shared group-source implementation is used for EP32/64 and must
    # agree with its single-block and multi-block reductions.
    for test_ranks in (32, 64):
        test_experts = test_ranks
        test_tokens = 513
        test_k = 8
        test_source = torch.arange(test_tokens, device="cuda") % test_ranks
        test_topk = (
            torch.arange(test_tokens * test_k, device="cuda")
            .view(test_tokens, test_k)
            .remainder(test_experts)
            .to(torch.int64)
        )
        test_count = (torch.arange(test_tokens, device="cuda") % 11 + 1).to(
            torch.int64
        )
        test_groups = torch.arange(test_experts, device="cuda") % test_ranks
        single_group_source = torch.empty(
            (test_ranks, test_ranks), device="cuda", dtype=torch.int64
        )
        multi_group_source = torch.empty_like(single_group_source)
        _C.group_source_into(
            test_source,
            test_topk,
            test_count,
            test_groups,
            single_group_source,
            1,
        )
        _C.group_source_into(
            test_source,
            test_topk,
            test_count,
            test_groups,
            multi_group_source,
            4,
        )
        assert torch.equal(single_group_source, multi_group_source)
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

    swap_embedding = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
        device="cuda",
        dtype=torch.float64,
    )
    swap_affinity = torch.zeros((4, 4), device="cuda", dtype=torch.int64)
    swap_affinity[0, 2] = swap_affinity[2, 0] = 10
    swap_affinity[1, 3] = swap_affinity[3, 1] = 10
    swap_groups = torch.empty(4, device="cuda", dtype=torch.int64)
    _C.spectral_groups_into(
        swap_embedding,
        swap_affinity,
        torch.empty((2, 2), device="cuda", dtype=torch.float64),
        torch.empty(4, device="cuda", dtype=torch.float64),
        swap_groups,
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty((4, 2), device="cuda", dtype=torch.int64),
        torch.empty((4, 4), device="cuda", dtype=torch.int64),
    )
    assert swap_groups[0].item() == swap_groups[2].item()
    assert swap_groups[1].item() == swap_groups[3].item()

    balance_demand = torch.tensor(
        [[9, 0], [8, 0], [1, 0], [0, 0]], device="cuda", dtype=torch.int64
    )
    balance_groups = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64)
    _C.balance_affinity_groups_into(
        balance_demand,
        torch.zeros((4, 4), device="cuda", dtype=torch.int64),
        balance_groups,
        torch.empty(4, device="cuda", dtype=torch.int64),
        torch.empty(2, device="cuda", dtype=torch.int64),
        torch.empty((4, 2), device="cuda", dtype=torch.int64),
    )
    balance_loads = torch.zeros(2, device="cuda", dtype=torch.int64)
    balance_loads.scatter_add_(0, balance_groups, balance_demand.sum(1))
    assert balance_loads.cpu().tolist() == [9, 9]
    assert torch.bincount(balance_groups, minlength=2).cpu().tolist() == [2, 2]

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
        torch.empty_like(demand),
        fused_all_replicas,
        fused_all_routing,
    )
    assert torch.equal(fused_all_demand, demand)
    assert torch.equal(fused_all_replicas, replicas)
    assert torch.equal(fused_all_routing, fused_routing)

    # Two independently removable ranks beat copying both experts from one
    # shared destination, even though the shared experts have higher demand.
    grouped_source = torch.tensor([0, 0, 0], device="cuda", dtype=torch.int64)
    grouped_topk = torch.tensor(
        [[0, 1], [2, 4], [3, 4]], device="cuda", dtype=torch.int64
    )
    grouped_count = torch.tensor([8, 7, 7], device="cuda", dtype=torch.int64)
    grouped_primary = torch.tensor(
        [1, 1, 2, 3, 0], device="cuda", dtype=torch.int64
    )
    grouped_demand = _C.source_demand(
        grouped_source, grouped_topk, grouped_count, 5, 4
    )
    grouped_replicas = torch.empty((5, 4), device="cuda", dtype=torch.bool)
    grouped_routing = torch.empty((4, 5), device="cuda", dtype=torch.int64)
    grouped_workspace = 4 * 5 * 6
    _C.select_rank_group_topn_routing_into(
        grouped_source,
        grouped_topk,
        grouped_count,
        grouped_demand,
        grouped_primary,
        2,
        torch.empty_like(grouped_demand),
        torch.empty_like(grouped_demand),
        torch.empty_like(grouped_demand),
        torch.empty(grouped_workspace, device="cuda", dtype=torch.int64),
        torch.empty(grouped_workspace, device="cuda", dtype=torch.int64),
        grouped_replicas,
        grouped_routing,
    )
    grouped_traffic, _ = _C.traffic(
        grouped_source,
        grouped_topk,
        grouped_count,
        grouped_primary,
        grouped_replicas,
        4,
    )
    demand_replicas = _C.select_topn(grouped_demand, grouped_primary, 2)
    demand_traffic, _ = _C.traffic(
        grouped_source,
        grouped_topk,
        grouped_count,
        grouped_primary,
        demand_replicas,
        4,
    )
    assert grouped_replicas[:, 0].cpu().tolist() == [False, False, True, True, True]
    assert grouped_traffic.sum().item() == 8
    assert demand_traffic.sum().item() == 14

    # Replicating either expert would only split one shared remote bundle;
    # bundle-aware selection correctly spends no replica on it.
    shared_source = torch.tensor([0], device="cuda", dtype=torch.int64)
    shared_topk = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    shared_count = torch.tensor([10], device="cuda", dtype=torch.int64)
    shared_primary = torch.tensor([1, 1], device="cuda", dtype=torch.int64)
    _, shared_replicas, _ = _C.fused_source_topn(
        shared_source, shared_topk, shared_count, shared_primary, 2, 2, 1
    )
    assert shared_replicas.cpu().tolist() == [[False, True], [False, True]]
    shared_gains = torch.empty((2, 2), device="cuda", dtype=torch.int64)
    shared_covers = torch.empty((2, 2, 2), device="cuda", dtype=torch.int64)
    _C.current_bundle_gains_into(
        shared_source,
        shared_topk,
        shared_count,
        shared_primary,
        torch.tensor([[False, True], [False, True]], device="cuda"),
        shared_gains,
        shared_covers,
        1,
    )
    assert shared_gains.cpu().tolist() == [[0, 0], [0, 0]]
    assert shared_covers[0].cpu().tolist() == [[0, 10], [0, 10]]

    # The fast current-gain path must remain bit-for-bit equivalent for the
    # compile-time K specializations and for the dynamic fallback.
    gain_tokens = 19
    gain_experts = 5
    gain_ranks = 4
    gain_source = (
        torch.arange(gain_tokens, device="cuda", dtype=torch.int64) % gain_ranks
    )
    gain_count = torch.arange(1, gain_tokens + 1, device="cuda", dtype=torch.int64)
    gain_primary = torch.tensor([0, 1, 2, 3, 0], device="cuda", dtype=torch.int64)
    gain_replicas = torch.nn.functional.one_hot(
        gain_primary, num_classes=gain_ranks
    ).bool()
    gain_replicas[1, 0] = True
    gain_replicas[4, 2] = True
    for gain_k in (1, 2, 4, 8, 16, 10):
        gain_topk = (
            torch.arange(gain_tokens * gain_k, device="cuda", dtype=torch.int64)
            % gain_experts
        ).view(gain_tokens, gain_k)
        ordinal_keys = (gain_source[:, None] * gain_experts + gain_topk).reshape(
            -1
        )
        ordinal_weights = gain_count[:, None].expand_as(gain_topk).reshape(-1)
        sorted_keys, ordinal_order = torch.sort(ordinal_keys, stable=True)
        sorted_weights = ordinal_weights[ordinal_order]
        ordinal_before = sorted_weights.cumsum(0) - sorted_weights
        ordinal_starts = torch.empty_like(sorted_keys, dtype=torch.bool)
        ordinal_starts[0] = True
        ordinal_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
        ordinal_base = torch.cummax(
            torch.where(ordinal_starts, ordinal_before, 0), dim=0
        ).values
        expected_ordinals = torch.empty_like(ordinal_before)
        expected_ordinals.scatter_(
            0, ordinal_order, ordinal_before - ordinal_base
        )
        expected_ordinals = expected_ordinals.view_as(gain_topk)
        actual_ordinals = torch.empty_like(gain_topk)
        _C.bundle_ordinals_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_experts,
            gain_ranks,
            torch.empty(
                (gain_experts, gain_ranks), device="cuda", dtype=torch.int64
            ),
            actual_ordinals,
        )
        assert torch.equal(actual_ordinals, expected_ordinals)
        reference = torch.empty_like(gain_replicas, dtype=torch.int64)
        reference_covers = torch.empty(
            (gain_ranks, gain_experts, gain_ranks),
            device="cuda",
            dtype=torch.int64,
        )
        fast = torch.empty_like(reference)
        _C.current_bundle_gains_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_replicas,
            reference,
            reference_covers,
            1,
        )
        _C.current_bundle_gains_fast_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_replicas,
            fast,
            1,
        )
        assert torch.equal(fast, reference)
        fast_multi = torch.empty_like(reference)
        _C.current_bundle_gains_fast_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_replicas,
            fast_multi,
            2,
        )
        assert torch.equal(fast_multi, reference)

        # The indexed path starts from primary-only gains and applies deltas
        # only for bundles touched by newly local replicas.
        indexed_demand = torch.empty_like(reference)
        indexed_initial = torch.empty_like(reference)
        indexed_replicas = torch.empty_like(gain_replicas)
        indexed_routing = torch.empty(
            (gain_ranks, gain_experts), device="cuda", dtype=torch.int64
        )
        bundle_heads = torch.empty_like(reference, dtype=torch.int32)
        bundle_next = torch.empty(
            gain_topk.numel(), device="cuda", dtype=torch.int32
        )
        _C.fused_source_topn_index_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_experts,
            gain_ranks,
            0,
            indexed_demand,
            indexed_initial,
            indexed_replicas,
            indexed_routing,
            bundle_heads,
            bundle_next,
        )
        assert torch.equal(
            indexed_demand,
            _C.source_demand(
                gain_source, gain_topk, gain_count, gain_experts, gain_ranks
            ),
        )
        bundle_marks = torch.zeros(
            gain_tokens, device="cuda", dtype=torch.int32
        )
        assert bundle_heads.element_size() == 4
        assert bundle_next.element_size() == 4
        assert bundle_marks.element_size() == 4
        for incremental_sms, epoch in ((1, 1), (2, 2)):
            incremental = indexed_initial.clone()
            _C.incremental_bundle_gains_fast_into(
                gain_source,
                gain_topk,
                gain_count,
                gain_primary,
                gain_replicas,
                incremental,
                bundle_heads,
                bundle_next,
                bundle_marks,
                epoch,
                incremental_sms,
            )
            assert torch.equal(incremental, reference)

        fused_demand_k, fused_replicas_k, fused_routing_k = _C.fused_source_topn(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_experts,
            gain_ranks,
            2,
        )
        expected_demand_k = _C.source_demand(
            gain_source, gain_topk, gain_count, gain_experts, gain_ranks
        )
        expected_gains_k = torch.empty_like(expected_demand_k)
        expected_replicas_k = torch.empty_like(gain_replicas)
        expected_routing_k = torch.empty_like(fused_routing_k)
        _C.select_bundle_topn_routing_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            2,
            expected_gains_k,
            expected_replicas_k,
            expected_routing_k,
            1,
        )
        assert torch.equal(fused_demand_k, expected_demand_k)
        assert torch.equal(fused_replicas_k, expected_replicas_k)
        assert torch.equal(fused_routing_k, expected_routing_k)

        selected_initial = torch.empty_like(reference)
        selected_replicas = torch.empty_like(gain_replicas)
        selected_routing = torch.empty_like(fused_routing_k)
        _C.select_bundle_topn_routing_index_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            2,
            selected_initial,
            selected_replicas,
            selected_routing,
            bundle_heads,
            bundle_next,
            1,
        )
        assert torch.equal(selected_replicas, fused_replicas_k)
        assert torch.equal(selected_routing, fused_routing_k)
        selected_reference = torch.empty_like(reference)
        _C.current_bundle_gains_fast_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            selected_replicas,
            selected_reference,
            1,
        )
        selected_incremental = selected_initial.clone()
        _C.incremental_bundle_gains_fast_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            selected_replicas,
            selected_incremental,
            bundle_heads,
            bundle_next,
            bundle_marks,
            3,
            1,
        )
        assert torch.equal(selected_incremental, selected_reference)

        csr_offsets = torch.empty(
            gain_experts * gain_ranks + 1, device="cuda", dtype=torch.int32
        )
        csr_entries = torch.empty(
            gain_topk.numel(), device="cuda", dtype=torch.int32
        )
        csr_counts = torch.empty(
            gain_experts * gain_ranks, device="cuda", dtype=torch.int32
        )
        csr_cursors = torch.empty_like(csr_counts)
        _C.build_bundle_incidence_csr_into(
            gain_source,
            gain_topk,
            csr_offsets,
            csr_entries,
            csr_counts,
            csr_cursors,
            gain_experts,
        )
        csr_incremental = indexed_initial.clone()
        _C.incremental_bundle_gains_csr_fast_into(
            gain_source,
            gain_topk,
            gain_count,
            gain_primary,
            gain_replicas,
            csr_incremental,
            csr_offsets,
            csr_entries,
            bundle_marks,
            4,
            2,
        )
        assert torch.equal(csr_incremental, reference)

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

    v2_demand = torch.tensor([[0, 10], [0, 10]], device="cuda", dtype=torch.int64)
    v2_replicas = initial.clone()
    v2_instance = torch.empty_like(v2_demand)
    v2_loads = torch.empty((2,), device="cuda", dtype=torch.int64)
    v2_slots = torch.empty_like(v2_loads)
    v2_order = torch.empty_like(v2_demand)
    v2_quota = torch.empty((2, 2, 2), device="cuda", dtype=torch.int64)
    v2_routing = torch.zeros((2, 2), device="cuda", dtype=torch.int64)
    v2_added = torch.empty((1,), device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_v2_into(
        v2_demand,
        torch.tensor([[0, 0], [0, 10]], device="cuda", dtype=torch.int64),
        torch.zeros((2, 2, 2), device="cuda", dtype=torch.int64),
        v2_replicas,
        torch.tensor([0, 0], device="cuda"),
        1,
        1.0,
        v2_instance,
        v2_loads,
        v2_slots,
        v2_order,
        v2_quota,
        v2_routing,
        v2_added,
    )
    assert v2_replicas.cpu().tolist() == [[True, False], [True, True]]
    assert v2_added.item() == 1
    assert v2_order.cpu().tolist() == [[0, 0], [0, 1]]
    assert v2_quota.sum(dim=(0, 1)).cpu().tolist() == [10, 10]
    assert v2_loads.cpu().tolist() == [10, 10]
    assert torch.equal(v2_quota.sum(dim=2), v2_demand.t())

    # Sparse export must reconstruct the dense fast-capacity plan exactly.
    fast_replicas = initial.clone()
    fast_instance = torch.empty_like(v2_demand)
    fast_loads = torch.empty_like(v2_loads)
    fast_slots = torch.empty_like(v2_slots)
    fast_order = torch.empty_like(v2_order)
    fast_move_plan = torch.empty_like(v2_demand)
    fast_quota = torch.empty_like(v2_quota)
    fast_routing = torch.empty_like(v2_routing)
    fast_added = torch.empty_like(v2_added)
    fast_candidates = torch.empty(44, device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_fast_into(
        v2_demand,
        torch.tensor([[0, 0], [0, 10]], device="cuda", dtype=torch.int64),
        fast_replicas,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        1,
        1.0,
        fast_instance,
        fast_loads,
        fast_slots,
        fast_order,
        fast_move_plan,
        fast_quota,
        fast_routing,
        fast_added,
        fast_candidates,
        1,
    )
    multi_replicas = initial.clone()
    multi_instance = torch.empty_like(v2_demand)
    multi_loads = torch.empty_like(v2_loads)
    multi_slots = torch.empty_like(v2_slots)
    multi_order = torch.empty_like(v2_order)
    multi_move_plan = torch.empty_like(v2_demand)
    multi_quota = torch.empty_like(v2_quota)
    multi_routing = torch.empty_like(v2_routing)
    multi_added = torch.empty_like(v2_added)
    multi_candidates = torch.empty(164, device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_fast_into(
        v2_demand,
        torch.tensor([[0, 0], [0, 10]], device="cuda", dtype=torch.int64),
        multi_replicas,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        1,
        1.0,
        multi_instance,
        multi_loads,
        multi_slots,
        multi_order,
        multi_move_plan,
        multi_quota,
        multi_routing,
        multi_added,
        multi_candidates,
        4,
    )
    for single_value, multi_value in zip(
        (fast_replicas, fast_instance, fast_loads, fast_slots, fast_order,
         fast_move_plan, fast_quota, fast_routing, fast_added),
        (multi_replicas, multi_instance, multi_loads, multi_slots, multi_order,
         multi_move_plan, multi_quota, multi_routing, multi_added),
    ):
        assert torch.equal(single_value, multi_value)
    sparse_replicas = initial.clone()
    sparse_instance = torch.empty_like(v2_demand)
    sparse_loads = torch.empty_like(v2_loads)
    sparse_slots = torch.empty_like(v2_slots)
    sparse_order = torch.empty_like(v2_order)
    sparse_move_plan = torch.empty_like(v2_demand)
    sparse_added = torch.empty_like(v2_added)
    _C.select_compute_replicas_fast_sparse_into(
        v2_demand,
        torch.tensor([[0, 0], [0, 10]], device="cuda", dtype=torch.int64),
        sparse_replicas,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        1,
        1.0,
        sparse_instance,
        sparse_loads,
        sparse_slots,
        sparse_order,
        sparse_move_plan,
        sparse_added,
        fast_candidates,
        1,
    )
    sparse_prefix = torch.empty_like(v2_quota)
    sparse_targets = torch.empty_like(v2_quota)
    sparse_routing = torch.empty_like(v2_routing)
    csr_move_plan = sparse_move_plan.clone()
    _C.materialize_fast_sparse_quota_into(
        v2_demand,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        sparse_replicas,
        sparse_order,
        sparse_move_plan,
        sparse_prefix,
        sparse_targets,
        sparse_routing,
    )
    sparse_delta = sparse_prefix.clone()
    sparse_delta[:, :, 1:] = torch.where(
        sparse_prefix[:, :, 1:] > 0,
        sparse_prefix[:, :, 1:] - sparse_prefix[:, :, :-1],
        torch.zeros_like(sparse_prefix[:, :, 1:]),
    )
    reconstructed = torch.zeros_like(v2_quota)
    reconstructed.scatter_add_(2, sparse_targets, sparse_delta)
    assert torch.equal(reconstructed, fast_quota)
    assert torch.equal(sparse_replicas, fast_replicas)
    assert torch.equal(sparse_order, fast_order)
    assert torch.equal(sparse_routing, fast_routing)

    csr_offsets = torch.empty(
        (v2_demand.numel() + 1,), device="cuda", dtype=torch.int32
    )
    csr_boundaries = torch.empty(
        v2_quota.numel(), device="cuda", dtype=torch.int64
    )
    csr_targets = torch.empty(
        csr_boundaries.shape, device="cuda", dtype=torch.int32
    )
    csr_routing = torch.empty_like(v2_routing)
    _C.materialize_fast_csr_quota_into(
        v2_demand,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        sparse_replicas,
        sparse_order,
        csr_move_plan,
        torch.empty_like(v2_demand),
        torch.empty(v2_demand.size(0), device="cuda", dtype=torch.int64),
        csr_offsets,
        csr_boundaries,
        csr_targets,
        csr_routing,
    )
    csr_offsets_cpu = csr_offsets.cpu().tolist()
    csr_nnz = csr_offsets_cpu[-1]
    assert csr_nnz == 6
    assert csr_nnz < v2_quota.numel()
    csr_boundaries_cpu = csr_boundaries[:csr_nnz].cpu().tolist()
    csr_targets_cpu = csr_targets[:csr_nnz].cpu().tolist()
    csr_reconstructed = torch.zeros_like(v2_quota).view(-1, 2)
    for row in range(v2_demand.numel()):
        previous = 0
        for position in range(csr_offsets_cpu[row], csr_offsets_cpu[row + 1]):
            boundary = csr_boundaries_cpu[position]
            csr_reconstructed[row, csr_targets_cpu[position]] = (
                boundary - previous
            )
            previous = boundary
    assert torch.equal(csr_reconstructed.view_as(v2_quota), fast_quota)
    assert torch.equal(csr_routing, fast_routing)

    # EP32 exercises both supported CTA modes. Their result must be
    # independent of the configured number of solver CTAs.
    large_ranks = 32
    large_experts = 32
    large_demand = torch.zeros(
        (large_experts, large_ranks), device="cuda", dtype=torch.int64
    )
    large_demand[:, 0] = 10
    large_gain = torch.zeros_like(large_demand)
    large_primary = torch.zeros(
        large_experts, device="cuda", dtype=torch.int64
    )
    large_initial = torch.zeros_like(large_demand, dtype=torch.bool)
    large_initial[:, 0] = True

    def run_large_aggregate(solver_ctas):
        replicas_out = large_initial.clone()
        instance_out = torch.empty_like(large_demand)
        loads_out = torch.empty(
            large_ranks, device="cuda", dtype=torch.int64
        )
        slots_out = torch.empty_like(loads_out)
        order_out = torch.empty_like(large_demand)
        moves_out = torch.empty_like(large_demand)
        added_out = torch.empty(1, device="cuda", dtype=torch.int64)
        candidates = torch.empty(
            solver_ctas * 8 * 5 + 4, device="cuda", dtype=torch.int64
        )
        _C.select_compute_replicas_fast_sparse_into(
            large_demand,
            large_gain,
            replicas_out,
            large_primary,
            1,
            1.0,
            instance_out,
            loads_out,
            slots_out,
            order_out,
            moves_out,
            added_out,
            candidates,
            solver_ctas,
        )
        return (
            replicas_out,
            loads_out,
            slots_out,
            order_out,
            moves_out,
            added_out,
        )

    large_single = run_large_aggregate(1)
    large_multi = run_large_aggregate(4)
    # The single-CTA path is valid even when EP is larger than the
    # shared-memory-specialized range.
    assert large_single[1].cpu().tolist() == [10] * large_ranks
    assert large_single[5].item() == large_ranks - 1
    for single_value, multi_value in zip(large_single, large_multi):
        assert torch.equal(single_value, multi_value)
    assert large_multi[1].cpu().tolist() == [10] * large_ranks
    assert large_multi[2].cpu().tolist() == [0] + [1] * (large_ranks - 1)
    assert large_multi[5].item() == large_ranks - 1

    sparse_source = torch.tensor([1, 1], device="cuda", dtype=torch.int64)
    sparse_topk = torch.tensor([[0], [1]], device="cuda", dtype=torch.int64)
    sparse_count = torch.tensor([10, 10], device="cuda", dtype=torch.int64)
    sparse_ordinals = torch.zeros_like(sparse_topk)
    dense_traffic, dense_compute = _C.quota_traffic(
        sparse_source,
        sparse_topk,
        sparse_count,
        fast_quota,
        fast_replicas,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        fast_order,
        sparse_ordinals,
        2,
    )
    sparse_traffic, sparse_compute = _C.sparse_quota_traffic(
        sparse_source,
        sparse_topk,
        sparse_count,
        sparse_prefix,
        sparse_targets,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        sparse_ordinals,
        2,
    )
    csr_traffic, csr_compute = _C.csr_quota_traffic(
        sparse_source,
        sparse_topk,
        sparse_count,
        csr_offsets,
        csr_boundaries,
        csr_targets,
        torch.tensor([0, 0], device="cuda", dtype=torch.int64),
        sparse_ordinals,
        2,
    )
    assert torch.equal(sparse_traffic, dense_traffic)
    assert torch.equal(sparse_compute, dense_compute)
    assert torch.equal(csr_traffic, dense_traffic)
    assert torch.equal(csr_compute, dense_compute)

    # A scarce replica slot must go to the expert that can remove the
    # overload, even when a tiny candidate has a better communication score.
    slot_demand = torch.tensor(
        [[0, 1], [100, 0]], device="cuda", dtype=torch.int64
    )
    slot_replicas = torch.tensor(
        [[True, False], [True, False]], device="cuda", dtype=torch.bool
    )
    slot_loads = torch.empty(2, device="cuda", dtype=torch.int64)
    slot_quota = torch.empty((2, 2, 2), device="cuda", dtype=torch.int64)
    _C.select_compute_replicas_v2_into(
        slot_demand,
        torch.tensor([[0, 1], [0, 0]], device="cuda", dtype=torch.int64),
        torch.zeros((2, 2, 2), device="cuda", dtype=torch.int64),
        slot_replicas,
        torch.tensor([0, 0], device="cuda"),
        1,
        1.0,
        torch.empty_like(slot_demand),
        slot_loads,
        torch.empty_like(slot_loads),
        torch.empty_like(slot_demand),
        slot_quota,
        torch.empty((2, 2), device="cuda", dtype=torch.int64),
        torch.empty(1, device="cuda", dtype=torch.int64),
    )
    assert slot_replicas.cpu().tolist() == [[True, False], [True, True]]
    assert slot_loads.cpu().tolist() == [51, 50]
    assert torch.equal(slot_quota.sum(dim=2), slot_demand.t())

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
