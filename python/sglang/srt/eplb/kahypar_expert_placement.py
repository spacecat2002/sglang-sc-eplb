"""Optional KaHyPar placement for source-terminal Top-K hypergraphs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

from .cable_expert_placement import evaluate_cable_placement
from .expert_affinity_graph import RoutedToken


def kahypar_expert_placement(
    tokens: Sequence[RoutedToken],
    *,
    experts: Sequence[int],
    num_ranks: int,
    config: str | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Partition bundles with KaHyPar while pinning one source terminal/rank.

    KaHyPar is intentionally optional. Fixed source terminals make the
    connectivity cut equal to the distinct remote destination-rank metric;
    older bindings without fixed vertices use heavy anchors as a fallback.
    """

    try:
        import kahypar
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "KaHyPar is not installed; install the optional 'kahypar' package"
        ) from exc
    experts = tuple(sorted(experts))
    if not tokens or not experts:
        raise ValueError("tokens and experts must not be empty")
    if num_ranks < 1 or any(token.source_rank >= num_ranks for token in tokens):
        raise ValueError("invalid num_ranks or source rank")
    expert_index = {expert: index for index, expert in enumerate(experts)}
    if any(
        expert not in expert_index
        for token in tokens
        for expert in token.topk_experts
    ):
        raise ValueError("tokens contain an expert outside experts")

    terminal_base = len(experts)
    edges: list[int] = []
    offsets = [0]
    edge_weights: list[int] = []
    for token in tokens:
        edges.append(terminal_base + token.source_rank)
        edges.extend(expert_index[expert] for expert in token.topk_experts)
        offsets.append(len(edges))
        edge_weights.append(int(token.count))
    demand = {expert: 0 for expert in experts}
    for token in tokens:
        for expert in token.topk_experts:
            demand[expert] += token.count
    # KaHyPar Python wheels have used both six- and seven-argument constructors.
    def build_hypergraph(vertex_weights: list[int]):
        try:
            return kahypar.Hypergraph(
                len(experts) + num_ranks,
                len(tokens),
                offsets,
                edges,
                num_ranks,
                edge_weights,
                vertex_weights,
            )
        except TypeError:  # pragma: no cover - version-specific API
            return kahypar.Hypergraph(
                len(experts) + num_ranks,
                len(tokens),
                offsets,
                edges,
                edge_weights,
                vertex_weights,
            )

    # KaHyPar requires strictly positive weights. Unit terminal weights are
    # enough when the binding supports fixed vertices.
    vertex_weights = [max(1, demand[expert]) for expert in experts] + [1] * num_ranks
    hypergraph = build_hypergraph(vertex_weights)

    context = kahypar.Context()
    if config:
        context.loadINIconfiguration(str(Path(config)))
    if hasattr(context, "setK"):
        context.setK(num_ranks)
    if hasattr(context, "setSeed"):
        context.setSeed(int(seed))

    pin = getattr(hypergraph, "setFixedVertex", None) or getattr(
        hypergraph, "fixVertex", None
    )
    if pin is None:
        pin = getattr(context, "setFixedVertex", None) or getattr(
            context, "fixVertex", None
        )
    terminal_mode = "fixed"
    if pin is not None:
        for rank in range(num_ranks):
            pin(terminal_base + rank, rank)
    else:
        # Older wheels omit fixed-vertex support. Equal heavy anchors force one
        # source terminal into each block; blocks are relabelled below.
        warnings.warn(
            "KaHyPar binding has no fixed-vertex API; using heavy source anchors",
            RuntimeWarning,
            stacklevel=2,
        )
        anchor_weight = max(1, sum(demand.values()) + 1)
        vertex_weights = [max(1, demand[expert]) for expert in experts] + [
            anchor_weight
        ] * num_ranks
        hypergraph = build_hypergraph(vertex_weights)
        terminal_mode = "anchor"
    kahypar.partition(hypergraph, context)

    block_id = getattr(hypergraph, "blockID", None) or getattr(
        hypergraph, "block_id", None
    )
    if block_id is None:  # pragma: no cover - version-specific API
        raise RuntimeError("installed KaHyPar binding has no blockID accessor")
    block_remap = {block: block for block in range(num_ranks)}
    if terminal_mode == "anchor":
        terminal_blocks = [block_id(terminal_base + rank) for rank in range(num_ranks)]
        block_remap = {}
        remaining_ranks = iter(
            rank for rank in range(num_ranks) if rank not in terminal_blocks
        )
        for rank, block in enumerate(terminal_blocks):
            block_remap.setdefault(block, rank)
        for block in range(num_ranks):
            if block not in block_remap:
                block_remap[block] = next(remaining_ranks)
    placement = {
        expert: int(block_remap[int(block_id(index))])
        for index, expert in enumerate(experts)
    }
    metrics = evaluate_cable_placement(tokens, placement, num_ranks=num_ranks)
    return {
        "rank_by_expert": placement,
        "experts_by_rank": {
            rank: tuple(expert for expert in experts if placement[expert] == rank)
            for rank in range(num_ranks)
        },
        "metrics": metrics,
        "terminal_mode": terminal_mode,
    }
