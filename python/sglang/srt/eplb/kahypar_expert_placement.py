"""Optional KaHyPar placement for source-terminal Top-K hypergraphs."""

from __future__ import annotations

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

    KaHyPar is intentionally optional.  The fixed source terminals make the
    connectivity cut equal to the distinct remote destination-rank metric.
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
    vertex_weights = [demand[expert] for expert in experts] + [0] * num_ranks

    # KaHyPar Python wheels have used both six- and seven-argument constructors.
    try:
        hypergraph = kahypar.Hypergraph(
            len(experts) + num_ranks,
            len(tokens),
            offsets,
            edges,
            num_ranks,
            edge_weights,
            vertex_weights,
        )
    except TypeError:  # pragma: no cover - version-specific API
        hypergraph = kahypar.Hypergraph(
            len(experts) + num_ranks,
            len(tokens),
            offsets,
            edges,
            edge_weights,
            vertex_weights,
        )

    pin = getattr(hypergraph, "setFixedVertex", None) or getattr(
        hypergraph, "fixVertex", None
    )
    if pin is None:  # pragma: no cover - version-specific API
        raise RuntimeError("installed KaHyPar binding does not support fixed vertices")
    for rank in range(num_ranks):
        pin(terminal_base + rank, rank)

    context = kahypar.Context()
    if config:
        context.loadINIconfiguration(str(Path(config)))
    if hasattr(context, "setK"):
        context.setK(num_ranks)
    if hasattr(context, "setSeed"):
        context.setSeed(int(seed))
    kahypar.partition(hypergraph, context)

    block_id = getattr(hypergraph, "blockID", None) or getattr(
        hypergraph, "block_id", None
    )
    if block_id is None:  # pragma: no cover - version-specific API
        raise RuntimeError("installed KaHyPar binding has no blockID accessor")
    placement = {expert: int(block_id(index)) for index, expert in enumerate(experts)}
    metrics = evaluate_cable_placement(tokens, placement, num_ranks=num_ranks)
    return {
        "rank_by_expert": placement,
        "experts_by_rank": {
            rank: tuple(expert for expert in experts if placement[expert] == rank)
            for rank in range(num_ranks)
        },
        "metrics": metrics,
    }
