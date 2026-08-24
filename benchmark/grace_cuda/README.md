# GRACE CUDA planner

This extension implements the source-aware replication algorithm used by the
simulation benchmark. It does not import or call UltraEP's placement solver.

Build from the repository root:

    cd benchmark/grace_cuda
    python setup.py build_ext --inplace

The extension has separate CUDA translation units for demand histogram,
source-aware Top-N placement, capacity/export planning, quota routing, and
traffic evaluation. Compute replicas are selected by probing the ideal rank
capacity first, then binary-searching only when that threshold is infeasible.
The final export pass prefers source-local targets at the same feasible
capacity. If direct exports stop above ideal capacity, the solver adds only
the replica edges needed to expose multi-hop augmenting paths. The GPU-resident
runtime consumes that export quota directly, so the hot path does not rerun
the old per-expert quota solver.
