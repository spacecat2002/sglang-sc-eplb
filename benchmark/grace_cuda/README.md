# GRACE CUDA planner

This extension implements the source-aware replication algorithm used by the
simulation benchmark. It does not import or call UltraEP's placement solver.

Build from the repository root:

    cd benchmark/grace_cuda
    python setup.py build_ext --inplace

The extension has separate CUDA translation units for demand histogram,
source-aware Top-N placement, and traffic evaluation. ptx.cuh contains the
cache-load helper; tma.cuh is reserved for contiguous quota-prefix tiles,
where TMA is appropriate. Sparse trace histogram and routing use coalesced
loads and atomics.
