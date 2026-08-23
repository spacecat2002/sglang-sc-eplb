#pragma once

// Irregular trace fields use coalesced/PTX loads. This contiguous staging
// helper is reserved for future quota-prefix tiles on SM90/SM100.
namespace grace_cuda {
template <typename T>
__device__ __forceinline__ void tma_stage_contiguous(T* shared_dst,
                                                      const T* global_src,
                                                      int count) {
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    shared_dst[i] = global_src[i];
  }
  __syncthreads();
}
}  // namespace grace_cuda
