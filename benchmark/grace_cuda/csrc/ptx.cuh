#pragma once

#include <cuda_runtime.h>

namespace grace_cuda {

__device__ __forceinline__ int64_t ld_global_i64(const int64_t* ptr) {
#if defined(__CUDA_ARCH__)
  int64_t value;
  asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
  return value;
#else
  return *ptr;
#endif
}

}  // namespace grace_cuda
