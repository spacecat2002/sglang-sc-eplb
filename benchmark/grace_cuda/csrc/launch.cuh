#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace grace_cuda {

inline void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename Kernel, typename... Args>
inline void launch(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream,
                   Args... args) {
  void* packed[] = {reinterpret_cast<void*>(&args)...};
  check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block,
                              packed, 0, stream));
}

template <typename Kernel, typename... Args>
inline void launch_cooperative(Kernel kernel, dim3 grid, dim3 block,
                               cudaStream_t stream, Args... args) {
  void* packed[] = {reinterpret_cast<void*>(&args)...};
  check_cuda(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel), grid, block, packed, 0, stream));
}

}  // namespace grace_cuda
