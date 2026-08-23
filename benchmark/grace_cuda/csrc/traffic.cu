#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

#include "launch.cuh"

namespace grace_cuda {

__global__ void traffic_kernel(const int64_t* source, const int64_t* topk,
                               const int64_t* count, const int64_t* primary,
                               const bool* replicas, int64_t* traffic,
                               int64_t* compute, int64_t tokens, int64_t k,
                               int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t src = source[token];
  const int64_t weight = count[token];
  bool seen[128];
  for (int r = 0; r < ranks; ++r) seen[r] = false;
  for (int col = 0; col < k; ++col) {
    const int64_t expert = topk[token * k + col];
    const int64_t destination =
        replicas[expert * ranks + src] ? src : primary[expert];
    atomicAdd(reinterpret_cast<unsigned long long*>(compute + destination),
              static_cast<unsigned long long>(weight));
    if (destination != src && !seen[destination]) {
      seen[destination] = true;
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    traffic + src * ranks + destination),
                static_cast<unsigned long long>(weight));
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> traffic(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, int64_t num_ranks) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda());
  auto traffic_out = torch::zeros({num_ranks, num_ranks},
                                  source.options().dtype(torch::kInt64));
  auto compute = torch::zeros({num_ranks},
                              source.options().dtype(torch::kInt64));
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  launch(traffic_kernel, dim3((tokens + 255) / 256), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), traffic_out.data_ptr<int64_t>(),
         compute.data_ptr<int64_t>(), tokens, topk.size(1), num_ranks);
  check_cuda(cudaGetLastError());
  return {traffic_out, compute};
}

}  // namespace grace_cuda
