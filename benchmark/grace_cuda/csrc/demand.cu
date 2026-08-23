#include <torch/extension.h>

#include "launch.cuh"
#include "ptx.cuh"

namespace grace_cuda {

__global__ void source_demand_kernel(const int64_t* source, const int64_t* topk,
                                     const int64_t* count, int64_t* demand,
                                     int64_t tokens, int64_t k, int64_t ranks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = tokens * k;
  if (index >= total) return;
  const int64_t token = index / k;
  const int64_t expert = topk[index];
  const int64_t rank = source[token];
  atomicAdd(reinterpret_cast<unsigned long long*>(demand + expert * ranks + rank),
            static_cast<unsigned long long>(ld_global_i64(count + token)));
}

torch::Tensor source_demand(torch::Tensor source, torch::Tensor topk,
                            torch::Tensor count, int64_t num_experts,
                            int64_t num_ranks) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1);
  TORCH_CHECK(topk.size(0) == source.size(0) &&
              count.size(0) == source.size(0));
  auto demand = torch::zeros({num_experts, num_ranks},
                             source.options().dtype(torch::kInt64));
  auto stream = at::cuda::getDefaultCUDAStream();
  const int64_t total = source.size(0) * topk.size(1);
  launch(source_demand_kernel, dim3((total + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(), source.size(0),
         topk.size(1), num_ranks);
  check_cuda(cudaGetLastError());
  return demand;
}

}  // namespace grace_cuda
