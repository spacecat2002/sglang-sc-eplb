#include <torch/extension.h>

#include "launch.cuh"

namespace grace_cuda {

__global__ void topn_kernel(const int64_t* demand, const int64_t* primary,
                            bool* replicas, int64_t experts, int64_t ranks,
                            int64_t max_extra) {
  const int source = blockIdx.x;
  if (source >= ranks || threadIdx.x != 0) return;
  for (int expert = 0; expert < experts; ++expert) {
    replicas[expert * ranks + primary[expert]] = true;
  }
  for (int pick = 0; pick < max_extra; ++pick) {
    int best = -1;
    int64_t best_demand = 0;
    for (int expert = 0; expert < experts; ++expert) {
      if (primary[expert] == source || replicas[expert * ranks + source]) continue;
      const int64_t value = demand[expert * ranks + source];
      if (value > best_demand || (value == best_demand && value > 0 &&
                                  (best < 0 || expert < best))) {
        best = expert;
        best_demand = value;
      }
    }
    if (best < 0 || best_demand == 0) break;
    replicas[best * ranks + source] = true;
  }
}

torch::Tensor select_topn(torch::Tensor demand, torch::Tensor primary,
                          int64_t max_extra) {
  TORCH_CHECK(demand.is_cuda() && primary.is_cuda());
  const auto experts = demand.size(0);
  const auto ranks = demand.size(1);
  auto replicas = torch::zeros({experts, ranks},
                               demand.options().dtype(torch::kBool));
  auto stream = at::cuda::getDefaultCUDAStream();
  launch(topn_kernel, dim3(ranks), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), experts, ranks, max_extra);
  check_cuda(cudaGetLastError());
  return replicas;
}

}  // namespace grace_cuda
