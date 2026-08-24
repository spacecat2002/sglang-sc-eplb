#include <c10/cuda/CUDAStream.h>
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

__global__ void topn_routing_kernel(
    const int64_t* demand, const int64_t* primary, bool* replicas,
    int64_t* routing, int64_t experts, int64_t ranks, int64_t max_extra) {
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
  for (int expert = 0; expert < experts; ++expert) {
    routing[source * experts + expert] =
        replicas[expert * ranks + source] ? source : primary[expert];
  }
}

void select_topn_into(torch::Tensor demand, torch::Tensor primary,
                      int64_t max_extra, torch::Tensor replicas) {
  TORCH_CHECK(demand.is_cuda() && primary.is_cuda());
  const auto experts = demand.size(0);
  const auto ranks = demand.size(1);
  TORCH_CHECK(replicas.is_cuda() && replicas.scalar_type() == torch::kBool &&
              replicas.sizes() == demand.sizes());
  replicas.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(topn_kernel, dim3(ranks), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), experts, ranks, max_extra);
  check_cuda(cudaGetLastError());
}

void select_topn_routing_into(torch::Tensor demand, torch::Tensor primary,
                              int64_t max_extra, torch::Tensor replicas,
                              torch::Tensor routing) {
  TORCH_CHECK(demand.is_cuda() && primary.is_cuda());
  const auto experts = demand.size(0);
  const auto ranks = demand.size(1);
  TORCH_CHECK(replicas.is_cuda() && replicas.scalar_type() == torch::kBool &&
              replicas.sizes() == demand.sizes());
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt64 &&
              routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  replicas.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(topn_routing_kernel, dim3(ranks), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), routing.data_ptr<int64_t>(), experts, ranks,
         max_extra);
  check_cuda(cudaGetLastError());
}

torch::Tensor select_topn(torch::Tensor demand, torch::Tensor primary,
                          int64_t max_extra) {
  const auto experts = demand.size(0);
  const auto ranks = demand.size(1);
  auto replicas = torch::zeros({experts, ranks},
                               demand.options().dtype(torch::kBool));
  select_topn_into(demand, primary, max_extra, replicas);
  return replicas;
}

}  // namespace grace_cuda
