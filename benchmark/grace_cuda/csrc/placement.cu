#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "launch.cuh"

namespace grace_cuda {

__global__ void bundle_gain_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, int64_t* gains, int64_t tokens, int64_t k,
    int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t src = source[token];
  const int64_t weight = count[token];
  unsigned long long seen_low = 0;
  unsigned long long seen_high = 0;
  unsigned long long duplicate_low = 0;
  unsigned long long duplicate_high = 0;
  for (int64_t column = 0; column < k; ++column) {
    const int64_t expert = topk[token * k + column];
    const int64_t destination = primary[expert];
    const auto bit = 1ULL << (destination & 63);
    auto& seen = destination < 64 ? seen_low : seen_high;
    auto& duplicate = destination < 64 ? duplicate_low : duplicate_high;
    if (seen & bit) duplicate |= bit;
    seen |= bit;
  }
  for (int64_t column = 0; column < k; ++column) {
    const int64_t expert = topk[token * k + column];
    const int64_t destination = primary[expert];
    const auto bit = 1ULL << (destination & 63);
    const auto duplicate = destination < 64 ? duplicate_low : duplicate_high;
    if (destination != src && !(duplicate & bit)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(gains + expert * ranks + src),
                static_cast<unsigned long long>(weight));
    }
  }
}

__global__ void topn_kernel(const int64_t* demand, const int64_t* primary,
                            bool* replicas, int64_t experts, int64_t ranks,
                            int64_t max_extra) {
  const int source = blockIdx.x;
  if (source >= ranks) return;
  __shared__ int64_t candidate_values[128];
  __shared__ int candidate_experts[128];
  __shared__ int stop;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    if (primary[expert] == source) replicas[expert * ranks + source] = true;
  }
  __syncthreads();
  for (int pick = 0; pick < max_extra; ++pick) {
    int best = -1;
    int64_t best_demand = 0;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
      if (primary[expert] == source || replicas[expert * ranks + source]) continue;
      const int64_t value = demand[expert * ranks + source];
      if (value > best_demand || (value == best_demand && value > 0 &&
                                  (best < 0 || expert < best))) {
        best = expert;
        best_demand = value;
      }
    }
    candidate_values[threadIdx.x] = best_demand;
    candidate_experts[threadIdx.x] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      best = -1;
      best_demand = 0;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        const int expert = candidate_experts[lane];
        const int64_t value = candidate_values[lane];
        if (expert >= 0 &&
            (value > best_demand ||
             (value == best_demand && (best < 0 || expert < best)))) {
          best = expert;
          best_demand = value;
        }
      }
      stop = best < 0 || best_demand == 0;
      if (!stop) {
        replicas[best * ranks + source] = true;
      }
    }
    __syncthreads();
    if (stop) break;
  }
}

__global__ void topn_routing_kernel(
    const int64_t* demand, const int64_t* primary, bool* replicas,
    int64_t* routing, int64_t experts, int64_t ranks, int64_t max_extra) {
  const int source = blockIdx.x;
  if (source >= ranks) return;
  __shared__ int64_t candidate_values[128];
  __shared__ int candidate_experts[128];
  __shared__ int stop;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    if (primary[expert] == source) replicas[expert * ranks + source] = true;
  }
  __syncthreads();
  for (int pick = 0; pick < max_extra; ++pick) {
    int best = -1;
    int64_t best_demand = 0;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
      if (primary[expert] == source || replicas[expert * ranks + source]) continue;
      const int64_t value = demand[expert * ranks + source];
      if (value > best_demand || (value == best_demand && value > 0 &&
                                  (best < 0 || expert < best))) {
        best = expert;
        best_demand = value;
      }
    }
    candidate_values[threadIdx.x] = best_demand;
    candidate_experts[threadIdx.x] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      best = -1;
      best_demand = 0;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        const int expert = candidate_experts[lane];
        const int64_t value = candidate_values[lane];
        if (expert >= 0 &&
            (value > best_demand ||
             (value == best_demand && (best < 0 || expert < best)))) {
          best = expert;
          best_demand = value;
        }
      }
      stop = best < 0 || best_demand == 0;
      if (!stop) replicas[best * ranks + source] = true;
    }
    __syncthreads();
    if (stop) break;
  }
  __syncthreads();
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
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
  launch(topn_kernel, dim3(ranks), dim3(128), stream.stream(),
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
  launch(topn_routing_kernel, dim3(ranks), dim3(128), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), routing.data_ptr<int64_t>(), experts, ranks,
         max_extra);
  check_cuda(cudaGetLastError());
}

void select_bundle_topn_routing_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t max_extra, torch::Tensor gains,
    torch::Tensor replicas, torch::Tensor routing) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(gains.is_cuda() && gains.scalar_type() == torch::kInt64 &&
              gains.dim() == 2 && gains.size(0) == primary.numel());
  const int64_t ranks = gains.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= 128,
              "bundle-aware replication supports at most 128 ranks");
  gains.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  launch(bundle_gain_kernel, dim3((source.size(0) + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         gains.data_ptr<int64_t>(), source.size(0), topk.size(1), ranks);
  check_cuda(cudaGetLastError());
  select_topn_routing_into(gains, primary, max_extra, replicas, routing);
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
