#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

#include "launch.cuh"

namespace grace_cuda {
namespace {

__global__ void capacity_v2_kernel(
    const int64_t* demand, const int64_t* primary, bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* quota, int64_t* routing,
    int64_t* added_out, int64_t experts, int ranks,
    int64_t max_extra_per_rank, double imbalance_limit) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  const int64_t matrix_size = experts * ranks;
  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    instance[index] = 0;
    addition_order[index] = 0;
  }
  for (int64_t index = tid; index < matrix_size * ranks; index += blockDim.x)
    quota[index] = 0;
  for (int rank = tid; rank < ranks; rank += blockDim.x)
    added_by_rank[rank] = 0;
  __syncthreads();

  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int source = index % ranks;
    const int target = replicas[index] ? source : primary[expert];
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  instance + expert * ranks + target),
              static_cast<unsigned long long>(demand[index]));
  }
  __syncthreads();

  if (tid == 0) {
    int64_t total = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      loads[rank] = 0;
      for (int64_t expert = 0; expert < experts; ++expert)
        loads[rank] += instance[expert * ranks + rank];
      total += loads[rank];
    }
    const double limit = imbalance_limit >= 1.0 ? imbalance_limit : 1.0;
    const int64_t capacity = static_cast<int64_t>(
        ceil(static_cast<double>(total) / ranks * limit));
    int64_t added = 0;
    while (true) {
      int over = -1;
      int target = -1;
      int64_t expert = -1;
      int64_t amount = 0;
      int64_t local_gain = -1;
      bool is_new = true;
      for (int candidate_over = 0; candidate_over < ranks; ++candidate_over) {
        if (loads[candidate_over] <= capacity) continue;
        for (int candidate_target = 0; candidate_target < ranks;
             ++candidate_target) {
          if (candidate_target == candidate_over ||
              loads[candidate_target] >= capacity)
            continue;
          const int64_t slack = capacity - loads[candidate_target];
          for (int64_t candidate_expert = 0; candidate_expert < experts;
               ++candidate_expert) {
            const int64_t available =
                instance[candidate_expert * ranks + candidate_over];
            if (!available) continue;
            const bool present =
                replicas[candidate_expert * ranks + candidate_target] ||
                addition_order[candidate_expert * ranks + candidate_target];
            if (!present &&
                added_by_rank[candidate_target] >= max_extra_per_rank)
              continue;
            const int64_t candidate_amount = min(
                min(loads[candidate_over] - capacity, slack), available);
            const int64_t candidate_local = min(
                candidate_amount,
                demand[candidate_expert * ranks + candidate_target]);
            const bool candidate_new = !present;
            if (expert < 0 || loads[candidate_over] > loads[over] ||
                (loads[candidate_over] == loads[over] &&
                 candidate_amount > amount) ||
                (loads[candidate_over] == loads[over] &&
                 candidate_amount == amount && candidate_local > local_gain) ||
                (loads[candidate_over] == loads[over] &&
                 candidate_amount == amount && candidate_local == local_gain &&
                 candidate_new < is_new)) {
              over = candidate_over;
              target = candidate_target;
              expert = candidate_expert;
              amount = candidate_amount;
              local_gain = candidate_local;
              is_new = candidate_new;
            }
          }
        }
      }
      if (expert < 0) break;
      if (is_new) {
        addition_order[expert * ranks + target] = ++added;
        ++added_by_rank[target];
      }
      instance[expert * ranks + over] -= amount;
      instance[expert * ranks + target] += amount;
      loads[over] -= amount;
      loads[target] += amount;
    }
    *added_out = added;
  }
  __syncthreads();

  for (int64_t index = tid; index < matrix_size; index += blockDim.x)
    replicas[index] = replicas[index] || addition_order[index];
  __syncthreads();

  // Fixed instance capacities are allocated source-local first, then to the
  // communication solver's preferred destination, then to the largest remainder.
  for (int64_t expert = tid; expert < experts; expert += blockDim.x) {
    for (int source = 0; source < ranks; ++source) {
      const int64_t offset = (source * experts + expert) * ranks;
      const int64_t local = min(demand[expert * ranks + source],
                                instance[expert * ranks + source]);
      quota[offset + source] = local;
      instance[expert * ranks + source] -= local;
    }
    for (int source = 0; source < ranks; ++source) {
      const int64_t offset = (source * experts + expert) * ranks;
      if (!demand[expert * ranks + source]) continue;
      int64_t remaining = demand[expert * ranks + source] - quota[offset + source];
      const int preferred = routing[source * experts + expert];
      if (remaining && preferred != source) {
        const int64_t moved = min(remaining, instance[expert * ranks + preferred]);
        quota[offset + preferred] += moved;
        instance[expert * ranks + preferred] -= moved;
        remaining -= moved;
      }
      while (remaining) {
        int target = -1;
        for (int rank = 0; rank < ranks; ++rank)
          if (instance[expert * ranks + rank] &&
              (target < 0 || instance[expert * ranks + rank] >
                                 instance[expert * ranks + target]))
            target = rank;
        if (target < 0) break;
        const int64_t moved = min(remaining, instance[expert * ranks + target]);
        quota[offset + target] += moved;
        instance[expert * ranks + target] -= moved;
        remaining -= moved;
      }
      int target = 0;
      for (int rank = 1; rank < ranks; ++rank)
        if (quota[offset + rank] > quota[offset + target]) target = rank;
      routing[source * experts + expert] = target;
    }
  }
}

}  // namespace

void select_compute_replicas_v2_into(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor primary,
    int64_t max_extra_per_rank, double imbalance_limit,
    torch::Tensor instance, torch::Tensor loads,
    torch::Tensor added_by_rank, torch::Tensor addition_order,
    torch::Tensor quota, torch::Tensor routing, torch::Tensor added) {
  TORCH_CHECK(demand.is_cuda() && replicas.is_cuda() && primary.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes());
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= 128 && max_extra_per_rank >= 0);
  TORCH_CHECK(primary.numel() == experts && instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.numel() == ranks && added_by_rank.numel() == ranks &&
              addition_order.sizes() == demand.sizes() && added.numel() == 1);
  TORCH_CHECK(quota.dim() == 3 && quota.size(0) == ranks &&
              quota.size(1) == experts && quota.size(2) == ranks);
  TORCH_CHECK(routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(capacity_v2_kernel, dim3(1), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), instance.data_ptr<int64_t>(),
         loads.data_ptr<int64_t>(), added_by_rank.data_ptr<int64_t>(),
         addition_order.data_ptr<int64_t>(), quota.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), added.data_ptr<int64_t>(), experts,
         static_cast<int>(ranks), max_extra_per_rank, imbalance_limit);
  check_cuda(cudaGetLastError());
}

}  // namespace grace_cuda
