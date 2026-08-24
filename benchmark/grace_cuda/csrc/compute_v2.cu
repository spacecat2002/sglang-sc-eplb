#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>

#include "launch.cuh"

namespace grace_cuda {
namespace {

__global__ void current_bundle_gain_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const bool* replicas, int64_t* gains,
    int64_t* covers, int64_t experts,
    int64_t tokens, int64_t k, int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t src = source[token];
  unsigned long long seen_low = 0;
  unsigned long long seen_high = 0;
  unsigned long long duplicate_low = 0;
  unsigned long long duplicate_high = 0;
  for (int64_t column = 0; column < k; ++column) {
    const int64_t expert = topk[token * k + column];
    const int64_t destination =
        replicas[expert * ranks + src] ? src : primary[expert];
    const auto bit = 1ULL << (destination & 63);
    auto& seen = destination < 64 ? seen_low : seen_high;
    auto& duplicate = destination < 64 ? duplicate_low : duplicate_high;
    if (seen & bit) duplicate |= bit;
    seen |= bit;
  }
  for (int64_t column = 0; column < k; ++column) {
    const int64_t expert = topk[token * k + column];
    const int64_t destination =
        replicas[expert * ranks + src] ? src : primary[expert];
    const auto bit = 1ULL << (destination & 63);
    const auto duplicate = destination < 64 ? duplicate_low : duplicate_high;
    if (destination != src && !(duplicate & bit))
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    gains + expert * ranks + src),
                static_cast<unsigned long long>(count[token]));
    unsigned long long covered_low = seen_low;
    unsigned long long covered_high = seen_high;
    if (src < 64)
      covered_low &= ~(1ULL << src);
    else
      covered_high &= ~(1ULL << (src & 63));
    if (!(duplicate & bit)) {
      if (destination < 64)
        covered_low &= ~bit;
      else
        covered_high &= ~bit;
    }
    while (covered_low) {
      const int target = __ffsll(static_cast<long long>(covered_low)) - 1;
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    covers + (src * experts + expert) * ranks + target),
                static_cast<unsigned long long>(count[token]));
      covered_low &= covered_low - 1;
    }
    while (covered_high) {
      const int target =
          64 + __ffsll(static_cast<long long>(covered_high)) - 1;
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    covers + (src * experts + expert) * ranks + target),
                static_cast<unsigned long long>(count[token]));
      covered_high &= covered_high - 1;
    }
  }
}

struct CapacityCandidate {
  int source;
  int over;
  int target;
  int64_t expert;
  int64_t amount;
  int64_t potential;
  int64_t penalty;
  int64_t gain;
  bool is_new;
};

__device__ bool better_capacity_candidate(
    const CapacityCandidate& candidate, const CapacityCandidate& current,
    const int64_t* loads) {
  if (current.expert < 0) return candidate.expert >= 0;
  if (candidate.expert < 0) return false;
  if (loads[candidate.over] != loads[current.over])
    return loads[candidate.over] > loads[current.over];
  if (candidate.is_new != current.is_new)
    return candidate.is_new < current.is_new;
  if (candidate.potential != current.potential)
    return candidate.potential > current.potential;
  const int64_t candidate_scaled = candidate.penalty * current.amount;
  const int64_t current_scaled = current.penalty * candidate.amount;
  if (candidate_scaled != current_scaled)
    return candidate_scaled < current_scaled;
  if (candidate.penalty != current.penalty)
    return candidate.penalty < current.penalty;
  if (candidate.gain != current.gain) return candidate.gain > current.gain;
  if (candidate.amount != current.amount) return candidate.amount > current.amount;
  if (candidate.source != current.source) return candidate.source < current.source;
  if (candidate.expert != current.expert) return candidate.expert < current.expert;
  if (candidate.target != current.target) return candidate.target < current.target;
  return candidate.over < current.over;
}

__global__ void capacity_v2_kernel(
    const int64_t* demand, const int64_t* bundle_gain,
    const int64_t* bundle_cover,
    const int64_t* primary, bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* quota, int64_t* routing,
    int64_t* added_out, int64_t experts, int ranks,
    int64_t max_extra_per_rank, double imbalance_limit) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  __shared__ CapacityCandidate candidates[256];
  __shared__ int64_t capacity;
  __shared__ int64_t added;
  __shared__ int stop;
  const int64_t matrix_size = experts * ranks;
  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    instance[index] = 0;
    addition_order[index] = 0;
  }
  for (int64_t index = tid; index < matrix_size * ranks; index += blockDim.x)
    quota[index] = 0;
  for (int rank = tid; rank < ranks; rank += blockDim.x) {
    added_by_rank[rank] = 0;
    loads[rank] = 0;
  }
  __syncthreads();

  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    const int source = index / experts;
    const int64_t expert = index % experts;
    const int target = replicas[expert * ranks + source]
                           ? source
                           : primary[expert];
    const int64_t value = demand[expert * ranks + source];
    quota[index * ranks + target] = value;
    routing[index] = target;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  instance + expert * ranks + target),
              static_cast<unsigned long long>(value));
    atomicAdd(reinterpret_cast<unsigned long long*>(loads + target),
              static_cast<unsigned long long>(value));
  }
  __syncthreads();

  if (tid == 0) {
    int64_t total = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      total += loads[rank];
    }
    const double limit = imbalance_limit >= 1.0 ? imbalance_limit : 1.0;
    capacity = static_cast<int64_t>(
        ceil(static_cast<double>(total) / ranks * limit));
    added = 0;
  }
  __syncthreads();

  while (true) {
    CapacityCandidate best = {-1, -1, -1, -1, 0, 0, LLONG_MAX, -1, true};
    const int64_t candidate_count = experts * ranks * ranks;
    for (int64_t index = tid; index < candidate_count; index += blockDim.x) {
      const int target = index % ranks;
      const int64_t pair = index / ranks;
      const int64_t expert = pair % experts;
      const int source = pair / experts;
      const int64_t route_index = source * experts + expert;
      const int64_t offset = route_index * ranks;
      int over = -1;
      int64_t potential = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        if (rank == target || loads[rank] <= capacity) continue;
        potential += min(instance[expert * ranks + rank],
                         loads[rank] - capacity);
        if (!quota[offset + rank]) continue;
        if (over < 0 || loads[rank] > loads[over] ||
            (loads[rank] == loads[over] && quota[offset + rank] > quota[offset + over]))
          over = rank;
      }
      if (over < 0 || loads[target] >= capacity) continue;
      const int64_t available = quota[offset + over];
      const bool present = replicas[expert * ranks + target] ||
                           addition_order[expert * ranks + target];
      if (!present && added_by_rank[target] >= max_extra_per_rank) continue;
      const int64_t amount = min(
          min(loads[over] - capacity, capacity - loads[target]), available);
      const int64_t headroom = capacity - loads[target];
      potential = min(potential, headroom);
      const int original = replicas[expert * ranks + source]
                               ? source
                               : primary[expert];
      const int64_t gain =
          over == original && amount == available
              ? min(amount, bundle_gain[expert * ranks + source])
              : 0;
      const int64_t covered =
          target == source
              ? amount
              : min(amount,
                    bundle_cover[(source * experts + expert) * ranks + target]);
      const int64_t penalty = amount - covered - gain;
      CapacityCandidate candidate = {
          source, over, target, expert, amount, potential, penalty, gain,
          !present};
      if (better_capacity_candidate(candidate, best, loads)) best = candidate;
    }
    candidates[tid] = best;
    __syncthreads();
    if (tid == 0) {
      best = candidates[0];
      for (int lane = 1; lane < blockDim.x; ++lane)
        if (better_capacity_candidate(candidates[lane], best, loads))
          best = candidates[lane];
      stop = best.expert < 0;
      if (!stop) {
        if (best.is_new) {
          addition_order[best.expert * ranks + best.target] = ++added;
          ++added_by_rank[best.target];
        }
        const int64_t offset =
            (best.source * experts + best.expert) * ranks;
        quota[offset + best.over] -= best.amount;
        quota[offset + best.target] += best.amount;
        instance[best.expert * ranks + best.over] -= best.amount;
        instance[best.expert * ranks + best.target] += best.amount;
        loads[best.over] -= best.amount;
        loads[best.target] += best.amount;
        int next = 0;
        for (int rank = 1; rank < ranks; ++rank)
          if (quota[offset + rank] > quota[offset + next]) next = rank;
        routing[best.source * experts + best.expert] = next;
      } else {
        *added_out = added;
      }
    }
    __syncthreads();
    if (stop) break;
  }

  for (int64_t index = tid; index < matrix_size; index += blockDim.x)
    replicas[index] = replicas[index] || addition_order[index];
}

}  // namespace

void current_bundle_gains_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    torch::Tensor covers) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda() && gains.is_cuda() &&
              covers.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              gains.scalar_type() == torch::kInt64 &&
              covers.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(replicas.sizes() == gains.sizes() &&
              replicas.size(0) == primary.numel());
  TORCH_CHECK(covers.dim() == 3 && covers.size(0) == gains.size(1) &&
              covers.size(1) == gains.size(0) &&
              covers.size(2) == gains.size(1));
  TORCH_CHECK(gains.size(1) <= 128,
              "bundle gain supports at most 128 ranks");
  gains.zero_();
  covers.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  launch(current_bundle_gain_kernel,
         dim3((source.size(0) + 255) / 256), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), gains.data_ptr<int64_t>(),
         covers.data_ptr<int64_t>(), gains.size(0), source.size(0),
         topk.size(1), gains.size(1));
  check_cuda(cudaGetLastError());
}

void select_compute_replicas_v2_into(
    torch::Tensor demand, torch::Tensor bundle_gain,
    torch::Tensor bundle_cover,
    torch::Tensor replicas, torch::Tensor primary,
    int64_t max_extra_per_rank, double imbalance_limit,
    torch::Tensor instance, torch::Tensor loads,
    torch::Tensor added_by_rank, torch::Tensor addition_order,
    torch::Tensor quota, torch::Tensor routing, torch::Tensor added) {
  TORCH_CHECK(demand.is_cuda() && bundle_gain.is_cuda() &&
              bundle_cover.is_cuda() && replicas.is_cuda() && primary.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes() &&
              bundle_gain.scalar_type() == torch::kInt64 &&
              bundle_gain.sizes() == demand.sizes() &&
              bundle_cover.scalar_type() == torch::kInt64);
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= 128 && max_extra_per_rank >= 0);
  TORCH_CHECK(primary.numel() == experts && instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.numel() == ranks && added_by_rank.numel() == ranks &&
              addition_order.sizes() == demand.sizes() && added.numel() == 1);
  TORCH_CHECK(quota.dim() == 3 && quota.size(0) == ranks &&
              quota.size(1) == experts && quota.size(2) == ranks);
  TORCH_CHECK(bundle_cover.sizes() == quota.sizes());
  TORCH_CHECK(routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(capacity_v2_kernel, dim3(1), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), bundle_gain.data_ptr<int64_t>(),
         bundle_cover.data_ptr<int64_t>(),
         primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), instance.data_ptr<int64_t>(),
         loads.data_ptr<int64_t>(), added_by_rank.data_ptr<int64_t>(),
         addition_order.data_ptr<int64_t>(), quota.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), added.data_ptr<int64_t>(), experts,
         static_cast<int>(ranks), max_extra_per_rank, imbalance_limit);
  check_cuda(cudaGetLastError());
}

}  // namespace grace_cuda
