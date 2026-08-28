#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <climits>
#include <cmath>

#include "launch.cuh"
#include "limits.cuh"

namespace grace_cuda {
namespace {

namespace cg = cooperative_groups;

__global__ void current_bundle_gain_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const bool* replicas, int64_t* gains,
    int64_t* covers, int64_t experts,
    int64_t tokens, int64_t k, int64_t ranks, bool clear_output) {
  if (clear_output) {
    for (int64_t index = threadIdx.x; index < experts * ranks;
         index += blockDim.x)
      gains[index] = 0;
    for (int64_t index = threadIdx.x; index < experts * ranks * ranks;
         index += blockDim.x)
      covers[index] = 0;
    __syncthreads();
  }
  for (int64_t token =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
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
      const auto duplicate =
          destination < 64 ? duplicate_low : duplicate_high;
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
        const int target =
            __ffsll(static_cast<long long>(covered_low)) - 1;
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
}

template <int FixedK>
__global__ void current_bundle_gain_fast_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const bool* replicas, int64_t* gains,
    int64_t experts, int64_t tokens, int64_t k, int64_t ranks,
    bool clear_output) {
  if (clear_output) {
    for (int64_t index = threadIdx.x; index < experts * ranks;
         index += blockDim.x)
      gains[index] = 0;
    __syncthreads();
  }
  for (int64_t token =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t src = source[token];
    unsigned long long seen_low = 0;
    unsigned long long seen_high = 0;
    unsigned long long duplicate_low = 0;
    unsigned long long duplicate_high = 0;
    const int64_t actual_k = FixedK ? FixedK : k;
    int64_t cached_experts[FixedK > 0 ? FixedK : 1];
    int cached_destinations[FixedK > 0 ? FixedK : 1];
#pragma unroll
    for (int64_t column = 0; column < actual_k; ++column) {
      const int64_t expert = topk[token * actual_k + column];
      const int64_t destination =
          replicas[expert * ranks + src] ? src : primary[expert];
      if constexpr (FixedK > 0) {
        cached_experts[column] = expert;
        cached_destinations[column] = destination;
      }
      const auto bit = 1ULL << (destination & 63);
      auto& seen = destination < 64 ? seen_low : seen_high;
      auto& duplicate = destination < 64 ? duplicate_low : duplicate_high;
      if (seen & bit) duplicate |= bit;
      seen |= bit;
    }
#pragma unroll
    for (int64_t column = 0; column < actual_k; ++column) {
      const int64_t expert = FixedK ? cached_experts[column]
                                    : topk[token * actual_k + column];
      const int64_t destination =
          FixedK ? cached_destinations[column]
                 : (replicas[expert * ranks + src] ? src : primary[expert]);
      const auto bit = 1ULL << (destination & 63);
      const auto duplicate =
          destination < 64 ? duplicate_low : duplicate_high;
      if (destination != src && !(duplicate & bit))
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      gains + expert * ranks + src),
                  static_cast<unsigned long long>(count[token]));
    }
  }
}

template <int FixedK>
__global__ void incremental_bundle_gain_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const bool* replicas, int64_t* gains,
    const int32_t* bundle_heads, const int32_t* bundle_next,
    int32_t* bundle_marks, int32_t epoch, int64_t experts, int64_t tokens,
    int64_t runtime_k, int64_t ranks) {
  const int64_t actual_k = FixedK ? FixedK : runtime_k;
  const int64_t matrix_size = experts * ranks;
  for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       pair < matrix_size;
       pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t expert = pair / ranks;
    const int64_t src = pair % ranks;
    if (!replicas[pair] || primary[expert] == src) continue;

    for (int32_t entry = bundle_heads[pair]; entry >= 0;
         entry = bundle_next[entry]) {
      const int64_t token = entry / actual_k;
      if (token >= tokens ||
          atomicExch(bundle_marks + token, epoch) == epoch)
        continue;

      const int64_t token_src = source[token];
      const auto weight = static_cast<unsigned long long>(count[token]);
      unsigned long long old_seen_low = 0;
      unsigned long long old_seen_high = 0;
      unsigned long long old_duplicate_low = 0;
      unsigned long long old_duplicate_high = 0;
      unsigned long long new_seen_low = 0;
      unsigned long long new_seen_high = 0;
      unsigned long long new_duplicate_low = 0;
      unsigned long long new_duplicate_high = 0;
      int64_t cached_experts[FixedK > 0 ? FixedK : 1];
      int cached_old_destinations[FixedK > 0 ? FixedK : 1];
      int cached_new_destinations[FixedK > 0 ? FixedK : 1];

#pragma unroll
      for (int64_t column = 0; column < actual_k; ++column) {
        const int64_t current_expert = topk[token * actual_k + column];
        const int old_destination = primary[current_expert];
        const int new_destination =
            replicas[current_expert * ranks + token_src]
                ? static_cast<int>(token_src)
                : old_destination;
        if constexpr (FixedK > 0) {
          cached_experts[column] = current_expert;
          cached_old_destinations[column] = old_destination;
          cached_new_destinations[column] = new_destination;
        }

        const auto old_bit = 1ULL << (old_destination & 63);
        auto& old_seen =
            old_destination < 64 ? old_seen_low : old_seen_high;
        auto& old_duplicate = old_destination < 64 ? old_duplicate_low
                                                   : old_duplicate_high;
        if (old_seen & old_bit) old_duplicate |= old_bit;
        old_seen |= old_bit;

        const auto new_bit = 1ULL << (new_destination & 63);
        auto& new_seen =
            new_destination < 64 ? new_seen_low : new_seen_high;
        auto& new_duplicate = new_destination < 64 ? new_duplicate_low
                                                   : new_duplicate_high;
        if (new_seen & new_bit) new_duplicate |= new_bit;
        new_seen |= new_bit;
      }

#pragma unroll
      for (int64_t column = 0; column < actual_k; ++column) {
        const int64_t current_expert =
            FixedK ? cached_experts[column]
                   : topk[token * actual_k + column];
        const int old_destination =
            FixedK ? cached_old_destinations[column] : primary[current_expert];
        const int new_destination =
            FixedK ? cached_new_destinations[column]
                   : (replicas[current_expert * ranks + token_src]
                          ? static_cast<int>(token_src)
                          : old_destination);
        const auto old_bit = 1ULL << (old_destination & 63);
        const auto new_bit = 1ULL << (new_destination & 63);
        const bool old_contributes =
            old_destination != token_src &&
            !((old_destination < 64 ? old_duplicate_low : old_duplicate_high) &
              old_bit);
        const bool new_contributes =
            new_destination != token_src &&
            !((new_destination < 64 ? new_duplicate_low : new_duplicate_high) &
              new_bit);
        if (old_contributes == new_contributes) continue;
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      gains + current_expert * ranks + token_src),
                  new_contributes ? weight : 0ULL - weight);
      }
    }
  }
}

template <int FixedK>
__global__ void incremental_bundle_gain_csr_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const bool* replicas, int64_t* gains,
    const int32_t* offsets, const int32_t* incidence_entries,
    int32_t* bundle_marks, int32_t epoch, int64_t experts, int64_t tokens,
    int64_t runtime_k, int64_t ranks) {
  const int64_t actual_k = FixedK ? FixedK : runtime_k;
  const int64_t matrix_size = experts * ranks;
  for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       pair < matrix_size; pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t expert = pair / ranks;
    const int64_t src = pair % ranks;
    if (!replicas[pair] || primary[expert] == src) continue;
    for (int32_t position = offsets[pair]; position < offsets[pair + 1];
         ++position) {
      const int64_t entry = incidence_entries[position];
      const int64_t token = entry / actual_k;
      if (token >= tokens || atomicExch(bundle_marks + token, epoch) == epoch)
        continue;
      const int64_t token_src = source[token];
      const auto weight = static_cast<unsigned long long>(count[token]);
      unsigned long long old_seen_low = 0, old_seen_high = 0;
      unsigned long long old_duplicate_low = 0, old_duplicate_high = 0;
      unsigned long long new_seen_low = 0, new_seen_high = 0;
      unsigned long long new_duplicate_low = 0, new_duplicate_high = 0;
      int64_t cached_experts[FixedK > 0 ? FixedK : 1];
      int cached_old[FixedK > 0 ? FixedK : 1];
      int cached_new[FixedK > 0 ? FixedK : 1];
#pragma unroll
      for (int64_t column = 0; column < actual_k; ++column) {
        const int64_t current_expert = topk[token * actual_k + column];
        const int old_destination = primary[current_expert];
        const int new_destination = replicas[current_expert * ranks + token_src]
                                        ? static_cast<int>(token_src)
                                        : old_destination;
        if constexpr (FixedK > 0) {
          cached_experts[column] = current_expert;
          cached_old[column] = old_destination;
          cached_new[column] = new_destination;
        }
        const auto old_bit = 1ULL << (old_destination & 63);
        auto& old_seen = old_destination < 64 ? old_seen_low : old_seen_high;
        auto& old_duplicate = old_destination < 64 ? old_duplicate_low
                                                   : old_duplicate_high;
        if (old_seen & old_bit) old_duplicate |= old_bit;
        old_seen |= old_bit;
        const auto new_bit = 1ULL << (new_destination & 63);
        auto& new_seen = new_destination < 64 ? new_seen_low : new_seen_high;
        auto& new_duplicate = new_destination < 64 ? new_duplicate_low
                                                   : new_duplicate_high;
        if (new_seen & new_bit) new_duplicate |= new_bit;
        new_seen |= new_bit;
      }
#pragma unroll
      for (int64_t column = 0; column < actual_k; ++column) {
        const int64_t current_expert = FixedK ? cached_experts[column]
                                              : topk[token * actual_k + column];
        const int old_destination = FixedK ? cached_old[column]
                                           : primary[current_expert];
        const int new_destination = FixedK
                                        ? cached_new[column]
                                        : (replicas[current_expert * ranks + token_src]
                                               ? static_cast<int>(token_src)
                                               : old_destination);
        const auto old_bit = 1ULL << (old_destination & 63);
        const auto new_bit = 1ULL << (new_destination & 63);
        const bool old_contributes = old_destination != token_src &&
            !((old_destination < 64 ? old_duplicate_low : old_duplicate_high) &
              old_bit);
        const bool new_contributes = new_destination != token_src &&
            !((new_destination < 64 ? new_duplicate_low : new_duplicate_high) &
              new_bit);
        if (old_contributes != new_contributes)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        gains + current_expert * ranks + token_src),
                    new_contributes ? weight : 0ULL - weight);
      }
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

struct FastCandidate {
  int64_t expert;
  int target;
  int64_t amount;
  int64_t gain;
  bool is_new;
};

__global__ void materialize_fast_quota_kernel(
    const int64_t* demand, const int64_t* primary, const bool* replicas,
    const int64_t* addition_order, int64_t* move_plan, int64_t* quota,
    int64_t* routing, int64_t experts, int ranks);

__device__ bool better_fast_candidate(const FastCandidate& candidate,
                                      const FastCandidate& current) {
  if (current.expert < 0) return candidate.expert >= 0;
  if (candidate.expert < 0) return false;
  if (candidate.amount != current.amount)
    return candidate.amount > current.amount;
  if (candidate.is_new != current.is_new)
    return candidate.is_new < current.is_new;
  if (candidate.gain != current.gain) return candidate.gain > current.gain;
  if (candidate.expert != current.expert)
    return candidate.expert < current.expert;
  return candidate.target < current.target;
}

__device__ FastCandidate warp_best_fast_candidate(FastCandidate best) {
  const unsigned mask = __activemask();
  const int lane = threadIdx.x & 31;
  for (int offset = 16; offset; offset >>= 1) {
    FastCandidate other = {
        __shfl_down_sync(mask, best.expert, offset),
        __shfl_down_sync(mask, best.target, offset),
        __shfl_down_sync(mask, best.amount, offset),
        __shfl_down_sync(mask, best.gain, offset),
        static_cast<bool>(
            __shfl_down_sync(mask, static_cast<int>(best.is_new), offset))};
    if (lane + offset < 32 && better_fast_candidate(other, best)) best = other;
  }
  return best;
}

// The first refresh has to scan the trace once to build current gains. Keep
// that scan and the single-CTA capacity solve in one launch; later refreshes
// use the incremental incidence index below instead.
template <int FixedK>
__global__ void fused_current_gain_aggregate_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, bool* replicas, int64_t* gains,
    const int64_t* demand, int64_t* instance, int64_t* loads,
    int64_t* added_by_rank, int64_t* addition_order, int64_t* move_plan,
    int64_t* added_out, int64_t experts, int ranks, int64_t tokens,
    int64_t runtime_k, int64_t max_extra_per_rank, double imbalance_limit) {
  const int64_t actual_k = FixedK ? FixedK : runtime_k;
  const int tid = threadIdx.x;
  const int64_t matrix_size = experts * ranks;
  __shared__ FastCandidate warp_candidates[8];
  __shared__ FastCandidate selected_candidates[8];
  __shared__ int64_t warp_totals[8];
  __shared__ int64_t capacity;
  __shared__ int64_t added;
  __shared__ int over;
  __shared__ int stop;
  __shared__ int64_t shared_loads[kMaxEpSize];
  __shared__ int64_t shared_added_by_rank[kMaxEpSize];
  __shared__ int64_t shared_over_instance[256];
  __shared__ int active_experts[256];
  __shared__ int active_count;

  for (int64_t index = tid; index < matrix_size; index += blockDim.x)
    gains[index] = 0;
  __syncthreads();
  for (int64_t token = tid; token < tokens; token += blockDim.x) {
    const int64_t src = source[token];
    const auto weight = static_cast<unsigned long long>(count[token]);
    unsigned long long seen_low = 0;
    unsigned long long seen_high = 0;
    unsigned long long duplicate_low = 0;
    unsigned long long duplicate_high = 0;
    int64_t cached_experts[FixedK > 0 ? FixedK : 1];
    int cached_destinations[FixedK > 0 ? FixedK : 1];
#pragma unroll
    for (int64_t column = 0; column < actual_k; ++column) {
      const int64_t expert = topk[token * actual_k + column];
      const int destination = replicas[expert * ranks + src]
                                  ? static_cast<int>(src)
                                  : static_cast<int>(primary[expert]);
      if constexpr (FixedK > 0) {
        cached_experts[column] = expert;
        cached_destinations[column] = destination;
      }
      const auto bit = 1ULL << (destination & 63);
      auto& seen = destination < 64 ? seen_low : seen_high;
      auto& duplicate = destination < 64 ? duplicate_low : duplicate_high;
      if (seen & bit) duplicate |= bit;
      seen |= bit;
    }
#pragma unroll
    for (int64_t column = 0; column < actual_k; ++column) {
      const int64_t expert = FixedK ? cached_experts[column]
                                    : topk[token * actual_k + column];
      const int destination = FixedK
                                 ? cached_destinations[column]
                                 : (replicas[expert * ranks + src]
                                        ? static_cast<int>(src)
                                        : static_cast<int>(primary[expert]));
      const auto bit = 1ULL << (destination & 63);
      const auto duplicate = destination < 64 ? duplicate_low : duplicate_high;
      if (destination != src && !(duplicate & bit))
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      gains + expert * ranks + src), weight);
    }
  }
  __syncthreads();

  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    instance[index] = 0;
    addition_order[index] = 0;
    move_plan[index] = 0;
  }
  for (int rank = tid; rank < ranks; rank += blockDim.x) {
    shared_loads[rank] = 0;
    shared_added_by_rank[rank] = 0;
    loads[rank] = 0;
    added_by_rank[rank] = 0;
  }
  __syncthreads();
  int64_t local_total = 0;
  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int source_rank = index % ranks;
    const int target = replicas[index] ? source_rank : primary[expert];
    const int64_t value = demand[index];
    local_total += value;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  instance + expert * ranks + target),
              static_cast<unsigned long long>(value));
    atomicAdd(reinterpret_cast<unsigned long long*>(shared_loads + target),
              static_cast<unsigned long long>(value));
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int offset = 16; offset; offset >>= 1)
    local_total += __shfl_down_sync(0xffffffff, local_total, offset);
  if (lane == 0) warp_totals[warp] = local_total;
  __syncthreads();
  if (warp == 0) {
    local_total = lane < 8 ? warp_totals[lane] : 0;
    for (int offset = 16; offset; offset >>= 1)
      local_total += __shfl_down_sync(0xffffffff, local_total, offset);
  }
  if (tid == 0) {
    const double limit = imbalance_limit >= 1.0 ? imbalance_limit : 1.0;
    capacity = static_cast<int64_t>(
        ceil(static_cast<double>(local_total) / ranks * limit));
    added = 0;
  }
  __syncthreads();

  while (true) {
    if (tid == 0) {
      over = -1;
      for (int rank = 0; rank < ranks; ++rank)
        if (shared_loads[rank] > capacity &&
            (over < 0 || shared_loads[rank] > shared_loads[over]))
          over = rank;
      stop = over < 0;
    }
    __syncthreads();
    if (stop) break;
    if (experts <= 256)
      for (int64_t expert = tid; expert < experts; expert += blockDim.x)
        shared_over_instance[expert] = instance[expert * ranks + over];
    __syncthreads();
    FastCandidate best = {-1, -1, 0, 0, true};
    for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
      const int64_t expert = index / ranks;
      const int target = index % ranks;
      const int64_t available = experts <= 256
                                    ? shared_over_instance[expert]
                                    : instance[expert * ranks + over];
      const int64_t slack = capacity - shared_loads[target];
      const bool present = replicas[index] || addition_order[index];
      if (!available || target == over || slack <= 0 ||
          (!present && shared_added_by_rank[target] >= max_extra_per_rank))
        continue;
      const FastCandidate candidate = {
          expert, target, min(available, slack), gains[index], !present};
      if (better_fast_candidate(candidate, best)) best = candidate;
    }
    best = warp_best_fast_candidate(best);
    if (lane == 0) warp_candidates[warp] = best;
    __syncthreads();
    if (warp == 0 && lane == 0) {
      int selected_count = 0;
      for (int i = 0; i < 8; ++i) {
        const FastCandidate candidate = warp_candidates[i];
        if (candidate.expert < 0) continue;
        int pos;
        if (selected_count < 8)
          pos = selected_count++;
        else if (better_fast_candidate(candidate, selected_candidates[7]))
          pos = 7;
        else
          continue;
        selected_candidates[pos] = candidate;
        while (pos > 0 && better_fast_candidate(selected_candidates[pos],
                                                  selected_candidates[pos - 1])) {
          const FastCandidate tmp = selected_candidates[pos - 1];
          selected_candidates[pos - 1] = selected_candidates[pos];
          selected_candidates[pos] = tmp;
          --pos;
        }
      }
      int applied = 0;
      for (int i = 0; i < selected_count; ++i) {
        const FastCandidate candidate = selected_candidates[i];
        const int64_t index = candidate.expert * ranks + candidate.target;
        const int64_t available = instance[candidate.expert * ranks + over];
        const int64_t slack = capacity - shared_loads[candidate.target];
        const bool present = replicas[index] || addition_order[index];
        if (!available || candidate.target == over || slack <= 0 ||
            (!present && shared_added_by_rank[candidate.target] >=
                             max_extra_per_rank))
          continue;
        const int64_t amount = min(available, slack);
        if (!amount) continue;
        if (!present) {
          addition_order[index] = ++added;
          ++shared_added_by_rank[candidate.target];
        }
        instance[candidate.expert * ranks + over] -= amount;
        instance[index] += amount;
        move_plan[candidate.expert * ranks + over] += amount;
        move_plan[index] -= amount;
        shared_loads[over] -= amount;
        shared_loads[candidate.target] += amount;
        if (experts <= 256) shared_over_instance[candidate.expert] -= amount;
        ++applied;
        if (shared_loads[over] <= capacity) break;
      }
      stop = applied == 0;
    }
    __syncthreads();
    if (stop) break;
  }
  for (int64_t index = tid; index < matrix_size; index += blockDim.x)
    replicas[index] = replicas[index] || addition_order[index];
  for (int rank = tid; rank < ranks; rank += blockDim.x) {
    loads[rank] = shared_loads[rank];
    added_by_rank[rank] = shared_added_by_rank[rank];
  }
  if (tid == 0) *added_out = added;
}

__device__ bool better_capacity_candidate(
    const CapacityCandidate& candidate, const CapacityCandidate& current,
    const int64_t* loads) {
  if (current.expert < 0) return candidate.expert >= 0;
  if (candidate.expert < 0) return false;
  if (loads[candidate.over] != loads[current.over])
    return loads[candidate.over] > loads[current.over];
  if (candidate.penalty != current.penalty)
    return candidate.penalty < current.penalty;
  if (candidate.gain != current.gain) return candidate.gain > current.gain;
  if (candidate.potential != current.potential)
    return candidate.potential > current.potential;
  if (candidate.is_new != current.is_new)
    return candidate.is_new < current.is_new;
  if (candidate.amount != current.amount) return candidate.amount > current.amount;
  if (candidate.source != current.source) return candidate.source < current.source;
  if (candidate.expert != current.expert) return candidate.expert < current.expert;
  if (candidate.target != current.target) return candidate.target < current.target;
  return candidate.over < current.over;
}

template <int FixedExperts, int FixedRanks>
__global__ void capacity_v2_kernel(
    const int64_t* demand, const int64_t* bundle_gain,
    const int64_t* bundle_cover,
    const int64_t* primary, bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* quota, int64_t* routing,
    int64_t* added_out, int64_t runtime_experts, int runtime_ranks,
    int64_t max_extra_per_rank, double imbalance_limit) {
  if (blockIdx.x) return;
  const int64_t experts = FixedExperts ? FixedExperts : runtime_experts;
  const int ranks = FixedRanks ? FixedRanks : runtime_ranks;
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
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
      if (tid < offset &&
          better_capacity_candidate(candidates[tid + offset], candidates[tid],
                                    loads))
        candidates[tid] = candidates[tid + offset];
      __syncthreads();
    }
    if (tid == 0) {
      best = candidates[0];
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

__global__ void aggregate_capacity_single_kernel(
    const int64_t* demand, const int64_t* bundle_gain,
    const int64_t* primary, bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* move_plan, int64_t* added_out,
    int64_t experts, int ranks,
    int64_t max_extra_per_rank, double imbalance_limit) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  const int64_t matrix_size = experts * ranks;
  __shared__ FastCandidate warp_candidates[8];
  __shared__ FastCandidate selected_candidates[8];
  __shared__ int64_t warp_totals[8];
  __shared__ int64_t capacity;
  __shared__ int64_t added;
  __shared__ int over;
  __shared__ int stop;
  __shared__ int64_t shared_loads[kMaxEpSize];
  __shared__ int64_t shared_added_by_rank[kMaxEpSize];
  __shared__ int64_t shared_over_instance[256];

  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    instance[index] = 0;
    addition_order[index] = 0;
    move_plan[index] = 0;
  }
  for (int rank = tid; rank < ranks; rank += blockDim.x) {
    shared_loads[rank] = 0;
    shared_added_by_rank[rank] = 0;
  }
  __syncthreads();

  int64_t local_total = 0;
  for (int64_t index = tid; index < matrix_size; index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int source = index % ranks;
    const int target = replicas[index] ? source : primary[expert];
    const int64_t value = demand[index];
    local_total += value;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  instance + expert * ranks + target),
              static_cast<unsigned long long>(value));
    atomicAdd(reinterpret_cast<unsigned long long*>(shared_loads + target),
              static_cast<unsigned long long>(value));
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int offset = 16; offset; offset >>= 1)
    local_total += __shfl_down_sync(0xffffffff, local_total, offset);
  if (lane == 0) warp_totals[warp] = local_total;
  __syncthreads();
  if (warp == 0) {
    local_total = lane < 8 ? warp_totals[lane] : 0;
    for (int offset = 16; offset; offset >>= 1)
      local_total += __shfl_down_sync(0xffffffff, local_total, offset);
  }
  if (tid == 0) {
    const double limit = imbalance_limit >= 1.0 ? imbalance_limit : 1.0;
    capacity = static_cast<int64_t>(
        ceil(static_cast<double>(local_total) / ranks * limit));
    added = 0;
  }
  __syncthreads();

  while (true) {
    if (tid == 0) {
      over = -1;
      for (int rank = 0; rank < ranks; ++rank)
        if (shared_loads[rank] > capacity &&
            (over < 0 || shared_loads[rank] > shared_loads[over]))
          over = rank;
      stop = over < 0;
    }
    __syncthreads();
    if (stop) break;

    if (experts <= 256) {
      for (int64_t expert = tid; expert < experts; expert += blockDim.x)
        shared_over_instance[expert] = instance[expert * ranks + over];
      if (tid == 0) active_count = 0;
    }
    __syncthreads();
    if (experts <= 256) {
      for (int64_t expert = tid; expert < experts; expert += blockDim.x) {
        if (shared_over_instance[expert] > 0)
          active_experts[atomicAdd(&active_count, 1)] = static_cast<int>(expert);
      }
    }
    __syncthreads();

    // Each warp returns its best candidate. Applying the top candidates in
    // one synchronized batch preserves the same capacity/slot checks while
    // avoiding one full CTA reduction and barrier per individual move.
    FastCandidate best = {-1, -1, 0, 0, true};
    const int64_t active_matrix = experts <= 256
                                      ? static_cast<int64_t>(active_count) * ranks
                                      : matrix_size;
    for (int64_t index = tid; index < active_matrix; index += blockDim.x) {
      const int64_t expert = experts <= 256 ? active_experts[index / ranks]
                                           : index / ranks;
      const int target = index % ranks;
      const int64_t pair_index = expert * ranks + target;
      const int64_t available = experts <= 256
                                    ? shared_over_instance[expert]
                                    : instance[expert * ranks + over];
      const int64_t slack = capacity - shared_loads[target];
      const bool present = replicas[pair_index] || addition_order[pair_index];
      if (!available || target == over || slack <= 0 ||
          (!present &&
           shared_added_by_rank[target] >= max_extra_per_rank))
        continue;
      FastCandidate candidate = {
          expert, target, min(available, slack), bundle_gain[index], !present};
      if (better_fast_candidate(candidate, best)) best = candidate;
    }
    best = warp_best_fast_candidate(best);
    if (lane == 0) warp_candidates[warp] = best;
    __syncthreads();
    if (warp == 0 && lane == 0) {
      int selected_count = 0;
      for (int i = 0; i < 8; ++i) {
        const FastCandidate candidate = warp_candidates[i];
        if (candidate.expert < 0) continue;
        int pos;
        if (selected_count < 8) {
          pos = selected_count++;
        } else if (better_fast_candidate(candidate, selected_candidates[7])) {
          pos = 7;
        } else {
          continue;
        }
        selected_candidates[pos] = candidate;
        while (pos > 0 &&
               better_fast_candidate(selected_candidates[pos],
                                     selected_candidates[pos - 1])) {
          const FastCandidate tmp = selected_candidates[pos - 1];
          selected_candidates[pos - 1] = selected_candidates[pos];
          selected_candidates[pos] = tmp;
          --pos;
        }
      }

      int applied = 0;
      for (int i = 0; i < selected_count; ++i) {
        const FastCandidate candidate = selected_candidates[i];
        const int64_t index = candidate.expert * ranks + candidate.target;
        const int64_t available = instance[candidate.expert * ranks + over];
        const int64_t slack = capacity - shared_loads[candidate.target];
        const bool present = replicas[index] || addition_order[index];
        if (!available || candidate.target == over || slack <= 0 ||
            (!present && shared_added_by_rank[candidate.target] >=
                             max_extra_per_rank))
          continue;
        const int64_t amount = min(available, slack);
        if (!amount) continue;
        if (!present) {
          addition_order[index] = ++added;
          ++shared_added_by_rank[candidate.target];
        }
        instance[candidate.expert * ranks + over] -= amount;
        instance[index] += amount;
        move_plan[candidate.expert * ranks + over] += amount;
        move_plan[index] -= amount;
        shared_loads[over] -= amount;
        shared_loads[candidate.target] += amount;
        if (experts <= 256) shared_over_instance[candidate.expert] -= amount;
        ++applied;
        if (shared_loads[over] <= capacity) break;
      }
      stop = applied == 0;
    }
    __syncthreads();
    if (stop) break;
  }

  for (int64_t index = tid; index < matrix_size; index += blockDim.x)
    replicas[index] = replicas[index] || addition_order[index];
  for (int rank = tid; rank < ranks; rank += blockDim.x) {
    loads[rank] = shared_loads[rank];
    added_by_rank[rank] = shared_added_by_rank[rank];
  }
  if (tid == 0) *added_out = added;
}

// The multi-CTA path scores disjoint portions of E x EP and uses grid
// synchronization to separate scoring from the deterministic move commit.
__global__ void aggregate_capacity_cooperative_kernel(
    const int64_t* demand, const int64_t* bundle_gain,
    const int64_t* primary, bool* replicas, int64_t* instance,
    int64_t* loads, int64_t* added_by_rank, int64_t* addition_order,
    int64_t* move_plan, int64_t* added_out, int64_t* candidate_workspace,
    int64_t experts, int ranks, int64_t max_extra_per_rank,
    double imbalance_limit) {
  __shared__ FastCandidate selected_candidates[8];
  cg::grid_group grid = cg::this_grid();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t global_tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const int64_t global_stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t matrix_size = experts * ranks;
  const int total_warps = gridDim.x * (blockDim.x / 32);
  int64_t* state = candidate_workspace + total_warps * 5;
  __shared__ int64_t shared_loads[kMaxEpSize];
  __shared__ int64_t shared_added_by_rank[kMaxEpSize];
  __shared__ int64_t shared_over_instance[256];
  __shared__ int active_experts[256];
  __shared__ int active_count;
  constexpr int kCapacity = 0;
  constexpr int kAdded = 1;
  constexpr int kOver = 2;
  constexpr int kStop = 3;

  for (int64_t index = global_tid; index < matrix_size;
       index += global_stride) {
    instance[index] = 0;
    addition_order[index] = 0;
    move_plan[index] = 0;
  }
  for (int rank = global_tid; rank < ranks; rank += global_stride) {
    loads[rank] = 0;
    added_by_rank[rank] = 0;
  }
  if (global_tid == 0) {
    state[kCapacity] = 0;
    state[kAdded] = 0;
    state[kStop] = 0;
  }
  grid.sync();

  int64_t local_total = 0;
  for (int64_t index = global_tid; index < matrix_size;
       index += global_stride) {
    const int64_t expert = index / ranks;
    const int source = index % ranks;
    const int target = replicas[index] ? source : primary[expert];
    const int64_t value = demand[index];
    local_total += value;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  instance + expert * ranks + target),
              static_cast<unsigned long long>(value));
    atomicAdd(reinterpret_cast<unsigned long long*>(loads + target),
              static_cast<unsigned long long>(value));
  }
  if (local_total)
    atomicAdd(reinterpret_cast<unsigned long long*>(state + kCapacity),
              static_cast<unsigned long long>(local_total));
  grid.sync();
  if (global_tid == 0) {
    const double limit = imbalance_limit >= 1.0 ? imbalance_limit : 1.0;
    state[kCapacity] = static_cast<int64_t>(
        ceil(static_cast<double>(state[kCapacity]) / ranks * limit));
  }
  grid.sync();

  while (true) {
    if (global_tid == 0) {
      int over = -1;
      for (int rank = 0; rank < ranks; ++rank)
        if (loads[rank] > state[kCapacity] &&
            (over < 0 || loads[rank] > loads[over]))
          over = rank;
      state[kOver] = over;
      state[kStop] = over < 0;
    }
    grid.sync();
    if (state[kStop]) break;

    const int over = static_cast<int>(state[kOver]);
    for (int rank = tid; rank < ranks; rank += blockDim.x) {
      shared_loads[rank] = loads[rank];
      shared_added_by_rank[rank] = added_by_rank[rank];
    }
    for (int64_t expert = tid; expert < experts && expert < 256;
         expert += blockDim.x)
      shared_over_instance[expert] = instance[expert * ranks + over];
    if (experts <= 256 && tid == 0) active_count = 0;
    __syncthreads();
    if (experts <= 256) {
      for (int64_t expert = tid; expert < experts; expert += blockDim.x) {
        if (shared_over_instance[expert] > 0)
          active_experts[atomicAdd(&active_count, 1)] = static_cast<int>(expert);
      }
    }
    __syncthreads();
    FastCandidate best = {-1, -1, 0, 0, true};
    const int64_t active_matrix = experts <= 256
                                      ? static_cast<int64_t>(active_count) * ranks
                                      : matrix_size;
    for (int64_t index = global_tid; index < active_matrix;
         index += global_stride) {
      const int64_t expert = experts <= 256 ? active_experts[index / ranks]
                                           : index / ranks;
      const int target = index % ranks;
      const int64_t pair_index = expert * ranks + target;
      const int64_t available = experts <= 256
                                    ? shared_over_instance[expert]
                                    : instance[expert * ranks + over];
      const int64_t slack = state[kCapacity] - shared_loads[target];
      const bool present = replicas[pair_index] || addition_order[pair_index];
      if (!available || target == over || slack <= 0 ||
          (!present &&
           shared_added_by_rank[target] >= max_extra_per_rank))
        continue;
      FastCandidate candidate = {
          expert, target, min(available, slack), bundle_gain[index], !present};
      if (better_fast_candidate(candidate, best)) best = candidate;
    }
    best = warp_best_fast_candidate(best);
    if (lane == 0) {
      int64_t* output =
          candidate_workspace + (blockIdx.x * (blockDim.x / 32) + warp) * 5;
      output[0] = best.expert;
      output[1] = best.target;
      output[2] = best.amount;
      output[3] = best.gain;
      output[4] = best.is_new;
    }
    grid.sync();

    if (global_tid == 0) {
      int selected_count = 0;
      for (int i = 0; i < total_warps; ++i) {
        const int64_t* input = candidate_workspace + i * 5;
        FastCandidate candidate = {
            input[0], static_cast<int>(input[1]), input[2], input[3],
            static_cast<bool>(input[4])};
        if (candidate.expert < 0) continue;
        int position;
        if (selected_count < 8) {
          position = selected_count++;
        } else if (better_fast_candidate(candidate, selected_candidates[7])) {
          position = 7;
        } else {
          continue;
        }
        selected_candidates[position] = candidate;
        while (position > 0 && better_fast_candidate(
                                   selected_candidates[position],
                                   selected_candidates[position - 1])) {
          const FastCandidate tmp = selected_candidates[position - 1];
          selected_candidates[position - 1] = selected_candidates[position];
          selected_candidates[position] = tmp;
          --position;
        }
      }

      int applied = 0;
      for (int i = 0; i < selected_count; ++i) {
        const FastCandidate candidate = selected_candidates[i];
        const int64_t index = candidate.expert * ranks + candidate.target;
        const int64_t available = instance[candidate.expert * ranks + over];
        const int64_t slack = state[kCapacity] - loads[candidate.target];
        const bool present = replicas[index] || addition_order[index];
        if (!available || candidate.target == over || slack <= 0 ||
            (!present && added_by_rank[candidate.target] >=
                             max_extra_per_rank))
          continue;
        const int64_t amount = min(available, slack);
        if (!amount) continue;
        if (!present) {
          addition_order[index] = ++state[kAdded];
          ++added_by_rank[candidate.target];
        }
        instance[candidate.expert * ranks + over] -= amount;
        instance[index] += amount;
        move_plan[candidate.expert * ranks + over] += amount;
        move_plan[index] -= amount;
        loads[over] -= amount;
        loads[candidate.target] += amount;
        ++applied;
        if (loads[over] <= state[kCapacity]) break;
      }
      state[kStop] = applied == 0;
    }
    grid.sync();
    if (state[kStop]) break;
  }

  for (int64_t index = global_tid; index < matrix_size;
       index += global_stride)
    replicas[index] = replicas[index] || addition_order[index];
  if (global_tid == 0) *added_out = state[kAdded];
}

__global__ void materialize_fast_quota_kernel(
    const int64_t* demand, const int64_t* primary, const bool* replicas,
    const int64_t* addition_order, int64_t* move_plan, int64_t* quota,
    int64_t* routing,
    int64_t experts, int ranks) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  const int64_t rows = experts * ranks;
  for (int64_t index = tid; index < rows * ranks; index += blockDim.x)
    quota[index] = 0;
  for (int64_t row = tid; row < rows; row += blockDim.x) {
    const int source = row / experts;
    const int64_t expert = row % experts;
    const int original = replicas[expert * ranks + source] &&
                                 !addition_order[expert * ranks + source]
                             ? source
                             : primary[expert];
    quota[row * ranks + original] = demand[expert * ranks + source];
  }
  __syncthreads();

  for (int64_t expert = tid; expert < experts; expert += blockDim.x)
    for (int target = 0; target < ranks; ++target) {
        int64_t needed = -move_plan[expert * ranks + target];
        if (needed <= 0) continue;
        for (int donor = 0; donor < ranks && needed > 0; ++donor) {
          int64_t transfer = min(needed, move_plan[expert * ranks + donor]);
          if (transfer <= 0) continue;
          for (int source = 0; source < ranks && transfer > 0; ++source) {
            const int64_t row = source * experts + expert;
            const int64_t available = quota[row * ranks + donor];
            const int64_t amount = min(transfer, available);
            quota[row * ranks + donor] -= amount;
            quota[row * ranks + target] += amount;
            transfer -= amount;
            needed -= amount;
            move_plan[expert * ranks + donor] -= amount;
          }
        }
    }
  __syncthreads();

  for (int64_t row = tid; row < rows; row += blockDim.x) {
    int best = 0;
    for (int rank = 1; rank < ranks; ++rank)
      if (quota[row * ranks + rank] > quota[row * ranks + best]) best = rank;
    routing[row] = best;
  }
}

__global__ void materialize_fast_sparse_quota_kernel(
    const int64_t* demand, const int64_t* primary, const bool* replicas,
    const int64_t* addition_order, int64_t* prefix, int64_t* targets,
    int64_t* routing, int64_t experts, int ranks) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows = experts * ranks;
  if (row >= rows) return;
  const int source = row / experts;
  const int64_t expert = row % experts;
  const int64_t offset = row * ranks;
  int ordered[64];
  int ordered_count = 0;
  if (replicas[expert * ranks + source]) ordered[ordered_count++] = source;
  const int primary_rank = primary[expert];
  if (primary_rank != source) ordered[ordered_count++] = primary_rank;
  for (int rank = 0; rank < ranks; ++rank) {
    if (rank != source && rank != primary_rank &&
        replicas[expert * ranks + rank] && !addition_order[expert * ranks + rank])
      ordered[ordered_count++] = rank;
  }
  int64_t last_order = 0;
  while (ordered_count < ranks) {
    int next_rank = -1;
    int64_t next_order = LLONG_MAX;
    for (int rank = 0; rank < ranks; ++rank) {
      const int64_t order = addition_order[expert * ranks + rank];
      if (rank != source && order > last_order && order < next_order) {
        next_rank = rank;
        next_order = order;
      }
    }
    if (next_rank < 0) break;
    ordered[ordered_count++] = next_rank;
    last_order = next_order;
  }

  int best_target = 0;
  int64_t best_amount = prefix[offset];
  for (int rank = 1; rank < ranks; ++rank) {
    if (prefix[offset + rank] > best_amount) {
      best_amount = prefix[offset + rank];
      best_target = rank;
    }
  }
  int sparse_count = 0;
  int64_t running = 0;
  for (int position = 0; position < ordered_count; ++position) {
    const int target = ordered[position];
    const int64_t amount = prefix[offset + target];
    if (amount > 0) {
      running += amount;
      targets[offset + sparse_count] = target;
      prefix[offset + sparse_count] = running;
      ++sparse_count;
    }
  }
  for (int position = sparse_count; position < ranks; ++position) {
    prefix[offset + position] = 0;
    targets[offset + position] = 0;
  }
  routing[row] = best_target;
}

__device__ void add_csr_amount(const int32_t* offsets, int64_t* amounts,
                               const int32_t* targets, int64_t row,
                               int target, int64_t amount) {
  if (!amount) return;
  for (int64_t position = offsets[row]; position < offsets[row + 1];
       ++position) {
    if (targets[position] == target) {
      amounts[position] += amount;
      return;
    }
  }
}

__global__ void layout_fast_csr_quota_kernel(
    const int64_t* primary, const bool* replicas,
    const int64_t* addition_order, int64_t* replica_counts, int32_t* offsets,
    int64_t* amounts, int32_t* targets, int64_t experts, int ranks,
    int64_t capacity) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  const int64_t rows = experts * ranks;
  for (int64_t expert = tid; expert < experts; expert += blockDim.x) {
    int64_t count = 0;
    for (int rank = 0; rank < ranks; ++rank)
      count += replicas[expert * ranks + rank];
    replica_counts[expert] = count;
  }
  __syncthreads();
  if (tid == 0) {
    int64_t cursor = 0;
    for (int64_t row = 0; row < rows; ++row) {
      offsets[row] = static_cast<int32_t>(cursor);
      cursor += replica_counts[row % experts];
    }
    offsets[rows] = static_cast<int32_t>(cursor);
  }
  __syncthreads();
  if (offsets[rows] > capacity) return;

  for (int64_t row = tid; row < rows; row += blockDim.x) {
    const int source = row / experts;
    const int64_t expert = row % experts;
    int32_t position = offsets[row];
    if (replicas[expert * ranks + source]) targets[position++] = source;
    const int primary_rank = primary[expert];
    if (primary_rank != source) targets[position++] = primary_rank;
    for (int rank = 0; rank < ranks; ++rank) {
      if (rank != source && rank != primary_rank &&
          replicas[expert * ranks + rank] &&
          !addition_order[expert * ranks + rank])
        targets[position++] = rank;
    }
    int64_t last_order = 0;
    while (position < offsets[row + 1]) {
      int next_rank = -1;
      int64_t next_order = LLONG_MAX;
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t order = addition_order[expert * ranks + rank];
        if (rank != source && order > last_order && order < next_order) {
          next_rank = rank;
          next_order = order;
        }
      }
      if (next_rank < 0) break;
      targets[position++] = next_rank;
      last_order = next_order;
    }
    for (int64_t index = offsets[row]; index < offsets[row + 1]; ++index)
      amounts[index] = 0;
  }
}

__global__ void fill_fast_csr_quota_kernel(
    const int64_t* demand, const int64_t* primary, const bool* replicas,
    const int64_t* addition_order, int64_t* move_plan, int64_t* remaining,
    const int32_t* offsets, int64_t* amounts, const int32_t* targets,
    int64_t experts, int ranks, int64_t capacity) {
  const int64_t expert =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (expert >= experts || offsets[experts * ranks] > capacity) return;

  for (int source = 0; source < ranks; ++source)
    remaining[expert * ranks + source] = demand[expert * ranks + source];

  for (int target = 0; target < ranks; ++target) {
    int64_t needed = -move_plan[expert * ranks + target];
    if (needed <= 0) continue;
    for (int donor = 0; donor < ranks && needed > 0; ++donor) {
      int64_t transfer =
          min(needed, move_plan[expert * ranks + donor]);
      if (transfer <= 0) continue;
      for (int source = 0; source < ranks && transfer > 0; ++source) {
        const int original =
            replicas[expert * ranks + source] &&
                    !addition_order[expert * ranks + source]
                ? source
                : primary[expert];
        if (original != donor) continue;
        const int64_t index = expert * ranks + source;
        const int64_t amount = min(transfer, remaining[index]);
        if (!amount) continue;
        add_csr_amount(offsets, amounts, targets, source * experts + expert,
                       target, amount);
        remaining[index] -= amount;
        transfer -= amount;
        needed -= amount;
        move_plan[expert * ranks + donor] -= amount;
      }
    }
  }

  for (int source = 0; source < ranks; ++source) {
    const int original = replicas[expert * ranks + source] &&
                                 !addition_order[expert * ranks + source]
                             ? source
                             : primary[expert];
    const int64_t index = expert * ranks + source;
    add_csr_amount(offsets, amounts, targets, source * experts + expert,
                   original, remaining[index]);
  }
}

__global__ void finalize_fast_csr_quota_kernel(
    const int32_t* offsets, int64_t* boundaries, const int32_t* targets,
    int64_t* routing, int64_t rows, int64_t capacity) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || offsets[rows] > capacity) return;
  int best_target = 0;
  int64_t best_amount = 0;
  int64_t running = 0;
  for (int64_t position = offsets[row]; position < offsets[row + 1];
       ++position) {
    const int target = targets[position];
    const int64_t amount = boundaries[position];
    if (amount > best_amount ||
        (amount == best_amount && amount > 0 && target < best_target)) {
      best_amount = amount;
      best_target = target;
    }
    running += amount;
    boundaries[position] = running;
  }
  routing[row] = best_target;
}

void launch_aggregate_capacity(
    const int64_t* demand, const int64_t* bundle_gain,
    const int64_t* primary, bool* replicas, int64_t* instance,
    int64_t* loads, int64_t* added_by_rank, int64_t* addition_order,
    int64_t* move_plan, int64_t* added_out, int64_t* candidate_workspace,
    int64_t experts, int ranks, int64_t max_extra_per_rank,
    double imbalance_limit, int solver_ctas, int device, cudaStream_t stream) {
  const auto launch_single = [&](auto kernel) {
    launch(kernel, dim3(1), dim3(256), stream, demand, bundle_gain, primary,
           replicas, instance, loads, added_by_rank, addition_order, move_plan,
           added_out, experts, ranks, max_extra_per_rank, imbalance_limit);
  };
  // A single CTA is a supported execution mode for every EP size. Keep this
  // branch explicit so callers can opt out of cooperative launch without
  // paying device capability/occupancy queries or requiring grid sync.
  if (solver_ctas == 1) {
    launch_single(aggregate_capacity_single_kernel);
    return;
  }

  int cooperative = 0;
  check_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                    device));
  if (cooperative && solver_ctas > 1) {
    int blocks_per_sm = 0;
    int multiprocessors = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, aggregate_capacity_cooperative_kernel, 256, 0));
    check_cuda(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int resident_ctas =
        std::min(solver_ctas, blocks_per_sm * multiprocessors);
    if (resident_ctas > 1) {
      launch_cooperative(
          aggregate_capacity_cooperative_kernel, dim3(resident_ctas),
          dim3(256), stream, demand, bundle_gain, primary, replicas, instance,
          loads, added_by_rank, addition_order, move_plan, added_out,
          candidate_workspace, experts, ranks, max_extra_per_rank,
          imbalance_limit);
      return;
    }
  }
  launch_single(aggregate_capacity_single_kernel);
}

}  // namespace

void current_bundle_gains_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    torch::Tensor covers, int64_t solver_sms) {
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
  TORCH_CHECK(gains.size(1) > 0 && gains.size(1) <= kMaxEpSize,
              "bundle gain supports at most 64 ranks");
  TORCH_CHECK(solver_sms > 0, "solver_sms must be positive");
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  if (solver_sms > 1) {
    check_cuda(cudaMemsetAsync(gains.data_ptr<int64_t>(), 0,
                               gains.numel() * sizeof(int64_t), stream.stream()));
    check_cuda(cudaMemsetAsync(covers.data_ptr<int64_t>(), 0,
                               covers.numel() * sizeof(int64_t), stream.stream()));
  }
  launch(current_bundle_gain_kernel, dim3(solver_sms), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), gains.data_ptr<int64_t>(),
         covers.data_ptr<int64_t>(), gains.size(0), source.size(0),
         topk.size(1), gains.size(1), solver_sms == 1);
  check_cuda(cudaGetLastError());
}

void current_bundle_gains_fast_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    int64_t solver_sms) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda() && gains.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              gains.scalar_type() == torch::kInt64 && solver_sms > 0);
  const int64_t experts = primary.numel();
  const int64_t ranks = gains.size(1);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(gains.dim() == 2 && gains.size(0) == experts &&
              replicas.sizes() == gains.sizes() && ranks > 0 &&
              ranks <= kMaxEpSize);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  if (solver_sms > 1)
    check_cuda(cudaMemsetAsync(gains.data_ptr<int64_t>(), 0,
                               gains.numel() * sizeof(int64_t), stream.stream()));
  const dim3 blocks(solver_sms);
  const dim3 threads(256);
  const auto launch_args = [&](auto kernel) {
    launch(kernel, blocks, threads, stream.stream(), source.data_ptr<int64_t>(),
           topk.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           gains.data_ptr<int64_t>(), experts, source.numel(), topk.size(1),
           ranks, solver_sms == 1);
  };
  switch (topk.size(1)) {
    case 1:
      launch_args(current_bundle_gain_fast_kernel<1>);
      break;
    case 2:
      launch_args(current_bundle_gain_fast_kernel<2>);
      break;
    case 4:
      launch_args(current_bundle_gain_fast_kernel<4>);
      break;
    case 8:
      launch_args(current_bundle_gain_fast_kernel<8>);
      break;
    case 16:
      launch_args(current_bundle_gain_fast_kernel<16>);
      break;
    default:
      launch_args(current_bundle_gain_fast_kernel<0>);
      break;
  }
  check_cuda(cudaGetLastError());
}

void current_bundle_gains_and_select_compute_replicas_fast_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    torch::Tensor demand, int64_t max_extra_per_rank, double imbalance_limit,
    torch::Tensor instance, torch::Tensor loads, torch::Tensor added_by_rank,
    torch::Tensor addition_order, torch::Tensor move_plan, torch::Tensor quota,
    torch::Tensor routing, torch::Tensor added) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda() && gains.is_cuda() &&
              demand.is_cuda() && instance.is_cuda() && loads.is_cuda() &&
              added_by_rank.is_cuda() && addition_order.is_cuda() &&
              move_plan.is_cuda() && quota.is_cuda() && routing.is_cuda() &&
              added.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              gains.scalar_type() == torch::kInt64 &&
              demand.scalar_type() == torch::kInt64);
  const int64_t experts = primary.numel();
  const int ranks = gains.size(1);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0) &&
              demand.sizes() == gains.sizes() && replicas.sizes() == gains.sizes() &&
              instance.sizes() == demand.sizes() && loads.numel() == ranks &&
              added_by_rank.numel() == ranks && addition_order.sizes() == demand.sizes() &&
              move_plan.sizes() == demand.sizes() && quota.dim() == 3 &&
              quota.size(0) == ranks && quota.size(1) == experts &&
              quota.size(2) == ranks && routing.dim() == 2 &&
              routing.size(0) == ranks && routing.size(1) == experts &&
              added.numel() == 1 && ranks > 0 && ranks <= kMaxEpSize &&
              max_extra_per_rank >= 0);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const dim3 blocks(1);
  const dim3 threads(256);
  const auto launch_args = [&](auto kernel) {
    launch(kernel, blocks, threads, stream.stream(), source.data_ptr<int64_t>(),
           topk.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           gains.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           move_plan.data_ptr<int64_t>(), added.data_ptr<int64_t>(), experts,
           ranks, source.numel(), topk.size(1), max_extra_per_rank,
           imbalance_limit);
  };
  switch (topk.size(1)) {
    case 1:
      launch_args(fused_current_gain_aggregate_kernel<1>);
      break;
    case 2:
      launch_args(fused_current_gain_aggregate_kernel<2>);
      break;
    case 4:
      launch_args(fused_current_gain_aggregate_kernel<4>);
      break;
    case 8:
      launch_args(fused_current_gain_aggregate_kernel<8>);
      break;
    case 16:
      launch_args(fused_current_gain_aggregate_kernel<16>);
      break;
    default:
      launch_args(fused_current_gain_aggregate_kernel<0>);
      break;
  }
  launch(materialize_fast_quota_kernel, dim3(1), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), addition_order.data_ptr<int64_t>(),
         move_plan.data_ptr<int64_t>(), quota.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
}

void incremental_bundle_gains_fast_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    torch::Tensor bundle_heads, torch::Tensor bundle_next,
    torch::Tensor bundle_marks, int64_t epoch, int64_t solver_sms) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda() && gains.is_cuda() &&
              bundle_heads.is_cuda() && bundle_next.is_cuda() &&
              bundle_marks.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              gains.scalar_type() == torch::kInt64 &&
              bundle_heads.scalar_type() == torch::kInt32 &&
              bundle_next.scalar_type() == torch::kInt32 &&
              bundle_marks.scalar_type() == torch::kInt32);
  const int64_t experts = primary.numel();
  const int64_t ranks = gains.size(1);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(gains.dim() == 2 && gains.size(0) == experts &&
              replicas.sizes() == gains.sizes() &&
              bundle_heads.sizes() == gains.sizes() &&
              bundle_next.numel() >= topk.numel() &&
              bundle_marks.numel() >= source.numel() && ranks > 0 &&
              ranks <= kMaxEpSize && epoch > 0 && epoch <= INT_MAX &&
              solver_sms > 0);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const dim3 blocks(solver_sms);
  const dim3 threads(256);
  const auto launch_args = [&](auto kernel) {
    launch(kernel, blocks, threads, stream.stream(), source.data_ptr<int64_t>(),
           topk.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           gains.data_ptr<int64_t>(), bundle_heads.data_ptr<int32_t>(),
           bundle_next.data_ptr<int32_t>(), bundle_marks.data_ptr<int32_t>(),
           static_cast<int32_t>(epoch), experts, source.numel(), topk.size(1),
           ranks);
  };
  switch (topk.size(1)) {
    case 1:
      launch_args(incremental_bundle_gain_kernel<1>);
      break;
    case 2:
      launch_args(incremental_bundle_gain_kernel<2>);
      break;
    case 4:
      launch_args(incremental_bundle_gain_kernel<4>);
      break;
    case 8:
      launch_args(incremental_bundle_gain_kernel<8>);
      break;
    case 16:
      launch_args(incremental_bundle_gain_kernel<16>);
      break;
    default:
      launch_args(incremental_bundle_gain_kernel<0>);
      break;
  }
  check_cuda(cudaGetLastError());
}

void incremental_bundle_gains_csr_fast_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, torch::Tensor replicas, torch::Tensor gains,
    torch::Tensor offsets, torch::Tensor incidence_entries,
    torch::Tensor bundle_marks, int64_t epoch, int64_t solver_sms) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && replicas.is_cuda() && gains.is_cuda() &&
              offsets.is_cuda() && incidence_entries.is_cuda() &&
              bundle_marks.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              gains.scalar_type() == torch::kInt64 &&
              offsets.scalar_type() == torch::kInt32 &&
              incidence_entries.scalar_type() == torch::kInt32 &&
              bundle_marks.scalar_type() == torch::kInt32);
  const int64_t experts = primary.numel();
  const int64_t ranks = gains.size(1);
  const int64_t rows = experts * ranks;
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0) &&
              gains.dim() == 2 && gains.size(0) == experts &&
              replicas.sizes() == gains.sizes() && offsets.numel() == rows + 1 &&
              incidence_entries.numel() >= topk.numel() &&
              bundle_marks.numel() >= source.numel() && ranks > 0 &&
              ranks <= kMaxEpSize && epoch > 0 && epoch <= INT_MAX &&
              solver_sms > 0 && topk.numel() <= INT_MAX);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const dim3 blocks(solver_sms);
  const dim3 threads(256);
  const auto launch_args = [&](auto kernel) {
    launch(kernel, blocks, threads, stream.stream(), source.data_ptr<int64_t>(),
           topk.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           gains.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
           incidence_entries.data_ptr<int32_t>(), bundle_marks.data_ptr<int32_t>(),
           static_cast<int32_t>(epoch), experts, source.numel(), topk.size(1),
           ranks);
  };
  switch (topk.size(1)) {
    case 1: launch_args(incremental_bundle_gain_csr_kernel<1>); break;
    case 2: launch_args(incremental_bundle_gain_csr_kernel<2>); break;
    case 4: launch_args(incremental_bundle_gain_csr_kernel<4>); break;
    case 8: launch_args(incremental_bundle_gain_csr_kernel<8>); break;
    case 16: launch_args(incremental_bundle_gain_csr_kernel<16>); break;
    default: launch_args(incremental_bundle_gain_csr_kernel<0>); break;
  }
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
  TORCH_CHECK(ranks > 0 && ranks <= kMaxEpSize && max_extra_per_rank >= 0);
  TORCH_CHECK(primary.numel() == experts && instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.numel() == ranks && added_by_rank.numel() == ranks &&
              addition_order.sizes() == demand.sizes() && added.numel() == 1);
  TORCH_CHECK(quota.dim() == 3 && quota.size(0) == ranks &&
              quota.size(1) == experts && quota.size(2) == ranks);
  TORCH_CHECK(bundle_cover.sizes() == quota.sizes());
  TORCH_CHECK(routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  const auto kernel = experts == 256 && ranks == 4
                          ? capacity_v2_kernel<256, 4>
                          : capacity_v2_kernel<0, 0>;
  launch(kernel, dim3(1), dim3(256), stream.stream(),
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

void select_compute_replicas_fast_into(
    torch::Tensor demand, torch::Tensor bundle_gain,
    torch::Tensor replicas, torch::Tensor primary,
    int64_t max_extra_per_rank, double imbalance_limit,
    torch::Tensor instance, torch::Tensor loads,
    torch::Tensor added_by_rank, torch::Tensor addition_order,
    torch::Tensor move_plan, torch::Tensor quota, torch::Tensor routing,
    torch::Tensor added, torch::Tensor candidate_workspace,
    int64_t solver_ctas) {
  TORCH_CHECK(demand.is_cuda() && bundle_gain.is_cuda() && replicas.is_cuda() &&
              primary.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              bundle_gain.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes() &&
              bundle_gain.sizes() == demand.sizes());
  const int64_t experts = demand.size(0);
  const int ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxEpSize && max_extra_per_rank >= 0);
  TORCH_CHECK(instance.scalar_type() == torch::kInt64 &&
              loads.scalar_type() == torch::kInt64 &&
              added_by_rank.scalar_type() == torch::kInt64 &&
              addition_order.scalar_type() == torch::kInt64 &&
              routing.scalar_type() == torch::kInt64 &&
              added.scalar_type() == torch::kInt64 && added.numel() == 1 &&
              primary.numel() == experts && instance.sizes() == demand.sizes() &&
              loads.numel() == ranks && added_by_rank.numel() == ranks &&
              addition_order.sizes() == demand.sizes() &&
              move_plan.sizes() == demand.sizes() &&
              quota.dim() == 3 && quota.size(0) == ranks &&
              quota.size(1) == experts && quota.size(2) == ranks &&
              routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts && added.numel() == 1 &&
              candidate_workspace.is_cuda() &&
              candidate_workspace.scalar_type() == torch::kInt64 &&
              solver_ctas > 0 && solver_ctas <= INT_MAX &&
              candidate_workspace.numel() >= solver_ctas * 8 * 5 + 4);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch_aggregate_capacity(
      demand.data_ptr<int64_t>(), bundle_gain.data_ptr<int64_t>(),
      primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
      instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
      added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
      move_plan.data_ptr<int64_t>(), added.data_ptr<int64_t>(),
      candidate_workspace.data_ptr<int64_t>(), experts, ranks,
      max_extra_per_rank, imbalance_limit, static_cast<int>(solver_ctas),
      demand.get_device(), stream.stream());
  launch(materialize_fast_quota_kernel, dim3(1), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), addition_order.data_ptr<int64_t>(),
         move_plan.data_ptr<int64_t>(),
         quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
}

void select_compute_replicas_fast_sparse_into(
    torch::Tensor demand, torch::Tensor bundle_gain, torch::Tensor replicas,
    torch::Tensor primary, int64_t max_extra_per_rank, double imbalance_limit,
    torch::Tensor instance, torch::Tensor loads, torch::Tensor added_by_rank,
    torch::Tensor addition_order, torch::Tensor move_plan,
    torch::Tensor added, torch::Tensor candidate_workspace,
    int64_t solver_ctas) {
  TORCH_CHECK(demand.is_cuda() && bundle_gain.is_cuda() && replicas.is_cuda() &&
              primary.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              bundle_gain.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes() &&
              bundle_gain.sizes() == demand.sizes());
  const int64_t experts = demand.size(0);
  const int ranks = demand.size(1);
  TORCH_CHECK(primary.numel() == experts && ranks > 0 && ranks <= kMaxEpSize &&
              max_extra_per_rank >= 0);
  TORCH_CHECK(instance.is_cuda() && instance.scalar_type() == torch::kInt64 &&
              instance.sizes() == demand.sizes() && loads.is_cuda() &&
              loads.scalar_type() == torch::kInt64 && loads.numel() == ranks &&
              added_by_rank.is_cuda() &&
              added_by_rank.scalar_type() == torch::kInt64 &&
              added_by_rank.numel() == ranks && addition_order.is_cuda() &&
              addition_order.scalar_type() == torch::kInt64 &&
              addition_order.sizes() == demand.sizes() && move_plan.is_cuda() &&
              move_plan.scalar_type() == torch::kInt64 &&
              move_plan.sizes() == demand.sizes() && added.is_cuda() &&
              added.scalar_type() == torch::kInt64 && added.numel() == 1 &&
              candidate_workspace.is_cuda() &&
              candidate_workspace.scalar_type() == torch::kInt64 &&
              solver_ctas > 0 && solver_ctas <= INT_MAX &&
              candidate_workspace.numel() >= solver_ctas * 8 * 5 + 4);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch_aggregate_capacity(
      demand.data_ptr<int64_t>(), bundle_gain.data_ptr<int64_t>(),
      primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
      instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
      added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
      move_plan.data_ptr<int64_t>(), added.data_ptr<int64_t>(),
      candidate_workspace.data_ptr<int64_t>(), experts, ranks,
      max_extra_per_rank, imbalance_limit, static_cast<int>(solver_ctas),
      demand.get_device(), stream.stream());
  check_cuda(cudaGetLastError());
}

void materialize_fast_sparse_quota_into(
    torch::Tensor demand, torch::Tensor primary, torch::Tensor replicas,
    torch::Tensor addition_order, torch::Tensor move_plan,
    torch::Tensor prefix, torch::Tensor targets, torch::Tensor routing) {
  TORCH_CHECK(demand.is_cuda() && primary.is_cuda() && replicas.is_cuda() &&
              addition_order.is_cuda() && move_plan.is_cuda() &&
              prefix.is_cuda() && targets.is_cuda() && routing.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              addition_order.scalar_type() == torch::kInt64 &&
              move_plan.scalar_type() == torch::kInt64 &&
              prefix.scalar_type() == torch::kInt64 &&
              targets.scalar_type() == torch::kInt64 &&
              routing.scalar_type() == torch::kInt64 && demand.dim() == 2 &&
              replicas.sizes() == demand.sizes() &&
              addition_order.sizes() == demand.sizes() &&
              move_plan.sizes() == demand.sizes());
  const int64_t experts = demand.size(0);
  const int ranks = demand.size(1);
  TORCH_CHECK(primary.numel() == experts && ranks > 0 && ranks <= kMaxEpSize &&
              prefix.dim() == 3 && prefix.size(0) == ranks &&
              prefix.size(1) == experts && prefix.size(2) == ranks &&
              targets.sizes() == prefix.sizes() && routing.dim() == 2 &&
              routing.size(0) == ranks && routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  // The prefix buffer is used as dense scratch while the move plan is applied;
  // the following kernel compacts each row in-place into cumulative boundaries.
  launch(materialize_fast_quota_kernel, dim3(1), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), addition_order.data_ptr<int64_t>(),
         move_plan.data_ptr<int64_t>(), prefix.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), experts, ranks);
  launch(materialize_fast_sparse_quota_kernel,
         dim3((experts * ranks + 255) / 256), dim3(256), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), addition_order.data_ptr<int64_t>(),
         prefix.data_ptr<int64_t>(), targets.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
}

void materialize_fast_csr_quota_into(
    torch::Tensor demand, torch::Tensor primary, torch::Tensor replicas,
    torch::Tensor addition_order, torch::Tensor move_plan,
    torch::Tensor remaining, torch::Tensor replica_counts,
    torch::Tensor offsets, torch::Tensor boundaries, torch::Tensor targets,
    torch::Tensor routing) {
  TORCH_CHECK(demand.is_cuda() && primary.is_cuda() && replicas.is_cuda() &&
              addition_order.is_cuda() && move_plan.is_cuda() &&
              remaining.is_cuda() && replica_counts.is_cuda() &&
              offsets.is_cuda() && boundaries.is_cuda() && targets.is_cuda() &&
              routing.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              addition_order.scalar_type() == torch::kInt64 &&
              move_plan.scalar_type() == torch::kInt64 &&
              remaining.scalar_type() == torch::kInt64 &&
              replica_counts.scalar_type() == torch::kInt64 &&
              offsets.scalar_type() == torch::kInt32 &&
              boundaries.scalar_type() == torch::kInt64 &&
              targets.scalar_type() == torch::kInt32 &&
              routing.scalar_type() == torch::kInt64);
  const int64_t experts = demand.size(0);
  const int ranks = demand.size(1);
  const int64_t rows = experts * ranks;
  TORCH_CHECK(demand.dim() == 2 && primary.numel() == experts && ranks > 0 &&
              ranks <= kMaxEpSize && replicas.sizes() == demand.sizes() &&
              addition_order.sizes() == demand.sizes() &&
              move_plan.sizes() == demand.sizes() &&
              remaining.sizes() == demand.sizes() &&
              replica_counts.numel() == experts && offsets.numel() == rows + 1 &&
              boundaries.dim() == 1 && targets.dim() == 1 &&
              targets.numel() == boundaries.numel() &&
              boundaries.numel() >= rows && boundaries.numel() <= INT_MAX &&
              routing.dim() == 2 &&
              routing.size(0) == ranks && routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  const int64_t capacity = boundaries.numel();
  launch(layout_fast_csr_quota_kernel, dim3(1), dim3(256), stream.stream(),
         primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         addition_order.data_ptr<int64_t>(),
         replica_counts.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
         boundaries.data_ptr<int64_t>(), targets.data_ptr<int32_t>(), experts,
         ranks, capacity);
  launch(fill_fast_csr_quota_kernel, dim3((experts + 255) / 256), dim3(256),
         stream.stream(), demand.data_ptr<int64_t>(),
         primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         addition_order.data_ptr<int64_t>(), move_plan.data_ptr<int64_t>(),
         remaining.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
         boundaries.data_ptr<int64_t>(), targets.data_ptr<int32_t>(), experts,
         ranks, capacity);
  launch(finalize_fast_csr_quota_kernel, dim3((rows + 255) / 256), dim3(256),
         stream.stream(), offsets.data_ptr<int32_t>(),
         boundaries.data_ptr<int64_t>(), targets.data_ptr<int32_t>(),
         routing.data_ptr<int64_t>(), rows, capacity);
  check_cuda(cudaGetLastError());
}

}  // namespace grace_cuda
