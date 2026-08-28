#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "launch.cuh"
#include "limits.cuh"

namespace grace_cuda {
namespace {

constexpr int kMaxRanks = kMaxEpSize;

template <int MaxRanks = kMaxRanks, int FixedRanks = 0>
__device__ void waterfill(const int64_t* loads, const bool* replicas,
                          int64_t expert, int extra_rank, int64_t total,
                          int64_t* allocation, int64_t ranks) {
  const int count = FixedRanks ? FixedRanks : static_cast<int>(ranks);
  int rank_order[MaxRanks];
  bool selected[MaxRanks];
  int replica_count = 0;
#pragma unroll
  for (int rank = 0; rank < count; ++rank) {
    allocation[rank] = 0;
    selected[rank] = false;
    replica_count += replicas[expert * ranks + rank] || rank == extra_rank;
  }
  for (int position = 0; position < replica_count; ++position) {
    int best = -1;
#pragma unroll
    for (int rank = 0; rank < count; ++rank) {
      if ((!replicas[expert * ranks + rank] && rank != extra_rank) || selected[rank]) {
        continue;
      }
      if (best < 0 || loads[rank] < loads[best] ||
          (loads[rank] == loads[best] && rank < best)) {
        best = rank;
      }
    }
    rank_order[position] = best;
    selected[best] = true;
  }
  int64_t remaining = total;
  int64_t level = loads[rank_order[0]];
  int width = 1;
  for (; width < replica_count; ++width) {
    const int64_t next_level = loads[rank_order[width]];
    const int64_t needed = (next_level - level) * width;
    if (remaining < needed) break;
    remaining -= needed;
    level = next_level;
  }
  const int64_t quotient = remaining / width;
  const int64_t remainder = remaining % width;
  for (int position = 0; position < width; ++position) {
    const int rank = rank_order[position];
    allocation[rank] = level - loads[rank] + quotient +
                       static_cast<int64_t>(position < remainder);
  }
}

template <int MaxRanks>
__device__ void waterfill_fast(const int64_t* loads, const bool* replicas,
                               int64_t expert, int extra_rank, int64_t total,
                               int64_t* allocation, int64_t ranks) {
  if constexpr (MaxRanks == 4) {
    if (ranks == 4) {
      waterfill<4, 4>(loads, replicas, expert, extra_rank, total, allocation,
                      ranks);
      return;
    }
  }
  waterfill<MaxRanks>(loads, replicas, expert, extra_rank, total, allocation,
                      ranks);
}

template <int MaxRanks>
__device__ void rebalance_compute_quota(
    int64_t* quota, const bool* replicas, int64_t* loads, int64_t experts,
    int ranks, int sources, int64_t capacity) {
  while (true) {
    int previous[MaxRanks];
    int edge_source[MaxRanks];
    int64_t edge_expert[MaxRanks];
    int queue[MaxRanks];
    int over = -1;
    int under = -1;
    for (int candidate_over = 0;
         candidate_over < ranks && under < 0; ++candidate_over) {
      if (loads[candidate_over] <= capacity) continue;
      for (int rank = 0; rank < ranks; ++rank) previous[rank] = -2;
      previous[candidate_over] = -1;
      int head = 0;
      int tail = 1;
      queue[0] = candidate_over;
      while (head < tail && under < 0) {
        const int from = queue[head++];
        for (int source = 0; source < sources && under < 0; ++source) {
          for (int64_t expert = 0; expert < experts && under < 0; ++expert) {
            if (!quota[(source * experts + expert) * ranks + from]) continue;
            for (int target = 0; target < ranks; ++target) {
              if (previous[target] != -2 ||
                  !replicas[expert * ranks + target]) {
                continue;
              }
              previous[target] = from;
              edge_source[target] = source;
              edge_expert[target] = expert;
              if (loads[target] < capacity) {
                over = candidate_over;
                under = target;
                break;
              }
              queue[tail++] = target;
            }
          }
        }
      }
    }
    if (under < 0) return;

    int64_t moved = min(loads[over] - capacity, capacity - loads[under]);
    for (int rank = under; previous[rank] >= 0; rank = previous[rank]) {
      moved = min(
          moved,
          quota[(edge_source[rank] * experts + edge_expert[rank]) * ranks +
                previous[rank]]);
    }
    for (int rank = under; previous[rank] >= 0; rank = previous[rank]) {
      const int64_t offset =
          (edge_source[rank] * experts + edge_expert[rank]) * ranks;
      quota[offset + previous[rank]] -= moved;
      quota[offset + rank] += moved;
    }
    loads[over] -= moved;
    loads[under] += moved;
  }
}

struct ExportCandidate {
  int64_t amount;
  int64_t target_ingress;
  int64_t expert;
  int source;
  int target;
  int delta;
  bool is_new;
};

__device__ bool better_export_candidate(const ExportCandidate& candidate,
                                        const ExportCandidate& current,
                                        bool communication_first) {
  if (current.expert < 0) return candidate.expert >= 0;
  if (candidate.expert < 0) return false;
  if (communication_first && candidate.delta != current.delta) {
    return candidate.delta < current.delta;
  }
  if (communication_first &&
      candidate.target_ingress != current.target_ingress) {
    return candidate.target_ingress < current.target_ingress;
  }
  if (candidate.amount != current.amount) return candidate.amount > current.amount;
  if (candidate.is_new != current.is_new) return candidate.is_new < current.is_new;
  if (!communication_first && candidate.delta != current.delta) {
    return candidate.delta < current.delta;
  }
  if (candidate.expert != current.expert) return candidate.expert < current.expert;
  if (candidate.source != current.source) return candidate.source < current.source;
  return candidate.target < current.target;
}

template <int MaxRanks>
__device__ bool build_export_plan(
    const int64_t* demand, const int64_t* primary, const bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* slots, int64_t* additions,
    int64_t* plan_quota, int64_t experts, int ranks,
    int64_t max_extra_per_rank, int64_t threshold, bool communication_first,
    int64_t* added_out, ExportCandidate* candidates, int* state,
    int64_t* shared_added, const int* rank_order, int64_t* ingress) {
  const int thread = threadIdx.x;
  for (int rank = thread; rank < ranks; rank += blockDim.x) {
    loads[rank] = slots[rank] = ingress[rank] = 0;
  }
  for (int64_t index = thread; index < experts * ranks; index += blockDim.x) {
    instance[index] = additions[index] = 0;
  }
  for (int64_t index = thread; index < experts * ranks * ranks;
       index += blockDim.x) {
    plan_quota[index] = 0;
  }
  __syncthreads();
  for (int64_t index = thread; index < experts * ranks; index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int source = index % ranks;
    const int target = replicas[index] ? source : primary[expert];
    const int64_t value = demand[index];
    atomicAdd(reinterpret_cast<unsigned long long*>(instance + expert * ranks + target),
              static_cast<unsigned long long>(value));
    atomicAdd(reinterpret_cast<unsigned long long*>(loads + target),
              static_cast<unsigned long long>(value));
    if (target != source) {
      atomicAdd(reinterpret_cast<unsigned long long*>(ingress + target),
                static_cast<unsigned long long>(value));
    }
    plan_quota[(source * experts + expert) * ranks + target] = value;
  }
  if (thread == 0) *shared_added = 0;
  __syncthreads();

  while (true) {
    if (thread == 0) {
      state[0] = -1;
      for (int position = 0; position < ranks; ++position) {
        const int rank = rank_order[position];
        if (loads[rank] > threshold) {
          state[0] = rank;
          break;
        }
      }
    }
    __syncthreads();
    const int over = state[0];
    if (over < 0) {
      if (thread == 0) *added_out = *shared_added;
      __syncthreads();
      return true;
    }

    const int64_t need = loads[over] - threshold;
    ExportCandidate best = {0, 0, -1, -1, -1, 0, false};
    const int64_t candidate_count = experts * ranks * ranks;
    for (int64_t index = thread; index < candidate_count; index += blockDim.x) {
      const int target = index % ranks;
      const int64_t pair = index / ranks;
      const int64_t expert = pair % experts;
      const int source = pair / experts;
      const int64_t available =
          plan_quota[(source * experts + expert) * ranks + over];
      const int64_t slack = threshold - loads[target];
      const bool present = replicas[expert * ranks + target] ||
                           additions[expert * ranks + target];
      const bool is_new = !present;
      if (!available || target == over || slack <= 0 ||
          (is_new && slots[target] >= max_extra_per_rank)) {
        continue;
      }
      ExportCandidate candidate = {
          min(min(need, available), slack), 0, expert, source, target,
          (target != source) - (over != source), is_new};
      candidate.target_ingress =
          ingress[target] + (target != source ? candidate.amount : 0);
      if (better_export_candidate(candidate, best, communication_first)) {
        best = candidate;
      }
    }
    candidates[thread] = best;
    __syncthreads();
    if (thread == 0) {
      best = candidates[0];
      for (int lane = 1; lane < blockDim.x; ++lane) {
        if (better_export_candidate(candidates[lane], best,
                                    communication_first)) {
          best = candidates[lane];
        }
      }
      state[1] = best.expert < 0 ? 0 : 1;
      if (best.expert >= 0) {
        if (best.is_new) {
          additions[best.expert * ranks + best.target] = ++*shared_added;
          ++slots[best.target];
        }
        instance[best.expert * ranks + over] -= best.amount;
        instance[best.expert * ranks + best.target] += best.amount;
        const int64_t offset =
            (best.source * experts + best.expert) * ranks;
        plan_quota[offset + over] -= best.amount;
        plan_quota[offset + best.target] += best.amount;
        loads[over] -= best.amount;
        loads[best.target] += best.amount;
        if (over != best.source) ingress[over] -= best.amount;
        if (best.target != best.source) ingress[best.target] += best.amount;
      }
    }
    __syncthreads();
    if (!state[1]) return false;
  }
}

template <int MaxRanks>
__global__ void select_compute_replicas_kernel(
    const int64_t* demand, const int64_t* primary, bool* replicas,
    int64_t* instance, int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* plan_quota, int64_t* routing,
    int64_t* added_out, int64_t experts, int64_t ranks, int64_t max_extra_per_rank) {
  if (blockIdx.x) return;
  __shared__ ExportCandidate candidates[128];
  __shared__ int state[2];
  __shared__ int64_t totals[128];
  __shared__ int64_t shared_added;
  __shared__ int64_t bounds[2];
  __shared__ int64_t ideal_capacity;
  __shared__ int64_t ingress[MaxRanks];
  __shared__ int rank_order[MaxRanks];
  int64_t local_total = 0;
  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x) {
    local_total += demand[index];
  }
  totals[threadIdx.x] = local_total;
  __syncthreads();
  if (threadIdx.x == 0) {
    int64_t total = 0;
    for (int lane = 0; lane < blockDim.x; ++lane) total += totals[lane];
    bounds[0] = (total + ranks - 1) / ranks;
    ideal_capacity = bounds[0];
  }
  for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x) {
    int64_t load = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
      for (int source = 0; source < ranks; ++source) {
        const int target = replicas[expert * ranks + source]
                               ? source
                               : primary[expert];
        if (target == rank) load += demand[expert * ranks + source];
      }
    }
    loads[rank] = load;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    bounds[1] = 0;
    for (int rank = 0; rank < ranks; ++rank) bounds[1] = max(bounds[1], loads[rank]);
    for (int rank = 0; rank < ranks; ++rank) rank_order[rank] = rank;
    for (int position = 1; position < ranks; ++position) {
      const int value = rank_order[position];
      int cursor = position;
      while (cursor > 0) {
        const int previous = rank_order[cursor - 1];
        if (loads[previous] > loads[value] ||
            (loads[previous] == loads[value] && previous < value)) {
          break;
        }
        rank_order[cursor--] = previous;
      }
      rank_order[cursor] = value;
    }
  }
  __syncthreads();
  int64_t ignored = 0;
  if (!build_export_plan<MaxRanks>(
          demand, primary, replicas, instance, loads, added_by_rank,
          addition_order, plan_quota, experts, ranks, max_extra_per_rank,
          bounds[0], false, &ignored, candidates, state, &shared_added,
          rank_order, ingress)) {
    if (threadIdx.x == 0) ++bounds[0];
    __syncthreads();
    while (bounds[0] < bounds[1]) {
      const int64_t middle = (bounds[0] + bounds[1]) / 2;
      if (build_export_plan<MaxRanks>(
              demand, primary, replicas, instance, loads, added_by_rank,
              addition_order, plan_quota, experts, ranks, max_extra_per_rank,
              middle, false, &ignored, candidates, state, &shared_added,
              rank_order, ingress)) {
        if (threadIdx.x == 0) bounds[1] = middle;
      } else {
        if (threadIdx.x == 0) bounds[0] = middle + 1;
      }
      __syncthreads();
    }
  }
  if (!build_export_plan<MaxRanks>(
          demand, primary, replicas, instance, loads, added_by_rank,
          addition_order, plan_quota, experts, ranks, max_extra_per_rank,
          bounds[0], true, added_out, candidates, state, &shared_added,
          rank_order, ingress)) {
    build_export_plan<MaxRanks>(
        demand, primary, replicas, instance, loads, added_by_rank,
        addition_order, plan_quota, experts, ranks, max_extra_per_rank,
        bounds[0], false, added_out, candidates, state, &shared_added,
        rank_order, ingress);
  }
  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x) {
    replicas[index] = replicas[index] || addition_order[index];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (true) {
      rebalance_compute_quota<MaxRanks>(plan_quota, replicas, loads, experts,
                                        ranks, ranks, ideal_capacity);
      for (int target = 0; target < ranks; ++target) {
        ingress[target] = 0;
        for (int source = 0; source < ranks; ++source) {
          if (source == target) continue;
          for (int64_t expert = 0; expert < experts; ++expert) {
            ingress[target] +=
                plan_quota[(source * experts + expert) * ranks + target];
          }
        }
      }
      int over = -1;
      for (int rank = 0; rank < ranks; ++rank) {
        if (loads[rank] > ideal_capacity) {
          over = rank;
          break;
        }
      }
      if (over < 0) break;

      int64_t best_amount = 0;
      int64_t best_target_load = 0;
      int64_t best_target_ingress = 0;
      int best_delta = 0;
      int best_source = -1;
      int64_t best_expert = -1;
      int best_target = -1;
      for (int source = 0; source < ranks; ++source) {
        for (int64_t expert = 0; expert < experts; ++expert) {
          const int64_t available =
              plan_quota[(source * experts + expert) * ranks + over];
          if (!available) continue;
          for (int target = 0; target < ranks; ++target) {
            if (target == over || replicas[expert * ranks + target] ||
                added_by_rank[target] >= max_extra_per_rank) {
              continue;
            }
            const int delta = (target != source) - (over != source);
            const int64_t target_ingress =
                ingress[target] + (target != source ? available : 0);
            bool better = best_expert < 0;
            if (!better && delta != best_delta) {
              better = delta < best_delta;
            } else if (!better && target_ingress != best_target_ingress) {
              better = target_ingress < best_target_ingress;
            } else if (!better && loads[target] != best_target_load) {
              better = loads[target] < best_target_load;
            } else if (!better && available != best_amount) {
              better = available > best_amount;
            } else if (!better && expert != best_expert) {
              better = expert < best_expert;
            } else if (!better && source != best_source) {
              better = source < best_source;
            } else if (!better) {
              better = target < best_target;
            }
            if (better) {
              best_amount = available;
              best_target_load = loads[target];
              best_target_ingress = target_ingress;
              best_delta = delta;
              best_source = source;
              best_expert = expert;
              best_target = target;
            }
          }
        }
      }
      if (best_expert < 0) break;
      addition_order[best_expert * ranks + best_target] = ++*added_out;
      replicas[best_expert * ranks + best_target] = true;
      ++added_by_rank[best_target];
    }
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x) {
    const int source = index / experts;
    const int64_t expert = index % experts;
    if (!demand[expert * ranks + source]) {
      routing[index] = replicas[expert * ranks + source] ? source : primary[expert];
      continue;
    }
    const int64_t offset = index * ranks;
    int target = 0;
    for (int rank = 1; rank < ranks; ++rank) {
      if (plan_quota[offset + rank] > plan_quota[offset + target]) target = rank;
    }
    routing[index] = target;
  }
}

}  // namespace

void select_pure_compute_replicas_into(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor primary,
    int64_t max_extra_per_rank,
    torch::Tensor instance, torch::Tensor loads, torch::Tensor added_by_rank,
    torch::Tensor addition_order, torch::Tensor plan_quota,
    torch::Tensor routing, torch::Tensor added) {
  TORCH_CHECK(demand.is_cuda() && replicas.is_cuda() && primary.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes());
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxRanks, "compute replicas support 1-64 ranks");
  TORCH_CHECK(primary.numel() == experts && max_extra_per_rank >= 0);
  TORCH_CHECK(instance.is_cuda() && instance.scalar_type() == torch::kInt64 &&
              instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.is_cuda() && loads.scalar_type() == torch::kInt64 &&
              loads.numel() == ranks);
  TORCH_CHECK(added_by_rank.is_cuda() && added_by_rank.scalar_type() == torch::kInt64 &&
              added_by_rank.numel() == ranks);
  TORCH_CHECK(addition_order.is_cuda() && addition_order.scalar_type() == torch::kInt64 &&
              addition_order.sizes() == demand.sizes());
  TORCH_CHECK(plan_quota.is_cuda() && plan_quota.scalar_type() == torch::kInt64 &&
              plan_quota.dim() == 3 && plan_quota.size(0) == ranks &&
              plan_quota.size(1) == experts && plan_quota.size(2) == ranks);
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt64 &&
              routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  TORCH_CHECK(added.is_cuda() && added.scalar_type() == torch::kInt64 && added.numel() == 1);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  if (ranks <= 4) {
    launch(select_compute_replicas_kernel<4>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 8) {
    launch(select_compute_replicas_kernel<8>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 16) {
    launch(select_compute_replicas_kernel<16>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 32) {
    launch(select_compute_replicas_kernel<32>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 64) {
    launch(select_compute_replicas_kernel<64>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else {
    launch(select_compute_replicas_kernel<64>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           plan_quota.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  }
  check_cuda(cudaGetLastError());
}

}  // namespace grace_cuda
