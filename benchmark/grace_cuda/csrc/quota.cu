#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "launch.cuh"

namespace grace_cuda {
namespace {

constexpr int kMaxRanks = 128;

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

__global__ void instance_quota_kernel(
    const int64_t* demand, const bool* replicas, const int64_t* expert_order,
    int64_t* instance_quota, int64_t* loads, int64_t experts, int64_t ranks) {
  if (blockIdx.x || threadIdx.x) return;
  int64_t allocation[kMaxRanks];
  for (int64_t index = 0; index < experts; ++index) {
    const int64_t expert = expert_order[index];
    int64_t total = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      total += demand[expert * ranks + rank];
    }
    waterfill(loads, replicas, expert, -1, total, allocation, ranks);
    for (int rank = 0; rank < ranks; ++rank) {
      instance_quota[expert * ranks + rank] = allocation[rank];
      loads[rank] += allocation[rank];
    }
  }
}

__global__ void source_quota_kernel(
    const int64_t* demand, int64_t* capacity, const int64_t* preferred,
    int64_t* quota, int64_t experts, int64_t ranks) {
  const int64_t expert = blockIdx.x;
  if (expert >= experts || threadIdx.x) return;
  int64_t remaining[kMaxRanks];
  bool selected[kMaxRanks];
  for (int source = 0; source < ranks; ++source) {
    const int64_t value = demand[expert * ranks + source];
    const int64_t local = min(value, capacity[expert * ranks + source]);
    quota[(source * experts + expert) * ranks + source] = local;
    remaining[source] = value - local;
    capacity[expert * ranks + source] -= local;
    selected[source] = false;
  }

  for (int position = 0; position < ranks; ++position) {
    int source = -1;
    for (int candidate = 0; candidate < ranks; ++candidate) {
      if (selected[candidate]) continue;
      if (source < 0 || remaining[candidate] > remaining[source] ||
          (remaining[candidate] == remaining[source] && candidate < source)) {
        source = candidate;
      }
    }
    selected[source] = true;
    int64_t amount = remaining[source];
    while (amount) {
      int target = -1;
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t available = capacity[expert * ranks + rank];
        if (!available) continue;
        if (target < 0 || rank == preferred[source * experts + expert] ||
            (target != preferred[source * experts + expert] &&
             (available > capacity[expert * ranks + target] ||
              (available == capacity[expert * ranks + target] && rank < target)))) {
          target = rank;
        }
      }
      const int64_t moved = min(amount, capacity[expert * ranks + target]);
      quota[(source * experts + expert) * ranks + target] += moved;
      capacity[expert * ranks + target] -= moved;
      amount -= moved;
    }
  }
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

__device__ bool better_candidate(
    int64_t next_max, int64_t next_square, int64_t remote,
    int64_t local_demand, int64_t expert, int target, int64_t best_max,
    int64_t best_square, int64_t best_remote, int64_t best_local_demand,
    int64_t best_expert, int best_target) {
  if (next_max != best_max) return next_max < best_max;
  if (next_square != best_square) return next_square < best_square;
  if (remote != best_remote) return remote < best_remote;
  if (local_demand != best_local_demand) return local_demand > best_local_demand;
  if (expert != best_expert) return expert < best_expert;
  return target < best_target;
}

struct ComputeCandidate {
  int64_t next_max;
  int64_t next_square;
  int64_t remote;
  int64_t local_demand;
  int64_t expert;
  int target;
};

__device__ bool better_candidate(const ComputeCandidate& candidate,
                                 const ComputeCandidate& best) {
  if (candidate.expert < 0) return false;
  if (best.expert < 0) return true;
  return better_candidate(
      candidate.next_max, candidate.next_square, candidate.remote,
      candidate.local_demand, candidate.expert, candidate.target, best.next_max,
      best.next_square, best.remote, best.local_demand, best.expert,
      best.target);
}

template <int MaxRanks>
__global__ __launch_bounds__(128, 1) void select_compute_replicas_kernel(
    const int64_t* demand, const int64_t* expert_demand,
    const int64_t* demand_order, bool* replicas, int64_t* instance_quota,
    int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* added_out, int64_t experts, int64_t ranks,
    int64_t max_extra_per_rank) {
  if (blockIdx.x) return;
  constexpr int kThreads = 128;
  __shared__ ComputeCandidate candidates[kThreads];
  __shared__ int64_t current_max;
  __shared__ int64_t ideal_capacity;
  __shared__ int64_t current_square;
  __shared__ int64_t current_remote;
  __shared__ int64_t added;
  __shared__ bool done;
  int64_t allocation[MaxRanks];
  int64_t base[MaxRanks];
  if (threadIdx.x == 0) {
    added = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      loads[rank] = 0;
      added_by_rank[rank] = 0;
    }
    for (int64_t index = 0; index < experts * ranks; ++index) {
      instance_quota[index] = 0;
      addition_order[index] = 0;
    }
  }
  __syncthreads();
  while (added < ranks * max_extra_per_rank) {
    if (threadIdx.x == 0) {
      for (int rank = 0; rank < ranks; ++rank) loads[rank] = 0;
      // The Python reference rebuilds the stable (fixed, flexible) order
      // after every added replica.  Keep that contract here; using the order
      // from the first iteration changes the waterfill result.
      for (int flexible = 0; flexible < 2; ++flexible) {
        for (int64_t position = 0; position < experts; ++position) {
          const int64_t expert = demand_order[position];
          int replica_count = 0;
          for (int rank = 0; rank < ranks; ++rank) {
            replica_count += replicas[expert * ranks + rank];
          }
          if ((replica_count > 1) != flexible) continue;
          waterfill_fast<MaxRanks>(loads, replicas, expert, -1,
                                   expert_demand[expert], allocation, ranks);
          for (int rank = 0; rank < ranks; ++rank) {
            instance_quota[expert * ranks + rank] = allocation[rank];
            loads[rank] += allocation[rank];
          }
        }
      }
      int64_t total = 0;
      for (int64_t expert = 0; expert < experts; ++expert) {
        total += expert_demand[expert];
      }
      ideal_capacity = (total + ranks - 1) / ranks;
      rebalance_compute_quota<MaxRanks>(instance_quota, replicas, loads, experts,
                                        ranks, 1, ideal_capacity);
      current_max = 0;
      current_square = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        current_max = max(current_max, loads[rank]);
        current_square += loads[rank] * loads[rank];
      }
      current_remote = 0;
      for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t total = expert_demand[expert];
        int64_t local = 0;
        for (int rank = 0; rank < ranks; ++rank) {
          local += min(demand[expert * ranks + rank],
                       instance_quota[expert * ranks + rank]);
        }
        current_remote += total - local;
      }
    }
    __syncthreads();
    if (current_max <= ideal_capacity) break;

    ComputeCandidate best = {0, 0, 0, 0, -1, -1};
    for (int64_t index = threadIdx.x; index < experts * ranks;
         index += blockDim.x) {
      const int64_t expert = index / ranks;
      const int target = index % ranks;
      const int64_t total = expert_demand[expert];
      if (!total || replicas[expert * ranks + target] ||
          added_by_rank[target] >= max_extra_per_rank) {
        continue;
      }
      for (int rank = 0; rank < ranks; ++rank) {
        base[rank] = loads[rank] - instance_quota[expert * ranks + rank];
      }
      waterfill_fast<MaxRanks>(base, replicas, expert, target, total, allocation,
                               ranks);
      ComputeCandidate candidate = {
          0, 0, 0, demand[expert * ranks + target], expert, target};
      int64_t local = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t next = base[rank] + allocation[rank];
        candidate.next_max = max(candidate.next_max, next);
        candidate.next_square += next * next;
        local += min(demand[expert * ranks + rank], allocation[rank]);
      }
      if (candidate.next_max > current_max ||
          (candidate.next_max == current_max &&
           candidate.next_square > current_square)) {
        continue;
      }
      int64_t current_local = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        current_local += min(demand[expert * ranks + rank],
                             instance_quota[expert * ranks + rank]);
      }
      // Source-demand remote bound for the whole plan, replacing the
      // candidate expert's contribution without scanning the trace.
      candidate.remote = current_remote - (total - current_local) + (total - local);
      if (better_candidate(candidate, best)) best = candidate;
    }
    candidates[threadIdx.x] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      ComputeCandidate winner = candidates[0];
      for (int index = 1; index < kThreads; ++index) {
        if (better_candidate(candidates[index], winner)) {
          winner = candidates[index];
        }
      }
      done = winner.expert < 0;
      if (!done) {
        replicas[winner.expert * ranks + winner.target] = true;
        addition_order[winner.expert * ranks + winner.target] = added + 1;
        ++added_by_rank[winner.target];
        ++added;
      }
    }
    __syncthreads();
    if (done) break;
  }
  if (threadIdx.x == 0) added_out[0] = added;
}

__global__ void localize_quota_kernel(
    const int64_t* demand, const bool* replicas, const int64_t* primary,
    const int64_t* source_order, int64_t* quota, int64_t* loads,
    double imbalance_limit, int64_t experts, int64_t ranks) {
  if (blockIdx.x || threadIdx.x || imbalance_limit < 1.0) return;
  int64_t total = 0;
  for (int64_t index = 0; index < experts * ranks; ++index) total += demand[index];
  const int64_t capacity = static_cast<int64_t>(ceil(
      static_cast<double>(total) / ranks * imbalance_limit));

  bool changed = true;
  while (changed) {
    changed = false;
    for (int source = 0; source < ranks; ++source) {
      int64_t room = capacity - loads[source];
      for (int64_t index = 0; index < experts && room > 0; ++index) {
        const int64_t expert = source_order[source * experts + index];
        if (!replicas[expert * ranks + source]) continue;
        for (int pass = 0; pass < ranks && room > 0; ++pass) {
          const int primary_rank = primary[expert];
          const int secondary = pass - 1;
          const int target =
              pass == 0 ? primary_rank : secondary + (secondary >= primary_rank);
          if (target == source ||
              !replicas[expert * ranks + target]) {
            continue;
          }
          const int64_t offset = (source * experts + expert) * ranks;
          const int64_t moved = min(room, quota[offset + target]);
          quota[offset + target] -= moved;
          quota[offset + source] += moved;
          loads[target] -= moved;
          loads[source] += moved;
          room -= moved;
          changed |= moved != 0;
        }
      }
    }
  }
}

template <int MaxRanks>
__global__ void fused_quota_kernel(
    const int64_t* demand, const bool* replicas, const int64_t* primary,
    const int64_t* routing, const int64_t* expert_order,
    const int64_t* source_order, double imbalance_limit, int64_t* quota,
    int64_t* next_routing, int64_t* instance, int64_t* loads, int64_t experts,
    int64_t ranks) {
  if (blockIdx.x) return;
  const int tid = threadIdx.x;
  int64_t allocation[MaxRanks];
  int64_t remaining[MaxRanks];
  bool selected[MaxRanks];

  const int64_t quota_size = experts * ranks * ranks;
  for (int64_t index = tid; index < quota_size; index += blockDim.x) {
    quota[index] = 0;
  }
  const int64_t routing_size = experts * ranks;
  for (int64_t index = tid; index < routing_size; index += blockDim.x) {
    next_routing[index] = routing[index];
  }
  if (tid == 0) {
    for (int rank = 0; rank < ranks; ++rank) loads[rank] = 0;
    for (int64_t position = 0; position < experts; ++position) {
      const int64_t expert = expert_order[position];
      int64_t total = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        total += demand[expert * ranks + rank];
      }
      waterfill_fast<MaxRanks>(loads, replicas, expert, -1, total, allocation,
                               ranks);
      for (int rank = 0; rank < ranks; ++rank) {
        instance[expert * ranks + rank] = allocation[rank];
        loads[rank] += allocation[rank];
      }
    }
  }
  __syncthreads();

  for (int64_t expert = tid; expert < experts; expert += blockDim.x) {
    for (int source = 0; source < ranks; ++source) {
      const int64_t value = demand[expert * ranks + source];
      const int64_t local = min(value, instance[expert * ranks + source]);
      const int64_t offset = (source * experts + expert) * ranks;
      quota[offset + source] = local;
      remaining[source] = value - local;
      instance[expert * ranks + source] -= local;
      selected[source] = false;
    }
    for (int position = 0; position < ranks; ++position) {
      int source = -1;
      for (int candidate = 0; candidate < ranks; ++candidate) {
        if (selected[candidate]) continue;
        if (source < 0 || remaining[candidate] > remaining[source] ||
            (remaining[candidate] == remaining[source] && candidate < source)) {
          source = candidate;
        }
      }
      selected[source] = true;
      int64_t amount = remaining[source];
      while (amount) {
        int target = -1;
        const int preferred_target = routing[source * experts + expert];
        for (int rank = 0; rank < ranks; ++rank) {
          const int64_t available = instance[expert * ranks + rank];
          if (!available) continue;
          if (target < 0 || rank == preferred_target ||
              (target != preferred_target &&
               (available > instance[expert * ranks + target] ||
                (available == instance[expert * ranks + target] && rank < target)))) {
            target = rank;
          }
        }
        const int64_t moved = min(amount, instance[expert * ranks + target]);
        const int64_t offset = (source * experts + expert) * ranks;
        quota[offset + target] += moved;
        instance[expert * ranks + target] -= moved;
        amount -= moved;
      }
    }
  }
  __syncthreads();

  // Rebalance the global compute load before source-local localization.  The
  // initial per-expert waterfill is greedy and can miss a feasible global
  // capacity (for example, [11, 9, 10] when [10, 10, 10] is possible).
  if (tid == 0 && imbalance_limit >= 1.0) {
    int64_t total = 0;
    for (int64_t index = 0; index < experts * ranks; ++index) {
      total += demand[index];
    }
    const int64_t capacity = static_cast<int64_t>(ceil(
        static_cast<double>(total) / ranks * imbalance_limit));
    rebalance_compute_quota<MaxRanks>(quota, replicas, loads, experts, ranks,
                                      ranks, capacity);
  }
  __syncthreads();

  if (tid == 0 && imbalance_limit >= 1.0) {
    int64_t total = 0;
    for (int64_t index = 0; index < experts * ranks; ++index) total += demand[index];
    const int64_t capacity = static_cast<int64_t>(ceil(
        static_cast<double>(total) / ranks * imbalance_limit));
    bool changed = true;
    while (changed) {
      changed = false;
      for (int source = 0; source < ranks; ++source) {
        int64_t room = capacity - loads[source];
        for (int64_t index = 0; index < experts && room > 0; ++index) {
          const int64_t expert = source_order[source * experts + index];
          if (!replicas[expert * ranks + source]) continue;
          for (int pass = 0; pass < ranks && room > 0; ++pass) {
            const int primary_rank = primary[expert];
            const int secondary = pass - 1;
            const int target =
                pass == 0 ? primary_rank
                          : secondary + (secondary >= primary_rank);
            if (target == source || !replicas[expert * ranks + target]) {
              continue;
            }
            const int64_t offset = (source * experts + expert) * ranks;
            const int64_t moved = min(room, quota[offset + target]);
            quota[offset + target] -= moved;
            quota[offset + source] += moved;
            loads[target] -= moved;
            loads[source] += moved;
            room -= moved;
            changed |= moved != 0;
          }
        }
      }
    }
  }
  __syncthreads();

  for (int64_t index = tid; index < routing_size; index += blockDim.x) {
    const int64_t source = index / experts;
    const int64_t expert = index % experts;
    const int64_t offset = index * ranks;
    int target = 0;
    int64_t best = quota[offset];
    for (int rank = 1; rank < ranks; ++rank) {
      if (quota[offset + rank] > best) {
        best = quota[offset + rank];
        target = rank;
      }
    }
    if (best) next_routing[index] = target;
  }
}

template <int MaxRanks>
void launch_fused_quota(
    const int64_t* demand, const bool* replicas, const int64_t* primary,
    const int64_t* routing, const int64_t* expert_order,
    const int64_t* source_order, double imbalance_limit, int64_t* quota,
    int64_t* next_routing, int64_t* instance, int64_t* loads, int64_t experts,
    int64_t ranks, cudaStream_t stream) {
  launch(fused_quota_kernel<MaxRanks>, dim3(1), dim3(128), stream,
         demand, replicas, primary, routing, expert_order, source_order,
         imbalance_limit, quota, next_routing, instance, loads, experts, ranks);
}

__global__ void quota_routing_kernel(const int64_t* quota, int64_t* routing,
                                     int64_t experts, int64_t ranks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= experts * ranks) return;
  const int64_t source = index / experts;
  const int64_t expert = index % experts;
  const int64_t offset = index * ranks;
  int target = 0;
  int64_t best = quota[offset];
  for (int rank = 1; rank < ranks; ++rank) {
    if (quota[offset + rank] > best) {
      best = quota[offset + rank];
      target = rank;
    }
  }
  if (best) routing[source * experts + expert] = target;
}

__device__ int quota_destination(const int64_t* quota, const bool* replicas,
                                 const int64_t* primary,
                                 const int64_t* addition_order, int64_t ordinal,
                                 int64_t source, int64_t expert,
                                 int64_t experts, int64_t ranks) {
  const int64_t offset = (source * experts + expert) * ranks;
  int64_t prefix = 0;
  if (replicas[expert * ranks + source]) {
    prefix += quota[offset + source];
    if (ordinal < prefix) return source;
  }
  const int primary_rank = primary[expert];
  if (primary_rank != source) {
    prefix += quota[offset + primary_rank];
    if (ordinal < prefix) return primary_rank;
  }
  for (int rank = 0; rank < ranks; ++rank) {
    if (rank == source || rank == primary_rank ||
        !replicas[expert * ranks + rank] ||
        addition_order[expert * ranks + rank]) {
      continue;
    }
    prefix += quota[offset + rank];
    if (ordinal < prefix) return rank;
  }
  int64_t last_order = 0;
  while (true) {
    int target = -1;
    int64_t next_order = std::numeric_limits<int64_t>::max();
    for (int rank = 0; rank < ranks; ++rank) {
      const int64_t order = addition_order[expert * ranks + rank];
      if (rank != source && order > last_order && order < next_order) {
        target = rank;
        next_order = order;
      }
    }
    if (target < 0) break;
    prefix += quota[offset + target];
    if (ordinal < prefix) return target;
    last_order = next_order;
  }
  return primary_rank;
}

__device__ int64_t quota_next_boundary(
    const int64_t* quota, const bool* replicas, const int64_t* primary,
    const int64_t* addition_order, int64_t ordinal, int64_t source,
    int64_t expert, int64_t experts, int64_t ranks) {
  const int64_t offset = (source * experts + expert) * ranks;
  int64_t prefix = 0;
  if (replicas[expert * ranks + source]) {
    prefix += quota[offset + source];
    if (prefix > ordinal) return prefix;
  }
  const int primary_rank = primary[expert];
  if (primary_rank != source) {
    prefix += quota[offset + primary_rank];
    if (prefix > ordinal) return prefix;
  }
  for (int rank = 0; rank < ranks; ++rank) {
    if (rank == source || rank == primary_rank ||
        !replicas[expert * ranks + rank] ||
        addition_order[expert * ranks + rank]) {
      continue;
    }
    prefix += quota[offset + rank];
    if (prefix > ordinal) return prefix;
  }
  int64_t last_order = 0;
  while (true) {
    int target = -1;
    int64_t next_order = std::numeric_limits<int64_t>::max();
    for (int rank = 0; rank < ranks; ++rank) {
      const int64_t order = addition_order[expert * ranks + rank];
      if (rank != source && order > last_order && order < next_order) {
        target = rank;
        next_order = order;
      }
    }
    if (target < 0) break;
    prefix += quota[offset + target];
    if (prefix > ordinal) return prefix;
    last_order = next_order;
  }
  return std::numeric_limits<int64_t>::max();
}

__global__ void quota_traffic_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* quota, const bool* replicas, const int64_t* primary,
    const int64_t* addition_order, const int64_t* ordinals, int64_t* traffic,
    int64_t* compute,
    int64_t tokens, int64_t k, int64_t experts, int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t src = source[token];
  const int64_t weight = count[token];
  int64_t position = 0;
  while (position < weight) {
    int64_t next = weight;
    for (int64_t col = 0; col < k; ++col) {
      const int64_t index = token * k + col;
      const int64_t boundary = quota_next_boundary(
          quota, replicas, primary, addition_order, ordinals[index] + position,
          src, topk[index], experts, ranks);
      if (boundary > ordinals[index] + position &&
          boundary - ordinals[index] < next) {
        next = boundary - ordinals[index];
      }
    }
    if (next <= position) next = position + 1;
    const int64_t segment = next - position;
    unsigned long long seen_low = 0;
    unsigned long long seen_high = 0;
    for (int64_t col = 0; col < k; ++col) {
      const int64_t index = token * k + col;
      const int destination = quota_destination(
          quota, replicas, primary, addition_order, ordinals[index] + position,
          src, topk[index], experts, ranks);
      atomicAdd(reinterpret_cast<unsigned long long*>(compute + destination),
                static_cast<unsigned long long>(segment));
      const auto bit = 1ULL << (destination & 63);
      auto& seen = destination < 64 ? seen_low : seen_high;
      if (destination != src && !(seen & bit)) {
        seen |= bit;
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      traffic + src * ranks + destination),
                  static_cast<unsigned long long>(segment));
      }
    }
    position = next;
  }
}

}  // namespace

void solve_quota_into(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor primary,
    torch::Tensor routing, torch::Tensor expert_order,
    torch::Tensor source_order, double imbalance_limit, torch::Tensor quota,
    torch::Tensor next_routing, torch::Tensor instance, torch::Tensor loads) {
  TORCH_CHECK(demand.is_cuda() && replicas.is_cuda() && primary.is_cuda() &&
              routing.is_cuda() && expert_order.is_cuda() && source_order.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              routing.scalar_type() == torch::kInt64 &&
              expert_order.scalar_type() == torch::kInt64 &&
              source_order.scalar_type() == torch::kInt64);
  TORCH_CHECK(demand.dim() == 2 && replicas.sizes() == demand.sizes());
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxRanks, "quota supports 1-128 ranks");
  TORCH_CHECK(primary.numel() == experts && routing.size(0) == ranks &&
              routing.size(1) == experts && expert_order.numel() == experts &&
              source_order.sizes() == routing.sizes());

  TORCH_CHECK(quota.is_cuda() && quota.scalar_type() == torch::kInt64 &&
              quota.dim() == 3 && quota.size(0) == ranks &&
              quota.size(1) == experts && quota.size(2) == ranks);
  TORCH_CHECK(next_routing.is_cuda() && next_routing.scalar_type() == torch::kInt64 &&
              next_routing.sizes() == routing.sizes());
  TORCH_CHECK(instance.is_cuda() && instance.scalar_type() == torch::kInt64 &&
              instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.is_cuda() && loads.scalar_type() == torch::kInt64 &&
              loads.dim() == 1 && loads.size(0) == ranks);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  if (ranks <= 4) {
    launch_fused_quota<4>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  } else if (ranks <= 8) {
    launch_fused_quota<8>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  } else if (ranks <= 16) {
    launch_fused_quota<16>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  } else if (ranks <= 32) {
    launch_fused_quota<32>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  } else if (ranks <= 64) {
    launch_fused_quota<64>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  } else {
    launch_fused_quota<128>(
        demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
        primary.data_ptr<int64_t>(), routing.data_ptr<int64_t>(),
        expert_order.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
        imbalance_limit, quota.data_ptr<int64_t>(),
        next_routing.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
        loads.data_ptr<int64_t>(), experts, ranks, stream.stream());
  }
  check_cuda(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor> solve_quota(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor primary,
    torch::Tensor routing, torch::Tensor expert_order,
    torch::Tensor source_order, double imbalance_limit) {
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  auto quota = torch::zeros({ranks, experts, ranks}, demand.options());
  auto next_routing = routing.clone();
  auto instance = torch::zeros_like(demand);
  auto loads = torch::zeros({ranks}, demand.options());
  solve_quota_into(demand, replicas, primary, routing, expert_order, source_order,
                   imbalance_limit, quota, next_routing, instance, loads);
  return {quota, next_routing};
}

void select_compute_replicas_into(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor expert_demand,
    torch::Tensor demand_order, int64_t max_extra_per_rank,
    torch::Tensor instance, torch::Tensor loads, torch::Tensor added_by_rank,
    torch::Tensor addition_order, torch::Tensor added) {
  TORCH_CHECK(demand.is_cuda() && replicas.is_cuda() &&
              expert_demand.is_cuda() && demand_order.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              expert_demand.scalar_type() == torch::kInt64 &&
              demand_order.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes());
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxRanks, "compute replicas support 1-128 ranks");
  TORCH_CHECK(expert_demand.numel() == experts &&
              demand_order.numel() == experts && max_extra_per_rank >= 0);
  TORCH_CHECK(instance.is_cuda() && instance.scalar_type() == torch::kInt64 &&
              instance.sizes() == demand.sizes());
  TORCH_CHECK(loads.is_cuda() && loads.scalar_type() == torch::kInt64 &&
              loads.numel() == ranks);
  TORCH_CHECK(added_by_rank.is_cuda() && added_by_rank.scalar_type() == torch::kInt64 &&
              added_by_rank.numel() == ranks);
  TORCH_CHECK(addition_order.is_cuda() && addition_order.scalar_type() == torch::kInt64 &&
              addition_order.sizes() == demand.sizes());
  TORCH_CHECK(added.is_cuda() && added.scalar_type() == torch::kInt64 && added.numel() == 1);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  if (ranks <= 4) {
    launch(select_compute_replicas_kernel<4>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 8) {
    launch(select_compute_replicas_kernel<8>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 16) {
    launch(select_compute_replicas_kernel<16>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 32) {
    launch(select_compute_replicas_kernel<32>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 64) {
    launch(select_compute_replicas_kernel<64>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else {
    launch(select_compute_replicas_kernel<128>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           demand_order.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  }
  check_cuda(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> select_compute_replicas(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor expert_demand,
    torch::Tensor demand_order, int64_t max_extra_per_rank) {
  auto next_replicas = replicas.clone();
  auto instance = torch::zeros_like(demand);
  auto loads = torch::zeros({demand.size(1)}, demand.options());
  auto added_by_rank = torch::zeros_like(loads);
  auto addition_order = torch::zeros_like(demand);
  auto added = torch::zeros({1}, demand.options());
  select_compute_replicas_into(demand, next_replicas, expert_demand, demand_order,
                               max_extra_per_rank, instance, loads, added_by_rank,
                               addition_order, added);
  return {next_replicas, added, addition_order};
}

std::tuple<torch::Tensor, torch::Tensor> quota_traffic(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor quota, torch::Tensor replicas, torch::Tensor primary,
    torch::Tensor addition_order, torch::Tensor ordinals, int64_t num_ranks) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              quota.is_cuda() && replicas.is_cuda() && primary.is_cuda() &&
              addition_order.is_cuda() && ordinals.is_cuda());
  TORCH_CHECK(num_ranks > 0 && num_ranks <= kMaxRanks,
              "quota traffic supports 1-128 ranks");
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              quota.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 &&
              addition_order.scalar_type() == torch::kInt64 &&
              ordinals.scalar_type() == torch::kInt64);
  const int64_t experts = replicas.size(0);
  TORCH_CHECK(quota.size(0) == num_ranks && quota.size(1) == experts &&
              quota.size(2) == num_ranks &&
              addition_order.sizes() == replicas.sizes() &&
              topk.sizes() == ordinals.sizes());
  auto traffic = torch::zeros({num_ranks, num_ranks}, source.options());
  auto compute = torch::zeros({num_ranks}, source.options());
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  launch(quota_traffic_kernel, dim3((tokens + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), quota.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), primary.data_ptr<int64_t>(),
         addition_order.data_ptr<int64_t>(), ordinals.data_ptr<int64_t>(),
         traffic.data_ptr<int64_t>(), compute.data_ptr<int64_t>(), tokens,
         topk.size(1), experts, num_ranks);
  check_cuda(cudaGetLastError());
  return {traffic, compute};
}

}  // namespace grace_cuda
