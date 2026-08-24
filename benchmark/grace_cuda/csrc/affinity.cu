#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cfloat>
#include <climits>

#include "launch.cuh"

namespace grace_cuda {
namespace {

__global__ void affinity_histogram_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    int64_t* demand, int64_t* affinity, int64_t* degree, int64_t tokens,
    int64_t k, int64_t experts, int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t weight = count[token];
  for (int64_t left = 0; left < k; ++left) {
    const int64_t a = topk[token * k + left];
    atomicAdd(reinterpret_cast<unsigned long long*>(demand + a * ranks + source[token]),
              static_cast<unsigned long long>(weight));
    for (int64_t right = left + 1; right < k; ++right) {
      const int64_t b = topk[token * k + right];
      atomicAdd(reinterpret_cast<unsigned long long*>(affinity + a * experts + b),
                static_cast<unsigned long long>(weight));
      atomicAdd(reinterpret_cast<unsigned long long*>(affinity + b * experts + a),
                static_cast<unsigned long long>(weight));
      atomicAdd(reinterpret_cast<unsigned long long*>(degree + a),
                static_cast<unsigned long long>(weight));
      atomicAdd(reinterpret_cast<unsigned long long*>(degree + b),
                static_cast<unsigned long long>(weight));
    }
  }
}

// ponytail: one thread is intentional for <=256 experts; replace with a
// parallel partitioner only if this O(E^2) step becomes measurable.
__global__ void affinity_groups_kernel(
    const int64_t* demand, const int64_t* affinity, const int64_t* degree,
    int64_t* score, int64_t* groups, int64_t experts, int ranks) {
  if (blockIdx.x || threadIdx.x) return;
  for (int64_t expert = 0; expert < experts; ++expert) groups[expert] = -1;

  const int64_t base = experts / ranks;
  const int64_t remainder = experts % ranks;
  for (int group = 0; group < ranks; ++group) {
    const int64_t capacity = base + (group < remainder);
    if (!capacity) continue;
    int64_t seed = -1;
    int64_t best_degree = -1;
    int64_t best_demand = -1;
    for (int64_t expert = 0; expert < experts; ++expert) {
      if (groups[expert] >= 0) continue;
      int64_t total = 0;
      for (int rank = 0; rank < ranks; ++rank)
        total += demand[expert * ranks + rank];
      if (degree[expert] > best_degree ||
          (degree[expert] == best_degree &&
           (total > best_demand ||
            (total == best_demand && (seed < 0 || expert < seed))))) {
        seed = expert;
        best_degree = degree[expert];
        best_demand = total;
      }
    }
    groups[seed] = group;
    for (int64_t expert = 0; expert < experts; ++expert)
      score[expert] = affinity[expert * experts + seed];

    for (int64_t slot = 1; slot < capacity; ++slot) {
      int64_t selected = -1;
      int64_t best_score = -1;
      best_demand = -1;
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (groups[expert] >= 0) continue;
        int64_t total = 0;
        for (int rank = 0; rank < ranks; ++rank)
          total += demand[expert * ranks + rank];
        if (score[expert] > best_score ||
            (score[expert] == best_score &&
             (total > best_demand ||
              (total == best_demand &&
               (selected < 0 || expert < selected))))) {
          selected = expert;
          best_score = score[expert];
          best_demand = total;
        }
      }
      groups[selected] = group;
      for (int64_t expert = 0; expert < experts; ++expert)
        if (groups[expert] < 0)
          score[expert] += affinity[expert * experts + selected];
    }
  }
}

__global__ void group_source_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* groups, int64_t* group_source, int64_t tokens, int64_t k,
    int ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  for (int64_t column = 0; column < k; ++column) {
    const int64_t group = groups[topk[token * k + column]];
    bool first = true;
    for (int64_t previous = 0; previous < column; ++previous)
      if (groups[topk[token * k + previous]] == group) first = false;
    if (first)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    group_source + group * ranks + source[token]),
                static_cast<unsigned long long>(count[token]));
  }
}

__global__ void map_groups_kernel(const int64_t* group_source,
                                  const int64_t* groups,
                                  int64_t* group_to_rank, int64_t* primary,
                                  int64_t experts, int ranks) {
  if (blockIdx.x || threadIdx.x) return;
  bool used_group[128] = {};
  bool used_rank[128] = {};
  int64_t egress[128] = {};
  int64_t max_ingress = 0;
  int64_t max_pair = 0;

  for (int step = 0; step < ranks; ++step) {
    int group = -1;
    int64_t group_total = -1;
    int64_t group_peak = -1;
    for (int candidate = 0; candidate < ranks; ++candidate) {
      if (used_group[candidate]) continue;
      int64_t total = 0;
      int64_t peak = 0;
      for (int source = 0; source < ranks; ++source) {
        const int64_t value = group_source[candidate * ranks + source];
        total += value;
        if (value > peak) peak = value;
      }
      if (total > group_total ||
          (total == group_total &&
           (peak > group_peak ||
            (peak == group_peak && (group < 0 || candidate < group))))) {
        group = candidate;
        group_total = total;
        group_peak = peak;
      }
    }

    int best_rank = -1;
    int64_t best_bottleneck = LLONG_MAX;
    int64_t best_pair = LLONG_MAX;
    int64_t best_remote = LLONG_MAX;
    for (int rank = 0; rank < ranks; ++rank) {
      if (used_rank[rank]) continue;
      const int64_t remote =
          group_total - group_source[group * ranks + rank];
      int64_t projected_egress = 0;
      int64_t projected_pair = max_pair;
      for (int source = 0; source < ranks; ++source) {
        const int64_t added = source == rank
                                  ? 0
                                  : group_source[group * ranks + source];
        const int64_t value = egress[source] + added;
        if (value > projected_egress) projected_egress = value;
        if (added > projected_pair) projected_pair = added;
      }
      const int64_t ingress = remote > max_ingress ? remote : max_ingress;
      const int64_t bottleneck =
          ingress > projected_egress ? ingress : projected_egress;
      if (bottleneck < best_bottleneck ||
          (bottleneck == best_bottleneck &&
           (projected_pair < best_pair ||
            (projected_pair == best_pair &&
             (remote < best_remote ||
              (remote == best_remote && (best_rank < 0 || rank < best_rank))))))) {
        best_rank = rank;
        best_bottleneck = bottleneck;
        best_pair = projected_pair;
        best_remote = remote;
      }
    }
    used_group[group] = true;
    used_rank[best_rank] = true;
    group_to_rank[group] = best_rank;
    if (best_remote > max_ingress) max_ingress = best_remote;
    for (int source = 0; source < ranks; ++source)
      if (source != best_rank) {
        const int64_t added = group_source[group * ranks + source];
        egress[source] += added;
        if (added > max_pair) max_pair = added;
      }
  }
  for (int64_t expert = 0; expert < experts; ++expert)
    primary[expert] = group_to_rank[groups[expert]];
}

__global__ void spectral_exact_groups_kernel(
    const double* embedding, const int64_t* affinity, double* centers,
    double* distances,
    int64_t* labels, int64_t* next_labels, int64_t* sizes, int64_t* overflow,
    int64_t experts, int ranks) {
  if (blockIdx.x) return;
  __shared__ int unchanged;
  __shared__ int64_t selected_item;
  __shared__ int overflow_count;
  __shared__ int chosen_group;
  for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x)
    labels[expert] = 0;
  if (threadIdx.x == 0) overflow[0] = 0;
  __syncthreads();

  for (int center_count = 1; center_count < ranks; ++center_count) {
    for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
      bool is_center = false;
      for (int center = 0; center < center_count; ++center)
        is_center |= overflow[center] == expert;
      double nearest = DBL_MAX;
      if (!is_center) {
        for (int center = 0; center < center_count; ++center) {
          const int64_t center_expert = overflow[center];
          double distance = 0;
          for (int dim = 0; dim < ranks; ++dim) {
            const double delta = embedding[expert * ranks + dim] -
                                 embedding[center_expert * ranks + dim];
            distance += delta * delta;
          }
          nearest = min(nearest, distance);
        }
      }
      distances[expert] = is_center ? -1.0 : nearest;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int64_t selected = -1;
      double best = -1.0;
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (distances[expert] > best) {
          best = distances[expert];
          selected = expert;
        }
      }
      overflow[center_count] = selected;
    }
    __syncthreads();
  }
  for (int64_t index = threadIdx.x; index < ranks * ranks;
       index += blockDim.x) {
    const int center = index / ranks;
    const int dim = index % ranks;
    centers[index] = embedding[overflow[center] * ranks + dim];
  }
  __syncthreads();

  for (int iteration = 0; iteration < 32; ++iteration) {
    for (int group = threadIdx.x; group < ranks; group += blockDim.x)
      sizes[group] = 0;
    __syncthreads();
    for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
      int best_group = 0;
      double best_distance = DBL_MAX;
      for (int group = 0; group < ranks; ++group) {
        double distance = 0;
        for (int dim = 0; dim < ranks; ++dim) {
          const double delta = embedding[expert * ranks + dim] -
                               centers[group * ranks + dim];
          distance += delta * delta;
        }
        if (distance < best_distance) {
          best_distance = distance;
          best_group = group;
        }
      }
      next_labels[expert] = best_group;
      atomicAdd(reinterpret_cast<unsigned long long*>(sizes + best_group), 1ULL);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int empty = 0; empty < ranks; ++empty) {
        if (sizes[empty]) continue;
        int donor = 0;
        for (int group = 1; group < ranks; ++group)
          if (sizes[group] > sizes[donor]) donor = group;
        int64_t moved = -1;
        double farthest = -1;
        for (int64_t expert = 0; expert < experts; ++expert) {
          if (next_labels[expert] != donor) continue;
          double distance = 0;
          for (int dim = 0; dim < ranks; ++dim) {
            const double delta = embedding[expert * ranks + dim] -
                                 centers[donor * ranks + dim];
            distance += delta * delta;
          }
          if (distance > farthest) {
            farthest = distance;
            moved = expert;
          }
        }
        next_labels[moved] = empty;
        --sizes[donor];
        ++sizes[empty];
      }
      unchanged = 1;
    }
    __syncthreads();
    for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x)
      if (next_labels[expert] != labels[expert]) atomicExch(&unchanged, 0);
    for (int64_t index = threadIdx.x; index < ranks * ranks;
         index += blockDim.x) {
      const int group = index / ranks;
      const int dim = index % ranks;
      double total = 0;
      for (int64_t expert = 0; expert < experts; ++expert)
        if (next_labels[expert] == group)
          total += embedding[expert * ranks + dim];
      centers[index] = total / sizes[group];
    }
    for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x)
      labels[expert] = next_labels[expert];
    __syncthreads();
    if (unchanged) break;
  }

  const int64_t target = experts / ranks;
  if (threadIdx.x == 0) overflow_count = 0;
  __syncthreads();
  for (int group = 0; group < ranks; ++group) {
    for (int64_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      int64_t value = 0;
      if (labels[expert] == group)
        for (int64_t other = 0; other < experts; ++other)
          if (other != expert && labels[other] == group)
            value += affinity[expert * experts + other];
      next_labels[expert] = value;
    }
    __syncthreads();
    while (true) {
      if (threadIdx.x == 0) {
        selected_item = -1;
        int64_t least = LLONG_MAX;
        if (sizes[group] > target)
          for (int64_t expert = 0; expert < experts; ++expert)
            if (labels[expert] == group &&
                (next_labels[expert] < least ||
                 (next_labels[expert] == least && expert < selected_item))) {
              least = next_labels[expert];
              selected_item = expert;
            }
        if (selected_item >= 0) {
          labels[selected_item] = -1;
          overflow[overflow_count++] = selected_item;
          --sizes[group];
        }
      }
      __syncthreads();
      if (selected_item < 0) break;
      for (int64_t expert = threadIdx.x; expert < experts;
           expert += blockDim.x)
        if (labels[expert] == group)
          next_labels[expert] -= affinity[expert * experts + selected_item];
      __syncthreads();
    }
  }
  for (int index = threadIdx.x; index < overflow_count; index += blockDim.x) {
    const int64_t expert = overflow[index];
    int64_t degree = 0;
    for (int64_t other = 0; other < experts; ++other)
      degree += affinity[expert * experts + other];
    next_labels[expert] = degree;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int64_t left = 1; left < overflow_count; ++left) {
      const int64_t value = overflow[left];
      int64_t cursor = left;
      const int64_t value_degree = next_labels[value];
      while (cursor > 0) {
        const int64_t previous = overflow[cursor - 1];
        const int64_t previous_degree = next_labels[previous];
        if (previous_degree > value_degree ||
            (previous_degree == value_degree && previous < value))
          break;
        overflow[cursor--] = previous;
      }
      overflow[cursor] = value;
    }
  }
  __syncthreads();
  for (int64_t index = 0; index < overflow_count; ++index) {
    const int64_t expert = overflow[index];
    for (int group = threadIdx.x; group < ranks; group += blockDim.x) {
      int64_t value = 0;
      for (int64_t other = 0; other < experts; ++other)
        if (labels[other] == group)
          value += affinity[expert * experts + other];
      next_labels[group] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      chosen_group = -1;
      int64_t best_affinity = -1;
      for (int group = 0; group < ranks; ++group) {
        if (sizes[group] >= target) continue;
        const int64_t value = next_labels[group];
        if (value > best_affinity ||
            (value == best_affinity &&
             (chosen_group < 0 || sizes[group] < sizes[chosen_group] ||
              (sizes[group] == sizes[chosen_group] && group < chosen_group)))) {
          best_affinity = value;
          chosen_group = group;
        }
      }
      labels[expert] = chosen_group;
      ++sizes[chosen_group];
    }
    __syncthreads();
  }
}

__device__ void hungarian(const int64_t* cost, int64_t* work,
                          int64_t* assignment, int size) {
  int64_t* u = work;
  int64_t* v = work + size + 1;
  int64_t* matched = work + 2 * (size + 1);
  int64_t* previous = work + 3 * (size + 1);
  int64_t* minimum = work + 4 * (size + 1);
  int64_t* used = work + 5 * (size + 1);
  for (int i = 0; i <= size; ++i) u[i] = v[i] = matched[i] = 0;
  for (int row = 1; row <= size; ++row) {
    matched[0] = row;
    int column = 0;
    for (int i = 0; i <= size; ++i) {
      minimum[i] = LLONG_MAX;
      used[i] = 0;
    }
    while (true) {
      used[column] = 1;
      const int current_row = matched[column];
      int64_t delta = LLONG_MAX;
      int next_column = 0;
      for (int candidate = 1; candidate <= size; ++candidate) {
        if (used[candidate]) continue;
        const int64_t reduced = cost[(current_row - 1) * size + candidate - 1] -
                                u[current_row] - v[candidate];
        if (reduced < minimum[candidate]) {
          minimum[candidate] = reduced;
          previous[candidate] = column;
        }
        if (minimum[candidate] < delta) {
          delta = minimum[candidate];
          next_column = candidate;
        }
      }
      for (int candidate = 0; candidate <= size; ++candidate) {
        if (used[candidate]) {
          u[matched[candidate]] += delta;
          v[candidate] -= delta;
        } else if (candidate) {
          minimum[candidate] -= delta;
        }
      }
      column = next_column;
      if (!matched[column]) break;
    }
    while (true) {
      const int next_column = previous[column];
      matched[column] = matched[next_column];
      column = next_column;
      if (!column) break;
    }
  }
  for (int column = 1; column <= size; ++column)
    assignment[matched[column] - 1] = column - 1;
}

__global__ void congestion_hungarian_kernel(
    const int64_t* local, const int64_t* groups, bool* allowed, int64_t* values,
    int64_t* cost, int64_t* work, int64_t* assignment, int64_t* primary,
    int64_t experts, int ranks) {
  if (blockIdx.x || threadIdx.x) return;
  for (int index = 0; index < ranks * ranks; ++index) allowed[index] = true;
  for (int objective = 0; objective < 2; ++objective) {
    for (int group = 0; group < ranks; ++group) {
      int64_t total = 0;
      for (int source = 0; source < ranks; ++source)
        total += local[group * ranks + source];
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t ingress = total - local[group * ranks + rank];
        int64_t egress = 0;
        int64_t pair = 0;
        for (int other_group = 0; other_group < ranks; ++other_group)
          egress += local[other_group * ranks + rank];
        egress -= local[group * ranks + rank];
        for (int source = 0; source < ranks; ++source)
          if (source != rank)
            pair = max(pair, local[group * ranks + source]);
        values[group * ranks + rank] =
            objective == 0 ? max(ingress, egress) : pair;
      }
    }
    int64_t low = 0;
    int64_t high = 0;
    for (int index = 0; index < ranks * ranks; ++index)
      if (allowed[index]) high = max(high, values[index]);
    while (low < high) {
      const int64_t threshold = low + (high - low) / 2;
      for (int index = 0; index < ranks * ranks; ++index)
        cost[index] = allowed[index] && values[index] <= threshold ? 0 : 1;
      hungarian(cost, work, assignment, ranks);
      bool feasible = true;
      for (int group = 0; group < ranks; ++group)
        feasible &= allowed[group * ranks + assignment[group]] &&
                    values[group * ranks + assignment[group]] <= threshold;
      if (feasible)
        high = threshold;
      else
        low = threshold + 1;
    }
    for (int index = 0; index < ranks * ranks; ++index)
      allowed[index] &= values[index] <= low;
  }
  int64_t maximum = 0;
  for (int group = 0; group < ranks; ++group) {
    int64_t total = 0;
    for (int source = 0; source < ranks; ++source)
      total += local[group * ranks + source];
    for (int rank = 0; rank < ranks; ++rank) {
      const int64_t ingress = total - local[group * ranks + rank];
      cost[group * ranks + rank] = ingress;
      maximum = max(maximum, ingress);
    }
  }
  for (int index = 0; index < ranks * ranks; ++index)
    if (!allowed[index]) cost[index] = maximum * ranks + 1;
  hungarian(cost, work, assignment, ranks);
  for (int64_t expert = 0; expert < experts; ++expert)
    primary[expert] = assignment[groups[expert]];
}

__global__ void occurrence_count_kernel(const int64_t* topk, int64_t* counts,
                                        int64_t entries) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < entries)
    atomicAdd(reinterpret_cast<unsigned long long*>(counts + topk[index]), 1ULL);
}

__global__ void occurrence_prefix_kernel(int64_t* counts, int64_t* offsets,
                                         int64_t* cursors, int64_t experts) {
  if (blockIdx.x || threadIdx.x) return;
  int64_t offset = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    offsets[expert] = offset;
    cursors[expert] = 0;
    offset += counts[expert];
  }
  offsets[experts] = offset;
}

__global__ void occurrence_fill_kernel(const int64_t* topk,
                                       const int64_t* offsets,
                                       int64_t* cursors, int64_t* tokens,
                                       int64_t entries, int64_t k) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= entries) return;
  const int64_t expert = topk[index];
  const int64_t position = atomicAdd(
      reinterpret_cast<unsigned long long*>(cursors + expert), 1ULL);
  tokens[offsets[expert] + position] = index / k;
}

__global__ void refinement_state_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* demand, const int64_t* primary, int64_t* slots,
    int64_t* loads, int64_t* traffic, int64_t tokens, int64_t k,
    int64_t experts, int ranks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < experts) {
    const int rank = primary[index];
    int64_t total = 0;
    for (int source_rank = 0; source_rank < ranks; ++source_rank)
      total += demand[index * ranks + source_rank];
    atomicAdd(reinterpret_cast<unsigned long long*>(slots + rank), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(loads + rank),
              static_cast<unsigned long long>(total));
  }
  if (index >= tokens) return;
  for (int64_t column = 0; column < k; ++column) {
    const int rank = primary[topk[index * k + column]];
    bool first = true;
    for (int64_t previous = 0; previous < column; ++previous)
      first &= primary[topk[index * k + previous]] != rank;
    if (first && source[index] != rank)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    traffic + source[index] * ranks + rank),
                static_cast<unsigned long long>(count[index]));
  }
}

__global__ void congestion_move_candidates_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* demand, const int64_t* primary, const int64_t* offsets,
    const int64_t* occurrence_tokens, const int64_t* slots,
    const int64_t* loads, const int64_t* traffic, int64_t* candidates,
    int64_t k, int64_t experts, int ranks, int64_t minimum_capacity,
    int64_t maximum_capacity, double compute_limit) {
  const int64_t candidate = blockIdx.x;
  const int64_t expert = candidate / ranks;
  const int target = candidate % ranks;
  const int old = primary[expert];
  __shared__ unsigned long long delta[256];
  for (int index = threadIdx.x; index < 2 * ranks; index += blockDim.x)
    delta[index] = 0;
  __syncthreads();
  if (target == old || slots[old] <= minimum_capacity ||
      slots[target] >= maximum_capacity) {
    if (!threadIdx.x) candidates[candidate * 5] = 0;
    return;
  }
  for (int64_t position = offsets[expert] + threadIdx.x;
       position < offsets[expert + 1]; position += blockDim.x) {
    const int64_t token = occurrence_tokens[position];
    int old_count = 0;
    int target_count = 0;
    for (int64_t column = 0; column < k; ++column) {
      const int rank = primary[topk[token * k + column]];
      old_count += rank == old;
      target_count += rank == target;
    }
    const int source_rank = source[token];
    const int64_t weight = count[token];
    if (old_count == 1 && source_rank != old)
      atomicAdd(delta + source_rank, static_cast<unsigned long long>(-weight));
    if (!target_count && source_rank != target)
      atomicAdd(delta + ranks + source_rank,
                static_cast<unsigned long long>(weight));
  }
  __syncthreads();
  if (threadIdx.x) return;
  int64_t expert_demand = 0;
  int64_t total_demand = 0;
  int64_t current_peak = 0;
  for (int rank = 0; rank < ranks; ++rank) {
    expert_demand += demand[expert * ranks + rank];
    total_demand += loads[rank];
    current_peak = max(current_peak, loads[rank]);
  }
  const int64_t next_old = loads[old] - expert_demand;
  const int64_t next_target = loads[target] + expert_demand;
  int64_t compute_peak = 0;
  for (int rank = 0; rank < ranks; ++rank)
    compute_peak = max(compute_peak, rank == old ? next_old :
                       rank == target ? next_target : loads[rank]);
  const double limit_peak = compute_limit * total_demand / ranks;
  const double allowed_peak =
      limit_peak > current_peak ? limit_peak : static_cast<double>(current_peak);
  if (compute_peak > allowed_peak) {
    candidates[candidate * 5] = 0;
    return;
  }
  int64_t remote = 0;
  int64_t pair = 0;
  int64_t max_ingress = 0;
  int64_t max_egress = 0;
  for (int source_rank = 0; source_rank < ranks; ++source_rank) {
    int64_t egress = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      int64_t value = traffic[source_rank * ranks + rank];
      if (rank == old) value += static_cast<int64_t>(delta[source_rank]);
      if (rank == target)
        value += static_cast<int64_t>(delta[ranks + source_rank]);
      remote += value;
      egress += value;
      pair = max(pair, value);
    }
    max_egress = max(max_egress, egress);
  }
  for (int rank = 0; rank < ranks; ++rank) {
    int64_t ingress = 0;
    for (int source_rank = 0; source_rank < ranks; ++source_rank) {
      int64_t value = traffic[source_rank * ranks + rank];
      if (rank == old) value += static_cast<int64_t>(delta[source_rank]);
      if (rank == target)
        value += static_cast<int64_t>(delta[ranks + source_rank]);
      ingress += value;
    }
    max_ingress = max(max_ingress, ingress);
  }
  candidates[candidate * 5] = 1;
  candidates[candidate * 5 + 1] = max(max_ingress, max_egress);
  candidates[candidate * 5 + 2] = pair;
  candidates[candidate * 5 + 3] = remote;
  candidates[candidate * 5 + 4] = compute_peak;
}

__global__ void select_congestion_move_kernel(
    const int64_t* traffic, const int64_t* candidates, int64_t* selected,
    int64_t experts, int ranks) {
  if (blockIdx.x || threadIdx.x) return;
  int64_t current_remote = 0;
  int64_t current_pair = 0;
  int64_t current_ingress = 0;
  int64_t current_egress = 0;
  for (int source = 0; source < ranks; ++source) {
    int64_t egress = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      const int64_t value = traffic[source * ranks + rank];
      current_remote += value;
      egress += value;
      current_pair = max(current_pair, value);
    }
    current_egress = max(current_egress, egress);
  }
  for (int rank = 0; rank < ranks; ++rank) {
    int64_t ingress = 0;
    for (int source = 0; source < ranks; ++source)
      ingress += traffic[source * ranks + rank];
    current_ingress = max(current_ingress, ingress);
  }
  const int64_t current_bottleneck = max(current_ingress, current_egress);
  int64_t best = -1;
  for (int64_t candidate = 0; candidate < experts * ranks; ++candidate) {
    const int64_t* value = candidates + candidate * 5;
    if (!value[0]) continue;
    const bool improves = value[1] < current_bottleneck ||
        (value[1] == current_bottleneck &&
         (value[2] < current_pair ||
          (value[2] == current_pair && value[3] < current_remote)));
    if (!improves) continue;
    if (best < 0) {
      best = candidate;
      continue;
    }
    const int64_t* previous = candidates + best * 5;
    bool better = false;
    for (int key = 1; key < 5; ++key) {
      if (value[key] != previous[key]) {
        better = value[key] < previous[key];
        break;
      }
    }
    if (better) best = candidate;
  }
  selected[0] = best < 0 ? -1 : best / ranks;
  selected[1] = best < 0 ? -1 : best % ranks;
}

__global__ void apply_congestion_move_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* demand, int64_t* primary, const int64_t* offsets,
    const int64_t* occurrence_tokens, int64_t* slots, int64_t* loads,
    int64_t* traffic, const int64_t* selected, int64_t k, int ranks) {
  const int64_t expert = selected[0];
  if (expert < 0) return;
  const int target = selected[1];
  const int old = primary[expert];
  for (int64_t position = offsets[expert] + threadIdx.x;
       position < offsets[expert + 1]; position += blockDim.x) {
    const int64_t token = occurrence_tokens[position];
    int old_count = 0;
    int target_count = 0;
    for (int64_t column = 0; column < k; ++column) {
      const int rank = primary[topk[token * k + column]];
      old_count += rank == old;
      target_count += rank == target;
    }
    const int source_rank = source[token];
    if (old_count == 1 && source_rank != old)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    traffic + source_rank * ranks + old),
                static_cast<unsigned long long>(-count[token]));
    if (!target_count && source_rank != target)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    traffic + source_rank * ranks + target),
                static_cast<unsigned long long>(count[token]));
  }
  __syncthreads();
  if (!threadIdx.x) {
    int64_t expert_demand = 0;
    for (int rank = 0; rank < ranks; ++rank)
      expert_demand += demand[expert * ranks + rank];
    --slots[old];
    ++slots[target];
    loads[old] -= expert_demand;
    loads[target] += expert_demand;
    primary[expert] = target;
  }
}

}  // namespace

void affinity_primary_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor demand, torch::Tensor affinity, torch::Tensor degree,
    torch::Tensor score, torch::Tensor groups, torch::Tensor group_source,
    torch::Tensor group_to_rank, torch::Tensor primary) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.is_contiguous() && topk.is_contiguous() &&
              count.is_contiguous());
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              topk.size(0) == source.size(0) && count.size(0) == source.size(0));
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(experts >= ranks && ranks > 0 && ranks <= 128);
  TORCH_CHECK(demand.is_cuda() && demand.scalar_type() == torch::kInt64 &&
              demand.is_contiguous());
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.dim() == 2 && affinity.size(0) == experts &&
              affinity.size(1) == experts);
  TORCH_CHECK(degree.numel() == experts && score.numel() == experts &&
              groups.numel() == experts && primary.numel() == experts);
  TORCH_CHECK(group_source.dim() == 2 && group_source.size(0) == ranks &&
              group_source.size(1) == ranks &&
              group_to_rank.numel() == ranks);
  TORCH_CHECK(degree.is_cuda() && score.is_cuda() && groups.is_cuda() &&
              group_source.is_cuda() && group_to_rank.is_cuda() &&
              primary.is_cuda());
  TORCH_CHECK(degree.scalar_type() == torch::kInt64 &&
              score.scalar_type() == torch::kInt64 &&
              groups.scalar_type() == torch::kInt64 &&
              group_source.scalar_type() == torch::kInt64 &&
              group_to_rank.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);

  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  demand.zero_();
  affinity.zero_();
  degree.zero_();
  group_source.zero_();
  const int64_t tokens = source.size(0);
  launch(affinity_histogram_kernel, dim3((tokens + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
         affinity.data_ptr<int64_t>(), degree.data_ptr<int64_t>(), tokens,
         topk.size(1), experts, ranks);
  launch(affinity_groups_kernel, dim3(1), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), affinity.data_ptr<int64_t>(),
         degree.data_ptr<int64_t>(), score.data_ptr<int64_t>(),
         groups.data_ptr<int64_t>(), experts, ranks);
  launch(group_source_kernel, dim3((tokens + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks);
  launch(map_groups_kernel, dim3(1), dim3(1), stream.stream(),
         group_source.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         group_to_rank.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), experts,
         ranks);
  check_cuda(cudaGetLastError());
}

void affinity_histogram_into(torch::Tensor source, torch::Tensor topk,
                             torch::Tensor count, torch::Tensor demand,
                             torch::Tensor affinity, torch::Tensor degree) {
  demand.zero_();
  affinity.zero_();
  degree.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  launch(affinity_histogram_kernel, dim3((tokens + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
         affinity.data_ptr<int64_t>(), degree.data_ptr<int64_t>(), tokens,
         topk.size(1), demand.size(0), demand.size(1));
  check_cuda(cudaGetLastError());
}

void spectral_groups_into(torch::Tensor embedding, torch::Tensor affinity,
                          torch::Tensor centers, torch::Tensor distances,
                          torch::Tensor groups,
                          torch::Tensor next_groups, torch::Tensor sizes,
                          torch::Tensor overflow) {
  const int64_t experts = embedding.size(0);
  const int64_t ranks = embedding.size(1);
  TORCH_CHECK(embedding.is_cuda() && embedding.scalar_type() == torch::kFloat64 &&
              embedding.is_contiguous() && experts % ranks == 0);
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.size(0) == experts && affinity.size(1) == experts);
  TORCH_CHECK(centers.is_cuda() && centers.scalar_type() == torch::kFloat64 &&
              centers.numel() == ranks * ranks);
  TORCH_CHECK(distances.is_cuda() &&
              distances.scalar_type() == torch::kFloat64 &&
              distances.numel() == experts);
  TORCH_CHECK(groups.is_cuda() && groups.scalar_type() == torch::kInt64 &&
              groups.numel() == experts && next_groups.numel() == experts &&
              sizes.numel() == ranks && overflow.numel() == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(embedding.get_device());
  launch(spectral_exact_groups_kernel, dim3(1), dim3(256), stream.stream(),
         embedding.data_ptr<double>(), affinity.data_ptr<int64_t>(),
         centers.data_ptr<double>(), distances.data_ptr<double>(),
         groups.data_ptr<int64_t>(),
         next_groups.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(),
         overflow.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
}

void group_source_into(torch::Tensor source, torch::Tensor topk,
                       torch::Tensor count, torch::Tensor groups,
                       torch::Tensor group_source) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              groups.is_cuda() && group_source.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              groups.scalar_type() == torch::kInt64 &&
              group_source.scalar_type() == torch::kInt64);
  group_source.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  launch(group_source_kernel, dim3((tokens + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         group_source.data_ptr<int64_t>(), tokens, topk.size(1),
         group_source.size(0));
  check_cuda(cudaGetLastError());
}

void congestion_hungarian_into(
    torch::Tensor group_source, torch::Tensor groups, torch::Tensor allowed,
    torch::Tensor values, torch::Tensor cost, torch::Tensor work,
    torch::Tensor assignment, torch::Tensor primary) {
  const int ranks = group_source.size(0);
  TORCH_CHECK(ranks > 0 && ranks <= 128 && group_source.size(1) == ranks);
  TORCH_CHECK(group_source.is_cuda() && groups.is_cuda() && allowed.is_cuda() &&
              values.is_cuda() && cost.is_cuda() && work.is_cuda() &&
              assignment.is_cuda() && primary.is_cuda());
  TORCH_CHECK(group_source.scalar_type() == torch::kInt64 &&
              groups.scalar_type() == torch::kInt64 &&
              allowed.scalar_type() == torch::kBool &&
              values.scalar_type() == torch::kInt64 &&
              cost.scalar_type() == torch::kInt64 &&
              work.scalar_type() == torch::kInt64 &&
              assignment.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);
  auto stream = c10::cuda::getCurrentCUDAStream(group_source.get_device());
  launch(congestion_hungarian_kernel, dim3(1), dim3(1), stream.stream(),
         group_source.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         allowed.data_ptr<bool>(), values.data_ptr<int64_t>(),
         cost.data_ptr<int64_t>(), work.data_ptr<int64_t>(),
         assignment.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         groups.numel(), ranks);
  check_cuda(cudaGetLastError());
}

void refine_congestion_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor demand, torch::Tensor primary, int64_t minimum_capacity,
    int64_t maximum_capacity, double compute_limit, int64_t rounds,
    torch::Tensor occurrence_counts, torch::Tensor occurrence_offsets,
    torch::Tensor occurrence_cursors, torch::Tensor occurrence_tokens,
    torch::Tensor slots, torch::Tensor loads, torch::Tensor traffic,
    torch::Tensor candidates, torch::Tensor selected) {
  const int64_t experts = demand.size(0);
  const int ranks = demand.size(1);
  const int64_t entries = topk.numel();
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              demand.is_cuda() && primary.is_cuda());
  TORCH_CHECK(occurrence_counts.is_cuda() && occurrence_offsets.is_cuda() &&
              occurrence_cursors.is_cuda() && occurrence_tokens.is_cuda() &&
              slots.is_cuda() && loads.is_cuda() && traffic.is_cuda() &&
              candidates.is_cuda() && selected.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              demand.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);
  TORCH_CHECK(experts >= ranks && ranks > 0 && ranks <= 128 &&
              minimum_capacity >= 1 && maximum_capacity >= minimum_capacity &&
              compute_limit >= 1 && rounds >= 0);
  TORCH_CHECK(occurrence_counts.numel() == experts &&
              occurrence_offsets.numel() == experts + 1 &&
              occurrence_cursors.numel() == experts &&
              occurrence_tokens.numel() == entries && slots.numel() == ranks &&
              loads.numel() == ranks && traffic.numel() == ranks * ranks &&
              candidates.numel() == experts * ranks * 5 && selected.numel() == 2);
  TORCH_CHECK(occurrence_counts.scalar_type() == torch::kInt64 &&
              occurrence_offsets.scalar_type() == torch::kInt64 &&
              occurrence_cursors.scalar_type() == torch::kInt64 &&
              occurrence_tokens.scalar_type() == torch::kInt64 &&
              slots.scalar_type() == torch::kInt64 &&
              loads.scalar_type() == torch::kInt64 &&
              traffic.scalar_type() == torch::kInt64 &&
              candidates.scalar_type() == torch::kInt64 &&
              selected.scalar_type() == torch::kInt64);
  occurrence_counts.zero_();
  slots.zero_();
  loads.zero_();
  traffic.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  launch(occurrence_count_kernel, dim3((entries + 255) / 256), dim3(256),
         stream.stream(), topk.data_ptr<int64_t>(),
         occurrence_counts.data_ptr<int64_t>(), entries);
  launch(occurrence_prefix_kernel, dim3(1), dim3(1), stream.stream(),
         occurrence_counts.data_ptr<int64_t>(),
         occurrence_offsets.data_ptr<int64_t>(),
         occurrence_cursors.data_ptr<int64_t>(), experts);
  launch(occurrence_fill_kernel, dim3((entries + 255) / 256), dim3(256),
         stream.stream(), topk.data_ptr<int64_t>(),
         occurrence_offsets.data_ptr<int64_t>(),
         occurrence_cursors.data_ptr<int64_t>(),
         occurrence_tokens.data_ptr<int64_t>(), entries, topk.size(1));
  const int64_t state_items = source.numel() > experts ? source.numel() : experts;
  launch(refinement_state_kernel, dim3((state_items + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
         primary.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
         loads.data_ptr<int64_t>(), traffic.data_ptr<int64_t>(), source.numel(),
         topk.size(1), experts, ranks);
  for (int64_t round = 0; round < rounds; ++round) {
    launch(congestion_move_candidates_kernel, dim3(experts * ranks), dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), occurrence_offsets.data_ptr<int64_t>(),
           occurrence_tokens.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
           loads.data_ptr<int64_t>(), traffic.data_ptr<int64_t>(),
           candidates.data_ptr<int64_t>(), topk.size(1), experts, ranks,
           minimum_capacity, maximum_capacity, compute_limit);
    launch(select_congestion_move_kernel, dim3(1), dim3(1), stream.stream(),
           traffic.data_ptr<int64_t>(), candidates.data_ptr<int64_t>(),
           selected.data_ptr<int64_t>(), experts, ranks);
    launch(apply_congestion_move_kernel, dim3(1), dim3(256), stream.stream(),
           source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
           primary.data_ptr<int64_t>(), occurrence_offsets.data_ptr<int64_t>(),
           occurrence_tokens.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
           loads.data_ptr<int64_t>(), traffic.data_ptr<int64_t>(),
           selected.data_ptr<int64_t>(), topk.size(1), ranks);
  }
  check_cuda(cudaGetLastError());
}

}  // namespace grace_cuda
