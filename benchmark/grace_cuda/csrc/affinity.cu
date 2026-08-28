#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cfloat>
#include <climits>

#include "launch.cuh"
#include "limits.cuh"

namespace grace_cuda {
namespace {

__global__ void affinity_histogram_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    int64_t* demand, int64_t* affinity, int64_t* degree, int64_t tokens,
    int64_t k, int64_t experts, int64_t ranks, bool clear_output) {
  if (clear_output) {
    for (int64_t index = threadIdx.x; index < experts * ranks;
         index += blockDim.x)
      demand[index] = 0;
    for (int64_t index = threadIdx.x; index < experts * experts;
         index += blockDim.x)
      affinity[index] = 0;
    __syncthreads();
  }
  for (int64_t token =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t weight = count[token];
    for (int64_t left = 0; left < k; ++left) {
      const int64_t a = topk[token * k + left];
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    demand + a * ranks + source[token]),
                static_cast<unsigned long long>(weight));
      for (int64_t right = left + 1; right < k; ++right) {
        const int64_t b = topk[token * k + right];
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      affinity + a * experts + b),
                  static_cast<unsigned long long>(weight));
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      affinity + b * experts + a),
                  static_cast<unsigned long long>(weight));
      }
    }
  }
}

__global__ void affinity_degree_kernel(const int64_t* affinity,
                                       int64_t* degree, int64_t experts) {
  for (int64_t expert =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       expert < experts;
       expert += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t total = 0;
    for (int64_t other = 0; other < experts; ++other)
      total += affinity[expert * experts + other];
    degree[expert] = total;
  }
}

__global__ void affinity_scale_kernel(const int64_t* degree, double* scale,
                                      int64_t experts) {
  const int64_t expert =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (expert >= experts) return;
  scale[expert] = degree[expert]
                      ? 1.0 / sqrt(static_cast<double>(degree[expert]))
                      : 0.0;
}

__global__ void normalize_affinity_kernel(const int64_t* affinity,
                                          const double* scale,
                                          double* normalized,
                                          int64_t elements,
                                          int64_t experts) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  normalized[index] = static_cast<double>(affinity[index]) *
                      scale[index / experts] * scale[index % experts];
}

__global__ void affinity_subspace_kernel(
    const int64_t* affinity, const int64_t* degree, const double* initial,
    double* embedding, int64_t experts, int iterations) {
  if (blockIdx.x) return;
  __shared__ double vectors[256][4];
  __shared__ double next[256][4];
  __shared__ double scale[256];
  __shared__ double reduction[256];

  const int expert = threadIdx.x;
  if (expert < experts) {
    scale[expert] = degree[expert]
                        ? rsqrt(static_cast<double>(degree[expert]))
                        : 0.0;
    for (int column = 0; column < 4; ++column)
      vectors[expert][column] = initial[expert * 4 + column];
  }
  __syncthreads();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (expert < experts)
      for (int column = 0; column < 4; ++column) {
        double value = 0.0;
        for (int64_t other = 0; other < experts; ++other)
          value += static_cast<double>(affinity[expert * experts + other]) *
                   scale[other] * vectors[other][column];
        next[expert][column] = value * scale[expert];
      }
    __syncthreads();

    for (int column = 0; column < 4; ++column) {
      for (int previous = 0; previous < column; ++previous) {
        reduction[expert] =
            expert < experts
                ? next[expert][column] * next[expert][previous]
                : 0.0;
        __syncthreads();
        for (int offset = 128; offset; offset >>= 1) {
          if (expert < offset)
            reduction[expert] += reduction[expert + offset];
          __syncthreads();
        }
        if (expert < experts)
          next[expert][column] -=
              reduction[0] * next[expert][previous];
        __syncthreads();
      }
      reduction[expert] =
          expert < experts ? next[expert][column] * next[expert][column] : 0.0;
      __syncthreads();
      for (int offset = 128; offset; offset >>= 1) {
        if (expert < offset)
          reduction[expert] += reduction[expert + offset];
        __syncthreads();
      }
      const double inverse_norm =
          reduction[0] > 0.0 ? rsqrt(reduction[0]) : 0.0;
      if (expert < experts)
        next[expert][column] *= inverse_norm;
      __syncthreads();
    }
    if (expert < experts)
      for (int column = 0; column < 4; ++column)
        vectors[expert][column] = next[expert][column];
    __syncthreads();
  }

  if (expert < experts) {
    double norm = 0.0;
    for (int column = 0; column < 4; ++column)
      norm += vectors[expert][column] * vectors[expert][column];
    const double inverse_norm = norm > 1e-24 ? rsqrt(norm) : 0.0;
    for (int column = 0; column < 4; ++column)
      embedding[expert * 4 + column] =
          vectors[expert][column] * inverse_norm;
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
    int ranks, bool clear_output) {
  if (clear_output) {
    for (int index = threadIdx.x; index < ranks * ranks;
         index += blockDim.x)
      group_source[index] = 0;
    __syncthreads();
  }
  for (int64_t token =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
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
}

template <int MaxRanks, int FixedK = 0>
__global__ void group_source_shared_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* groups, int64_t* group_source, int64_t tokens, int64_t k,
    int ranks, bool merge_output) {
  __shared__ int64_t local[MaxRanks * MaxRanks];
  const int bins = ranks * ranks;
  for (int index = threadIdx.x; index < bins; index += blockDim.x)
    local[index] = 0;
  __syncthreads();
  const int64_t actual_k = FixedK ? FixedK : k;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t first_token =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t iterations = tokens ? (tokens - 1) / stride + 1 : 0;
  // Keep every lane in the warp on the same control-flow path. In
  // particular, the last partial token batch must still execute all warp
  // intrinsics; invalid lanes contribute the sentinel key and zero weight.
  for (int64_t iteration = 0; iteration < iterations; ++iteration) {
    const int64_t token = first_token + iteration * stride;
    const bool valid = token < tokens;
    unsigned long long seen = 0;
    const int64_t src = valid ? source[token] : 0;
    const int64_t weight = valid ? count[token] : 0;
#pragma unroll
    for (int64_t column = 0; column < actual_k; ++column) {
      const int64_t group =
          valid ? groups[topk[token * actual_k + column]] : 0;
      const auto bit = 1ULL << group;
      const bool emit = valid && !(seen & bit);
      if (emit) seen |= bit;
      // Every lane participates in the warp intrinsics. Non-emitting lanes
      // use a sentinel key and contribute zero, avoiding divergent sync
      // behavior while retaining one atomic per real (group, source) key.
      const int key = emit ? static_cast<int>(group * MaxRanks + src) : -1;
      constexpr unsigned full_mask = 0xffffffffU;
      const unsigned peers = __match_any_sync(full_mask, key);
      const int leader = __ffs(peers) - 1;
      int64_t aggregate = 0;
      // Use a fixed 32-step exchange. A peer-list loop would execute a
      // different number of __shfl calls for different keys in the same warp,
      // which can deadlock on architectures requiring all lanes in the mask
      // to reach each intrinsic together. The peer mask already identifies
      // matching keys, so only the weight needs to be shuffled.
#pragma unroll
      for (int peer = 0; peer < 32; ++peer) {
        const unsigned peer_bit = 1U << peer;
        const int source_lane = (peers & peer_bit) ? peer : (threadIdx.x & 31);
        const int64_t peer_weight = __shfl_sync(full_mask, weight, source_lane);
        if ((threadIdx.x & 31) == leader && (peers & peer_bit))
          aggregate += peer_weight;
      }
      if (emit && (threadIdx.x & 31) == leader)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      local + group * ranks + src),
                  static_cast<unsigned long long>(aggregate));
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < bins; index += blockDim.x) {
    if (merge_output)
      atomicAdd(reinterpret_cast<unsigned long long*>(group_source + index),
                static_cast<unsigned long long>(local[index]));
    else
      group_source[index] = local[index];
  }
}

template <int MaxRanks>
void launch_group_source_shared(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* groups, int64_t* group_source, int64_t tokens, int64_t k,
    int ranks, bool merge_output, int blocks, cudaStream_t stream) {
  switch (k) {
    case 1:
      launch(group_source_shared_kernel<MaxRanks, 1>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
    case 2:
      launch(group_source_shared_kernel<MaxRanks, 2>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
    case 4:
      launch(group_source_shared_kernel<MaxRanks, 4>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
    case 8:
      launch(group_source_shared_kernel<MaxRanks, 8>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
    case 16:
      launch(group_source_shared_kernel<MaxRanks, 16>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
    default:
      launch(group_source_shared_kernel<MaxRanks, 0>, dim3(blocks), dim3(256),
             stream, source, topk, count, groups, group_source, tokens, k,
             ranks, merge_output);
      break;
  }
}

__global__ void map_groups_kernel(const int64_t* group_source,
                                  const int64_t* groups,
                                  int64_t* group_to_rank, int64_t* primary,
                                  int64_t experts, int ranks) {
  if (blockIdx.x) return;
  __shared__ bool used_group[kMaxEpSize];
  __shared__ bool used_rank[kMaxEpSize];
  __shared__ int64_t egress[kMaxEpSize];
  __shared__ int64_t group_totals[kMaxEpSize];
  __shared__ int64_t group_peaks[kMaxEpSize];
  __shared__ int64_t bottlenecks[kMaxEpSize];
  __shared__ int64_t pairs[kMaxEpSize];
  __shared__ int64_t remotes[kMaxEpSize];
  __shared__ int group_order[kMaxEpSize];
  __shared__ int selected_rank;
  __shared__ int64_t max_ingress;
  __shared__ int64_t max_pair;

  for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x) {
    used_group[rank] = false;
    used_rank[rank] = false;
    egress[rank] = 0;
  }
  if (threadIdx.x == 0) max_ingress = max_pair = 0;
  __syncthreads();

  for (int candidate = threadIdx.x; candidate < ranks;
       candidate += blockDim.x) {
    int64_t total = 0;
    int64_t peak = 0;
    for (int source = 0; source < ranks; ++source) {
      const int64_t value = group_source[candidate * ranks + source];
      total += value;
      if (value > peak) peak = value;
    }
    group_totals[candidate] = total;
    group_peaks[candidate] = peak;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int step = 0; step < ranks; ++step) {
      int selected_group = -1;
      int64_t selected_total = -1;
      int64_t selected_peak = -1;
      for (int candidate = 0; candidate < ranks; ++candidate) {
        if (used_group[candidate]) continue;
        const int64_t total = group_totals[candidate];
        const int64_t peak = group_peaks[candidate];
        if (total > selected_total ||
            (total == selected_total &&
             (peak > selected_peak ||
              (peak == selected_peak &&
               (selected_group < 0 || candidate < selected_group))))) {
          selected_group = candidate;
          selected_total = total;
          selected_peak = peak;
        }
      }
      group_order[step] = selected_group;
      used_group[selected_group] = true;
    }
  __syncthreads();

  for (int step = 0; step < ranks; ++step) {
    const int selected_group = group_order[step];
    const int64_t selected_total = group_totals[selected_group];
    for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x) {
      if (used_rank[rank]) {
        bottlenecks[rank] = pairs[rank] = remotes[rank] = LLONG_MAX;
        continue;
      }
      const int64_t remote = selected_total -
                             group_source[selected_group * ranks + rank];
      int64_t projected_egress = 0;
      int64_t projected_pair = max_pair;
      for (int source = 0; source < ranks; ++source) {
        const int64_t added = source == rank
                                  ? 0
                                  : group_source[selected_group * ranks + source];
        const int64_t value = egress[source] + added;
        if (value > projected_egress) projected_egress = value;
        if (added > projected_pair) projected_pair = added;
      }
      const int64_t ingress = remote > max_ingress ? remote : max_ingress;
      bottlenecks[rank] =
          ingress > projected_egress ? ingress : projected_egress;
      pairs[rank] = projected_pair;
      remotes[rank] = remote;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      selected_rank = -1;
      int64_t best_bottleneck = LLONG_MAX;
      int64_t best_pair = LLONG_MAX;
      int64_t best_remote = LLONG_MAX;
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t bottleneck = bottlenecks[rank];
        const int64_t pair = pairs[rank];
        const int64_t remote = remotes[rank];
        if (bottleneck < best_bottleneck ||
            (bottleneck == best_bottleneck &&
             (pair < best_pair ||
              (pair == best_pair &&
               (remote < best_remote ||
                (remote == best_remote &&
                 (selected_rank < 0 || rank < selected_rank))))))) {
          selected_rank = rank;
          best_bottleneck = bottleneck;
          best_pair = pair;
          best_remote = remote;
        }
      }
      used_rank[selected_rank] = true;
      group_to_rank[selected_group] = selected_rank;
      if (best_remote > max_ingress) max_ingress = best_remote;
      for (int source = 0; source < ranks; ++source)
        if (source != selected_rank) {
          const int64_t added =
              group_source[selected_group * ranks + source];
          egress[source] += added;
          if (added > max_pair) max_pair = added;
        }
    }
    __syncthreads();
  }
  for (int64_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x)
    primary[expert] = group_to_rank[groups[expert]];
}

__global__ void spectral_exact_groups_kernel(
    const double* embedding, const int64_t* affinity, double* centers,
    double* distances,
    int64_t* labels, int64_t* next_labels, int64_t* sizes, int64_t* overflow,
    int64_t experts, int ranks, int dimensions) {
  if (blockIdx.x) return;
  __shared__ int unchanged;
  __shared__ int64_t selected_item;
  __shared__ int overflow_count;
  __shared__ int chosen_group;
  __shared__ double center_best_dist[256];
  __shared__ int64_t center_best_index[256];
  for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x)
    labels[expert] = 0;
  if (threadIdx.x == 0) overflow[0] = 0;
  __syncthreads();

  for (int64_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    double distance = 0;
    for (int dim = 0; dim < dimensions; ++dim) {
      const double delta = embedding[expert * dimensions + dim] -
                           embedding[dim];
      distance += delta * delta;
    }
    distances[expert] = expert ? distance : -1.0;
  }
  __syncthreads();
  for (int center_count = 1; center_count < ranks; ++center_count) {
    double best_distance = -1.0;
    int64_t best_expert = -1;
    for (int64_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      const double value = distances[expert];
      if (value > best_distance ||
          (value == best_distance &&
           (best_expert < 0 || expert < best_expert))) {
        best_distance = value;
        best_expert = expert;
      }
    }
    center_best_dist[threadIdx.x] = best_distance;
    center_best_index[threadIdx.x] = best_expert;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
      if (threadIdx.x < offset) {
        const double candidate_distance = center_best_dist[threadIdx.x + offset];
        const int64_t candidate_expert = center_best_index[threadIdx.x + offset];
        if (candidate_distance > center_best_dist[threadIdx.x] ||
            (candidate_distance == center_best_dist[threadIdx.x] &&
             candidate_expert >= 0 &&
             (center_best_index[threadIdx.x] < 0 ||
              candidate_expert < center_best_index[threadIdx.x]))) {
          center_best_dist[threadIdx.x] = candidate_distance;
          center_best_index[threadIdx.x] = candidate_expert;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) overflow[center_count] = center_best_index[0];
    __syncthreads();
    const int64_t center_expert = overflow[center_count];
    for (int64_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      if (expert == center_expert) {
        distances[expert] = -1.0;
        continue;
      }
      double distance = 0;
      for (int dim = 0; dim < dimensions; ++dim) {
        const double delta = embedding[expert * dimensions + dim] -
                             embedding[center_expert * dimensions + dim];
        distance += delta * delta;
      }
      distances[expert] = min(distances[expert], distance);
    }
    __syncthreads();
  }
  for (int64_t index = threadIdx.x; index < ranks * dimensions;
       index += blockDim.x) {
    const int center = index / dimensions;
    const int dim = index % dimensions;
    centers[index] = embedding[overflow[center] * dimensions + dim];
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
        for (int dim = 0; dim < dimensions; ++dim) {
          const double delta = embedding[expert * dimensions + dim] -
                               centers[group * dimensions + dim];
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
          for (int dim = 0; dim < dimensions; ++dim) {
            const double delta = embedding[expert * dimensions + dim] -
                                 centers[donor * dimensions + dim];
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
    for (int64_t index = threadIdx.x; index < ranks * dimensions;
         index += blockDim.x) {
      const int group = index / dimensions;
      const int dim = index % dimensions;
      double total = 0;
      for (int64_t expert = 0; expert < experts; ++expert)
        if (next_labels[expert] == group)
          total += embedding[expert * dimensions + dim];
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


template <int FixedExperts, int FixedRanks>
__global__ void affinity_swaps_kernel(
    const int64_t* affinity, int64_t* labels, int64_t* group_affinity,
    int64_t* gains, int64_t runtime_experts, int runtime_ranks) {
  if (blockIdx.x) return;
  const int64_t experts = FixedExperts ? FixedExperts : runtime_experts;
  const int ranks = FixedRanks ? FixedRanks : runtime_ranks;
  __shared__ int64_t best_gains[1024];
  __shared__ int64_t best_indices[1024];
  __shared__ int64_t selected;
  __shared__ int selected_left_group;
  __shared__ int selected_right_group;
  __shared__ int member_counts[4];
  __shared__ int member_positions[256];
  __shared__ int64_t group_members[256];
  if constexpr (FixedExperts == 256 && FixedRanks == 4) {
    if (threadIdx.x < 4) member_counts[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x < 256) {
      const int group = labels[threadIdx.x];
      const int position = atomicAdd(member_counts + group, 1);
      group_members[group * 64 + position] = threadIdx.x;
      member_positions[threadIdx.x] = position;
    }
    __syncthreads();
  }
  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int group = index % ranks;
    int64_t total = 0;
    for (int64_t other = 0; other < experts; ++other)
      if (labels[other] == group)
        total += affinity[expert * experts + other];
    group_affinity[index] = total;
  }
  __syncthreads();

  for (int round = 0; round < 8; ++round) {
    int64_t best_gain = 0;
    int64_t best_index = -1;
    const int64_t candidate_count =
        FixedExperts == 256 && FixedRanks == 4 ? 6 * 64 * 64
                                               : experts * experts;
    for (int64_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      int64_t left_expert;
      int64_t right_expert;
      if constexpr (FixedExperts == 256 && FixedRanks == 4) {
        const int pair_index = candidate >> 12;
        const int position = candidate & 4095;
        const int left_group =
            pair_index < 3 ? 0 : pair_index < 5 ? 1 : 2;
        const int right_group =
            pair_index < 3 ? pair_index + 1
                           : pair_index < 5 ? pair_index - 1 : 3;
        left_expert =
            group_members[left_group * 64 + (position >> 6)];
        right_expert =
            group_members[right_group * 64 + (position & 63)];
      } else {
        left_expert = candidate / experts;
        right_expert = candidate % experts;
        if (labels[left_expert] >= labels[right_expert]) continue;
      }
      const int left = labels[left_expert];
      const int right = labels[right_expert];
      const int64_t index = left_expert * experts + right_expert;
      const int64_t pair = affinity[index];
      const int64_t gain =
          group_affinity[left_expert * ranks + right] - pair +
          group_affinity[right_expert * ranks + left] - pair -
          group_affinity[left_expert * ranks + left] -
          group_affinity[right_expert * ranks + right];
      if (gain > best_gain ||
          (gain == best_gain && gain > 0 && index > best_index)) {
        best_gain = gain;
        best_index = index;
      }
    }
    best_gains[threadIdx.x] = best_gain;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
      if (threadIdx.x < offset &&
          (best_gains[threadIdx.x + offset] > best_gains[threadIdx.x] ||
           (best_gains[threadIdx.x + offset] == best_gains[threadIdx.x] &&
            best_gains[threadIdx.x + offset] > 0 &&
            best_indices[threadIdx.x + offset] >
                best_indices[threadIdx.x]))) {
        best_gains[threadIdx.x] = best_gains[threadIdx.x + offset];
        best_indices[threadIdx.x] = best_indices[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected = best_indices[0];
      if (selected >= 0) {
        const int64_t left = selected / experts;
        const int64_t right = selected % experts;
        const int64_t group = labels[left];
        selected_left_group = group;
        selected_right_group = labels[right];
        if constexpr (FixedExperts == 256 && FixedRanks == 4) {
          const int left_position = member_positions[left];
          const int right_position = member_positions[right];
          group_members[selected_left_group * 64 + left_position] = right;
          group_members[selected_right_group * 64 + right_position] = left;
          member_positions[right] = left_position;
          member_positions[left] = right_position;
        }
        labels[left] = labels[right];
        labels[right] = group;
      }
    }
    __syncthreads();
    if (selected < 0) break;
    const int64_t left = selected / experts;
    const int64_t right = selected % experts;
    for (int64_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      const int64_t left_affinity = affinity[expert * experts + left];
      const int64_t right_affinity = affinity[expert * experts + right];
      group_affinity[expert * ranks + selected_left_group] +=
          right_affinity - left_affinity;
      group_affinity[expert * ranks + selected_right_group] +=
          left_affinity - right_affinity;
    }
    __syncthreads();
  }
}

__device__ bool better_group_swap(int64_t peak, int64_t squared,
                                  int64_t affinity_gain, int64_t index,
                                  int64_t best_peak, int64_t best_squared,
                                  int64_t best_affinity_gain,
                                  int64_t best_index) {
  return index >= 0 &&
         (best_index < 0 || peak < best_peak ||
          (peak == best_peak &&
           (squared < best_squared ||
            (squared == best_squared &&
             (affinity_gain > best_affinity_gain ||
              (affinity_gain == best_affinity_gain && index < best_index))))));
}

// ponytail: exhaustive O(32 * E^2) swaps are adequate for current <=256
// experts; use a candidate shortlist only if this becomes measurable.
template <int FixedExperts, int FixedRanks>
__global__ void balance_affinity_groups_kernel(
    const int64_t* demand, const int64_t* affinity, int64_t* labels,
    int64_t* expert_loads, int64_t* group_loads, int64_t* group_affinity,
    int64_t runtime_experts, int runtime_ranks) {
  if (blockIdx.x) return;
  const int64_t experts = FixedExperts ? FixedExperts : runtime_experts;
  const int ranks = FixedRanks ? FixedRanks : runtime_ranks;
  __shared__ int64_t best_peaks[1024];
  __shared__ int64_t best_squared[1024];
  __shared__ int64_t best_affinity_gains[1024];
  __shared__ int64_t best_indices[1024];
  __shared__ int64_t current_peak;
  __shared__ int64_t current_squared;
  __shared__ int64_t peak_values[3];
  __shared__ int peak_groups[3];
  __shared__ int64_t selected;
  __shared__ int selected_left_group;
  __shared__ int selected_right_group;
  __shared__ int member_counts[4];
  __shared__ int member_positions[256];
  __shared__ int64_t group_members[256];

  for (int rank = threadIdx.x; rank < ranks; rank += blockDim.x)
    group_loads[rank] = 0;
  __syncthreads();
  for (int64_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    int64_t load = 0;
    for (int rank = 0; rank < ranks; ++rank)
      load += demand[expert * ranks + rank];
    expert_loads[expert] = load;
    atomicAdd(reinterpret_cast<unsigned long long*>(group_loads + labels[expert]),
              static_cast<unsigned long long>(load));
  }
  __syncthreads();
  if constexpr (FixedExperts == 256 && FixedRanks == 4) {
    if (threadIdx.x < 4) member_counts[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x < 256) {
      const int group = labels[threadIdx.x];
      const int position = atomicAdd(member_counts + group, 1);
      group_members[group * 64 + position] = threadIdx.x;
      member_positions[threadIdx.x] = position;
    }
    __syncthreads();
  }

  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x) {
    const int64_t expert = index / ranks;
    const int group = index % ranks;
    int64_t total = 0;
    for (int64_t other = 0; other < experts; ++other)
      if (labels[other] == group)
        total += affinity[expert * experts + other];
    group_affinity[index] = total;
  }
  __syncthreads();

  const int max_rounds = ranks > 16 ? 16 : 32;
  for (int round = 0; round < max_rounds; ++round) {
    if (threadIdx.x == 0) {
      current_peak = 0;
      current_squared = 0;
      for (int slot = 0; slot < 3; ++slot) {
        peak_values[slot] = -1;
        peak_groups[slot] = -1;
      }
      for (int rank = 0; rank < ranks; ++rank) {
        current_peak = max(current_peak, group_loads[rank]);
        current_squared += group_loads[rank] * group_loads[rank];
        for (int slot = 0; slot < 3; ++slot)
          if (group_loads[rank] > peak_values[slot]) {
            for (int tail = 2; tail > slot; --tail) {
              peak_values[tail] = peak_values[tail - 1];
              peak_groups[tail] = peak_groups[tail - 1];
            }
            peak_values[slot] = group_loads[rank];
            peak_groups[slot] = rank;
            break;
          }
      }
    }
    __syncthreads();

    int64_t best_peak = LLONG_MAX;
    int64_t best_square = LLONG_MAX;
    int64_t best_gain = LLONG_MIN;
    int64_t best_index = -1;
    const int64_t candidate_count =
        FixedExperts == 256 && FixedRanks == 4 ? 6 * 64 * 64
                                               : experts * experts;
    for (int64_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      int64_t left_expert;
      int64_t right_expert;
      if constexpr (FixedExperts == 256 && FixedRanks == 4) {
        const int pair = candidate >> 12;
        const int position = candidate & 4095;
        const int left_group =
            pair < 3 ? 0 : pair < 5 ? 1 : 2;
        const int right_group =
            pair < 3 ? pair + 1 : pair < 5 ? pair - 1 : 3;
        left_expert =
            group_members[left_group * 64 + (position >> 6)];
        right_expert =
            group_members[right_group * 64 + (position & 63)];
        if (left_expert > right_expert) {
          const int64_t expert = left_expert;
          left_expert = right_expert;
          right_expert = expert;
        }
      } else {
        left_expert = candidate / experts;
        right_expert = candidate % experts;
        if (left_expert >= right_expert) continue;
      }
      const int64_t index = left_expert * experts + right_expert;
      const int left = labels[left_expert];
      const int right = labels[right_expert];
      if (left == right) continue;
      const int64_t next_left = group_loads[left] - expert_loads[left_expert] +
                                expert_loads[right_expert];
      const int64_t next_right = group_loads[right] - expert_loads[right_expert] +
                                 expert_loads[left_expert];
      int64_t peak = max(next_left, next_right);
      for (int slot = 0; slot < 3; ++slot)
        if (peak_groups[slot] != left && peak_groups[slot] != right)
          peak = max(peak, peak_values[slot]);
      const int64_t squared =
          current_squared - group_loads[left] * group_loads[left] -
          group_loads[right] * group_loads[right] + next_left * next_left +
          next_right * next_right;
      const int64_t pair = affinity[index];
      const int64_t gain =
          group_affinity[left_expert * ranks + right] - pair +
          group_affinity[right_expert * ranks + left] - pair -
          group_affinity[left_expert * ranks + left] -
          group_affinity[right_expert * ranks + right];
      const bool improves =
          peak < current_peak ||
          (peak == current_peak &&
           (squared < current_squared ||
            (squared == current_squared && gain > 0)));
      if (improves &&
          better_group_swap(peak, squared, gain, index, best_peak, best_square,
                            best_gain, best_index)) {
        best_peak = peak;
        best_square = squared;
        best_gain = gain;
        best_index = index;
      }
    }
    best_peaks[threadIdx.x] = best_peak;
    best_squared[threadIdx.x] = best_square;
    best_affinity_gains[threadIdx.x] = best_gain;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset >>= 1) {
      if (threadIdx.x < offset &&
          better_group_swap(
              best_peaks[threadIdx.x + offset],
              best_squared[threadIdx.x + offset],
              best_affinity_gains[threadIdx.x + offset],
              best_indices[threadIdx.x + offset], best_peaks[threadIdx.x],
              best_squared[threadIdx.x],
              best_affinity_gains[threadIdx.x], best_indices[threadIdx.x])) {
        best_peaks[threadIdx.x] = best_peaks[threadIdx.x + offset];
        best_squared[threadIdx.x] = best_squared[threadIdx.x + offset];
        best_affinity_gains[threadIdx.x] =
            best_affinity_gains[threadIdx.x + offset];
        best_indices[threadIdx.x] = best_indices[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected = best_indices[0];
      if (selected >= 0) {
        const int64_t left_expert = selected / experts;
        const int64_t right_expert = selected % experts;
        const int left = labels[left_expert];
        const int right = labels[right_expert];
        selected_left_group = left;
        selected_right_group = right;
        if constexpr (FixedExperts == 256 && FixedRanks == 4) {
          const int left_position = member_positions[left_expert];
          const int right_position = member_positions[right_expert];
          group_members[left * 64 + left_position] = right_expert;
          group_members[right * 64 + right_position] = left_expert;
          member_positions[right_expert] = left_position;
          member_positions[left_expert] = right_position;
        }
        group_loads[left] +=
            expert_loads[right_expert] - expert_loads[left_expert];
        group_loads[right] +=
            expert_loads[left_expert] - expert_loads[right_expert];
        labels[left_expert] = right;
        labels[right_expert] = left;
      }
    }
    __syncthreads();
    if (selected < 0) break;
    const int64_t left_expert = selected / experts;
    const int64_t right_expert = selected % experts;
    for (int64_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      const int64_t left_affinity =
          affinity[expert * experts + left_expert];
      const int64_t right_affinity =
          affinity[expert * experts + right_expert];
      group_affinity[expert * ranks + selected_left_group] +=
          right_affinity - left_affinity;
      group_affinity[expert * ranks + selected_right_group] +=
          left_affinity - right_affinity;
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
  TORCH_CHECK(experts >= ranks && ranks > 0 && ranks <= 64);
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
  const int64_t tokens = source.size(0);
  launch(affinity_histogram_kernel, dim3(1), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
         affinity.data_ptr<int64_t>(), degree.data_ptr<int64_t>(), tokens,
         topk.size(1), experts, ranks, true);
  launch(affinity_degree_kernel, dim3(1), dim3(256),
         stream.stream(), affinity.data_ptr<int64_t>(),
         degree.data_ptr<int64_t>(), experts);
  launch(affinity_groups_kernel, dim3(1), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), affinity.data_ptr<int64_t>(),
         degree.data_ptr<int64_t>(), score.data_ptr<int64_t>(),
         groups.data_ptr<int64_t>(), experts, ranks);
  if (ranks <= 16)
    launch_group_source_shared<16>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks, false, 1,
        stream.stream());
  else if (ranks <= 32)
    launch_group_source_shared<32>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks, false, 1,
        stream.stream());
  else if (ranks <= 64)
    launch_group_source_shared<64>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks, false, 1,
        stream.stream());
  else
    launch(group_source_kernel, dim3(1), dim3(256), stream.stream(),
           source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
           group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks, true);
  launch(map_groups_kernel, dim3(1), dim3(1), stream.stream(),
         group_source.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         group_to_rank.data_ptr<int64_t>(), primary.data_ptr<int64_t>(), experts,
         ranks);
  check_cuda(cudaGetLastError());
}

void affinity_histogram_into(torch::Tensor source, torch::Tensor topk,
                             torch::Tensor count, torch::Tensor demand,
                             torch::Tensor affinity, torch::Tensor degree,
                             int64_t solver_sms) {
  TORCH_CHECK(solver_sms > 0, "solver_sms must be positive");
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  if (solver_sms > 1) {
    check_cuda(cudaMemsetAsync(demand.data_ptr<int64_t>(), 0,
                               demand.numel() * sizeof(int64_t), stream.stream()));
    check_cuda(cudaMemsetAsync(affinity.data_ptr<int64_t>(), 0,
                               affinity.numel() * sizeof(int64_t), stream.stream()));
  }
  launch(affinity_histogram_kernel, dim3(solver_sms), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(),
         affinity.data_ptr<int64_t>(), degree.data_ptr<int64_t>(), tokens,
         topk.size(1), demand.size(0), demand.size(1), solver_sms == 1);
  const int64_t experts = demand.size(0);
  launch(affinity_degree_kernel, dim3(solver_sms), dim3(256),
         stream.stream(), affinity.data_ptr<int64_t>(),
         degree.data_ptr<int64_t>(), experts);
  check_cuda(cudaGetLastError());
}

void normalize_affinity_into(torch::Tensor affinity, torch::Tensor degree,
                             torch::Tensor scale, torch::Tensor normalized) {
  const int64_t experts = degree.numel();
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.is_contiguous() && affinity.dim() == 2 &&
              affinity.size(0) == experts && affinity.size(1) == experts);
  TORCH_CHECK(degree.is_cuda() && degree.scalar_type() == torch::kInt64 &&
              degree.is_contiguous());
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat64 &&
              scale.is_contiguous() && scale.numel() == experts);
  TORCH_CHECK(normalized.is_cuda() &&
              normalized.scalar_type() == torch::kFloat64 &&
              normalized.is_contiguous() && normalized.sizes() == affinity.sizes());
  auto stream = c10::cuda::getCurrentCUDAStream(affinity.get_device());
  launch(affinity_scale_kernel, dim3((experts + 255) / 256), dim3(256),
         stream.stream(), degree.data_ptr<int64_t>(), scale.data_ptr<double>(),
         experts);
  const int64_t elements = affinity.numel();
  launch(normalize_affinity_kernel, dim3((elements + 255) / 256), dim3(256),
         stream.stream(), affinity.data_ptr<int64_t>(), scale.data_ptr<double>(),
         normalized.data_ptr<double>(), elements, experts);
  check_cuda(cudaGetLastError());
}

void affinity_subspace_into(torch::Tensor affinity, torch::Tensor degree,
                            torch::Tensor initial, torch::Tensor embedding,
                            int64_t iterations) {
  const int64_t experts = degree.numel();
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.is_contiguous() && affinity.size(0) == experts &&
              affinity.size(1) == experts && experts <= 256);
  TORCH_CHECK(degree.is_cuda() && degree.scalar_type() == torch::kInt64 &&
              degree.is_contiguous());
  TORCH_CHECK(initial.is_cuda() && initial.scalar_type() == torch::kFloat64 &&
              initial.is_contiguous() && initial.size(0) == experts &&
              initial.size(1) == 4);
  TORCH_CHECK(embedding.is_cuda() &&
              embedding.scalar_type() == torch::kFloat64 &&
              embedding.is_contiguous() && embedding.sizes() == initial.sizes());
  TORCH_CHECK(iterations > 0);
  auto stream = c10::cuda::getCurrentCUDAStream(affinity.get_device());
  launch(affinity_subspace_kernel, dim3(1), dim3(256), stream.stream(),
         affinity.data_ptr<int64_t>(), degree.data_ptr<int64_t>(),
         initial.data_ptr<double>(), embedding.data_ptr<double>(), experts,
         static_cast<int>(iterations));
  check_cuda(cudaGetLastError());
}

void spectral_groups_into(torch::Tensor embedding, torch::Tensor affinity,
                          torch::Tensor centers, torch::Tensor distances,
                          torch::Tensor groups,
                          torch::Tensor next_groups, torch::Tensor sizes,
                          torch::Tensor overflow, torch::Tensor group_affinity,
                          torch::Tensor swap_gains) {
  const int64_t experts = embedding.size(0);
  const int dimensions = embedding.size(1);
  const int ranks = sizes.numel();
  TORCH_CHECK(embedding.is_cuda() && embedding.scalar_type() == torch::kFloat64 &&
              embedding.is_contiguous() && dimensions > 0 && ranks > 0 &&
              ranks <= kMaxEpSize &&
              experts % ranks == 0);
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.size(0) == experts && affinity.size(1) == experts);
  TORCH_CHECK(centers.is_cuda() && centers.scalar_type() == torch::kFloat64 &&
              centers.numel() == ranks * dimensions);
  TORCH_CHECK(distances.is_cuda() &&
              distances.scalar_type() == torch::kFloat64 &&
              distances.numel() == experts);
  TORCH_CHECK(groups.is_cuda() && groups.scalar_type() == torch::kInt64 &&
              groups.numel() == experts && next_groups.numel() == experts &&
              sizes.numel() == ranks && overflow.numel() == experts);
  TORCH_CHECK(group_affinity.is_cuda() &&
              group_affinity.scalar_type() == torch::kInt64 &&
              group_affinity.numel() == experts * ranks &&
              swap_gains.is_cuda() &&
              swap_gains.scalar_type() == torch::kInt64 &&
              swap_gains.numel() == experts * experts);
  auto stream = c10::cuda::getCurrentCUDAStream(embedding.get_device());
  launch(spectral_exact_groups_kernel, dim3(1), dim3(256), stream.stream(),
         embedding.data_ptr<double>(), affinity.data_ptr<int64_t>(),
         centers.data_ptr<double>(), distances.data_ptr<double>(),
         groups.data_ptr<int64_t>(),
         next_groups.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(),
         overflow.data_ptr<int64_t>(), experts, ranks, dimensions);
  const auto kernel = experts == 256 && ranks == 4
                          ? affinity_swaps_kernel<256, 4>
                          : affinity_swaps_kernel<0, 0>;
  launch(kernel, dim3(1), dim3(1024), stream.stream(),
         affinity.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
         group_affinity.data_ptr<int64_t>(), swap_gains.data_ptr<int64_t>(),
         experts, ranks);
  check_cuda(cudaGetLastError());
}

void balance_affinity_groups_into(
    torch::Tensor demand, torch::Tensor affinity, torch::Tensor groups,
    torch::Tensor expert_loads, torch::Tensor group_loads,
    torch::Tensor group_affinity) {
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(demand.is_cuda() && demand.scalar_type() == torch::kInt64 &&
              demand.is_contiguous() && ranks > 0 && ranks <= kMaxEpSize);
  TORCH_CHECK(affinity.is_cuda() && affinity.scalar_type() == torch::kInt64 &&
              affinity.is_contiguous() && affinity.size(0) == experts &&
              affinity.size(1) == experts);
  TORCH_CHECK(groups.is_cuda() && groups.scalar_type() == torch::kInt64 &&
              groups.is_contiguous() && groups.numel() == experts);
  TORCH_CHECK(expert_loads.is_cuda() &&
              expert_loads.scalar_type() == torch::kInt64 &&
              expert_loads.numel() == experts && group_loads.is_cuda() &&
              group_loads.scalar_type() == torch::kInt64 &&
              group_loads.numel() == ranks && group_affinity.is_cuda() &&
              group_affinity.scalar_type() == torch::kInt64 &&
              group_affinity.numel() == experts * ranks);
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  const auto kernel = experts == 256 && ranks == 4
                          ? balance_affinity_groups_kernel<256, 4>
                          : balance_affinity_groups_kernel<0, 0>;
  launch(kernel, dim3(1), dim3(1024), stream.stream(),
         demand.data_ptr<int64_t>(), affinity.data_ptr<int64_t>(),
         groups.data_ptr<int64_t>(), expert_loads.data_ptr<int64_t>(),
         group_loads.data_ptr<int64_t>(), group_affinity.data_ptr<int64_t>(),
         experts, ranks);
  check_cuda(cudaGetLastError());
}

void group_source_into(torch::Tensor source, torch::Tensor topk,
                       torch::Tensor count, torch::Tensor groups,
                       torch::Tensor group_source, int64_t solver_sms) {
  TORCH_CHECK(solver_sms > 0, "solver_sms must be positive");
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              groups.is_cuda() && group_source.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              groups.scalar_type() == torch::kInt64 &&
              group_source.scalar_type() == torch::kInt64);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  const int ranks = group_source.size(0);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxEpSize && group_source.size(1) == ranks);
  if (solver_sms > 1)
    check_cuda(cudaMemsetAsync(group_source.data_ptr<int64_t>(), 0,
                               group_source.numel() * sizeof(int64_t),
                               stream.stream()));
  if (ranks <= 16)
    launch_group_source_shared<16>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks,
        solver_sms > 1, solver_sms, stream.stream());
  else if (ranks <= 32)
    launch_group_source_shared<32>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks,
        solver_sms > 1, solver_sms, stream.stream());
  else if (ranks <= 64)
    launch_group_source_shared<64>(
        source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
        count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
        group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks,
        solver_sms > 1, solver_sms, stream.stream());
  else
    launch(group_source_kernel, dim3(solver_sms), dim3(256), stream.stream(),
           source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
           group_source.data_ptr<int64_t>(), tokens, topk.size(1), ranks,
           solver_sms == 1);
  check_cuda(cudaGetLastError());
}

void congestion_hungarian_into(
    torch::Tensor group_source, torch::Tensor groups, torch::Tensor allowed,
    torch::Tensor values, torch::Tensor cost, torch::Tensor work,
    torch::Tensor assignment, torch::Tensor primary) {
  const int ranks = group_source.size(0);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxEpSize && group_source.size(1) == ranks);
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
  if (ranks > 16) {
    launch(map_groups_kernel, dim3(1), dim3(128), stream.stream(),
           group_source.data_ptr<int64_t>(), groups.data_ptr<int64_t>(),
           assignment.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           groups.numel(), ranks);
    check_cuda(cudaGetLastError());
    return;
  }
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
  TORCH_CHECK(experts >= ranks && ranks > 0 && ranks <= kMaxEpSize &&
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
