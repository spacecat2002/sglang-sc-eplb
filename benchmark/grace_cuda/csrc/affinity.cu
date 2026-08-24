#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

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

}  // namespace grace_cuda
