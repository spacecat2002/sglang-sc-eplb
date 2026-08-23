#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "launch.cuh"

namespace grace_cuda {
namespace {

constexpr int kMaxRanks = 128;

template <int MaxRanks = kMaxRanks>
__device__ void waterfill(const int64_t* loads, const bool* replicas,
                          int64_t expert, int extra_rank, int64_t total,
                          int64_t* allocation, int64_t ranks) {
  int rank_order[MaxRanks];
  bool selected[MaxRanks];
  int replica_count = 0;
  for (int rank = 0; rank < ranks; ++rank) {
    allocation[rank] = 0;
    selected[rank] = false;
    replica_count += replicas[expert * ranks + rank] || rank == extra_rank;
  }
  for (int position = 0; position < replica_count; ++position) {
    int best = -1;
    for (int rank = 0; rank < ranks; ++rank) {
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
      candidate.local_demand, candidate.expert, candidate.target,
      best.next_max, best.next_square, best.remote, best.local_demand,
      best.expert, best.target);
}

template <int MaxRanks>
__global__ void select_compute_replicas_kernel(
    const int64_t* demand, const int64_t* expert_demand,
    const int64_t* expert_order, bool* replicas, int64_t* instance_quota,
    int64_t* loads, int64_t* added_by_rank,
    int64_t* addition_order, int64_t* added_out, int64_t experts, int64_t ranks,
    int64_t max_extra_per_rank) {
  if (blockIdx.x) return;
  constexpr int kThreads = 128;
  __shared__ ComputeCandidate candidates[kThreads];
  __shared__ int64_t current_max;
  __shared__ int64_t current_square;
  __shared__ int64_t added;
  __shared__ bool done;
  int64_t allocation[MaxRanks];
  int64_t base[MaxRanks];
  if (threadIdx.x == 0) added = 0;
  __syncthreads();
  while (added < ranks * max_extra_per_rank) {
    if (threadIdx.x == 0) {
      for (int rank = 0; rank < ranks; ++rank) loads[rank] = 0;
      for (int64_t expert = 0; expert < experts; ++expert) {
        for (int rank = 0; rank < ranks; ++rank) {
          instance_quota[expert * ranks + rank] = 0;
        }
      }
      // The Python key is (is_flexible, -demand, expert). Demand order is
      // fixed, so only the two flexibility passes change after each addition.
      for (int flexible = 0; flexible < 2; ++flexible) {
        for (int64_t position = 0; position < experts; ++position) {
          const int64_t expert = expert_order[position];
          int replica_count = 0;
          for (int rank = 0; rank < ranks; ++rank) {
            replica_count += replicas[expert * ranks + rank];
          }
          if ((replica_count > 1) != flexible) continue;
          waterfill<MaxRanks>(loads, replicas, expert, -1,
                              expert_demand[expert], allocation, ranks);
          for (int rank = 0; rank < ranks; ++rank) {
            instance_quota[expert * ranks + rank] = allocation[rank];
            loads[rank] += allocation[rank];
          }
        }
      }
      current_max = 0;
      current_square = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        current_max = max(current_max, loads[rank]);
        current_square += loads[rank] * loads[rank];
      }
    }
    __syncthreads();

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
      waterfill<MaxRanks>(base, replicas, expert, target, total, allocation,
                          ranks);
      ComputeCandidate candidate = {0, 0, 0, demand[expert * ranks + target],
                                    expert, target};
      int64_t local = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        const int64_t next = base[rank] + allocation[rank];
        candidate.next_max = max(candidate.next_max, next);
        candidate.next_square += next * next;
        local += min(demand[expert * ranks + rank], allocation[rank]);
      }
      if (candidate.next_max > current_max ||
          (candidate.next_max == current_max &&
           candidate.next_square >= current_square)) {
        continue;
      }
      candidate.remote = total - local;
      if (better_candidate(candidate, best)) best = candidate;
    }
    candidates[threadIdx.x] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
      if (threadIdx.x < stride &&
          better_candidate(candidates[threadIdx.x + stride],
                           candidates[threadIdx.x])) {
        candidates[threadIdx.x] = candidates[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const ComputeCandidate winner = candidates[0];
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
  int64_t capacity = 0;
  for (int rank = 0; rank < ranks; ++rank) capacity = max(capacity, loads[rank]);
  for (int64_t index = 0; index < experts * ranks; ++index) total += demand[index];
  capacity = max(
      capacity,
      static_cast<int64_t>(ceil(static_cast<double>(total) / ranks * imbalance_limit)));

  bool changed = true;
  while (changed) {
    changed = false;
    for (int source = 0; source < ranks; ++source) {
      int64_t room = capacity - loads[source];
      for (int64_t index = 0; index < experts && room; ++index) {
        const int64_t expert = source_order[source * experts + index];
        if (!replicas[expert * ranks + source]) continue;
        for (int pass = 0; pass < ranks && room; ++pass) {
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
  unsigned long long seen_low = 0;
  unsigned long long seen_high = 0;
  for (int64_t col = 0; col < k; ++col) {
    const int64_t index = token * k + col;
    const int64_t expert = topk[index];
    const int destination = quota_destination(
        quota, replicas, primary, addition_order, ordinals[index], src, expert,
        experts, ranks);
    atomicAdd(reinterpret_cast<unsigned long long*>(compute + destination),
              static_cast<unsigned long long>(weight));
    const auto bit = 1ULL << (destination & 63);
    auto& seen = destination < 64 ? seen_low : seen_high;
    if (destination != src && !(seen & bit)) {
      seen |= bit;
      atomicAdd(reinterpret_cast<unsigned long long*>(traffic + src * ranks + destination),
                static_cast<unsigned long long>(weight));
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> solve_quota(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor primary,
    torch::Tensor routing, torch::Tensor expert_order,
    torch::Tensor source_order, double imbalance_limit) {
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

  auto instance = torch::zeros_like(demand);
  auto loads = torch::zeros({ranks}, demand.options());
  auto quota = torch::zeros({ranks, experts, ranks}, demand.options());
  auto next_routing = routing.clone();
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(instance_quota_kernel, dim3(1), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         expert_order.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
         loads.data_ptr<int64_t>(), experts, ranks);
  launch(source_quota_kernel, dim3(experts), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), instance.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), quota.data_ptr<int64_t>(), experts, ranks);
  launch(localize_quota_kernel, dim3(1), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         primary.data_ptr<int64_t>(), source_order.data_ptr<int64_t>(),
         quota.data_ptr<int64_t>(), loads.data_ptr<int64_t>(), imbalance_limit,
         experts, ranks);
  const int64_t total = experts * ranks;
  launch(quota_routing_kernel, dim3((total + 255) / 256), dim3(256),
         stream.stream(), quota.data_ptr<int64_t>(),
         next_routing.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
  return {quota, next_routing};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> select_compute_replicas(
    torch::Tensor demand, torch::Tensor replicas, torch::Tensor expert_demand,
    torch::Tensor expert_order, int64_t max_extra_per_rank) {
  TORCH_CHECK(demand.is_cuda() && replicas.is_cuda() &&
              expert_demand.is_cuda() && expert_order.is_cuda());
  TORCH_CHECK(demand.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              expert_demand.scalar_type() == torch::kInt64 &&
              expert_order.scalar_type() == torch::kInt64 &&
              demand.sizes() == replicas.sizes());
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(ranks > 0 && ranks <= kMaxRanks, "compute replicas support 1-128 ranks");
  TORCH_CHECK(expert_demand.numel() == experts &&
              expert_order.numel() == experts && max_extra_per_rank >= 0);
  auto next_replicas = replicas.clone();
  auto instance = torch::zeros_like(demand);
  auto loads = torch::zeros({ranks}, demand.options());
  auto added_by_rank = torch::zeros({ranks}, demand.options());
  auto addition_order = torch::zeros_like(demand);
  auto added = torch::zeros({1}, demand.options());
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  if (ranks <= 4) {
    launch(select_compute_replicas_kernel<4>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 8) {
    launch(select_compute_replicas_kernel<8>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 16) {
    launch(select_compute_replicas_kernel<16>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 32) {
    launch(select_compute_replicas_kernel<32>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else if (ranks <= 64) {
    launch(select_compute_replicas_kernel<64>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  } else {
    launch(select_compute_replicas_kernel<128>, dim3(1), dim3(128), stream.stream(),
           demand.data_ptr<int64_t>(), expert_demand.data_ptr<int64_t>(),
           expert_order.data_ptr<int64_t>(), next_replicas.data_ptr<bool>(),
           instance.data_ptr<int64_t>(), loads.data_ptr<int64_t>(),
           added_by_rank.data_ptr<int64_t>(), addition_order.data_ptr<int64_t>(),
           added.data_ptr<int64_t>(), experts, ranks, max_extra_per_rank);
  }
  check_cuda(cudaGetLastError());
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
