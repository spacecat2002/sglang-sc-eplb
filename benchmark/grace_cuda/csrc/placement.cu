#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <climits>

#include "launch.cuh"
#include "limits.cuh"

namespace grace_cuda {

__global__ void bundle_gain_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, int64_t* gains, int64_t tokens, int64_t k,
    int64_t experts, int64_t ranks, bool clear_output, int32_t* bundle_heads,
    int32_t* bundle_next) {
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
    const int64_t weight = count[token];
    unsigned long long seen_low = 0;
    unsigned long long seen_high = 0;
    unsigned long long duplicate_low = 0;
    unsigned long long duplicate_high = 0;
    for (int64_t column = 0; column < k; ++column) {
      const int64_t entry = token * k + column;
      const int64_t expert = topk[entry];
      const int64_t destination = primary[expert];
      if (bundle_heads) {
        bundle_next[entry] = atomicExch(
            bundle_heads + expert * ranks + src, static_cast<int32_t>(entry));
      }
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
      if (destination != src && !(duplicate & bit))
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      gains + expert * ranks + src),
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

__global__ void topn_routing_single_block_kernel(
    const int64_t* demand, const int64_t* primary, bool* replicas,
    int64_t* routing, int64_t experts, int64_t ranks, int64_t max_extra) {
  if (blockIdx.x) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  for (int64_t index = threadIdx.x; index < experts * ranks;
       index += blockDim.x)
    replicas[index] = false;
  __syncthreads();
  for (int source = warp; source < ranks; source += warps) {
    for (int expert = lane; expert < experts; expert += 32)
      if (primary[expert] == source)
        replicas[expert * ranks + source] = true;
    __syncwarp();
    for (int pick = 0; pick < max_extra; ++pick) {
      int best = -1;
      int64_t best_demand = 0;
      for (int expert = lane; expert < experts; expert += 32) {
        if (primary[expert] == source ||
            replicas[expert * ranks + source])
          continue;
        const int64_t value = demand[expert * ranks + source];
        if (value > best_demand ||
            (value == best_demand && value > 0 &&
             (best < 0 || expert < best))) {
          best = expert;
          best_demand = value;
        }
      }
      for (int offset = 16; offset; offset >>= 1) {
        const int other = __shfl_down_sync(0xffffffff, best, offset);
        const int64_t value =
            __shfl_down_sync(0xffffffff, best_demand, offset);
        if (other >= 0 &&
            (value > best_demand ||
             (value == best_demand && (best < 0 || other < best)))) {
          best = other;
          best_demand = value;
        }
      }
      best = __shfl_sync(0xffffffff, best, 0);
      if (best < 0) break;
      if (lane == 0) replicas[best * ranks + source] = true;
      __syncwarp();
    }
    for (int expert = lane; expert < experts; expert += 32)
      routing[source * experts + expert] =
          replicas[expert * ranks + source] ? source : primary[expert];
  }
}

__global__ void rank_group_order_kernel(
    const int64_t* demand, const int64_t* primary, int64_t* ordinals,
    int64_t* group_experts, bool* replicas, int64_t experts, int64_t ranks) {
  const int source = blockIdx.x;
  if (source >= ranks || threadIdx.x) return;
  int64_t cursor = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    ordinals[expert * ranks + source] = -1;
    group_experts[expert * ranks + source] = -1;
    if (primary[expert] == source) replicas[expert * ranks + source] = true;
  }
  for (int destination = 0; destination < ranks; ++destination) {
    if (destination == source) continue;
    for (int64_t position = 0;; ++position) {
      int64_t best = -1;
      int64_t best_demand = 0;
      for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t value = demand[expert * ranks + source];
        if (primary[expert] != destination || value == 0 ||
            ordinals[expert * ranks + source] >= 0) {
          continue;
        }
        if (value > best_demand ||
            (value == best_demand && (best < 0 || expert < best))) {
          best = expert;
          best_demand = value;
        }
      }
      if (best < 0) break;
      ordinals[best * ranks + source] = position;
      group_experts[cursor++ * ranks + source] = best;
    }
  }
}

__global__ void rank_group_unlock_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, const int64_t* ordinals, int64_t* gains,
    int64_t tokens, int64_t k, int64_t ranks) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int src = source[token];
  for (int64_t column = 0; column < k; ++column) {
    const int64_t expert = topk[token * k + column];
    const int destination = primary[expert];
    if (destination == src) continue;
    const int64_t ordinal = ordinals[expert * ranks + src];
    bool last = true;
    for (int64_t other = 0; other < k; ++other) {
      const int64_t other_expert = topk[token * k + other];
      if (primary[other_expert] == destination &&
          (ordinals[other_expert * ranks + src] > ordinal ||
           (ordinals[other_expert * ranks + src] == ordinal && other < column))) {
        last = false;
        break;
      }
    }
    if (last) {
      atomicAdd(reinterpret_cast<unsigned long long*>(gains + expert * ranks + src),
                static_cast<unsigned long long>(count[token]));
    }
  }
}

__global__ void rank_group_select_kernel(
    const int64_t* primary, const int64_t* group_experts, int64_t* gains,
    int64_t* values, int64_t* choices, bool* replicas, int64_t* routing,
    int64_t experts, int ranks, int64_t max_extra) {
  const int source = blockIdx.x;
  if (source >= ranks || threadIdx.x) return;
  const int64_t budget = max_extra < experts ? max_extra : experts;
  const int64_t stride = experts + 1;
  const int64_t source_offset = source * (ranks + 1) * stride;
  int64_t starts[65];

  int64_t cursor = 0;
  for (int destination = 0; destination < ranks; ++destination) {
    starts[destination] = cursor;
    int64_t running = 0;
    while (cursor < experts) {
      const int64_t expert = group_experts[cursor * ranks + source];
      if (expert < 0 || primary[expert] != destination) break;
      running += gains[expert * ranks + source];
      gains[expert * ranks + source] = running;
      ++cursor;
    }
  }
  starts[ranks] = cursor;

  int64_t* first = values + source_offset;
  for (int64_t used = 0; used <= budget; ++used) first[used] = -1;
  first[0] = 0;
  for (int destination = 0; destination < ranks; ++destination) {
    const int64_t start = starts[destination];
    const int64_t group_size = starts[destination + 1] - start;
    int64_t* previous = values + source_offset + destination * stride;
    int64_t* current = previous + stride;
    int64_t* current_choices =
        choices + source_offset + (destination + 1) * stride;
    for (int64_t used = 0; used <= budget; ++used) {
      current[used] = -1;
      current_choices[used] = 0;
      const int64_t limit = group_size < used ? group_size : used;
      for (int64_t take = 0; take <= limit; ++take) {
        const int64_t before = previous[used - take];
        if (before < 0) continue;
        const int64_t benefit =
            take ? gains[group_experts[(start + take - 1) * ranks + source] * ranks +
                         source]
                 : 0;
        const int64_t value = before + benefit;
        if (value > current[used]) {
          current[used] = value;
          current_choices[used] = take;
        }
      }
    }
  }

  int64_t used = 0;
  int64_t best = 0;
  const int64_t* final_values = values + source_offset + ranks * stride;
  for (int64_t candidate = 1; candidate <= budget; ++candidate) {
    if (final_values[candidate] > best) {
      best = final_values[candidate];
      used = candidate;
    }
  }
  for (int destination = ranks - 1; destination >= 0; --destination) {
    const int64_t take =
        choices[source_offset + (destination + 1) * stride + used];
    for (int64_t offset = 0; offset < take; ++offset) {
      const int64_t expert =
          group_experts[(starts[destination] + offset) * ranks + source];
      replicas[expert * ranks + source] = true;
    }
    used -= take;
  }
  for (int64_t expert = 0; expert < experts; ++expert) {
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

void select_rank_group_topn_routing_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor demand, torch::Tensor primary, int64_t max_extra,
    torch::Tensor ordinals, torch::Tensor group_experts, torch::Tensor gains,
    torch::Tensor values, torch::Tensor choices, torch::Tensor replicas,
    torch::Tensor routing) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              demand.is_cuda() && primary.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              demand.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  const int64_t experts = demand.size(0);
  const int64_t ranks = demand.size(1);
  TORCH_CHECK(demand.dim() == 2 && primary.numel() == experts && ranks > 0 &&
              ranks <= kMaxEpSize && max_extra >= 0);
  TORCH_CHECK(ordinals.sizes() == demand.sizes() &&
              group_experts.sizes() == demand.sizes() &&
              gains.sizes() == demand.sizes() &&
              ordinals.scalar_type() == torch::kInt64 &&
              group_experts.scalar_type() == torch::kInt64 &&
              gains.scalar_type() == torch::kInt64);
  TORCH_CHECK(values.is_cuda() && choices.is_cuda() &&
              values.scalar_type() == torch::kInt64 &&
              choices.scalar_type() == torch::kInt64 && values.is_contiguous() &&
              choices.is_contiguous() &&
              values.numel() >= ranks * (ranks + 1) * (experts + 1) &&
              choices.numel() >= ranks * (ranks + 1) * (experts + 1));
  TORCH_CHECK(replicas.is_cuda() && replicas.scalar_type() == torch::kBool &&
              replicas.sizes() == demand.sizes());
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt64 &&
              routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  ordinals.fill_(-1);
  group_experts.fill_(-1);
  gains.zero_();
  replicas.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(demand.get_device());
  launch(rank_group_order_kernel, dim3(ranks), dim3(1), stream.stream(),
         demand.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         ordinals.data_ptr<int64_t>(), group_experts.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), experts, ranks);
  launch(rank_group_unlock_kernel, dim3((source.size(0) + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         ordinals.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), source.size(0),
         topk.size(1), ranks);
  launch(rank_group_select_kernel, dim3(ranks), dim3(1), stream.stream(),
         primary.data_ptr<int64_t>(), group_experts.data_ptr<int64_t>(),
         gains.data_ptr<int64_t>(), values.data_ptr<int64_t>(),
         choices.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         routing.data_ptr<int64_t>(), experts, ranks, max_extra);
  check_cuda(cudaGetLastError());
}

void select_bundle_topn_routing_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t max_extra, torch::Tensor gains,
    torch::Tensor replicas, torch::Tensor routing, int64_t solver_sms) {
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
  TORCH_CHECK(ranks > 0 && ranks <= kMaxEpSize,
              "bundle-aware replication supports at most 64 ranks");
  TORCH_CHECK(solver_sms > 0, "solver_sms must be positive");
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  if (solver_sms > 1)
    check_cuda(cudaMemsetAsync(gains.data_ptr<int64_t>(), 0,
                               gains.numel() * sizeof(int64_t), stream.stream()));
  launch(bundle_gain_kernel, dim3(solver_sms), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         gains.data_ptr<int64_t>(), source.size(0), topk.size(1),
         primary.numel(), ranks, solver_sms == 1,
         static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr));
  launch(topn_routing_single_block_kernel, dim3(1), dim3(256),
         stream.stream(), gains.data_ptr<int64_t>(),
         primary.data_ptr<int64_t>(), replicas.data_ptr<bool>(),
         routing.data_ptr<int64_t>(), primary.numel(), ranks, max_extra);
  check_cuda(cudaGetLastError());
}

void select_bundle_topn_routing_index_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t max_extra, torch::Tensor gains,
    torch::Tensor replicas, torch::Tensor routing, torch::Tensor bundle_heads,
    torch::Tensor bundle_next, int64_t solver_sms) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && gains.is_cuda() && replicas.is_cuda() &&
              routing.is_cuda() && bundle_heads.is_cuda() &&
              bundle_next.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              gains.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              routing.scalar_type() == torch::kInt64 &&
              bundle_heads.scalar_type() == torch::kInt32 &&
              bundle_next.scalar_type() == torch::kInt32);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  const int64_t experts = primary.numel();
  const int64_t ranks = gains.size(1);
  TORCH_CHECK(gains.dim() == 2 && gains.size(0) == experts && ranks > 0 &&
              ranks <= kMaxEpSize && replicas.sizes() == gains.sizes() &&
              bundle_heads.sizes() == gains.sizes() &&
              bundle_next.numel() >= topk.numel() && routing.dim() == 2 &&
              routing.size(0) == ranks && routing.size(1) == experts &&
              max_extra >= 0 && solver_sms > 0 && topk.numel() <= INT_MAX);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  check_cuda(cudaMemsetAsync(bundle_heads.data_ptr<int32_t>(), 0xff,
                             bundle_heads.numel() * sizeof(int32_t),
                             stream.stream()));
  if (solver_sms > 1)
    check_cuda(cudaMemsetAsync(gains.data_ptr<int64_t>(), 0,
                               gains.numel() * sizeof(int64_t), stream.stream()));
  launch(bundle_gain_kernel, dim3(solver_sms), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         gains.data_ptr<int64_t>(), source.size(0), topk.size(1), experts,
         ranks, solver_sms == 1, bundle_heads.data_ptr<int32_t>(),
         bundle_next.data_ptr<int32_t>());
  launch(topn_routing_single_block_kernel, dim3(1), dim3(256), stream.stream(),
         gains.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), routing.data_ptr<int64_t>(), experts, ranks,
         max_extra);
  check_cuda(cudaGetLastError());
}

void select_bundle_from_gains_into(torch::Tensor primary, int64_t max_extra,
                                   torch::Tensor gains, torch::Tensor replicas,
                                   torch::Tensor routing) {
  TORCH_CHECK(primary.is_cuda() && gains.is_cuda() && replicas.is_cuda() &&
              routing.is_cuda());
  TORCH_CHECK(primary.scalar_type() == torch::kInt64 &&
              gains.scalar_type() == torch::kInt64 &&
              replicas.scalar_type() == torch::kBool &&
              routing.scalar_type() == torch::kInt64 && gains.dim() == 2 &&
              replicas.sizes() == gains.sizes());
  const int64_t experts = gains.size(0);
  const int64_t ranks = gains.size(1);
  TORCH_CHECK(primary.numel() == experts && ranks > 0 && ranks <= kMaxEpSize &&
              max_extra >= 0 && routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(gains.get_device());
  launch(topn_routing_single_block_kernel, dim3(1), dim3(256), stream.stream(),
         gains.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
         replicas.data_ptr<bool>(), routing.data_ptr<int64_t>(), experts, ranks,
         max_extra);
  check_cuda(cudaGetLastError());
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
