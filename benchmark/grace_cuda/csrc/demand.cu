#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <climits>

#include "launch.cuh"
#include "limits.cuh"
#include "ptx.cuh"

namespace grace_cuda {

namespace {

__global__ void bundle_incidence_count_kernel(
    const int64_t* source, const int64_t* topk, int32_t* counts,
    int64_t tokens, int64_t k, int64_t experts) {
  const int64_t entry = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (entry >= tokens * k) return;
  const int64_t token = entry / k;
  const int64_t row = source[token] * experts + topk[entry];
  atomicAdd(counts + row, 1);
}

__global__ void bundle_incidence_prefix_kernel(
    const int32_t* counts, int32_t* offsets, int64_t rows) {
  if (blockIdx.x || threadIdx.x) return;
  int32_t total = 0;
  offsets[0] = 0;
  for (int64_t row = 0; row < rows; ++row) {
    total += counts[row];
    offsets[row + 1] = total;
  }
}

__global__ void bundle_incidence_fill_kernel(
    const int64_t* source, const int64_t* topk, const int32_t* offsets,
    int32_t* cursors, int32_t* entries, int64_t tokens, int64_t k,
    int64_t experts) {
  const int64_t entry = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (entry >= tokens * k) return;
  const int64_t token = entry / k;
  const int64_t row = source[token] * experts + topk[entry];
  const int32_t position = atomicAdd(cursors + row, 1);
  entries[offsets[row] + position] = static_cast<int32_t>(entry);
}

}  // namespace

void build_bundle_incidence_csr_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor offsets,
    torch::Tensor entries, torch::Tensor counts, int64_t experts) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && offsets.is_cuda() &&
              entries.is_cuda() && counts.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              offsets.scalar_type() == torch::kInt32 &&
              entries.scalar_type() == torch::kInt32 &&
              counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 &&
              source.size(0) == topk.size(0) && offsets.dim() == 1 &&
              counts.dim() == 1 && offsets.numel() == counts.numel() + 1 &&
              entries.numel() >= topk.numel() && topk.numel() <= INT_MAX &&
              experts > 0 && counts.numel() % experts == 0 &&
              counts.numel() / experts <= kMaxEpSize);
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  counts.zero_();
  launch(bundle_incidence_count_kernel,
         dim3((topk.numel() + 255) / 256), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         counts.data_ptr<int32_t>(), source.size(0), topk.size(1),
         experts);
  check_cuda(cudaGetLastError());
  launch(bundle_incidence_prefix_kernel, dim3(1), dim3(1), stream.stream(),
         counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
         counts.numel());
  check_cuda(cudaMemsetAsync(counts.data_ptr<int32_t>(), 0,
                             counts.numel() * sizeof(int32_t), stream.stream()));
  launch(bundle_incidence_fill_kernel,
         dim3((topk.numel() + 255) / 256), dim3(256), stream.stream(),
         source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         offsets.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
         entries.data_ptr<int32_t>(), source.size(0), topk.size(1), experts);
  check_cuda(cudaGetLastError());
}

__global__ void source_demand_kernel(const int64_t* source, const int64_t* topk,
                                     const int64_t* count, int64_t* demand,
                                     int64_t tokens, int64_t k, int64_t ranks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = tokens * k;
  if (index >= total) return;
  const int64_t token = index / k;
  const int64_t expert = topk[index];
  const int64_t rank = source[token];
  atomicAdd(reinterpret_cast<unsigned long long*>(demand + expert * ranks + rank),
            static_cast<unsigned long long>(ld_global_i64(count + token)));
}

template <int FixedK>
__global__ void fused_source_demand_bundle_kernel(
    const int64_t* source, const int64_t* topk, const int64_t* count,
    const int64_t* primary, int64_t* demand, int64_t* gains, int64_t tokens,
    int64_t runtime_k, int64_t ranks, int32_t* bundle_heads,
    int32_t* bundle_next) {
  const int64_t actual_k = FixedK ? FixedK : runtime_k;
  for (int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       token < tokens;
       token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
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
      const int64_t entry = token * actual_k + column;
      if constexpr (FixedK > 0) cached_experts[column] = expert;
      const int64_t destination = primary[expert];
      if constexpr (FixedK > 0) cached_destinations[column] = destination;
      if (bundle_heads) {
        bundle_next[entry] = atomicExch(
            bundle_heads + expert * ranks + src, static_cast<int32_t>(entry));
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    demand + expert * ranks + src),
                weight);
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
      const int64_t destination = FixedK
                                      ? cached_destinations[column]
                                      : primary[expert];
      const auto bit = 1ULL << (destination & 63);
      const auto duplicate = destination < 64 ? duplicate_low : duplicate_high;
      if (destination != src && !(duplicate & bit))
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      gains + expert * ranks + src),
                  weight);
    }
  }
}

void source_demand_into(torch::Tensor source, torch::Tensor topk,
                        torch::Tensor count, int64_t num_experts,
                        int64_t num_ranks, torch::Tensor demand) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1);
  TORCH_CHECK(topk.size(0) == source.size(0) &&
              count.size(0) == source.size(0));
  TORCH_CHECK(demand.is_cuda() && demand.scalar_type() == torch::kInt64 &&
              demand.dim() == 2 && demand.size(0) == num_experts &&
              demand.size(1) == num_ranks);
  demand.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t total = source.size(0) * topk.size(1);
  launch(source_demand_kernel, dim3((total + 255) / 256), dim3(256),
         stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
         count.data_ptr<int64_t>(), demand.data_ptr<int64_t>(), source.size(0),
         topk.size(1), num_ranks);
  check_cuda(cudaGetLastError());
}

void fused_source_demand_bundle_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t num_experts, int64_t num_ranks,
    torch::Tensor demand, torch::Tensor gains) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && demand.is_cuda() && gains.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              demand.scalar_type() == torch::kInt64 &&
              gains.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(primary.numel() == num_experts && num_ranks > 0 &&
              num_ranks <= kMaxEpSize && demand.sizes() == gains.sizes() &&
              demand.size(0) == num_experts && demand.size(1) == num_ranks);
  demand.zero_();
  gains.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  const int64_t tokens = source.size(0);
  const int64_t k = topk.size(1);
  const dim3 blocks((tokens + 255) / 256);
  if (k == 1) {
    launch(fused_source_demand_bundle_kernel<1>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  } else if (k == 2) {
    launch(fused_source_demand_bundle_kernel<2>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  } else if (k == 4) {
    launch(fused_source_demand_bundle_kernel<4>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  } else if (k == 8) {
    launch(fused_source_demand_bundle_kernel<8>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  } else if (k == 16) {
    launch(fused_source_demand_bundle_kernel<16>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  } else {
    launch(fused_source_demand_bundle_kernel<0>, blocks, dim3(256),
           stream.stream(), source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, static_cast<int32_t*>(nullptr),
           static_cast<int32_t*>(nullptr));
  }
  check_cuda(cudaGetLastError());
}

void fused_source_demand_bundle_index_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t num_experts, int64_t num_ranks,
    torch::Tensor demand, torch::Tensor gains, torch::Tensor bundle_heads,
    torch::Tensor bundle_next) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda() && demand.is_cuda() && gains.is_cuda() &&
              bundle_heads.is_cuda() && bundle_next.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64 &&
              demand.scalar_type() == torch::kInt64 &&
              gains.scalar_type() == torch::kInt64 &&
              bundle_heads.scalar_type() == torch::kInt32 &&
              bundle_next.scalar_type() == torch::kInt32);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(primary.numel() == num_experts && num_ranks > 0 &&
              num_ranks <= kMaxEpSize && demand.sizes() == gains.sizes() &&
              demand.size(0) == num_experts && demand.size(1) == num_ranks &&
              bundle_heads.sizes() == demand.sizes() &&
              bundle_next.numel() >= topk.numel() &&
              topk.numel() <= INT_MAX);
  demand.zero_();
  gains.zero_();
  auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  check_cuda(cudaMemsetAsync(bundle_heads.data_ptr<int32_t>(), 0xff,
                             bundle_heads.numel() * sizeof(int32_t),
                             stream.stream()));
  const int64_t tokens = source.size(0);
  const int64_t k = topk.size(1);
  const dim3 blocks((tokens + 255) / 256);
  const auto launch_args = [&](auto kernel) {
    launch(kernel, blocks, dim3(256), stream.stream(),
           source.data_ptr<int64_t>(), topk.data_ptr<int64_t>(),
           count.data_ptr<int64_t>(), primary.data_ptr<int64_t>(),
           demand.data_ptr<int64_t>(), gains.data_ptr<int64_t>(), tokens, k,
           num_ranks, bundle_heads.data_ptr<int32_t>(),
           bundle_next.data_ptr<int32_t>());
  };
  switch (k) {
    case 1:
      launch_args(fused_source_demand_bundle_kernel<1>);
      break;
    case 2:
      launch_args(fused_source_demand_bundle_kernel<2>);
      break;
    case 4:
      launch_args(fused_source_demand_bundle_kernel<4>);
      break;
    case 8:
      launch_args(fused_source_demand_bundle_kernel<8>);
      break;
    case 16:
      launch_args(fused_source_demand_bundle_kernel<16>);
      break;
    default:
      launch_args(fused_source_demand_bundle_kernel<0>);
      break;
  }
  check_cuda(cudaGetLastError());
}

torch::Tensor source_demand(torch::Tensor source, torch::Tensor topk,
                            torch::Tensor count, int64_t num_experts,
                            int64_t num_ranks) {
  auto demand = torch::zeros({num_experts, num_ranks},
                             source.options().dtype(torch::kInt64));
  source_demand_into(source, topk, count, num_experts, num_ranks, demand);
  return demand;
}

}  // namespace grace_cuda
