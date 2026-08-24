#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <tuple>

#include "launch.cuh"

namespace grace_cuda {

torch::Tensor source_demand(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                            int64_t);
void source_demand_into(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                        int64_t, torch::Tensor);
torch::Tensor select_topn(torch::Tensor, torch::Tensor, int64_t);
void select_topn_routing_into(torch::Tensor, torch::Tensor, int64_t,
                              torch::Tensor, torch::Tensor);
void select_bundle_topn_routing_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor);

namespace {

__global__ void default_routing_kernel(const bool* replicas,
                                       const int64_t* primary, int64_t* routing,
                                       int64_t experts, int64_t ranks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= ranks * experts) return;
  const int64_t source = index / experts;
  const int64_t expert = index % experts;
  routing[index] = replicas[expert * ranks + source] ? source : primary[expert];
}

}  // namespace

void fused_source_topn_into(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t experts, int64_t ranks,
    int64_t max_extra_per_rank, torch::Tensor demand, torch::Tensor gains,
    torch::Tensor replicas, torch::Tensor routing) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda());
  source_demand_into(source, topk, count, experts, ranks, demand);
  select_bundle_topn_routing_into(source, topk, count, primary,
                                  max_extra_per_rank, gains, replicas, routing);
}

void default_routing_into(torch::Tensor replicas, torch::Tensor primary,
                          torch::Tensor routing) {
  TORCH_CHECK(replicas.is_cuda() && primary.is_cuda());
  TORCH_CHECK(replicas.scalar_type() == torch::kBool &&
              primary.scalar_type() == torch::kInt64 && replicas.dim() == 2 &&
              primary.numel() == replicas.size(0));
  const int64_t experts = replicas.size(0);
  const int64_t ranks = replicas.size(1);
  TORCH_CHECK(routing.is_cuda() && routing.scalar_type() == torch::kInt64 &&
              routing.dim() == 2 && routing.size(0) == ranks &&
              routing.size(1) == experts);
  auto stream = c10::cuda::getCurrentCUDAStream(replicas.get_device());
  launch(default_routing_kernel, dim3((ranks * experts + 255) / 256), dim3(256),
         stream.stream(), replicas.data_ptr<bool>(), primary.data_ptr<int64_t>(),
         routing.data_ptr<int64_t>(), experts, ranks);
  check_cuda(cudaGetLastError());
}

torch::Tensor default_routing(torch::Tensor replicas, torch::Tensor primary) {
  const int64_t experts = replicas.size(0);
  const int64_t ranks = replicas.size(1);
  auto routing = torch::empty({ranks, experts}, primary.options());
  default_routing_into(replicas, primary, routing);
  return routing;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_source_topn(
    torch::Tensor source, torch::Tensor topk, torch::Tensor count,
    torch::Tensor primary, int64_t experts, int64_t ranks,
    int64_t max_extra_per_rank) {
  TORCH_CHECK(source.is_cuda() && topk.is_cuda() && count.is_cuda() &&
              primary.is_cuda());
  TORCH_CHECK(source.scalar_type() == torch::kInt64 &&
              topk.scalar_type() == torch::kInt64 &&
              count.scalar_type() == torch::kInt64 &&
              primary.scalar_type() == torch::kInt64);
  TORCH_CHECK(source.dim() == 1 && topk.dim() == 2 && count.dim() == 1 &&
              source.size(0) == topk.size(0) && source.size(0) == count.size(0));
  TORCH_CHECK(primary.numel() == experts && ranks > 0 && ranks <= 128 &&
              max_extra_per_rank >= 0);
  auto demand = source_demand(source, topk, count, experts, ranks);
  auto gains = torch::empty_like(demand);
  auto replicas = torch::empty({experts, ranks}, demand.options().dtype(torch::kBool));
  auto routing = torch::empty({ranks, experts}, primary.options());
  select_bundle_topn_routing_into(source, topk, count, primary,
                                  max_extra_per_rank, gains, replicas, routing);
  return {demand, replicas, routing};
}

}  // namespace grace_cuda
