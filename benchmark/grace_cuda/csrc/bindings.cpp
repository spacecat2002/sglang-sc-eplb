#include <torch/extension.h>

#include <tuple>

namespace grace_cuda {
torch::Tensor source_demand(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                            int64_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_source_topn(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    int64_t, int64_t);
torch::Tensor select_topn(torch::Tensor, torch::Tensor, int64_t);
std::tuple<torch::Tensor, torch::Tensor> traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t);
std::tuple<torch::Tensor, torch::Tensor> solve_quota(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, double);
std::tuple<torch::Tensor, torch::Tensor> quota_traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> select_compute_replicas(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
}  // namespace grace_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("source_demand", &grace_cuda::source_demand);
  m.def("fused_source_topn", &grace_cuda::fused_source_topn);
  m.def("select_topn", &grace_cuda::select_topn);
  m.def("traffic", &grace_cuda::traffic);
  m.def("solve_quota", &grace_cuda::solve_quota);
  m.def("quota_traffic", &grace_cuda::quota_traffic);
  m.def("select_compute_replicas", &grace_cuda::select_compute_replicas);
}
