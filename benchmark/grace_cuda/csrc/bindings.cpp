#include <torch/extension.h>

#include <tuple>

namespace grace_cuda {
void affinity_primary_into(torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor);
void affinity_histogram_into(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor);
void spectral_groups_into(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor);
void group_source_into(torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor);
void congestion_hungarian_into(torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor, torch::Tensor);
void refine_congestion_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, double, int64_t, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
torch::Tensor source_demand(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                            int64_t);
void source_demand_into(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                        int64_t, torch::Tensor);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_source_topn(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    int64_t, int64_t);
torch::Tensor select_topn(torch::Tensor, torch::Tensor, int64_t);
void select_topn_into(torch::Tensor, torch::Tensor, int64_t, torch::Tensor);
void select_topn_routing_into(torch::Tensor, torch::Tensor, int64_t,
                              torch::Tensor, torch::Tensor);
void fused_source_topn_into(torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor, int64_t, int64_t, int64_t,
                            torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor default_routing(torch::Tensor, torch::Tensor);
void default_routing_into(torch::Tensor, torch::Tensor, torch::Tensor);
std::tuple<torch::Tensor, torch::Tensor> traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t);
std::tuple<torch::Tensor, torch::Tensor> solve_quota(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, double);
void solve_quota_into(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, double, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor);
std::tuple<torch::Tensor, torch::Tensor> quota_traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> select_compute_replicas(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void select_compute_replicas_into(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
}  // namespace grace_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affinity_primary_into", &grace_cuda::affinity_primary_into);
  m.def("affinity_histogram_into", &grace_cuda::affinity_histogram_into);
  m.def("spectral_groups_into", &grace_cuda::spectral_groups_into);
  m.def("group_source_into", &grace_cuda::group_source_into);
  m.def("congestion_hungarian_into", &grace_cuda::congestion_hungarian_into);
  m.def("refine_congestion_into", &grace_cuda::refine_congestion_into);
  m.def("source_demand", &grace_cuda::source_demand);
  m.def("source_demand_into", &grace_cuda::source_demand_into);
  m.def("fused_source_topn", &grace_cuda::fused_source_topn);
  m.def("select_topn", &grace_cuda::select_topn);
  m.def("select_topn_into", &grace_cuda::select_topn_into);
  m.def("select_topn_routing_into", &grace_cuda::select_topn_routing_into);
  m.def("fused_source_topn_into", &grace_cuda::fused_source_topn_into);
  m.def("default_routing", &grace_cuda::default_routing);
  m.def("default_routing_into", &grace_cuda::default_routing_into);
  m.def("traffic", &grace_cuda::traffic);
  m.def("solve_quota", &grace_cuda::solve_quota);
  m.def("solve_quota_into", &grace_cuda::solve_quota_into);
  m.def("quota_traffic", &grace_cuda::quota_traffic);
  m.def("select_compute_replicas", &grace_cuda::select_compute_replicas);
  m.def("select_compute_replicas_into", &grace_cuda::select_compute_replicas_into);
}
