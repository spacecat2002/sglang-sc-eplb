#include <torch/extension.h>

#include <tuple>

namespace grace_cuda {
void affinity_primary_into(torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor);
void affinity_histogram_into(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, torch::Tensor, torch::Tensor,
                             int64_t);
void normalize_affinity_into(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor);
void affinity_subspace_into(torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor, int64_t);
void spectral_groups_into(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor);
void balance_affinity_groups_into(torch::Tensor, torch::Tensor, torch::Tensor,
                                  torch::Tensor, torch::Tensor, torch::Tensor);
void group_source_into(torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, int64_t);
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
void select_rank_group_topn_routing_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor);
void select_bundle_topn_routing_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void select_bundle_topn_routing_index_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t);
void fused_source_topn_into(torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor, int64_t, int64_t, int64_t,
                            torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor);
void fused_source_topn_index_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor,
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
void bundle_ordinals_into(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                          int64_t, torch::Tensor, torch::Tensor);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> select_compute_replicas(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void select_compute_replicas_into(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
void select_pure_compute_replicas_into(
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
void select_compute_replicas_v2_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, double,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
void select_compute_replicas_fast_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void select_compute_replicas_fast_sparse_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t);
void materialize_fast_sparse_quota_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor);
void materialize_fast_csr_quota_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor);
std::tuple<torch::Tensor, torch::Tensor> sparse_quota_traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t);
std::tuple<torch::Tensor, torch::Tensor> csr_quota_traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void current_bundle_gains_into(torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor, int64_t);
void current_bundle_gains_fast_into(torch::Tensor, torch::Tensor, torch::Tensor,
                                    torch::Tensor, torch::Tensor, torch::Tensor,
                                    int64_t);
void current_bundle_gains_and_select_compute_replicas_fast_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, double, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);
void incremental_bundle_gains_fast_into(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    int64_t);
}  // namespace grace_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affinity_primary_into", &grace_cuda::affinity_primary_into);
  m.def("affinity_histogram_into", &grace_cuda::affinity_histogram_into);
  m.def("normalize_affinity_into", &grace_cuda::normalize_affinity_into);
  m.def("affinity_subspace_into", &grace_cuda::affinity_subspace_into);
  m.def("spectral_groups_into", &grace_cuda::spectral_groups_into);
  m.def("balance_affinity_groups_into",
        &grace_cuda::balance_affinity_groups_into);
  m.def("group_source_into", &grace_cuda::group_source_into);
  m.def("congestion_hungarian_into", &grace_cuda::congestion_hungarian_into);
  m.def("refine_congestion_into", &grace_cuda::refine_congestion_into);
  m.def("source_demand", &grace_cuda::source_demand);
  m.def("source_demand_into", &grace_cuda::source_demand_into);
  m.def("fused_source_topn", &grace_cuda::fused_source_topn);
  m.def("select_topn", &grace_cuda::select_topn);
  m.def("select_topn_into", &grace_cuda::select_topn_into);
  m.def("select_topn_routing_into", &grace_cuda::select_topn_routing_into);
  m.def("select_rank_group_topn_routing_into",
        &grace_cuda::select_rank_group_topn_routing_into);
  m.def("select_bundle_topn_routing_into",
        &grace_cuda::select_bundle_topn_routing_into);
  m.def("select_bundle_topn_routing_index_into",
        &grace_cuda::select_bundle_topn_routing_index_into);
  m.def("fused_source_topn_into", &grace_cuda::fused_source_topn_into);
  m.def("fused_source_topn_index_into",
        &grace_cuda::fused_source_topn_index_into);
  m.def("default_routing", &grace_cuda::default_routing);
  m.def("default_routing_into", &grace_cuda::default_routing_into);
  m.def("traffic", &grace_cuda::traffic);
  m.def("solve_quota", &grace_cuda::solve_quota);
  m.def("solve_quota_into", &grace_cuda::solve_quota_into);
  m.def("quota_traffic", &grace_cuda::quota_traffic);
  m.def("bundle_ordinals_into", &grace_cuda::bundle_ordinals_into);
  m.def("select_compute_replicas", &grace_cuda::select_compute_replicas);
  m.def("select_compute_replicas_into", &grace_cuda::select_compute_replicas_into);
  m.def("select_pure_compute_replicas_into",
        &grace_cuda::select_pure_compute_replicas_into);
  m.def("select_compute_replicas_v2_into",
        &grace_cuda::select_compute_replicas_v2_into);
  m.def("select_compute_replicas_fast_into",
        &grace_cuda::select_compute_replicas_fast_into);
  m.def("select_compute_replicas_fast_sparse_into",
        &grace_cuda::select_compute_replicas_fast_sparse_into);
  m.def("materialize_fast_sparse_quota_into",
        &grace_cuda::materialize_fast_sparse_quota_into);
  m.def("materialize_fast_csr_quota_into",
        &grace_cuda::materialize_fast_csr_quota_into);
  m.def("sparse_quota_traffic", &grace_cuda::sparse_quota_traffic);
  m.def("csr_quota_traffic", &grace_cuda::csr_quota_traffic);
  m.def("current_bundle_gains_into", &grace_cuda::current_bundle_gains_into);
  m.def("current_bundle_gains_fast_into",
        &grace_cuda::current_bundle_gains_fast_into);
  m.def("current_bundle_gains_and_select_compute_replicas_fast_into",
        &grace_cuda::current_bundle_gains_and_select_compute_replicas_fast_into);
  m.def("incremental_bundle_gains_fast_into",
        &grace_cuda::incremental_bundle_gains_fast_into);
}
