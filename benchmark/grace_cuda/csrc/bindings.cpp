#include <torch/extension.h>

namespace grace_cuda {
torch::Tensor source_demand(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                            int64_t);
torch::Tensor select_topn(torch::Tensor, torch::Tensor, int64_t);
std::tuple<torch::Tensor, torch::Tensor> traffic(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t);
}  // namespace grace_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("source_demand", &grace_cuda::source_demand);
  m.def("select_topn", &grace_cuda::select_topn);
  m.def("traffic", &grace_cuda::traffic);
}
