// Minimal binding for the SSHdotCodes qwenfast NVFP4 decode MoE (Apache-2.0,
// https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000 serve/qwenfast/qf_moe.cu, 2026-09-24), exp-only.
#include <torch/extension.h>
at::Tensor nvfp4_moe_decode(const at::Tensor & x, const at::Tensor & topk_ids, const at::Tensor & topk_w,
                            const at::Tensor & w13, const at::Tensor & w13_sf, const at::Tensor & g1_alpha,
                            const at::Tensor & a1_gs, const at::Tensor & w2, const at::Tensor & w2_sf,
                            const at::Tensor & g2_alpha, const at::Tensor & a2_gs, bool pdl);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_moe_decode", &nvfp4_moe_decode, py::arg("x"), py::arg("topk_ids"), py::arg("topk_w"), py::arg("w13"),
          py::arg("w13_sf"), py::arg("g1_alpha"), py::arg("a1_gs"), py::arg("w2"), py::arg("w2_sf"), py::arg("g2_alpha"),
          py::arg("a2_gs"), py::arg("pdl") = true);
}
