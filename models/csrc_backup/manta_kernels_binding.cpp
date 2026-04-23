/*
 * MANTA CUDA extension: PyTorch C++ binding.
 * Calls C-linkage launch functions from manta_kernels.cu.
 * This file uses PyTorch headers (no /Zc:preprocessor needed).
 */

#include <torch/extension.h>
#include <vector>

// C-linkage launch functions (defined in manta_kernels.cu)
extern "C" {
void mdna_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int H, int N, int E, int max_K, float scale);
void mdna_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int H, int N, int E, int max_K, float scale);
void tanca_forward_launch(
    float* Q, float* K, float* V,
    int32_t* nb_idx, int8_t* nb_valid,
    float* Out, float* Lse,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, float scale);
void tanca_backward_launch(
    float* Q, float* K, float* V, float* dO,
    float* Lse, float* D_buf,
    int32_t* nb_idx, int8_t* nb_valid,
    float* dQ, float* dK, float* dV, float* Out,
    int B, int Vq, int Vkv, int H, int N, int E, int max_K, float scale);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ================================================================
//  MDNA wrappers
// ================================================================
std::vector<torch::Tensor> mdna_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor nb_idx, torch::Tensor nb_valid, float scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(nb_idx); CHECK_INPUT(nb_valid);
    const int B = Q.size(0), H = Q.size(1), N = Q.size(2), E = Q.size(3);
    const int max_K = nb_idx.size(2);
    auto Out = torch::zeros_like(Q);
    auto Lse = torch::empty({B, H, N}, Q.options());
    mdna_forward_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        nb_idx.data_ptr<int32_t>(), nb_valid.data_ptr<int8_t>(),
        Out.data_ptr<float>(), Lse.data_ptr<float>(),
        B, H, N, E, max_K, scale);
    return {Out, Lse};
}

std::vector<torch::Tensor> mdna_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Out, torch::Tensor Lse, torch::Tensor dO,
    torch::Tensor nb_idx, torch::Tensor nb_valid, float scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(Out); CHECK_INPUT(Lse); CHECK_INPUT(dO);
    CHECK_INPUT(nb_idx); CHECK_INPUT(nb_valid);
    const int B = Q.size(0), H = Q.size(1), N = Q.size(2), E = Q.size(3);
    const int max_K = nb_idx.size(2);
    const int total = B * H * N;
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto D = torch::empty({total}, Q.options());
    mdna_backward_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), Lse.data_ptr<float>(), D.data_ptr<float>(),
        nb_idx.data_ptr<int32_t>(), nb_valid.data_ptr<int8_t>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        Out.data_ptr<float>(),
        B, H, N, E, max_K, scale);
    return {dQ, dK, dV};
}

// ================================================================
//  TANCA wrappers
// ================================================================
std::vector<torch::Tensor> tanca_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor nb_idx, torch::Tensor nb_valid, float scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(nb_idx); CHECK_INPUT(nb_valid);
    const int B = Q.size(0), Vq = Q.size(1), H = Q.size(2);
    const int N = Q.size(3), E = Q.size(4), Vkv = K.size(1);
    const int max_K = nb_idx.size(2);
    auto Out = torch::zeros_like(Q);
    auto Lse = torch::empty({B, Vq, H, N}, Q.options());
    tanca_forward_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        nb_idx.data_ptr<int32_t>(), nb_valid.data_ptr<int8_t>(),
        Out.data_ptr<float>(), Lse.data_ptr<float>(),
        B, Vq, Vkv, H, N, E, max_K, scale);
    return {Out, Lse};
}

std::vector<torch::Tensor> tanca_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Out, torch::Tensor Lse, torch::Tensor dO,
    torch::Tensor nb_idx, torch::Tensor nb_valid, float scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_INPUT(Out); CHECK_INPUT(Lse); CHECK_INPUT(dO);
    CHECK_INPUT(nb_idx); CHECK_INPUT(nb_valid);
    const int B = Q.size(0), Vq = Q.size(1), H = Q.size(2);
    const int N = Q.size(3), E = Q.size(4), Vkv = K.size(1);
    const int max_K = nb_idx.size(2);
    const int total = B * Vq * H * N;
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto D = torch::empty({total}, Q.options());
    tanca_backward_launch(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), Lse.data_ptr<float>(), D.data_ptr<float>(),
        nb_idx.data_ptr<int32_t>(), nb_valid.data_ptr<int8_t>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        Out.data_ptr<float>(),
        B, Vq, Vkv, H, N, E, max_K, scale);
    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mdna_forward",   &mdna_forward,   "MDNA forward (CUDA)");
    m.def("mdna_backward",  &mdna_backward,  "MDNA backward (CUDA)");
    m.def("tanca_forward",  &tanca_forward,  "TANCA forward (CUDA)");
    m.def("tanca_backward", &tanca_backward, "TANCA backward (CUDA)");
}
