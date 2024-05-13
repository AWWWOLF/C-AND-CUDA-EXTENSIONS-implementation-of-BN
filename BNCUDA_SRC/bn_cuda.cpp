// bn_cuda.cpp
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> bn_cuda_forward(
    at::Tensor X,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor eps,
    at::Tensor momentum,
    at::Tensor mode);

std::vector<at::Tensor> bn_cuda_backward(
    at::Tensor dout,
    at::Tensor var,
    at::Tensor out_,
    at::Tensor gamma);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forward(
    at::Tensor X,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor eps,
    at::Tensor momentum,
    at::Tensor mode
    ){
        CHECK_INPUT(X);
        CHECK_INPUT(gamma);
        CHECK_INPUT(beta);
        CHECK_INPUT(running_mean);
        CHECK_INPUT(running_var);
        CHECK_INPUT(eps);
        CHECK_INPUT(momentum);
        CHECK_INPUT(mode);
        return bn_cuda_forward(X, gamma, beta, running_mean, running_var, eps, momentum, mode);
    }

std::vector<at::Tensor> backward(
    at::Tensor dout,
    at::Tensor var,
    at::Tensor out_,
    at::Tensor gamma){
        CHECK_INPUT(dout);
        CHECK_INPUT(var);
        CHECK_INPUT(out_);
        CHECK_INPUT(gamma);
        return bn_cuda_backward(dout, var, out_, gamma);
        }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "BN forward (CUDA)");
  m.def("backward", &backward, "BN backward (CUDA)");}