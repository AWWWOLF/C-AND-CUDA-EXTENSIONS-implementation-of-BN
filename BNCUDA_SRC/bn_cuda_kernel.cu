// bn_cuda_kernel.cu 
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void bn_cuda_forward_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ running_mean,
    scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ eps,
    const scalar_t* __restrict__ momentum,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ sample_mean,
    scalar_t* __restrict__ sample_var,
    scalar_t* __restrict__ out_,
    scalar_t* __restrict__ var,
    size_t M,
    size_t D,
    size_t H,
    size_t W,
    size_t state_size,
    size_t batch_size,
    size_t stats)
{   const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ scalar_t shared_mem[1024];
    for(int column = 0; column < batch_size; column++){
      if (tid >= column * state_size && tid < (column + 1) * state_size){
      if (stats == 3) {
        // 计算样本均值和方差并更新
        shared_mem[tid - column * state_size] = X[tid];
        __syncthreads();
        for(int i = state_size >> 1; i > 0;  i >>= 1){
          if((tid - column * state_size) < i){
            shared_mem[tid - column * state_size] += shared_mem[tid - column * state_size + i];}
            __syncthreads();}
        if(tid == column * state_size){
        sample_mean[column] = shared_mem[0] / state_size;
        running_mean[column] = momentum[0] * running_mean[column] + (1 - momentum[0]) * sample_mean[column];}
        __syncthreads();

        shared_mem[tid - column * state_size] = (X[tid] - sample_mean[column]) * (X[tid] - sample_mean[column]);
        __syncthreads();
        for(int i = state_size >> 1; i > 0;  i >>= 1){
          if((tid - column * state_size) < i){
            shared_mem[tid - column * state_size] += shared_mem[tid - column * state_size + i];}
            __syncthreads();}
        if(tid == column * state_size){
        sample_var[column] = shared_mem[0] / state_size;
        var[column] = 1 / sqrt(sample_var[column] + eps[0]);
        running_var[column] = momentum[0] * running_var[column] * state_size / (state_size - 1) + (1 - momentum[0]) * sample_var[column];}
        __syncthreads();
        
        // 计算归一化后的输出
        out_[tid] = (X[tid] - sample_mean[column]) / sqrt(sample_var[column] + eps[0]);
        output[tid] = gamma[column] * out_[tid] + beta[column];}
      else {
        output[tid] = (X[tid] - sample_mean[column]) / sqrt(sample_var[column] + eps[0]);
        output[tid] = gamma[column] * output[tid] + beta[column];}
      }
    }
}

std::vector<at::Tensor> bn_cuda_forward(
    at::Tensor X,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor eps,
    at::Tensor momentum,
    at::Tensor mode)
{
  //获取张量尺寸
  const auto M = X.size(0);
  const auto D = X.size(1);
  const auto H = X.size(2);
  const auto W = X.size(3);
  const auto stats = mode.numel();
  const auto state_size = M * H * W;
  const auto batch_size = X.size(1);

  //初始化张量
  auto output = at::empty({M, D, H, W}, X.options());
  auto out_ = at::empty({M, D, H, W}, X.options());
  auto sample_mean = at::empty(D, X.options());
  auto sample_var = at::empty(D, X.options());
  auto var = at::empty(D, X.options());

  //定义CUDA中线程和块的数量
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "bn_cuda_forward", ([&] {
    bn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        X.data<scalar_t>(),
        gamma.data<scalar_t>(),
        beta.data<scalar_t>(),
        running_mean.data<scalar_t>(),
        running_var.data<scalar_t>(),
        eps.data<scalar_t>(),
        momentum.data<scalar_t>(),
        output.data<scalar_t>(),
        sample_mean.data<scalar_t>(),
        sample_var.data<scalar_t>(),
        out_.data<scalar_t>(),
        var.data<scalar_t>(),
        M,
        D,
        H,
        W,
        state_size,
        batch_size,
        stats);
  }));

  return {output, var, out_, gamma};
}


template <typename scalar_t>
__global__ void bn_cuda_backward_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ out_,
    const scalar_t* __restrict__ dinput,
    const scalar_t* __restrict__ var,
    scalar_t* __restrict__ dgamma,
    scalar_t* __restrict__ dbeta,
    scalar_t* __restrict__ dX,
    scalar_t* __restrict__ dY,
    scalar_t* __restrict__ dYmean,
    scalar_t* __restrict__ dZ,
    scalar_t* __restrict__ dZmean,
    size_t M,
    size_t D,
    size_t H,
    size_t W,
    size_t state_size,
    size_t batch_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ scalar_t shared_mem[1024];
    __shared__ scalar_t shared_mem1[1024];
    __shared__ scalar_t shared_mem2[1024];
    __shared__ scalar_t shared_mem3[1024];
    for(int column = 0; column < batch_size; column++){
      if (tid >= column * state_size && tid < (column + 1) * state_size){
        shared_mem[tid - column * state_size] = dout[tid];
        shared_mem1[tid - column * state_size] = dinput[tid];
        shared_mem2[tid - column * state_size] = dY[tid];
        shared_mem3[tid - column * state_size] = dZ[tid];
        __syncthreads();
        for(int i = state_size >> 1; i > 0;  i >>= 1){
          if((tid - column * state_size) < i){
            shared_mem[tid - column * state_size] += shared_mem[tid - column * state_size + i];
            shared_mem1[tid - column * state_size] += shared_mem1[tid - column * state_size + i];
            shared_mem2[tid - column * state_size] += shared_mem2[tid - column * state_size + i];
            shared_mem3[tid - column * state_size] += shared_mem3[tid - column * state_size + i];}
            __syncthreads();}
        if(tid == column * state_size){
        dbeta[column] = shared_mem[0];
        dgamma[column] = shared_mem1[0];
        dYmean[column] = shared_mem2[0];
        dZmean[column] = shared_mem3[0];}
        __syncthreads();
        dX[tid] = var[column] * ((dY[tid] - dYmean[column]) - out_[tid] * dZmean[column]);
  }}
}

std::vector<at::Tensor> bn_cuda_backward(
    at::Tensor dout,
    at::Tensor var,
    at::Tensor out_,
    at::Tensor gamma)
{
  const auto M = dout.size(0);
  const auto D = dout.size(1);
  const auto H = dout.size(2);
  const auto W = dout.size(3);
  const auto state_size = M * H * W;
  const auto batch_size = dout.size(1);

  auto dbeta = at::empty(state_size, dout.options());
  auto dgamma = at::empty(state_size, dout.options());
  auto dX = at::empty_like(dout);
  auto dY = dout.mm(gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3)).contiguous();
  auto dYmean =  at::empty_like(dY);
  auto dZ = dY.mm(out_).contiguous();
  auto dZmean =  at::empty_like(dZ);
  auto dinput = dout.mm(out_);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(dout.type(), "bn_cuda_backward", ([&] {
    bn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        dout.data<scalar_t>(),
        out_.data<scalar_t>(),
        dinput.data<scalar_t>(),
        var.data<scalar_t>(),
        dgamma.contiguous().data<scalar_t>(),
        dbeta.contiguous().data<scalar_t>(),
        dX.contiguous().data<scalar_t>(),
        dY.contiguous().data<scalar_t>(),
        dYmean.contiguous().data<scalar_t>(),
        dZ.contiguous().data<scalar_t>(),
        dZmean.contiguous().data<scalar_t>(),
        M,
        D,
        H,
        W,
        batch_size,
        state_size);
  }));
  return {dX, dgamma, dbeta};
}