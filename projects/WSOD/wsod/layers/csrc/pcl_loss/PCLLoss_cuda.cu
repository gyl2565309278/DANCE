// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "PCLLoss.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

constexpr float EPSILON = 1e-12;

namespace wsod {

template <typename scalar_t, typename index_t>
__global__ void PCLLossForward_gpu_kernel(
    const int nthreads,
    const scalar_t* input,
    const int channels,
    const index_t* target,
    const scalar_t* weight,
    const index_t* cluster,
    const scalar_t* pc_input,
    const int ignore_index,
    scalar_t* output) {
  const scalar_t epsilon = EPSILON;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cur_target = target[index];
    if (cur_target != ignore_index) {
      if (cur_target == channels - 1) {
        output[index] = -weight[index] * std::log(std::max(input[index * channels + cur_target], epsilon));
      } else if (cur_target >= 0 && cur_target < channels - 1) {
        output[index] = -weight[index] * std::log(std::max(pc_input[cluster[index] * channels + cur_target], epsilon));
      }
    }
  } // CUDA_1D_KERNEL_LOOP
} // PCLLossForward_gpu_kernel

template <typename scalar_t, typename index_t>
__global__ void PCLLossBackward_gpu_kernel(
    const int nthreads,
    const scalar_t* input,
    const int channels,
    const index_t* target,
    const scalar_t* weight,
    const index_t* cluster,
    const scalar_t* pc_input,
    const int ignore_index,
    scalar_t* grad_input) {
  const scalar_t epsilon = EPSILON;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cur_target = target[index];
    if (cur_target != ignore_index) {
      if (cur_target == channels - 1) {
        grad_input[index * channels + cur_target] = -weight[index] / std::max(input[index * channels + cur_target], epsilon);
      } else if (cur_target >= 0 && cur_target < channels - 1) {
        grad_input[index * channels + cur_target] = -weight[index] / std::max(pc_input[cluster[index] * channels + cur_target], epsilon);
      }
    }
  } // CUDA_1D_KERNEL_LOOP
} // PCLLossBackward_gpu_kernel

at::Tensor PCLLoss_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(target.device().is_cuda(), "target must be a CUDA tensor");
  AT_ASSERTM(weight.device().is_cuda(), "weight must be a CUDA tensor");
  AT_ASSERTM(cluster.device().is_cuda(), "cluster must be a CUDA tensor");
  AT_ASSERTM(pc_input.device().is_cuda(), "pc_input must be a CUDA tensor");

  at::TensorArg input_t{input, "input", 1};
  at::TensorArg target_t{target, "target", 2};
  at::TensorArg weight_t{weight, "target", 3};
  at::TensorArg cluster_t{cluster, "cluster", 4};
  at::TensorArg pc_input_t{pc_input, "target", 5};
  at::CheckedFrom c = "PCLLoss_forward_cuda";
  at::checkAllSameGPU(c, {input_t, target_t, weight_t, cluster_t, pc_input_t});
  at::checkAllSameType(c, {input_t, weight_t, pc_input_t});
  at::checkAllSameType(c, {target_t, cluster_t});
  at::cuda::CUDAGuard device_guard(input.device());

  auto batch_size = input.size(0);
  auto channels = input.size(1);
  auto output = at::zeros({batch_size}, input.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(batch_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto weight_ = weight.contiguous();
  auto cluster_ = cluster.contiguous();
  auto pc_input_ = pc_input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "PCLLoss_forward",
      [&] {
        AT_DISPATCH_PCL_LOSS_INDEX_TYPES(
            target.scalar_type(),
            "PCLLoss_forward_index",
            [&] {
              PCLLossForward_gpu_kernel<scalar_t, index_t><<<grid, block, 0, stream>>>(
                  batch_size,
                  input_.data_ptr<scalar_t>(),
                  channels,
                  target_.data_ptr<index_t>(),
                  weight_.data_ptr<scalar_t>(),
                  cluster_.data_ptr<index_t>(),
                  pc_input_.data_ptr<scalar_t>(),
                  ignore_index,
                  output.data_ptr<scalar_t>());
            });
      });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  if (reduction_enum == 0) {
    return output;
  } else if (reduction_enum == 1) {
    return output.sum() / at::ne(target, at::full_like(target, ignore_index)).sum();
  } else {
    return output.sum();
  }
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor PCLLoss_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(target.device().is_cuda(), "target must be a CUDA tensor");
  AT_ASSERTM(weight.device().is_cuda(), "weight must be a CUDA tensor");
  AT_ASSERTM(cluster.device().is_cuda(), "cluster must be a CUDA tensor");
  AT_ASSERTM(pc_input.device().is_cuda(), "pc_input must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1};
  at::TensorArg input_t{input, "input", 2};
  at::TensorArg target_t{target, "target", 3};
  at::TensorArg weight_t{weight, "weight", 4};
  at::TensorArg cluster_t{cluster, "cluster", 5};
  at::TensorArg pc_input_t{pc_input, "pc_input", 6};
  at::CheckedFrom c = "PCLLoss_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, input_t, target_t, weight_t, cluster_t, pc_input_t});
  at::checkAllSameType(c, {grad_t, input_t, weight_t, pc_input_t});
  at::checkAllSameType(c, {target_t, cluster_t});
  at::cuda::CUDAGuard device_guard(grad.device());

  auto batch_size = input.size(0);
  auto channels = input.size(1);
  auto grad_input = at::zeros({batch_size, channels}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(batch_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  auto grad_ = grad.contiguous();
  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto weight_ = weight.contiguous();
  auto cluster_ = cluster.contiguous();
  auto pc_input_ = pc_input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "PCLLoss_backward",
      [&] {
        AT_DISPATCH_PCL_LOSS_INDEX_TYPES(
            target.scalar_type(),
            "PCLLoss_backward_index",
            [&] {
              PCLLossBackward_gpu_kernel<scalar_t, index_t><<<grid, block, 0, stream>>>(
                  batch_size,
                  input_.data_ptr<scalar_t>(),
                  channels,
                  target_.data_ptr<index_t>(),
                  weight_.data_ptr<scalar_t>(),
                  cluster_.data_ptr<index_t>(),
                  pc_input_.data_ptr<scalar_t>(),
                  ignore_index,
                  grad_input.data_ptr<scalar_t>());
            });
      });
  AT_CUDA_CHECK(cudaGetLastError());

  if (reduction_enum == 0) {
    return grad_input * grad.unsqueeze(1);
  } else if (reduction_enum == 1) {
    return grad_input * grad / at::ne(target, at::full_like(target, ignore_index)).sum();
  } else {
    return grad_input * grad;
  }
}

} // namespace wsod
