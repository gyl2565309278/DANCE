// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/TensorUtils.h>
#include "PCLLoss.h"

constexpr float EPSILON = 1e-12;

namespace wsod {

template <typename scalar_t, typename index_t>
void PCLLossForward_cpu_kernel(
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
  for (int index = 0; index < nthreads; index++) {
    int cur_target = target[index];
    if (cur_target != ignore_index) {
      if (cur_target == channels - 1) {
        output[index] = -weight[index] * std::log(std::max(input[index * channels + cur_target], epsilon));
      } else if (cur_target >= 0 && cur_target < channels - 1) {
        output[index] = -weight[index] * std::log(std::max(pc_input[cluster[index] * channels + cur_target], epsilon));
      }
    }
  }
} // PCLLossForward_cpu_kernel

template <typename scalar_t, typename index_t>
void PCLLossBackward_cpu_kernel(
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
  for (int index = 0; index < nthreads; index++) {
    int cur_target = target[index];
    if (cur_target != ignore_index) {
      if (cur_target == channels - 1) {
        grad_input[index * channels + cur_target] = -weight[index] / std::max(input[index * channels + cur_target], epsilon);
      } else if (cur_target >= 0 && cur_target < channels - 1) {
        grad_input[index * channels + cur_target] = -weight[index] / std::max(pc_input[cluster[index] * channels + cur_target], epsilon);
      }
    }
  }
} // PCLLossBackward_cpu_kernel

at::Tensor PCLLoss_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(target.device().is_cpu(), "target must be a CPU tensor");
  AT_ASSERTM(weight.device().is_cpu(), "weight must be a CPU tensor");
  AT_ASSERTM(cluster.device().is_cpu(), "cluster must be a CPU tensor");
  AT_ASSERTM(pc_input.device().is_cpu(), "pc_input must be a CPU tensor");

  at::TensorArg input_t{input, "input", 1};
  at::TensorArg target_t{target, "target", 2};
  at::TensorArg weight_t{weight, "weight", 3};
  at::TensorArg cluster_t{cluster, "cluster", 4};
  at::TensorArg pc_input_t{pc_input, "pc_input", 5};
  at::CheckedFrom c = "PCLLoss_forward_cpu";
  at::checkAllSameType(c, {input_t, weight_t, pc_input_t});
  at::checkAllSameType(c, {target_t, cluster_t});

  auto batch_size = input.size(0);
  auto channels = input.size(1);

  at::Tensor output = at::zeros({batch_size}, input.options());

  if (output.numel() == 0) {
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
              PCLLossForward_cpu_kernel<scalar_t, index_t>(
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

  if (reduction_enum == 0) {
    return output;
  } else if (reduction_enum == 1) {
    return output.sum() / at::ne(target, at::full_like(target, ignore_index)).sum();
  } else {
    return output.sum();
  }
}

at::Tensor PCLLoss_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(target.device().is_cpu(), "target must be a CPU tensor");
  AT_ASSERTM(weight.device().is_cpu(), "weight must be a CPU tensor");
  AT_ASSERTM(cluster.device().is_cpu(), "cluster must be a CPU tensor");
  AT_ASSERTM(pc_input.device().is_cpu(), "pc_input must be a CPU tensor");

  at::TensorArg grad_t{grad, "grad", 1};
  at::TensorArg input_t{input, "input", 2};
  at::TensorArg target_t{target, "target", 3};
  at::TensorArg weight_t{weight, "weight", 4};
  at::TensorArg cluster_t{cluster, "cluster", 5};
  at::TensorArg pc_input_t{pc_input, "pc_input", 6};
  at::CheckedFrom c = "PCLLoss_backward_cpu";
  at::checkAllSameType(c, {grad_t, input_t, weight_t, pc_input_t});
  at::checkAllSameType(c, {target_t, cluster_t});

  auto batch_size = input.size(0);
  auto channels = input.size(1);

  at::Tensor grad_input = at::zeros({batch_size, channels}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
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
      "PCLLoss_backward",
      [&] {
        AT_DISPATCH_PCL_LOSS_INDEX_TYPES(
            target.scalar_type(),
            "PCLLoss_backward_index",
            [&] {
              PCLLossBackward_cpu_kernel<scalar_t, index_t>(
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

  if (reduction_enum == 0) {
    return grad_input * grad.unsqueeze(1);
  } else if (reduction_enum == 1) {
    return grad_input * grad / at::ne(target, at::full_like(target, ignore_index)).sum();
  } else {
    return grad_input * grad;
  }
}

} // namespace wsod
