// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/types.h>
#include <torch/version.h>

namespace wsod {

#if ((TORCH_VERSION_MAJOR == 0) || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 12))
#define AT_DISPATCH_PCL_LOSS_INDEX_TYPES(TYPE, NAME, ...)                                         \
  [&] {                                                                                           \
    at::ScalarType _it = TYPE;                                                                    \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                                       \
    switch (_it) {                                                                                \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Byte, uint8_t, index_t, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Short, int16_t, index_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Int, int32_t, index_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)  \
      default:                                                                                    \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");                            \
    }                                                                                             \
  }()
#else
#define AT_DISPATCH_PCL_LOSS_INDEX_TYPES(TYPE, NAME, ...)                      \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                               \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Byte, index_t, __VA_ARGS__)  \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Short, index_t, __VA_ARGS__) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Int, index_t, __VA_ARGS__)   \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Long, index_t, __VA_ARGS__))
#endif

at::Tensor PCLLoss_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index);

at::Tensor PCLLoss_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index);

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor PCLLoss_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index);

at::Tensor PCLLoss_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index);
#endif

// Interface for Python
inline at::Tensor PCLLoss_forward(
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return PCLLoss_forward_cuda(
        input,
        target,
        weight,
        cluster,
        pc_input,
        reduction_enum,
        ignore_index);
#else
    AT_ERROR("WSOD is not compiled with GPU support!");
#endif
  }
  return PCLLoss_forward_cpu(
      input,
      target,
      weight,
      cluster,
      pc_input,
      reduction_enum,
      ignore_index);
}

inline at::Tensor PCLLoss_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& cluster,
    const at::Tensor& pc_input,
    const int reduction_enum,
    const int ignore_index) {
  if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return PCLLoss_backward_cuda(
        grad,
        input,
        target,
        weight,
        cluster,
        pc_input,
        reduction_enum,
        ignore_index);
#else
    AT_ERROR("WSOD is not compiled with GPU support!");
#endif
  }
  return PCLLoss_backward_cpu(
      grad,
      input,
      target,
      weight,
      cluster,
      pc_input,
      reduction_enum,
      ignore_index);
}

} // namespace wsod
