// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/extension.h>
#include "crf/crf.h"
#include "pcl_loss/PCLLoss.h"

namespace wsod {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("crf_forward", &crf_forward, "crf_forward");

  m.def("pcl_loss_forward", &PCLLoss_forward, "PCLLoss_forward");
  m.def("pcl_loss_backward", &PCLLoss_backward, "PCLLoss_backward");
}

} // namespace wsod
