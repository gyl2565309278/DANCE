# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec

from .build import TEST_TIME_AUG_REGISTRY, build_test_time_aug

from .backbone import (
    VGGNet,
    VGGNetBlockBase,
    build_vggnet_backbone,
    make_vggnet_stage,
    ResNetWS,
    ResNetWSBlockBase,
    build_resnet_ws_backbone,
    make_resnet_ws_stage,
)
from .meta_arch import (
    GeneralizedWSRCNN,
)
from .postprocessing import detector_postprocess
from .roi_heads import (
    WeakROIHeads,
    WSDDNROIHeads,
    WSBDNROIHeads,
    OICRROIHeads,
    PCLROIHeads,
    MISTROIHeads,
    SLVROIHeads,
    DTHCPROIHeads,
    WSDDNOutputLayers,
    WSBDNOutputLayers,
    PCLOutputLayers,
    HGPSOutputLayers,
)
from .test_time_augmentation_average import (
    DatasetMapperTTAAverage,
    GeneralizedRCNNWSWithTTAAverage,
)
from .test_time_augmentation_union import (
    DatasetMapperTTAUnion,
    GeneralizedRCNNWSWithTTAUnion,
)

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
