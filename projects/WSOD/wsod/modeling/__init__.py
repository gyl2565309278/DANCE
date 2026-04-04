# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec

from .backbone import (
    ResNetWS,
    ResNetWSBlockBase,
    build_resnet_ws_backbone,
    make_resnet_ws_stage,
)
from .roi_heads import (
    WeakROIHeads,
    WSDDNROIHeads,
    WSBDNROIHeads,
    OICRROIHeads,
    PCLROIHeads,
    MISTROIHeads,
    SLVROIHeads,
    DANCEROIHeads,
    WSDDNOutputLayers,
    WSBDNOutputLayers,
    PCLOutputLayers,
    NCSOutputLayers,
)

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
