# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .vggnet import (
    PlainBlock,
    VGGNet,
    VGGNetBlockBase,
    build_vggnet_backbone,
    make_vggnet_stage,
)
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
)
from .regnet import RegNet
from .fpn import FPN
from .vit import (
    ViT,
    SimpleFeaturePyramid,
    build_vit_backbone,
    build_vitdet_backbone,
    get_vit_lr_decay_rate,
)
from .mvit import MViT
from .swin import (
    SwinTransformer,
    build_swin_backbone,
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
