# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec

from .build import TEST_TIME_AUG_REGISTRY, build_test_time_aug

from .test_time_augmentation_average import (
    DatasetMapperTTAAverage,
    GeneralizedRCNNWithTTAAverage,
)
from .test_time_augmentation_union import (
    DatasetMapperTTAUnion,
    GeneralizedRCNNWithTTAUnion,
)

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
