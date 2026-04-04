# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn

from detectron2.utils.registry import Registry

TEST_TIME_AUG_REGISTRY = Registry("TEST_TIME_AUG")
TEST_TIME_AUG_REGISTRY.__doc__ = """
Registry for test time augmentation, which do multiple data augmentations for image

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`nn.Module`, which means the model you need to test.

Registered object must return instance of :class:`nn.Module`.
"""


def build_test_time_aug(cfg, model: nn.Module):
    """
    Build test time augmentation class from `cfg.TEST.AUG.NAME`.

    Returns:
        an instance of :test time augmentation class:`nn.Module`
    """
    test_time_aug_name = cfg.TEST.AUG.NAME
    test_time_aug = TEST_TIME_AUG_REGISTRY.get(test_time_aug_name)(cfg, model)
    return test_time_aug
