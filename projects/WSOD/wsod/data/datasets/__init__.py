# Copyright (c) Facebook, Inc. and its affiliates.
from .pascal_voc import load_voc_2012_test_instances, register_pascal_voc_2012_test
from . import builtin as _builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if not k.startswith("_")]
