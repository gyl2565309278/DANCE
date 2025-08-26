# Copyright (c) Facebook, Inc. and its affiliates.
from .evaluator import inference_on_dataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
