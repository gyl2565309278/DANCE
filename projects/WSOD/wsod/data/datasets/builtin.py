# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .pascal_voc import register_pascal_voc_2012_test

from detectron2.data import MetadataCatalog

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    register_pascal_voc_2012_test("voc_2012_test", os.path.join(root, "VOC2012"), "test", 2012)
    MetadataCatalog.get("voc_2012_test").evaluator_type = "pascal_voc"


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_pascal_voc(_root)
