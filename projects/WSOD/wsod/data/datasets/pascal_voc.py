# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

__all__ = ["load_voc_2012_test_instances", "register_pascal_voc_2012_test"]


# fmt: off
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
# fmt: on


def load_voc_2012_test_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC 2012 test dataset to Detectron2 format. (without annotation)

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): "test"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    dicts = []
    for fileid in fileids:
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
        }

        dicts.append(r)
    return dicts


def register_pascal_voc_2012_test(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_2012_test_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
