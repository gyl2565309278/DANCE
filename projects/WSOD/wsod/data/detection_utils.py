# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import torch

from detectron2.structures import Boxes, BoxMode

from wsod.structures import Instances

__all__ = ["transform_proposals"]


def transform_proposals(dataset_dict, image_shape, transforms, *, proposal_topk, min_box_size=0):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        proposal_topk (int): only keep top-K scoring proposals
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )
        clusters = None
        if "proposal_clusters" in dataset_dict:
            clusters = dataset_dict.pop("proposal_clusters")
            for cluster in clusters:
                cluster["proposal_ids"] = torch.as_tensor(
                    cluster["proposal_ids"].astype("int64")
                )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_size)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]
        if clusters is not None:
            keep_ids = keep.cumsum(0) - 1
            keep_ids[~keep] = -1
            clusters_keep = []
            for cluster in clusters:
                cluster_proposal_ids = keep_ids[cluster["proposal_ids"]]
                cluster["proposal_ids"] = cluster_proposal_ids[cluster_proposal_ids != -1]
                if cluster["proposal_ids"].numel() != 0:
                    clusters_keep.append(cluster)

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        if clusters is not None:
            clusters = {}
            for cluster_keep in clusters_keep:
                proposal_ids_topk = cluster_keep["proposal_ids"] < proposal_topk
                if cluster_keep["category_id"] in clusters:
                    clusters[cluster_keep["category_id"]].append(
                        cluster_keep["proposal_ids"][proposal_ids_topk]
                    )
                else:
                    clusters[cluster_keep["category_id"]] = [
                        cluster_keep["proposal_ids"][proposal_ids_topk]
                    ]
            proposals.clusters = clusters
        dataset_dict["proposals"] = proposals
