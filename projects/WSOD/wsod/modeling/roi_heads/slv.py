# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import math
from skimage.measure import label, regionprops
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances

from .roi_heads import WeakROIHeads
from .wsddn import WSDDNOutputLayers
from .pcl import PCLOutputLayers, PCLROIHeads

__all__ = ["SLVROIHeads"]

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class SLVROIHeads(WeakROIHeads):
    """
    It's SLV's head.
    """
    score_thresh: float = 0.001
    heatmap_binary_score_thresh: float = 0.5
    max_iter: int = 0
    cur_iter: int = 0

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: List[nn.Module],
        max_iter: int,
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
        score_thresh: float = 0.001,
        heatmap_binary_score_thresh: float = 0.5,
        refine_K: int = 3,
        **kwargs,
    ):
        """
        This interface is for WSDDN.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`WSDDNOutputLayers`.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.add_module("box_predictor_0", box_predictor[0])
        for k in range(1, refine_K + 1):
            self.add_module("box_refinery_{}".format(k), box_predictor[k])

        self.train_on_pred_boxes = train_on_pred_boxes
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.score_thresh = score_thresh
        self.heatmap_binary_score_thresh = heatmap_binary_score_thresh
        self.max_iter = max_iter
        self.cur_iter = 0

        self.refine_K = refine_K

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features                 = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution           = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales               = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio              = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                 = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg       = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        score_thresh                = cfg.WSOD.SLV.SCORE_THRESH
        heatmap_binary_score_thresh = cfg.WSOD.SLV.HEATMAP.BINARY_SCORE_THRESH
        max_iter                    = cfg.SOLVER.MAX_ITER
        refine_K                    = cfg.WSOD.REFINE_K
        refine_reg                  = cfg.WSOD.REFINE_REG
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = [WSDDNOutputLayers(cfg, box_head.output_shape)]

        assert refine_K == len(refine_reg), "{} != {}".format(refine_K, len(refine_reg))
        for k in range(1, refine_K):
            box_predictor.append(
                PCLOutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1], "pcl")
            )
        box_predictor.append(
            PCLOutputLayers(cfg, box_head.output_shape, refine_K, refine_reg[refine_K - 1], "slv")
        )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "score_thresh": score_thresh,
            "heatmap_binary_score_thresh": heatmap_binary_score_thresh,
            "max_iter": max_iter,
            "refine_K": refine_K,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Union[
        Tuple[List[Instances], Dict[str, torch.Tensor]],
        Tuple[List[Instances], Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
    ]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            img_classes, img_classes_oh = self.get_image_level_gt(targets)
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, img_classes_oh, img_classes)
            self.cur_iter += 1
            return proposals, losses
        else:
            pred_instances, all_boxes, all_scores = self._forward_box(features, proposals)
            return pred_instances, {}, all_boxes, all_scores

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        img_classes_oh: Optional[torch.Tensor] = None,
        img_classes: Optional[List[torch.Tensor]] = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        if self.training:
            predictions = self.box_predictor[0](box_features)
            losses = self.box_predictor[0].losses(predictions, proposals, img_classes_oh)

            with torch.no_grad():
                boxes = self.box_predictor[0].predict_boxes(predictions, proposals)
                scores = self.box_predictor[0].predict_weighted_probs(predictions, proposals)
                mean_scores = [torch.zeros_like(scores_per_image) for scores_per_image in scores]
            for k in range(1, self.refine_K):
                proposals, pseudo_targets = self.sample_pseudo_targets_and_label_proposals(
                    boxes, scores, proposals, img_classes,
                    PCLROIHeads.get_pseudo_targets,
                    suffix="_r{}".format(k),
                )

                predictions = self.box_predictor[k](box_features)
                losses_k = self.box_predictor[k].losses(
                    predictions, proposals, img_classes_oh, pseudo_targets
                )
                losses.update(losses_k)

                with torch.no_grad():
                    boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                    scores = self.box_predictor[k].predict_probs(predictions, proposals)
                    mean_scores = [
                        mean_scores_per_image + scores_per_image
                        for mean_scores_per_image, scores_per_image in zip(mean_scores, scores)
                    ]
            mean_scores = [
                mean_scores_per_image / (self.refine_K - 1) for mean_scores_per_image in mean_scores
            ]
            proposals, _ = self.sample_pseudo_targets_and_label_proposals(
                boxes, mean_scores, proposals, img_classes,
                SLVROIHeads.get_pseudo_targets,
                suffix="_r{}".format(self.refine_K),
            )

            predictions = self.box_predictor[-1](box_features)
            losses_k = self.box_predictor[-1].losses(predictions, proposals, img_classes_oh)
            losses.update(losses_k)
            del box_features

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor[-1].predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            predictions = self.box_predictor[-1](box_features)
            del box_features

            pred_instances, _, all_boxes, all_scores = self.box_predictor[-1].inference(
                predictions, proposals
            )
            return pred_instances, all_boxes, all_scores

    @classmethod
    @torch.no_grad()
    def get_pseudo_targets(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        proposals: List[Instances],
        img_classes: List[torch.Tensor],
    ) -> List[Instances]:
        gt_score = SLVROIHeads._get_gt_score(SLVROIHeads.cur_iter)
        gt = []

        for (
            boxes_per_image, scores_per_image, proposals_per_image, img_classes_per_image
        ) in zip(boxes, scores, proposals, img_classes):
            gt_boxes_per_image = []
            gt_scores_per_image = []
            gt_classes_per_image = []

            for i, cls_idx in enumerate(img_classes_per_image):
                boxes_per_class = boxes_per_image[:, i]
                scores_per_class = scores_per_image[:, i]

                top_score_idxs = torch.where(scores_per_class > SLVROIHeads.score_thresh)[0]
                if len(top_score_idxs) > 0:
                    top_score_boxes = boxes_per_class[top_score_idxs]
                    top_score_scores = scores_per_class[top_score_idxs]
                else:
                    top_score_boxes = boxes_per_class
                    top_score_scores = scores_per_class

                binary_mask = SLVROIHeads._build_score_mask(
                    top_score_boxes, top_score_scores, proposals_per_image.image_size, cls_idx.item()
                )
                labeled_mask = label(binary_mask.cpu().numpy())
                regions = regionprops(labeled_mask)
                for region in regions:
                    y1, x1, y2, x2 = region.bbox
                    gt_boxes_per_image.append([x1, y1, x2, y2])
                gt_classes_per_image.extend([cls_idx.item()] * len(regions))
                gt_scores_per_image.extend([gt_score] * len(regions))

            gt_boxes_per_image = boxes_per_image.new(gt_boxes_per_image)
            gt_scores_per_image = scores_per_image.new(gt_scores_per_image)
            gt_classes_per_image = img_classes_per_image.new(gt_classes_per_image)

            gt.append(
                Instances(
                    proposals_per_image.image_size,
                    gt_boxes = Boxes(gt_boxes_per_image),
                    gt_classes = gt_classes_per_image,
                    gt_scores = gt_scores_per_image,
                )
            )

        return gt

    @classmethod
    def _get_gt_score(self, cur_iter: int) -> float:
        w = (cur_iter - SLVROIHeads.max_iter / 2) / 1000
        return 1 / (1 + math.exp(-w))

    @classmethod
    @torch.no_grad()
    def _build_score_mask(
        self,
        top_ranking_boxes: torch.Tensor,
        top_ranking_scores: torch.Tensor,
        image_size: Tuple[int, int],
        class_index,
    ) -> torch.Tensor:
        heat_matrix = top_ranking_scores.new_zeros(image_size)
        for box, score in zip(top_ranking_boxes.to(torch.int64), top_ranking_scores):
            heat_matrix[box[1]: box[3] + 1, box[0]: box[2] + 1] += score
        min_val = heat_matrix.min()
        max_val = heat_matrix.max()
        heat_matrix = (heat_matrix - min_val) / (max_val - min_val)

        if class_index == 14:
            binary_mask = (heat_matrix > 0.2).to(torch.int64)
        else:
            binary_mask = (heat_matrix > SLVROIHeads.heatmap_binary_score_thresh).to(torch.int64)
        return binary_mask
