# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.box_regression import (
    Box2BoxTransform,
    Box2BoxTransformRotated,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage

from .roi_heads import WeakROIHeads

__all__ = ["WSDDNOutputLayers", "WSDDNROIHeads"]

logger = logging.getLogger(__name__)


def _log_classification_stats_wsddn(
    pred_logits: torch.Tensor,
    gt_classes: torch.Tensor,
    pred_logits_thresh: float,
    prefix: str = "wsddn",
) -> None:
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: RxK logits
        gt_classes: R labels
        pred_logits_thresh (float): Only the confidence score of the proposal exceeding this
            threshold is seen as foreground proposal.
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return

    bg_class_ind = pred_logits.shape[1]
    pred_values, pred_classes = pred_logits.max(dim=1)
    filter_mask = pred_values <= pred_logits_thresh
    filter_inds = filter_mask.nonzero().view(-1)
    pred_classes = pred_classes.scatter_(0, filter_inds, bg_class_ind)

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


class WSDDNOutputLayers(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform: Union[Box2BoxTransform, Box2BoxTransformRotated],
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        epsilon: float = 1e-9,
    ) -> None:
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        self.cls_score = nn.Linear(input_size, num_classes)
        self.det_score = nn.Linear(input_size, num_classes)

        self.num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.box_dim = len(box2box_transform.weights)

        nn.init.xavier_uniform_(self.cls_score.weight)
        nn.init.xavier_uniform_(self.det_score.weight)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.det_score.bias, 0)

        self.box2box_transform = box2box_transform
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        if isinstance(loss_weight, float):
            loss_weight = {"loss_img_cls": loss_weight}
        self.loss_weight = loss_weight

        self.epsilon = epsilon

    @classmethod
    def from_config(cls, cfg, input_shape: ShapeSpec) -> Dict[str, Any]:
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "epsilon"                   : cfg.EPSILON,
            # fmt: on
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        weights = self.det_score(x)
        proposal_deltas = scores.new_zeros(
            (scores.shape[0], self.num_bbox_reg_classes * self.box_dim),
            requires_grad=False,
        )
        return scores, weights, proposal_deltas

    def losses(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
        img_classes_oh: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
            to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
            ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, weights, _ = predictions
        num_preds = [len(p) for p in proposals]
        weighted_scores = cat(
            [
                F.softmax(scores_per_image, dim=1) * F.softmax(weights_per_image, dim=0)
                for scores_per_image, weights_per_image in zip(
                    scores.split(num_preds, dim=0), weights.split(num_preds, dim=0)
                )
            ],
            dim=0,
        )

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats_wsddn(weighted_scores, gt_classes, self.test_score_thresh)

        img_scores = cat(
            [
                torch.sum(weighted_scores_per_image, dim=0, keepdim=True)
                for weighted_scores_per_image in weighted_scores.split(num_preds, dim=0)
            ],
            dim=0,
        )
        loss_img_cls = self.binary_cross_entropy_loss(img_scores, img_classes_oh[:, :-1])

        losses = {"loss_img_cls": loss_img_cls}
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def binary_cross_entropy_loss(
        self,
        pred_class_logits: torch.Tensor,
        img_classes_oh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_class_logits: shape (N, K), scores for each of the N image. Each row contains the
            scores for K object categories
            img_classes_oh: shape (N, K), labels for each of the N image. Each row contains the
            class label of each image.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        pred_class_logits.clamp_(self.epsilon, 1.0 - self.epsilon)
        return F.binary_cross_entropy(pred_class_logits, img_classes_oh, reduction="mean")

    def inference(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[List[Instances], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_weighted_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, _, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop = [len(p) for p in proposals]
        return predict_boxes.split(num_prop)

    def predict_boxes(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, _, proposal_deltas = predictions
        num_prop = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop)

    def predict_probs(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _ = predictions
        probs = F.softmax(scores, dim=-1)
        probs = torch.cat((probs, probs.new_zeros(probs.size(0), 1)), dim=1)
        return probs.split([len(p) for p in proposals], dim=0)

    def predict_weighted_probs(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, weights, _ = predictions
        num_inst = [len(p) for p in proposals]
        probs = cat(
            [
                F.softmax(scores_per_image, dim=1) * F.softmax(weights_per_image, dim=0)
                for scores_per_image, weights_per_image in zip(
                    scores.split(num_inst, dim=0), weights.split(num_inst, dim=0)
                )
            ],
            dim=0,
        )
        probs = torch.cat((probs, probs.new_zeros(probs.size(0), 1)), dim=1)
        return probs.split(num_inst, dim=0)


@ROI_HEADS_REGISTRY.register()
class WSDDNROIHeads(WeakROIHeads):
    """
    It's WSDDN's head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        train_on_pred_boxes: bool = False,
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

        self.train_on_pred_boxes = train_on_pred_boxes

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
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
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
        box_predictor = WSDDNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
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
            _, img_classes_oh = self.get_image_level_gt(targets)
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, img_classes_oh)
            return proposals, losses
        else:
            pred_instances, all_boxes, all_scores = self._forward_box(features, proposals)
            return pred_instances, {}, all_boxes, all_scores

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        img_classes_oh: Optional[torch.Tensor] = None,
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
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, img_classes_oh)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _, all_boxes, all_scores = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, all_boxes, all_scores
