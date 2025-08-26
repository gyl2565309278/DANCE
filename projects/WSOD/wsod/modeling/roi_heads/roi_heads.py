# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import math
import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .output_layers import (
    WSDDNOutputLayers,
    WSBDNOutputLayers,
    PCLOutputLayers,
    HGPSOutputLayers,
)

logger = logging.getLogger(__name__)


class WeakROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        proposal_matcher: Matcher,
        proposal_append_gt: bool = True,
        use_pcl_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
            use_pcl_loss (bool): whether to use proposal cluster loss which calculates the loss
            for each cluster instead of each proposal
        """
        super().__init__()
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt
        self.use_pcl_loss = use_pcl_loss

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
            "use_pcl_loss": cfg.MODEL.ROI_BOX_HEAD.USE_PCL_LOSS,
        }

    @torch.no_grad()
    def get_image_level_gt(self, targets: List[Instances]) -> Tuple[torch.Tensor, torch.Tensor]:
        img_classes = [torch.unique(t.gt_classes, sorted=True) for t in targets]
        img_classes = [gt.to(torch.int64) for gt in img_classes]
        img_classes_oh = torch.cat(
            [
                torch.zeros(
                    (1, self.num_classes), dtype=torch.float, device=img_classes[0].device
                ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
                for gt in img_classes
            ],
            dim=0,
        )
        img_classes_oh = torch.cat(
            (img_classes_oh, img_classes_oh.new_ones((len(targets), 1))), dim=1
        )
        return img_classes, img_classes_oh

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
            gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
            (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
            each sampled proposal. Each sample is labeled as either a category in
            [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], suffix: str = ""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`WeakROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set gt_classes attribute of proposals:
            proposals_per_image.gt_classes = gt_classes

            # Set gt_boxes attribute of proposals:
            if has_gt:
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                proposals_per_image.gt_boxes = proposals_per_image.proposal_boxes.clone()

            # Set gt_scores attribute of proposals:
            if targets_per_image.has("gt_scores"):
                if has_gt:
                    gt_scores = targets_per_image.gt_scores[matched_idxs]
                else:
                    gt_scores = torch.ones_like(matched_idxs, dtype=targets_per_image.gt_scores.dtype)
                proposals_per_image.gt_scores = gt_scores

            # Set gt_clusters attribute of proposals:
            if self.use_pcl_loss:
                gt_clusters = matched_idxs.clone()
                gt_clusters[(matched_labels == 0) | (matched_labels == -1)] = -1
                proposals_per_image.gt_clusters = gt_clusters

            if has_gt:
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[matched_idxs])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_ig_samples[-1] - num_bg_samples[-1])

            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def _get_labeled_proposals_and_pseudo_targets(
        self,
        boxes: Tuple[torch.Tensor, ...],
        scores: Tuple[torch.Tensor, ...],
        proposals: List[Instances],
        img_classes: List[torch.Tensor],
        get_pseudo_targets: Callable,
        suffix: str = "",
    ) -> Tuple[List[Instances], List[Instances]]:
        scores = [
            scores_per_image.index_select(1, img_classes_per_image)
            for scores_per_image, img_classes_per_image in zip(scores, img_classes)
        ]

        if self.cls_agnostic_bbox_reg:
            boxes = [
                boxes_per_image.unsqueeze_(1).expand(
                    boxes_per_image.size(0), self.num_classes, boxes_per_image.size(2)
                )
                for boxes_per_image in boxes
            ]
        else:
            boxes = [
                boxes_per_image.view(-1, self.num_classes, 4)
                for boxes_per_image in boxes
            ]
        boxes = [
            boxes_per_image.index_select(1, img_classes_per_image)
            for boxes_per_image, img_classes_per_image in zip(boxes, img_classes)
        ]

        pseudo_targets = get_pseudo_targets(boxes, scores, proposals, img_classes)

        proposal_append_gt = self.proposal_append_gt
        self.proposal_append_gt = False
        proposals = self.label_and_sample_proposals(proposals, pseudo_targets, suffix=suffix)
        self.proposal_append_gt = proposal_append_gt

        return proposals, pseudo_targets


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


@ROI_HEADS_REGISTRY.register()
class WSBDNROIHeads(WeakROIHeads):
    """
    It's WSBDN's head.
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
        in_features          = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution    = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales        = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio       = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type          = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
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
        box_predictor = WSBDNOutputLayers(cfg, box_head.output_shape)
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
        Tuple[List[Instances], Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
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
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            pseudo_targets = WSBDNROIHeads.get_pseudo_targets(proposals, img_classes)
            proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(proposals, pseudo_targets)
            self.proposal_append_gt = proposal_append_gt

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

    @classmethod
    @torch.no_grad()
    def get_pseudo_targets(
        self,
        proposals: List[Instances],
        img_classes: List[torch.Tensor],
    ) -> List[Instances]:
        pseudo_targets = []
        for proposals_per_image, img_classes_per_image in zip(proposals, img_classes):
            gt_boxes_per_image = []
            gt_classes_per_image = []
            for cls_idx in img_classes_per_image.reshape(-1, 1):
                clusters_per_class = torch.cat(proposals_per_image.clusters[cls_idx.item()])
                gt_boxes_per_image.append(
                    proposals_per_image.proposal_boxes.tensor[clusters_per_class]
                )
                gt_classes_per_image.append(cls_idx.expand(clusters_per_class.shape[0]))
            gt_boxes_per_image = torch.cat(gt_boxes_per_image, dim=0)
            gt_classes_per_image = torch.cat(gt_classes_per_image, dim=0)
            pseudo_targets.append(
                Instances(
                    proposals_per_image.image_size,
                    gt_boxes = Boxes(gt_boxes_per_image),
                    gt_classes = gt_classes_per_image,
                )
            )
        return pseudo_targets


@ROI_HEADS_REGISTRY.register()
class OICRROIHeads(WeakROIHeads):
    """
    It's OICR's head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: List[nn.Module],
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
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
        in_features           = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales         = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        refine_K              = cfg.WSOD.REFINE_K
        refine_reg            = cfg.WSOD.REFINE_REG
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
        for k in range(1, refine_K + 1):
            box_predictor.append(
                PCLOutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1], "oicr")
            )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
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
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
            for k in range(1, self.refine_K + 1):
                proposals, _ = self._get_labeled_proposals_and_pseudo_targets(
                    boxes, scores, proposals, img_classes,
                    OICRROIHeads.get_pseudo_targets,
                    suffix="_r{}".format(k),
                )

                predictions = self.box_predictor[k](box_features)
                losses_k = self.box_predictor[k].losses(predictions, proposals, img_classes_oh)
                losses.update(losses_k)

                with torch.no_grad():
                    boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                    scores = self.box_predictor[k].predict_probs(predictions, proposals)
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
        gt_scores, gt_idxs = OICRROIHeads._get_highest_scoring_proposals(scores)

        gt_boxes = [
            boxes_per_image.index_select(0, gt_idxs_per_image)
            for boxes_per_image, gt_idxs_per_image in zip(boxes, gt_idxs)
        ]
        gt_boxes = [gt_boxes_per_image.view(-1, 4) for gt_boxes_per_image in gt_boxes]
        diags = [
            torch.tensor(
                [
                    i * gt_classes_per_image.numel() + i
                    for i in range(gt_classes_per_image.numel())
                ],
                dtype=torch.int64,
                device=gt_classes_per_image.device,
            )
            for gt_classes_per_image in img_classes
        ]
        gt_boxes = [
            gt_boxes_per_image.index_select(0, diags_per_image)
            for gt_boxes_per_image, diags_per_image in zip(gt_boxes, diags)
        ]
        gt_boxes = [Boxes(gt_boxes_per_image) for gt_boxes_per_image in gt_boxes]

        gt = [
            Instances(
                proposals_per_image.image_size,
                gt_boxes = gt_boxes_per_image,
                gt_classes = gt_classes_per_image,
                gt_scores = gt_scores_per_image,
            )
            for (
                proposals_per_image, gt_boxes_per_image, gt_classes_per_image, gt_scores_per_image
            ) in zip(proposals, gt_boxes, img_classes, gt_scores)
        ]

        return gt

    @classmethod
    @torch.no_grad()
    def _get_highest_scoring_proposals(
        self,
        scores: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        gt_scores = []
        gt_idxs = []

        for scores_per_image in scores:
            gt_scores_per_image, gt_idxs_per_image = torch.max(scores_per_image, dim=0)
            gt_scores.append(gt_scores_per_image)
            gt_idxs.append(gt_idxs_per_image)

        return gt_scores, gt_idxs


@ROI_HEADS_REGISTRY.register()
class PCLROIHeads(WeakROIHeads):
    """
    It's PCL's head.
    """
    kmeans_num_clusters: int = 3
    kmeans_seed: int = 2
    graph_iou_threshold: float = 0.4
    min_remain_count: int = 5
    max_num_pc: int = 5

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: List[nn.Module],
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
        kmeans_num_clusters: int = 3,
        kmeans_seed: int = 2,
        graph_iou_threshold: float = 0.4,
        min_remain_count: int = 5,
        max_num_pc: int = 5,
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

        self.kmeans_num_clusters = kmeans_num_clusters
        self.kmeans_seed = kmeans_seed
        self.graph_iou_threshold = graph_iou_threshold
        self.min_remain_count = min_remain_count
        self.max_num_pc = max_num_pc

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
        in_features           = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales         = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        kmeans_num_clusters   = cfg.WSOD.PCL.KMEANS.NUM_CLUSTERS
        kmeans_seed           = cfg.WSOD.PCL.KMEANS.SEED
        graph_iou_threshold   = cfg.WSOD.PCL.GRAPH.IOU_THRESHOLD
        min_remain_count      = cfg.WSOD.PCL.MIN_REMAIN_COUNT
        max_num_pc            = cfg.WSOD.PCL.MAX_NUM_PC
        refine_K              = cfg.WSOD.REFINE_K
        refine_reg            = cfg.WSOD.REFINE_REG
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
        for k in range(1, refine_K + 1):
            box_predictor.append(
                PCLOutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1], "pcl")
            )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "kmeans_num_clusters": kmeans_num_clusters,
            "kmeans_seed": kmeans_seed,
            "graph_iou_threshold": graph_iou_threshold,
            "min_remain_count": min_remain_count,
            "max_num_pc": max_num_pc,
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
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
            for k in range(1, self.refine_K + 1):
                proposals, pseudo_targets = self._get_labeled_proposals_and_pseudo_targets(
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
        gt = []
        for (
            boxes_per_image, scores_per_image, proposals_per_image, img_classes_per_image
        ) in zip(boxes, scores, proposals, img_classes):
            gt_boxes_per_image = []
            gt_scores_per_image = []
            gt_classes_per_image = []

            boxes_per_image_cp = boxes_per_image.clone()
            scores_per_image_cp = scores_per_image.clone().clamp_(1e-9, 1 - 1e-9)

            for i, cls_idx in enumerate(img_classes_per_image.reshape(-1, 1)):
                remain_idxs = torch.arange(scores_per_image_cp.size(0), device=scores_per_image_cp.device)
                boxes_per_class = boxes_per_image_cp[:, i]
                scores_per_class = scores_per_image_cp[:, i]

                top_ranking_idxs = PCLROIHeads._get_top_ranking_proposals(
                    scores_per_class.reshape(-1, 1)
                )
                top_ranking_boxes = boxes_per_class[top_ranking_idxs]
                top_ranking_scores = scores_per_class[top_ranking_idxs]

                graph = PCLROIHeads._build_graph(top_ranking_boxes)

                keep_idxs = []
                gt_scores_per_class = []
                remain_count = top_ranking_scores.size(0)
                while True:
                    max_idx = graph.sum(dim=0).argmax()
                    keep_idxs.append(max_idx.reshape(-1))
                    conj_idxs = torch.where(graph[max_idx, :] > 0)[0]
                    gt_scores_per_class.append(
                        top_ranking_scores[conj_idxs].max(dim=0, keepdim=True).values
                    )

                    graph[:, conj_idxs] = 0
                    graph[conj_idxs, :] = 0
                    remain_count -= conj_idxs.size(0)
                    if remain_count <= PCLROIHeads.min_remain_count:
                        break
                keep_idxs = torch.cat(keep_idxs, dim=0)
                gt_scores_per_class = torch.cat(gt_scores_per_class, dim=0)
                gt_boxes_per_class = top_ranking_boxes[keep_idxs]

                keep_idxs_new = gt_scores_per_class.argsort(descending=True)[
                    : min(keep_idxs.size(0), PCLROIHeads.max_num_pc)
                ]
                gt_scores_per_image.append(gt_scores_per_class[keep_idxs_new])
                gt_boxes_per_image.append(gt_boxes_per_class[keep_idxs_new])
                gt_classes_per_image.append(cls_idx.new_full((keep_idxs_new.size(0), ), cls_idx.item()))

                # If a proposal is chosen as a cluster center, we simply delete a proposal
                # from the candidata proposal pool, because we found that the results of
                # different strategies are similar and this strategy is more efficient.
                delete_idxs = top_ranking_idxs[keep_idxs][keep_idxs_new]
                remain_idxs = torch.isin(remain_idxs, delete_idxs, assume_unique=True, invert=True)
                boxes_per_image_cp = boxes_per_image_cp[remain_idxs]
                scores_per_image_cp = scores_per_image_cp[remain_idxs]

            gt_boxes_per_image = torch.cat(gt_boxes_per_image, dim=0)
            gt_scores_per_image = torch.cat(gt_scores_per_image, dim=0)
            gt_classes_per_image = torch.cat(gt_classes_per_image, dim=0)

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
    @torch.no_grad()
    def _get_top_ranking_proposals(
        self,
        scores_per_class: torch.Tensor,
    ) -> torch.Tensor:
        scores_per_class_np = scores_per_class.cpu().numpy()
        n_clusters = min(PCLROIHeads.kmeans_num_clusters, scores_per_class_np.shape[0])
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=PCLROIHeads.kmeans_seed
        ).fit(scores_per_class_np)

        highest_cluster_center_label = np.argmax(kmeans.cluster_centers_)
        top_ranking_idxs = np.where(kmeans.labels_ == highest_cluster_center_label)[0]
        if len(top_ranking_idxs) == 0:
            top_ranking_idxs = np.array([np.argmax(scores_per_class_np)])

        return torch.as_tensor(top_ranking_idxs, device=scores_per_class.device)

    @classmethod
    @torch.no_grad()
    def _build_graph(
        self,
        top_ranking_boxes: torch.Tensor,
    ) -> torch.Tensor:
        match_quality_matrix = pairwise_iou(Boxes(top_ranking_boxes), Boxes(top_ranking_boxes))
        return match_quality_matrix > PCLROIHeads.graph_iou_threshold


@ROI_HEADS_REGISTRY.register()
class MISTROIHeads(WeakROIHeads):
    """
    It's MIST's head.
    """
    top_score_percent: float = 0.15
    graph_iou_threshold: float = 0.2

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: List[nn.Module],
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
        top_score_percent: float = 0.15,
        graph_iou_threshold: float = 0.2,
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

        self.top_score_percent = top_score_percent
        self.graph_iou_threshold = graph_iou_threshold

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
        in_features           = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales         = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        top_score_percent     = cfg.WSOD.MIST.TOP_SCORE_PERCENT
        graph_iou_threshold   = cfg.WSOD.MIST.GRAPH.IOU_THRESHOLD
        refine_K              = cfg.WSOD.REFINE_K
        refine_reg            = cfg.WSOD.REFINE_REG
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
        for k in range(1, refine_K + 1):
            box_predictor.append(
                PCLOutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1], "mist")
            )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
            "top_score_percent": top_score_percent,
            "graph_iou_threshold": graph_iou_threshold,
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
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
            for k in range(1, self.refine_K + 1):
                proposals, _ = self._get_labeled_proposals_and_pseudo_targets(
                    boxes, scores, proposals, img_classes,
                    MISTROIHeads.get_pseudo_targets,
                    suffix="_r{}".format(k),
                )

                predictions = self.box_predictor[k](box_features)
                losses_k = self.box_predictor[k].losses(predictions, proposals, img_classes_oh)
                losses.update(losses_k)

                with torch.no_grad():
                    boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                    scores = self.box_predictor[k].predict_probs(predictions, proposals)
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
        gt = []
        for (
            boxes_per_image, scores_per_image, proposals_per_image, img_classes_per_image
        ) in zip(boxes, scores, proposals, img_classes):
            topk_per_image = int(len(proposals_per_image) * MISTROIHeads.top_score_percent)
            top_idxs_per_image = scores_per_image.argsort(dim=0, descending=True)[: topk_per_image]
            top_boxes_per_image = boxes_per_image.gather(
                dim=0, index=top_idxs_per_image.unsqueeze(2).expand(-1, -1, 4)
            )
            top_scores_per_image = scores_per_image.gather(dim=0, index=top_idxs_per_image)

            gt_boxes_per_image = []
            gt_scores_per_image = []
            gt_classes_per_image = []

            for i, cls_idx in enumerate(img_classes_per_image.reshape(-1, 1)):
                boxes_per_class = top_boxes_per_image[:, i]
                scores_per_class = top_scores_per_image[:, i]

                keep_idxs = cls_idx.new_zeros(topk_per_image, dtype=torch.bool)
                keep_idxs[0] = True
                match_quality_matrix = pairwise_iou(Boxes(boxes_per_class), Boxes(boxes_per_class))
                graph = match_quality_matrix < MISTROIHeads.graph_iou_threshold
                for k in range(1, topk_per_image):
                    keep_idxs[k] = graph[k, 0: k].all()

                gt_boxes_per_image.append(boxes_per_class[keep_idxs])
                gt_scores_per_image.append(scores_per_class[keep_idxs])
                gt_classes_per_image.append(cls_idx.expand(keep_idxs.sum()))

            gt_boxes_per_image = torch.cat(gt_boxes_per_image, dim=0)
            gt_scores_per_image = torch.cat(gt_scores_per_image, dim=0)
            gt_classes_per_image = torch.cat(gt_classes_per_image, dim=0)

            gt.append(
                Instances(
                    proposals_per_image.image_size,
                    gt_boxes = Boxes(gt_boxes_per_image),
                    gt_classes = gt_classes_per_image,
                    gt_scores = gt_scores_per_image,
                )
            )
        return gt


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
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
                mean_scores = [torch.zeros_like(scores_per_image) for scores_per_image in scores]
            for k in range(1, self.refine_K):
                proposals, pseudo_targets = self._get_labeled_proposals_and_pseudo_targets(
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
            proposals, _ = self._get_labeled_proposals_and_pseudo_targets(
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


@ROI_HEADS_REGISTRY.register()
class DTHCPROIHeads(WeakROIHeads):
    """
    It's HGPS's head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: List[nn.Module],
        train_on_pred_boxes: bool = False,
        cls_agnostic_bbox_reg: bool = False,
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
        in_features           = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales         = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        refine_K              = cfg.WSOD.REFINE_K
        refine_reg            = cfg.WSOD.REFINE_REG
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
        box_predictor = [WSBDNOutputLayers(cfg, box_head.output_shape)]

        assert refine_K == len(refine_reg), "{} != {}".format(refine_K, len(refine_reg))
        for k in range(1, refine_K + 1):
            box_predictor.append(
                HGPSOutputLayers(cfg, box_head.output_shape, k, refine_reg[k - 1], "hgps")
            )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "cls_agnostic_bbox_reg": cls_agnostic_bbox_reg,
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
            pseudo_targets = WSBDNROIHeads.get_pseudo_targets(proposals, img_classes)
            proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(proposals, pseudo_targets)
            self.proposal_append_gt = proposal_append_gt

            predictions = self.box_predictor[0](box_features)
            losses = self.box_predictor[0].losses(predictions, proposals, img_classes_oh)

            with torch.no_grad():
                boxes = self.box_predictor[0].predict_boxes(predictions, proposals)
                scores = self.box_predictor[0].predict_probs(predictions, proposals)
            for k in range(1, self.refine_K + 1):
                proposals, _ = self._get_labeled_proposals_and_pseudo_targets(
                    boxes, scores, proposals, img_classes,
                    DTHCPROIHeads.get_pseudo_targets,
                    suffix="_r{}".format(k),
                )

                predictions = self.box_predictor[k](box_features)
                losses_k = self.box_predictor[k].losses(predictions, proposals, img_classes_oh)
                losses.update(losses_k)

                with torch.no_grad():
                    boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                    scores = self.box_predictor[k].predict_probs(predictions, proposals)
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

                clusters_per_class = proposals_per_image.clusters[cls_idx.item()]
                for cluster in clusters_per_class:
                    boxes_per_cluster = boxes_per_class[cluster]
                    scores_per_cluster = scores_per_class[cluster]
                    top_idx = scores_per_cluster.argmax()

                    gt_boxes_per_image.append(boxes_per_cluster[top_idx])
                    gt_scores_per_image.append(scores_per_cluster[top_idx])
                    gt_classes_per_image.append(cls_idx)

            gt_boxes_per_image = torch.stack(gt_boxes_per_image, dim=0)
            gt_scores_per_image = torch.stack(gt_scores_per_image, dim=0)
            gt_classes_per_image = torch.stack(gt_classes_per_image, dim=0)

            gt.append(
                Instances(
                    proposals_per_image.image_size,
                    gt_boxes = Boxes(gt_boxes_per_image),
                    gt_classes = gt_classes_per_image,
                    gt_scores = gt_scores_per_image,
                )
            )
        return gt
