import argparse
import cv2
import numpy as np
import os
import pickle
import torch
import tqdm
from skimage.measure import label, regionprops
from typing import Any, Dict, List, Tuple

from detectron2.data.catalog import DatasetCatalog

import wsod.data.datasets


CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

CLASS_NAME_IDS = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}


def resize_box(
    x1: int, y1: int, x2: int, y2: int,
    resize_ratio: float,
    image_size: Tuple[int, int],
    align_corners=False,
) -> Tuple[int, int, int, int]:
    if align_corners:
        offset_x = (resize_ratio - 1) / 2 * (x2 - x1 - 1)
        offset_y = (resize_ratio - 1) / 2 * (y2 - y1 - 1)
    else:
        offset_x = (resize_ratio - 1) / 2 * (x2 - x1)
        offset_y = (resize_ratio - 1) / 2 * (y2 - y1)
    x1 -= offset_x
    x2 += offset_x
    y1 -= offset_y
    y2 += offset_y
    return (
        max(int(x1), 0),
        max(int(y1), 0),
        min(int(x2), image_size[1]),
        min(int(y2), image_size[0]),
    )


def mask_in(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return (mask1[:, None] & mask2).any(dim=(2, 3))


def pairwise_in(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    left_top = boxes1[:, None, : 2] >= boxes2[:, : 2]
    right_bottom = boxes1[:, None, 2:] <= boxes2[:, 2:]
    return (left_top & right_bottom).all(dim=2)


def pairwise_equal(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    return (boxes1[:, None, :] == boxes2[None, :, :]).all(dim=2)


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)

    return inter * 1.0 / (area1[:, None] + area2 - inter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", default="voc_2007_trainval", type=str)
    parser.add_argument("--heatmap-dir", default="datasets/heatmaps/", type=str)
    parser.add_argument("--proposal-dir", default="datasets/proposals/", type=str)
    parser.add_argument("--proposal-type", default="mcg", type=str)

    parser.add_argument("--low-score-thresh", default=0.30, type=float)
    parser.add_argument("--high-score-thresh", default=0.80, type=float)
    parser.add_argument("--resize-ratio", default=1.2, type=float)

    args = parser.parse_args()
    dataset_dicts = DatasetCatalog.get(args.dataset_name)

    with open(os.path.join(args.heatmap_dir, "{}_cams_d2.pkl".format(args.dataset_name)), 'rb') as f:
        cams = pickle.load(f)

    with open(os.path.join(args.proposal_dir, "{}_{}_proposals_d2.pkl".format(args.dataset_name, args.proposal_type)), 'rb') as f:
        proposals = pickle.load(f)
    proposals_new = {
        "boxes": [],
        "objectness_logits": [],
        "ids": proposals["ids"],
        "bbox_mode": proposals["bbox_mode"],
        "clusters": [],
    }

    for data_dict, cams_per_image, proposal_boxes, proposal_objectness_logits, id1, id2 in tqdm.tqdm(
        zip(dataset_dicts, cams["cams"], proposals["boxes"], proposals["objectness_logits"], cams["ids"], proposals["ids"])
    ):
        assert str(data_dict["image_id"]) == str(id1) == str(id2), "{} = {} = {}".format(data_dict["image_id"], id1, id2)

        img = cv2.imread(data_dict["file_name"])

        proposal_boxes = torch.from_numpy(proposal_boxes).to(torch.int64)
        proposal_objectness_logits = torch.from_numpy(proposal_objectness_logits).to(torch.float32)

        pseudo_gt_ids: List[Tuple[int, torch.Tensor]] = []
        clusters: List[Dict[str, Any]] = []
        for cls_idx, cam in cams_per_image.items():
            pseudo_gt_ids_per_class: List[Tuple[int, torch.Tensor]] = []
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heat_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

            binary_mask = np.int64(cam >= args.low_score_thresh)
            labeled_mask = label(binary_mask)
            regions = regionprops(labeled_mask)
            low_masks = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.bool)
            low_thresh_boxes = torch.zeros((0, 4), dtype=torch.int64)
            low_thresh_boxes_re = torch.zeros((0, 4), dtype=torch.int64)
            for r_id, region in enumerate(regions):
                if region.area < 50:
                    continue
                y1, x1, y2, x2 = region.bbox
                low_masks = torch.cat([low_masks, torch.from_numpy(labeled_mask == r_id + 1).unsqueeze(0)])
                low_thresh_boxes = torch.cat([low_thresh_boxes, torch.tensor([x1, y1, x2, y2], dtype=torch.int64).unsqueeze(0)])
                x1_re, y1_re, x2_re, y2_re = resize_box(x1, y1, x2, y2, args.resize_ratio, img.shape)
                low_thresh_boxes_re = torch.cat([low_thresh_boxes_re, torch.tensor([x1_re, y1_re, x2_re, y2_re], dtype=torch.int64).unsqueeze(0)])

            binary_mask = np.int64(cam >= args.high_score_thresh)
            labeled_mask = label(binary_mask)
            regions = regionprops(labeled_mask)
            high_masks = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.bool)
            high_thresh_boxes = torch.zeros((0, 4), dtype=torch.int64)
            high_thresh_boxes_re = torch.zeros((0, 4), dtype=torch.int64)
            for r_id, region in enumerate(regions):
                if region.area < 50:
                    continue
                y1, x1, y2, x2 = region.bbox
                high_masks = torch.cat([high_masks, torch.from_numpy(labeled_mask == r_id + 1).unsqueeze(0)])
                high_thresh_boxes = torch.cat([high_thresh_boxes, torch.tensor([x1, y1, x2, y2], dtype=torch.int64).unsqueeze(0)])
                x1_re, y1_re, x2_re, y2_re = resize_box(x1, y1, x2, y2, args.resize_ratio, img.shape)
                high_thresh_boxes_re = torch.cat([high_thresh_boxes_re, torch.tensor([x1_re, y1_re, x2_re, y2_re], dtype=torch.int64).unsqueeze(0)])

            high_in_low = mask_in(high_masks, low_masks)
            assert (high_in_low.sum(dim=1) == 1).all() == True, "Something is wrong."
            assert high_in_low.shape[1] > 0, "Must have low boxes."

            low_eq_prop = pairwise_equal(low_thresh_boxes, proposal_boxes)
            high_re_eq_prop = pairwise_equal(high_thresh_boxes_re, proposal_boxes)

            for i, low_count in enumerate(high_in_low.sum(dim=0)):
                if low_count == 0:
                    is_in = torch.where(low_eq_prop[i] == True)[0]
                    assert is_in.numel() <= 1, "The proposals in image {} is wrong.".format(data_dict["image_id"])
                    if is_in.numel() == 0:
                        proposal_boxes = torch.vstack([proposal_boxes, low_thresh_boxes[i]])
                        proposal_objectness_logits = torch.cat([proposal_objectness_logits, torch.tensor([0.7])])
                        pseudo_gt_ids_per_class.append((cls_idx, torch.tensor([proposal_boxes.shape[0] - 1], dtype=torch.int64)))
                    else:
                        pseudo_gt_ids_per_class.append((cls_idx, is_in))
                elif low_count == 1:
                    is_in = torch.where(low_eq_prop[i] == True)[0]
                    assert is_in.numel() <= 1, "The proposals in image {} is wrong.".format(data_dict["image_id"])
                    if is_in.numel() == 0:
                        proposal_boxes = torch.vstack([proposal_boxes, low_thresh_boxes[i]])
                        proposal_objectness_logits = torch.cat([proposal_objectness_logits, torch.tensor([0.7])])
                    high_box = high_thresh_boxes[high_in_low[:, i]]
                    low_box = low_thresh_boxes_re[[i]]
                    high_in_prop = pairwise_in(high_box, proposal_boxes)
                    low_contain_prop = pairwise_in(proposal_boxes, low_box).T
                    prop_fit = torch.where(high_in_prop & low_contain_prop == True)
                    pseudo_gt_ids_per_class.append((cls_idx, prop_fit[1]))
                else:
                    is_in = torch.where(high_re_eq_prop[high_in_low[:, i]] == True)
                    for l_c in range(low_count):
                        prop_eq_ids = is_in[1][is_in[0] == l_c]
                        assert prop_eq_ids.numel() <= 1, "The proposals in image {} is wrong.".format(data_dict["image_id"])
                        if prop_eq_ids.numel() == 0:
                            proposal_boxes = torch.vstack([proposal_boxes, high_thresh_boxes_re[high_in_low[:, i]][l_c]])
                            proposal_objectness_logits = torch.cat([proposal_objectness_logits, torch.tensor([0.7])])

                    prop_fit_list = []
                    for l_c in range(low_count):
                        high_box = high_thresh_boxes[high_in_low[:, i]][[l_c]]
                        low_box = low_thresh_boxes_re[[i]]
                        high_in_prop = pairwise_in(high_box, proposal_boxes)
                        low_contain_prop = pairwise_in(proposal_boxes, low_box).T
                        prop_fit = torch.where(high_in_prop & low_contain_prop == True)
                        prop_fit_list.append(prop_fit[1])

                    def find_elements_in_at_least_two(ids_list: List[torch.Tensor], max_id: int):
                        total_ids = torch.arange(max_id + 1, dtype=torch.int64)
                        matches_count = torch.cat(
                            [(total_ids[:, None] == ids).sum(1, keepdim=True) for ids in ids_list],
                            dim=1,
                        ).sum(1)
                        return total_ids[matches_count >= 2]
                    repetitive_ids = find_elements_in_at_least_two(prop_fit_list, proposal_boxes.shape[0] - 1)
                    ious = pairwise_iou(proposal_boxes[repetitive_ids], high_thresh_boxes_re[high_in_low[:, i]])
                    repetitive_belong_to = ious.argmax(1)
                    for l_c in range(low_count):
                        remain_ids = ~torch.isin(prop_fit_list[l_c], repetitive_ids[repetitive_belong_to != l_c], assume_unique=True)
                        pseudo_gt_ids_per_class.append((cls_idx, prop_fit_list[l_c][remain_ids]))

            pseudo_gt_ids.extend(pseudo_gt_ids_per_class)

        assert proposal_boxes.shape[0] == proposal_objectness_logits.shape[0], "Fuck!"
        inds = proposal_objectness_logits.argsort(descending=True)
        for pgi, pg in pseudo_gt_ids:
            find = (inds == pg.unsqueeze(1)).to(torch.int64)
            clusters.append({
                "category_id": pgi,
                "proposal_ids": find.argmax(1).sort().values.numpy(),
            })

        proposals_new["boxes"].append(proposal_boxes[inds].numpy())
        proposals_new["objectness_logits"].append(proposal_objectness_logits[inds].numpy())
        proposals_new["clusters"].append(clusters)
    with open(os.path.join(args.proposal_dir, "{}_{}_hgps_proposals_d2.pkl".format(args.dataset_name, args.proposal_type)), 'wb') as f:
        pickle.dump(proposals_new, f, pickle.HIGHEST_PROTOCOL)
