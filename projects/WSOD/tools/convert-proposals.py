import argparse
import cv2
import numpy as np
import os
import pickle
import scipy.io as sio
import tqdm
from argparse import Namespace

from detectron2.data.catalog import DatasetCatalog

import wsod.data.datasets

parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", default="voc_2007_trainval", type=str)
parser.add_argument("--proposal-type", default="mcg", choices=["ss", "mcg"], type=str)
parser.add_argument("--proposal-dir", default="datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes", choices=["ss", "mcg"], type=str)


def convert_ss_box(args: Namespace) -> None:
    dataset_name = args.dataset_name
    dataset_dicts = DatasetCatalog.get(dataset_name)

    raw_data = sio.loadmat(args.proposal_dir)["boxes"].ravel()
    assert raw_data.shape[0] == len(dataset_dicts)

    boxes = []
    scores = []
    ids = []
    for i in tqdm.tqdm(range(len(dataset_dicts))):
        if "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        image = cv2.imread(dataset_dicts[i]["file_name"])

        # selective search boxes are 1-indexed and (y1, x1, y2, x2)
        i_boxes = raw_data[i][:, (1, 0, 3, 2)]
        i_boxes[:, 0] -= 1
        i_boxes[:, 1] -= 1
        x1 = i_boxes[:, 0].clip(0, image.shape[1])
        y1 = i_boxes[:, 1].clip(0, image.shape[0])
        x2 = i_boxes[:, 2].clip(0, image.shape[1])
        y2 = i_boxes[:, 3].clip(0, image.shape[0])
        i_boxes = np.stack((x1, y1, x2, y2), axis=-1)
        widths = i_boxes[:, 2] - i_boxes[:, 0]
        heights = i_boxes[:, 3] - i_boxes[:, 1]
        keep = (widths > 0) & (heights > 0)
        i_scores = np.ones((i_boxes.shape[0], ), dtype=np.float32)

        boxes.append(i_boxes.astype(np.int64)[keep])
        scores.append(i_scores.astype(np.float32)[keep])
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(os.path.join(
        "datasets", "proposals", "{}_ss_proposals_d2.pkl".format(dataset_name)
    ), "wb") as f:
        pickle.dump(dict(
            boxes=boxes,
            objectness_logits=scores,
            ids=ids,
            bbox_mode=0,
        ), f, pickle.HIGHEST_PROTOCOL)


def convert_mcg_box(args: Namespace) -> None:
    dataset_name = args.dataset_name
    dataset_dicts = DatasetCatalog.get(dataset_name)

    boxes = []
    scores = []
    ids = []
    for i in tqdm.tqdm(range(len(dataset_dicts))):
        if "coco" in dataset_name:
            index = os.path.basename(dataset_dicts[i]["file_name"])[:-4]
        else:
            index = dataset_dicts[i]["image_id"]
        image = cv2.imread(dataset_dicts[i]["file_name"])
        box_file = os.path.join(args.proposal_dir, "{}.mat".format(index))
        mat_data = sio.loadmat(box_file)

        boxes_data = mat_data["boxes"]
        scores_data = mat_data["scores"]

        # Boxes from the MCG website are in (y1, x1, y2, x2) order
        boxes_data = boxes_data[:, (1, 0, 3, 2)]
        boxes_data[:, 0] -= 1
        boxes_data[:, 1] -= 1
        x1 = boxes_data[:, 0].clip(0, image.shape[1])
        y1 = boxes_data[:, 1].clip(0, image.shape[0])
        x2 = boxes_data[:, 2].clip(0, image.shape[1])
        y2 = boxes_data[:, 3].clip(0, image.shape[0])
        boxes_data = np.stack((x1, y1, x2, y2), axis=-1)
        widths = boxes_data[:, 2] - boxes_data[:, 0]
        heights = boxes_data[:, 3] - boxes_data[:, 1]
        keep = (widths > 0) & (heights > 0)

        boxes.append(boxes_data.astype(np.int64)[keep])
        scores.append(np.squeeze(scores_data.astype(np.float32))[keep])
        index = dataset_dicts[i]["image_id"]
        ids.append(index)

    with open(os.path.join(
        "datasets", "proposals", "{}_mcg_proposals_d2.pkl".format(dataset_name)
    ), "wb") as f:
        pickle.dump(dict(
            boxes=boxes,
            objectness_logits=scores,
            ids=ids,
            bbox_mode=0,
        ), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.proposal_type == "ss":
        convert_ss_box(args)
    elif args.proposal_type == "mcg":
        convert_mcg_box(args)
