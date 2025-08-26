#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from typing import Optional

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    DefaultTrainer,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    print_csv_format,
    verify_results,
)
from detectron2.visualization import DatasetVisualizer

import wsod.data.datasets
from wsod.config import add_wsod_cfg
from wsod.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from wsod.engine import default_setup
from wsod.evaluation import inference_on_dataset
from wsod.modeling import build_test_time_aug


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name, output_dir=output_folder)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name: str, output_folder: Optional[str] = None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_visualizer(cls, cfg, dataset_name: str, output_folder: str, score_thresh: float):
        return DatasetVisualizer(
            dataset_name, output_dir=output_folder, score_thresh=score_thresh
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None, visualizers=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            )
            cfg.freeze()

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        if isinstance(visualizers, DatasetVisualizer):
            visualizers = [visualizers]
        if visualizers is not None:
            assert len(cfg.DATASETS.TEST) == len(visualizers), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(visualizers)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)

            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            if visualizers is not None:
                visualizer = visualizers[idx]
            else:
                visualizer = cls.build_visualizer(
                    cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "vis"), cfg.TEST.VIS.SCORE_THRESH
                ) if cfg.TEST.VIS.ENABLED else None

            results_i = inference_on_dataset(model, data_loader, evaluator, visualizer)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            )
            cfg.freeze()

        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = build_test_time_aug(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        visualizers = [
            cls.build_visualizer(
                cfg,
                name,
                output_folder=os.path.join(cfg.OUTPUT_DIR, "vis_TTA"),
                score_thresh=cfg.TEST.VIS.SCORE_THRESH,
            )
            for name in cfg.DATASETS.TEST
        ] if cfg.TEST.VIS.ENABLED else None

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        res = cls.test(cfg, model, evaluators, visualizers)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_wsod_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
