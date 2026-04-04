import logging
import os
import torch

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import ColorMode, Visualizer

__all__ = ["DatasetVisualizer"]


class DatasetVisualizer:
    """
    Visualize the predictions and annotations for dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        score_thresh: float = 0.5,
        instance_mode: int = ColorMode.IMAGE,
    ) -> None:
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test".
            output_dir (str): an output directory to dump all
                visualization results on the dataset.
            score_thresh (float): score threshold used to de-visualize
                boxes exceeding its value.
        """
        self._metadata = MetadataCatalog.get(dataset_name)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._output_dir = os.path.join(output_dir, dataset_name)

        self._score_thresh = score_thresh
        self._instance_mode = instance_mode

    def reset(self):
        self._predictions = []  # (image info, instances to draw)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            input_ = {"file_name": input["file_name"], "image_id": input["image_id"]}

            preds = output["instances"]
            keep_idxs = preds.scores > self._score_thresh
            output_ = {"instances": preds[keep_idxs].to(self._cpu_device)}

            self._predictions.append((input_, output_))

    def visualize(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = []
        for predictions_per_rank in all_predictions:
            predictions.extend(predictions_per_rank)
        del all_predictions

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            for input, output in predictions:
                image = read_image(input["file_name"], "RGB")
                visualizer = Visualizer(image, self._metadata, instance_mode=self._instance_mode)
                vis_output = visualizer.draw_instance_predictions(predictions=output["instances"])
                vis_output.save(os.path.join(self._output_dir, "{}_pred.jpg".format(input["image_id"])))
                del visualizer
