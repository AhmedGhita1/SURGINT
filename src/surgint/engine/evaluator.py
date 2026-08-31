from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.dataset.transform import Transform
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.model.decode import decode
from surgint.model.detector import Detector


def evaluate(detector: Detector, loader: DataLoader, transform: Transform) -> Dict:
    """coco mAP over the loader's split, through the same decode the application runs"""
    dataset = loader.dataset
    predictions = []

    for batch in tqdm(loader, desc="eval", leave=False):
        logits, pred_boxes = detector.predict(batch["pixel_values"])
        detections = decode(logits, pred_boxes, score_threshold=0.0)

        for (boxes, scores, class_ids), image_id, scale, frame_size in zip(
            detections, batch["image_ids"], batch["scales"], batch["frame_sizes"]
        ):
            boxes = transform.postprocess(boxes, scale, frame_size)
            predictions += coco_predictions(image_id, boxes, scores, class_ids, dataset.mappings)

    return coco_evaluate(dataset.gt, predictions)
