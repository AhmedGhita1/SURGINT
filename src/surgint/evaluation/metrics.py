import numpy as np
import torch
from torch.utils.data import DataLoader

from surgint.dataset.coco import CocoDetection
from surgint.detection.postprocessing import decode, to_frame_boxes
from surgint.evaluation import MetricsFn
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.inference.detector import DetectionResult


def iou_matrix(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    others = np.asarray(others, dtype=float)

    top_left = np.maximum(boxes[:, None, :2], others[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], others[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, 0, None)
    intersection = overlap[..., 0] * overlap[..., 1]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = areas[:, None] + other_areas[None, :] - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def count_matches(predicted: np.ndarray, ground_truth: np.ndarray, iou_threshold: float = 0.5) -> int:
    """greedy match, highest scoring prediction first; each ground-truth box is claimed once"""
    if len(predicted) == 0 or len(ground_truth) == 0:
        return 0

    ious = iou_matrix(predicted, ground_truth)
    claimed = np.zeros(len(ground_truth), dtype=bool)
    matches = 0
    for row in ious:
        candidates = np.where(claimed, -1.0, row)
        best = candidates.argmax()
        if candidates[best] >= iou_threshold:
            claimed[best] = True
            matches += 1
            if claimed.all():
                break
    return matches


def build_metrics_fn(dataset: CocoDetection, loader: DataLoader) -> MetricsFn:
    """COCO mAP over a complete dataset, through the inference postprocessing."""
    to_category = {label: category for category, label in dataset.category_map.items()}
    width, height = dataset.width, dataset.height

    @torch.inference_mode()
    def metrics_fn(model: torch.nn.Module) -> dict:
        model.eval()
        device = next(model.parameters()).device
        predictions = []

        for batch in loader:
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            detections = decode(outputs.logits.cpu(), outputs.pred_boxes.cpu(), width, height, 0.0)
            for (boxes, scores, class_ids), image_id, scale, frame_size in zip(
                detections, batch["image_ids"], batch["scales"], batch["frame_sizes"]
            ):
                result = DetectionResult(to_frame_boxes(boxes, scale, frame_size), scores, class_ids)
                predictions += coco_predictions(image_id, result, to_category)

        return coco_evaluate(dataset.annotations, predictions)

    return metrics_fn
