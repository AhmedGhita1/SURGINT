import contextlib
import io

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from surgint.dataset.coco import CocoDetection
from surgint.detection.postprocessing import decode, to_frame_boxes
from surgint.evaluation import MetricsFn
from surgint.inference.detector import DetectionResult


def coco_predictions(image_id: int, result, label_to_category: dict[int, int]) -> list[dict]:
    records = []
    for box, score, label in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = (float(value) for value in box)
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label_to_category[int(label)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return records


def evaluate(annotations: dict, predictions: list[dict]) -> dict:

    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = annotations
        truth.createIndex()
        names = {category["id"]: category["name"] for category in truth.loadCats(truth.getCatIds())}
        if not predictions:
            return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "per_class": dict.fromkeys(names.values(), 0.0)}

        evaluation = COCOeval(truth, truth.loadRes(predictions), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()

    # precision is [iou, recall, category, area, max_detections]; area 0 and max 100 are the defaults
    precision = evaluation.eval["precision"][:, :, :, 0, 2]
    per_class = {}
    for index, category_id in enumerate(evaluation.params.catIds):
        scores = precision[:, :, index]
        scores = scores[scores > -1]
        per_class[names[category_id]] = float(scores.mean()) if scores.size else float("nan")

    return {
        "mAP50_95": float(evaluation.stats[0]),
        "mAP50": float(evaluation.stats[1]),
        "mAP75": float(evaluation.stats[2]),
        "per_class": per_class,
    }


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

        return evaluate(dataset.annotations, predictions)

    return metrics_fn
