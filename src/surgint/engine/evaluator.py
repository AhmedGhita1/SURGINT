from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.model.transform import Transform
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.evaluation.mot import mot_counts, mot_evaluate
from surgint.model.decode import decode
from surgint.model.detector import Detector
from surgint.runtime.pipeline import Pipeline


def evaluate(detector: Detector, loader: DataLoader, transform: Transform) -> Dict:
    """
    evaluate the detector on one split.

    returns mAP50_95, mAP50, mAP75, and per-class AP.
    """
    dataset = loader.dataset
    predictions = []

    batches = tqdm(loader, desc="eval", leave=False)
    for batch in batches:
        logits, pred_boxes = detector.predict(batch["pixel_values"])
        detections = decode(logits, pred_boxes, score_threshold=0.0)

        for (boxes, scores, class_ids), image_id, scale, frame_size in zip(
            detections, batch["image_ids"], batch["scales"], batch["frame_sizes"]
        ):
            boxes = transform.postprocess(boxes, scale, frame_size)
            predictions += coco_predictions(image_id, boxes, scores, class_ids, dataset.mappings)

        batches.set_postfix(predictions=len(predictions))

    return coco_evaluate(dataset.gt, predictions)


def evaluate_sessions(pipeline: Pipeline, sessions: Dict) -> Dict:
    """
    evaluate the detector and tracker on a set of sessions.

    sessions maps a session name to a SurgintDataset built without a transform.
    returns MOTA, IDF1, id switches, the counts behind them, and the same per session.
    """
    if pipeline.tracker is None:
        raise ValueError(f"evaluate_sessions requires a tracker, got task {pipeline.task!r}")

    counts = {}
    for name, dataset in sessions.items():
        # each session is one continuous camera path, so its ids start from scratch
        pipeline.reset()

        frames = []
        for index in tqdm(range(len(dataset)), desc=name, leave=False):
            sample = dataset[index]
            detections = pipeline.predict(sample["frame"], 0.0)
            frames.append(
                (sample["boxes"], sample["track_ids"], detections.boxes, detections.track_ids)
            )
        counts[name] = mot_counts(frames)

    return {
        **mot_evaluate(list(counts.values())),
        "per_session": {name: mot_evaluate(count) for name, count in counts.items()},
    }
