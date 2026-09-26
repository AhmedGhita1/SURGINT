from typing import Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from surgint.model.transform import Transform
from surgint.evaluation.coco_eval import coco_evaluate, coco_predictions
from surgint.evaluation.inventory import inventory_counts, inventory_evaluate
from surgint.evaluation.mot import mot_counts, mot_evaluate
from surgint.model.decode import decode
from surgint.model.detector import Detector
from surgint.runtime.inventory import Inventory
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


def evaluate_sessions(
    pipeline: Pipeline,
    sessions: Dict,
    iou_threshold: float = 0.5,
    class_ids=None,
) -> Dict:
    """
    evaluate the detector and tracker on a set of sessions.

    sessions maps a session name to a SurgintDataset built without a transform.
    returns MOT and finalized inventory metrics, their primitive counts, and the same
    metrics per session.
    """
    if pipeline.tracker is None:
        raise ValueError(f"evaluate_sessions requires a tracker, got task {pipeline.task!r}")

    mot = {}
    inventory_records = {}
    observed_classes = set()
    for name, dataset in sessions.items():
        # each session is one continuous camera path, so its ids start from scratch
        pipeline.reset()
        inventory = Inventory()

        frames = []
        inventory_frames = []
        for index in tqdm(range(len(dataset)), desc=name, leave=False):
            sample = dataset[index]
            if "instance_uids" not in sample:
                raise ValueError(
                    f"session {name!r} has no instance_uid ground truth for inventory evaluation"
                )
            detections = pipeline.predict(sample["frame"], 0.0)
            inventory.update(detections)
            frames.append((
                sample["boxes"],
                sample["track_ids"],
                sample["class_ids"],
                detections.boxes,
                detections.track_ids,
                detections.class_ids,
            ))
            inventory_frames.append((sample["instance_uids"], sample["class_ids"]))
            observed_classes.update(int(class_id) for class_id in sample["class_ids"])
            observed_classes.update(int(class_id) for class_id in detections.class_ids)
        mot[name] = mot_counts(frames, iou_threshold)
        inventory_records[name] = (inventory_frames, inventory.finalize().counts())

    classes = tuple(class_ids) if class_ids is not None else tuple(sorted(observed_classes))
    inventory = {
        name: inventory_counts(frames, predictions, classes)
        for name, (frames, predictions) in inventory_records.items()
    }

    per_session = {
        name: {
            **mot_evaluate(mot[name]),
            **inventory_evaluate(inventory[name]),
        }
        for name in sessions
    }

    return {
        **mot_evaluate(list(mot.values())),
        **inventory_evaluate(list(inventory.values())),
        "per_session": per_session,
    }
