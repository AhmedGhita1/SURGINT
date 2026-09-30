from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from surgint.model import TASKS, Detections
from surgint.model.boxes import nms
from surgint.model.decode import decode
from surgint.model.detector import Detector
from surgint.model.transform import Transform
from surgint.runtime.bytetrack import ByteTrack


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        transform: Transform,
        task: str,
        tracker: Optional[dict] = None,
        max_detections: Optional[int] = None,
        nms_iou: Optional[float] = None,
    ):

        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")

        self.detector = detector
        self.transform = transform
        self.task = task
        self.max_detections = max_detections
        self.nms_iou = nms_iou
        self.tracker = ByteTrack(**(tracker or {})) if task == "detection-tracking" else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        task: str = "detection-only",
        device: str = "cuda",
        tracker: Optional[dict] = None,
        max_detections: Optional[int] = None,
        nms_iou: Optional[float] = None,
    ) -> "Pipeline":
        detector = Detector.from_checkpoint(checkpoint).to(device)
        meta = detector.meta
        transform = Transform(meta["input_size"], meta["pad_color"], meta["rescale_factor"])
        return cls(detector, transform, task, tracker, max_detections, nms_iou)

    def predict(self, frame: np.ndarray, score_threshold: float) -> Detections:
        return self.predict_batch([frame], score_threshold)[0]

    def predict_batch(
        self,
        frames: List[np.ndarray],
        score_threshold: float,
    ) -> List[Detections]:
        """Run one model forward pass, then update tracking in frame order."""
        if not frames:
            return []

        # the tracker needs low scoring boxes for its second pass. under tracking,
        # decode runs at low_thresh and score_threshold is unused.
        threshold = self.tracker.low_thresh if self.tracker else score_threshold

        samples = [self.transform(frame) for frame in frames]
        pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
        logits, pred_boxes = self.detector.predict(pixel_values)
        decoded = decode(logits, pred_boxes, threshold, self.max_detections)

        results = []
        for sample, (boxes, scores, class_ids) in zip(samples, decoded):
            boxes = self.transform.postprocess(boxes, sample["scale"], sample["frame_size"])

            # the detector puts more than one query on the same instrument, and each extra
            # box opens a second track on it
            if self.nms_iou is not None:
                keep = nms(boxes, scores, self.nms_iou)
                boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

            detections = Detections(boxes, scores, class_ids)
            results.append(self.tracker.update(detections) if self.tracker else detections)
        return results

    def reset(self) -> None:
        """drop the tracking state. call between sequences"""
        if self.tracker is not None:
            self.tracker.reset()
