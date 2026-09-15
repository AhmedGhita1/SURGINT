from pathlib import Path
from typing import Union

import numpy as np

from surgint.model import TASKS, Detections
from surgint.model.decode import decode
from surgint.model.detector import Detector
from surgint.model.transform import Transform
from surgint.runtime.bytetrack import ByteTrack


class Pipeline:
    def __init__(self, detector: Detector, transform: Transform, task: str):
        
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")

        self.detector = detector
        self.transform = transform
        self.task = task
        self.tracker = ByteTrack() if task == "detection-tracking" else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        task: str = "detection-only",
        device: str = "cuda",
    ) -> "Pipeline":
        detector = Detector.from_checkpoint(checkpoint).to(device)
        meta = detector.meta
        transform = Transform(meta["input_size"], meta["pad_color"], meta["rescale_factor"])
        return cls(detector, transform, task)

    def predict(self, frame: np.ndarray, score_threshold: float) -> Detections:
        # under tracking the tracker owns the score policy. its second pass rescues lost
        # tracks with the weak boxes a threshold here would have thrown away, so
        # score_threshold applies to detection only
        threshold = self.tracker.low_thresh if self.tracker else score_threshold

        sample = self.transform(frame)
        logits, pred_boxes = self.detector.predict(sample["pixel_values"].unsqueeze(0))

        boxes, scores, class_ids = decode(logits, pred_boxes, threshold)[0]
        boxes = self.transform.postprocess(boxes, sample["scale"], sample["frame_size"])

        detections = Detections(boxes, scores, class_ids)
        return self.tracker.update(detections) if self.tracker else detections

    def reset(self) -> None:
        """drop the tracking state between sequences, or ids leak from one into the next"""
        if self.tracker is not None:
            self.tracker.reset()
