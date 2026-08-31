from pathlib import Path
from typing import Union

import numpy as np

from surgint.model import Detections
from surgint.model.decode import decode
from surgint.model.detector import Detector
from surgint.model.transform import Transform

TASKS = ("detection-only", "detection-tracking")


class Pipeline:
    def __init__(self, detector: Detector, transform: Transform, task: str):
        
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")

        self.detector = detector
        self.transform = transform
        self.task = task

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
        sample = self.transform(frame)
        logits, pred_boxes = self.detector.predict(sample["pixel_values"].unsqueeze(0))

        boxes, scores, class_ids = decode(logits, pred_boxes, score_threshold)[0]
        boxes = self.transform.postprocess(boxes, sample["scale"], sample["frame_size"])

        return Detections(boxes, scores, class_ids)
