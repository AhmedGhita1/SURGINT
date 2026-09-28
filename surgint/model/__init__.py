from dataclasses import dataclass
from typing import Optional

import numpy as np

# detection alone, or detection with track ids
TASKS = ("detection-only", "detection-tracking")


@dataclass(frozen=True)
class Detections:
    """
    one frame of detections, checked on construction.

    boxes are xyxy in absolute frame pixels, scores are in [0, 1], and class ids index
    the artifact's id2label. track_ids are set by the tracker and are unique within the
    frame. the arrays are made read-only, so a consumer cannot rewrite what the model
    reported.

    Raises:
        TypeError: a field is not a numpy array.
        ValueError: the shapes disagree, a value is not finite, a score is outside
            [0, 1], an id is negative, or a track id repeats.
    """

    boxes: np.ndarray  # xyxy in frame pixels
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        arrays = {"boxes": self.boxes, "scores": self.scores, "class_ids": self.class_ids}
        if self.track_ids is not None:
            arrays["track_ids"] = self.track_ids

        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")

        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError(f"boxes must have shape (n, 4), got {self.boxes.shape}")
        for name, array in arrays.items():
            if name != "boxes" and array.ndim != 1:
                raise ValueError(f"{name} must have shape (n,), got {array.shape}")

        counts = {name: len(array) for name, array in arrays.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(f"every array must describe the same detections, got {counts}")

        if not np.isfinite(self.boxes).all():
            raise ValueError("every box coordinate must be finite")
        if not np.isfinite(self.scores).all():
            raise ValueError("every score must be finite")
        if self.scores.size and (self.scores.min() < 0.0 or self.scores.max() > 1.0):
            raise ValueError(
                f"scores must be in [0, 1], got [{self.scores.min()}, {self.scores.max()}]"
            )
        if self.class_ids.size and self.class_ids.min() < 0:
            raise ValueError(f"class ids must be non-negative, got {self.class_ids.min()}")

        if self.track_ids is not None and self.track_ids.size:
            if self.track_ids.min() < 0:
                raise ValueError(f"track ids must be non-negative, got {self.track_ids.min()}")
            if len(np.unique(self.track_ids)) != len(self.track_ids):
                raise ValueError(f"track ids must be unique within a frame, got {self.track_ids}")

        # freezing the arrays is what makes the frozen dataclass hold: without it the
        # fields cannot be rebound but their contents can still be rewritten
        for array in arrays.values():
            array.setflags(write=False)
