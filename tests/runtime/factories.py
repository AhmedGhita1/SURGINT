"""Small factories shared by runtime tests."""

import numpy as np

from surgint.model import Detections


def tracked_frame(class_ids, track_ids, scores) -> Detections:
    """Build a tracked detection frame from compact Python sequences."""
    count = len(track_ids)
    return Detections(
        boxes=np.zeros((count, 4), dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        track_ids=np.asarray(track_ids, dtype=np.int64),
    )
