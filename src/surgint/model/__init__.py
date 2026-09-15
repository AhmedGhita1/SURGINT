from dataclasses import dataclass
from typing import Optional

import numpy as np

# what a run is for: detection alone, or detection with identities carried across frames
TASKS = ("detection-only", "detection-tracking")


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # xyxy in frame pixels
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: Optional[np.ndarray] = None
