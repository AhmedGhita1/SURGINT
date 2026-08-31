from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # xyxy in frame pixels
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: Optional[np.ndarray] = None
