from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MOTOutput:
    boxes: np.ndarray  # xyxy in original frame pixels
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray  # int32, stable across frames
