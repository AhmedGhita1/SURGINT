from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DETOutput:
    boxes: np.ndarray  # xyxy in original frame pixels
    scores: np.ndarray
    class_ids: np.ndarray
