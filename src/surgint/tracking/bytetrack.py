from enum import Enum

import numpy as np

from surgint.inference import DETOutput
from surgint.tracking import MOTOutput
from surgint.tracking.kalman import KalmanFilter


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class Track:
    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        score: float,
        class_id: int,
        kalman: KalmanFilter,
    ):
        raise NotImplementedError

    @property
    def box(self) -> np.ndarray:
        """current estimate as xyxy"""
        raise NotImplementedError

    @property
    def class_id(self) -> int:
        """majority class over matched detections; a per-frame flip must not split the track"""
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def update(self, box: np.ndarray, score: float, class_id: int) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        raise NotImplementedError


def iou_distance(tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
    """1 - IoU, as a cost matrix"""
    raise NotImplementedError


def associate(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment; returns matches, unmatched track rows, unmatched detection columns"""
    raise NotImplementedError


class ByteTrack:
    """association is class agnostic; each track votes its own class. see docs/decisions/0007"""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 0.0,
    ):
        raise NotImplementedError

    def update(self, detections: DETOutput) -> MOTOutput:
        """one frame. high scoring boxes associate first, then low scoring ones rescue lost tracks"""
        raise NotImplementedError

    def reset(self) -> None:
        """drop every track and reset the id counter; call between sequences or ids leak across them"""
        raise NotImplementedError
