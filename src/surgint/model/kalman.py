import numpy as np

# state is [cx, cy, aspect, height] and their velocities; only the first four are measured
STATE_DIM = 8
MEASUREMENT_DIM = 4


def to_measurement(box: np.ndarray) -> np.ndarray:
    """xyxy to [cx, cy, width / height, height]"""
    raise NotImplementedError


def to_box(measurement: np.ndarray) -> np.ndarray:
    """[cx, cy, width / height, height] to xyxy"""
    raise NotImplementedError


class KalmanFilter:
    """constant velocity over box centre, aspect, and height. assumes a uniform frame interval"""

    def __init__(self, position_weight: float = 1 / 20, velocity_weight: float = 1 / 160):
        raise NotImplementedError

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mean and covariance for a detection that starts a new track"""
        raise NotImplementedError

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """state space to measurement space, adding observation noise"""
        raise NotImplementedError

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
