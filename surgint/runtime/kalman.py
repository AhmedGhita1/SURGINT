import numpy as np

# state is [cx, cy, aspect, height] and their velocities; only the first four are measured
STATE_DIM = 8
MEASUREMENT_DIM = 4

# aspect is a ratio. its noise does not scale with box size.
ASPECT_NOISE = 1e-2
ASPECT_VELOCITY_NOISE = 1e-5
ASPECT_OBSERVATION_NOISE = 1e-1


def to_measurement(box: np.ndarray) -> np.ndarray:
    """xyxy to [cx, cy, width / height, height]"""
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    width, height = x2 - x1, y2 - y1
    return np.array([x1 + width / 2, y1 + height / 2, width / height, height])


def to_box(measurement: np.ndarray) -> np.ndarray:
    """[cx, cy, width / height, height] to xyxy"""
    cx, cy, aspect, height = np.asarray(measurement, dtype=float)
    width = aspect * height
    return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])


class KalmanFilter:
    """constant velocity over box centre, aspect, and height. assumes a uniform frame interval"""

    def __init__(self, position_weight: float = 1 / 20, velocity_weight: float = 1 / 160):
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

        # one frame per step. each position gains its velocity.
        self.motion_mat = np.eye(STATE_DIM)
        self.motion_mat[:MEASUREMENT_DIM, MEASUREMENT_DIM:] = np.eye(MEASUREMENT_DIM)

        # a detection observes the position half.
        self.update_mat = np.eye(MEASUREMENT_DIM, STATE_DIM)

    def _std(self, height: float, position_scale: float, velocity_scale: float) -> np.ndarray:
        """process noise for one step. box noise scales with height, aspect noise is fixed"""
        position = position_scale * self.position_weight * height
        velocity = velocity_scale * self.velocity_weight * height
        return np.array(
            [
                position,
                position,
                position_scale * ASPECT_NOISE,
                position,
                velocity,
                velocity,
                velocity_scale * ASPECT_VELOCITY_NOISE,
                velocity,
            ]
        )

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mean and covariance for a detection that starts a new track"""
        mean = np.zeros(STATE_DIM)
        mean[:MEASUREMENT_DIM] = measurement  # the track starts at rest

        # one detection is a weak prior. both halves start inflated, velocity more
        # than position.
        std = self._std(float(measurement[3]), position_scale=2.0, velocity_scale=10.0)
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = self._std(float(mean[3]), position_scale=1.0, velocity_scale=1.0)
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + np.diag(np.square(std))
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """state space to measurement space, adding observation noise"""
        position = self.position_weight * float(mean[3])
        std = np.array([position, position, ASPECT_OBSERVATION_NOISE, position])

        mean = self.update_mat @ mean
        covariance = self.update_mat @ covariance @ self.update_mat.T + np.diag(np.square(std))
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_covariance = self.project(mean, covariance)

        # gain is P H^T S^-1, computed with a linear solve
        gain = np.linalg.solve(projected_covariance, (covariance @ self.update_mat.T).T).T

        mean = mean + gain @ (measurement - projected_mean)
        covariance = covariance - gain @ projected_covariance @ gain.T
        return mean, covariance
