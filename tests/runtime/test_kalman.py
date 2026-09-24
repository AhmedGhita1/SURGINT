"""
Kalman filter tests
===================

tests the motion model a track carries between frames.

the filter works in measurement space, [cx, cy, aspect, height]. the conversions
handle boxes.

coverage:
- conversion: xyxy to [cx, cy, aspect, height] and back
- initiate:  velocities start at zero, covariance is diagonal
- predict:   constant velocity advances the centre, covariance grows
- project:   eight state dimensions to the four a detection observes
- update:    the estimate moves toward the measurement, covariance shrinks
- tracking:  velocity recovered from positions over 19 frames
"""

import numpy as np

from surgint.runtime.kalman import MEASUREMENT_DIM, STATE_DIM, KalmanFilter, to_box, to_measurement

MEASUREMENT = np.array([100.0, 100.0, 0.5, 40.0])   # cx, cy, aspect, height
BOX = np.array([90.0, 80.0, 110.0, 120.0])          # the same box as xyxy


def test_conversion():
    """xyxy to [cx, cy, aspect, height] and back"""

    # a 20x40 box centred at (100, 100). aspect is width / height.
    assert np.allclose(to_measurement(BOX), MEASUREMENT), f"got {to_measurement(BOX).tolist()}"
    assert to_measurement(BOX)[2] < 1, "aspect is width / height"

    # to_box is the exact inverse
    assert np.allclose(to_box(MEASUREMENT), BOX), f"got {to_box(MEASUREMENT).tolist()}"
    assert np.allclose(to_box(to_measurement(BOX)), BOX), "the round trip must be lossless"

    # the inverse must hold for non-integer values too
    drifted = MEASUREMENT + np.array([3.5, -2.25, 0.125, 6.0])
    assert np.allclose(to_measurement(to_box(drifted)), drifted), "the inverse must hold both ways"


def test_initiate():
    """mean and covariance for a new track"""

    mean, covariance = KalmanFilter().initiate(MEASUREMENT)

    # position half is the measurement, velocity half starts at zero
    assert mean.shape == (STATE_DIM,), f"state must be {STATE_DIM}-dimensional, got {mean.shape}"
    assert np.allclose(mean[:MEASUREMENT_DIM], MEASUREMENT), "the measurement must be unchanged"
    assert np.all(mean[MEASUREMENT_DIM:] == 0), "velocities start at zero"

    # one observation gives no correlations
    assert covariance.shape == (STATE_DIM, STATE_DIM), f"got {covariance.shape}"
    assert np.allclose(covariance, np.diag(np.diag(covariance))), "the initial prior must be diagonal"
    assert np.all(np.diag(covariance) > 0), "every variance must be positive"

    # initiate is wider than one predict step in both halves
    _, running = KalmanFilter().predict(mean, np.zeros((STATE_DIM, STATE_DIM)))
    assert np.all(np.diag(covariance) > np.diag(running)), "initiate must be wider than a running step"


def test_predict():
    """one frame of constant velocity"""

    kalman = KalmanFilter()
    mean = np.array([100.0, 100.0, 0.5, 40.0, 2.0, -1.0, 0.0, 0.0])
    covariance = np.diag(np.full(STATE_DIM, 0.1))

    predicted, predicted_covariance = kalman.predict(mean, covariance)

    # the centre advances by one frame of velocity. velocity is unchanged.
    assert np.allclose(predicted[:MEASUREMENT_DIM], [102.0, 99.0, 0.5, 40.0]), f"got {predicted[:4]}"
    assert np.allclose(predicted[MEASUREMENT_DIM:], mean[MEASUREMENT_DIM:]), "velocity must be constant"

    # predict adds process noise
    assert np.trace(predicted_covariance) > np.trace(covariance), "process noise must widen the prior"


def test_project():
    """state space to measurement space"""

    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(MEASUREMENT)
    projected, projected_covariance = kalman.project(mean, covariance)

    # projection keeps the position half
    assert projected.shape == (MEASUREMENT_DIM,), f"got {projected.shape}"
    assert projected_covariance.shape == (MEASUREMENT_DIM, MEASUREMENT_DIM), f"got {projected_covariance.shape}"
    assert np.allclose(projected, MEASUREMENT), "projection must not move the estimate"

    # projection adds observation noise
    position_block = np.diag(covariance)[:MEASUREMENT_DIM]
    assert np.all(np.diag(projected_covariance) > position_block), "observation noise must be added"


def test_update():
    """the correction step"""

    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(MEASUREMENT)

    observed = MEASUREMENT + np.array([10.0, 0.0, 0.0, 0.0])
    corrected, corrected_covariance = kalman.update(mean, covariance, observed)

    # the estimate moves partway toward the measurement
    assert corrected[0] > mean[0], "the estimate must move toward the measurement"
    assert corrected[0] < observed[0], "the estimate must stay below the measurement"

    # an observation shrinks the covariance
    assert np.trace(corrected_covariance) < np.trace(covariance), "covariance must shrink"

    # a measurement equal to the prediction changes nothing
    unchanged, _ = kalman.update(mean, covariance, MEASUREMENT)
    assert np.allclose(unchanged, mean), "the estimate must be unchanged"


def test_tracking():
    """velocity is recovered from positions"""

    kalman = KalmanFilter()
    velocity = 5.0

    mean, covariance = kalman.initiate(MEASUREMENT)
    for frame in range(1, 20):
        mean, covariance = kalman.predict(mean, covariance)
        observed = MEASUREMENT + np.array([velocity * frame, 0.0, 0.0, 0.0])
        mean, covariance = kalman.update(mean, covariance, observed)

    # only positions were supplied
    assert np.isclose(mean[4], velocity, atol=0.1), f"velocity should converge to {velocity}, got {mean[4]:.3f}"
    assert np.isclose(mean[5], 0.0, atol=0.1), f"y velocity should be 0, got {mean[5]:.3f}"

    # the next frame is predicted before it arrives
    predicted, _ = kalman.predict(mean, covariance)
    assert np.isclose(predicted[0], MEASUREMENT[0] + velocity * 20, atol=1.0), f"got {predicted[0]:.2f}"

