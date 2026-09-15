"""
Kalman filter tests
===================

tests the motion model a track carries between frames.

the filter itself works in measurement space, [cx, cy, aspect, height]. only the
conversions see a box.

coverage:
- conversion: xyxy to [cx, cy, aspect, height] and back, aspect is width over height
- initiate:  a new track starts at rest, with a diagonal and weak prior
- predict:   constant velocity moves the centre, uncertainty grows
- project:   eight state dimensions down to the four a detection observes
- update:    the estimate moves toward the measurement and the prior narrows
- tracking:  a box on a straight line, velocity recovered from positions alone
"""

import numpy as np

from surgint.runtime.kalman import MEASUREMENT_DIM, STATE_DIM, KalmanFilter, to_box, to_measurement

MEASUREMENT = np.array([100.0, 100.0, 0.5, 40.0])   # cx, cy, aspect, height
BOX = np.array([90.0, 80.0, 110.0, 120.0])          # the same box as xyxy


def test_unit_conversion():
    """the box space the tracker speaks against the measurement space the filter speaks"""

    # a 20x40 box centred at (100, 100): aspect is width over height, not the other way round
    assert np.allclose(to_measurement(BOX), MEASUREMENT), f"got {to_measurement(BOX).tolist()}"
    assert to_measurement(BOX)[2] < 1, "a tall box must have an aspect below one"

    # to_box is the exact inverse, or a track drifts every frame it is predicted
    assert np.allclose(to_box(MEASUREMENT), BOX), f"got {to_box(MEASUREMENT).tolist()}"
    assert np.allclose(to_box(to_measurement(BOX)), BOX), "the round trip must be lossless"

    # the filter predicts in measurement space, so the inverse has to hold off the grid too
    drifted = MEASUREMENT + np.array([3.5, -2.25, 0.125, 6.0])
    assert np.allclose(to_measurement(to_box(drifted)), drifted), "the inverse must hold both ways"


def test_unit_initiate():
    """a detection that opens a track"""

    mean, covariance = KalmanFilter().initiate(MEASUREMENT)

    # the position half is the detection, the velocity half is unknown and starts at rest
    assert mean.shape == (STATE_DIM,), f"state must be {STATE_DIM}-dimensional, got {mean.shape}"
    assert np.allclose(mean[:MEASUREMENT_DIM], MEASUREMENT), "the measurement must survive intact"
    assert np.all(mean[MEASUREMENT_DIM:] == 0), "a new track has no velocity estimate yet"

    # one observation correlates nothing, so the prior is diagonal and positive
    assert covariance.shape == (STATE_DIM, STATE_DIM), f"got {covariance.shape}"
    assert np.allclose(covariance, np.diag(np.diag(covariance))), "the initial prior must be diagonal"
    assert np.all(np.diag(covariance) > 0), "every variance must be positive"

    # the prior is weaker than one step of process noise, in both halves
    _, running = KalmanFilter().predict(mean, np.zeros((STATE_DIM, STATE_DIM)))
    assert np.all(np.diag(covariance) > np.diag(running)), "initiate must be wider than a running step"


def test_unit_predict():
    """one frame of constant velocity"""

    kalman = KalmanFilter()
    mean = np.array([100.0, 100.0, 0.5, 40.0, 2.0, -1.0, 0.0, 0.0])
    covariance = np.diag(np.full(STATE_DIM, 0.1))

    predicted, predicted_covariance = kalman.predict(mean, covariance)

    # the centre advances by exactly one frame of velocity; the velocity itself is unchanged
    assert np.allclose(predicted[:MEASUREMENT_DIM], [102.0, 99.0, 0.5, 40.0]), f"got {predicted[:4]}"
    assert np.allclose(predicted[MEASUREMENT_DIM:], mean[MEASUREMENT_DIM:]), "velocity must be constant"

    # predicting without observing can only lose certainty
    assert np.trace(predicted_covariance) > np.trace(covariance), "process noise must widen the prior"


def test_unit_project():
    """state space to measurement space"""

    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(MEASUREMENT)
    projected, projected_covariance = kalman.project(mean, covariance)

    # a detection sees the position half only
    assert projected.shape == (MEASUREMENT_DIM,), f"got {projected.shape}"
    assert projected_covariance.shape == (MEASUREMENT_DIM, MEASUREMENT_DIM), f"got {projected_covariance.shape}"
    assert np.allclose(projected, MEASUREMENT), "projection must not move the estimate"

    # projection adds observation noise on top of the state uncertainty
    position_block = np.diag(covariance)[:MEASUREMENT_DIM]
    assert np.all(np.diag(projected_covariance) > position_block), "observation noise must be added"


def test_unit_update():
    """folding a detection into the estimate"""

    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(MEASUREMENT)

    observed = MEASUREMENT + np.array([10.0, 0.0, 0.0, 0.0])
    corrected, corrected_covariance = kalman.update(mean, covariance, observed)

    # the estimate moves toward the detection without jumping onto it
    assert corrected[0] > mean[0], "the estimate must move toward the measurement"
    assert corrected[0] < observed[0], "the estimate must not discard its prior"

    # an observation can only narrow the estimate
    assert np.trace(corrected_covariance) < np.trace(covariance), "the prior must narrow"

    # measuring exactly what was predicted moves nothing
    unchanged, _ = kalman.update(mean, covariance, MEASUREMENT)
    assert np.allclose(unchanged, mean), "a perfect prediction must leave the estimate alone"


def test_unit_tracking():
    """velocity is never measured, so it has to be recovered from positions"""

    kalman = KalmanFilter()
    velocity = 5.0

    mean, covariance = kalman.initiate(MEASUREMENT)
    for frame in range(1, 20):
        mean, covariance = kalman.predict(mean, covariance)
        observed = MEASUREMENT + np.array([velocity * frame, 0.0, 0.0, 0.0])
        mean, covariance = kalman.update(mean, covariance, observed)

    # the filter has only ever seen positions
    assert np.isclose(mean[4], velocity, atol=0.1), f"velocity should converge to {velocity}, got {mean[4]:.3f}"
    assert np.isclose(mean[5], 0.0, atol=0.1), f"the box never moved in y, got {mean[5]:.3f}"

    # and the next frame is predicted before it arrives
    predicted, _ = kalman.predict(mean, covariance)
    assert np.isclose(predicted[0], MEASUREMENT[0] + velocity * 20, atol=1.0), f"got {predicted[0]:.2f}"


if __name__ == "__main__":
    test_unit_conversion()
    test_unit_initiate()
    test_unit_predict()
    test_unit_project()
    test_unit_update()
    test_unit_tracking()
    print("\nall passed")
