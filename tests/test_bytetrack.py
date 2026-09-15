"""
ByteTrack tests
===============

tests the tracking layer that turns per-frame detections into persistent identities.

coverage:
- lifecycle: tentative until believed, lost when missed, confirmed again when found
- class:     majority vote over matched detections, not the latest label
- motion:    the estimate follows the detections and carries on without one
- cost:      1 - IoU between track estimates and the frame's boxes
- associate: global assignment, and the threshold that rejects a bad pairing
"""

import numpy as np

from surgint.runtime.bytetrack import CONFIRM_HITS, Track, TrackState, associate, iou_distance
from surgint.runtime.kalman import KalmanFilter

BOX = np.array([90.0, 80.0, 110.0, 120.0])   # 20x40, centred at (100, 100)
SCORE = 0.9


def build(class_id: int = 0) -> Track:
    return Track(7, BOX, SCORE, class_id, KalmanFilter())


def test_unit_lifecycle():
    """what a track has to earn before it is believed, and what it survives"""

    tracked = build()
    assert tracked.track_id == 7, "the id is assigned once and never changes"
    assert tracked.state is TrackState.TENTATIVE, "one detection is a suspicion, not a track"
    assert tracked.hits == 1, f"got {tracked.hits}"

    # a track is confirmed only once it has been seen CONFIRM_HITS times
    for hit in range(2, CONFIRM_HITS):
        tracked.update(BOX, SCORE, 0)
        assert tracked.state is TrackState.TENTATIVE, f"confirmed early at {hit} hits"
    tracked.update(BOX, SCORE, 0)
    assert tracked.state is TrackState.CONFIRMED, f"{CONFIRM_HITS} hits must confirm the track"

    # a predicted frame is a frame without evidence, and counts against the track
    tracked.predict()
    tracked.predict()
    assert tracked.time_since_update == 2, f"got {tracked.time_since_update}"
    tracked.update(BOX, SCORE, 0)
    assert tracked.time_since_update == 0, "a detection clears the debt"

    # the low scoring pass rescues lost tracks, so being found again restores the
    # identity rather than opening a new one
    tracked.mark_lost()
    assert tracked.state is TrackState.LOST, "a missed track is lost, not deleted"
    tracked.update(BOX, SCORE, 0)
    assert tracked.state is TrackState.CONFIRMED, "a rescued track must not restart as tentative"
    assert tracked.track_id == 7, "a rescued track keeps its id"


def test_unit_class():
    """association is class agnostic, so the track has to decide its own label"""

    tracked = build(class_id=0)
    tracked.update(BOX, SCORE, 0)

    # one frame disagreeing with two does not change what the track is
    tracked.update(BOX, SCORE, 1)
    assert tracked.class_id == 0, f"a single flip must not relabel the track, got {tracked.class_id}"

    # but the vote is a count, not a lock on whatever arrived first
    for _ in range(3):
        tracked.update(BOX, SCORE, 1)
    assert tracked.class_id == 1, f"the majority must win, got {tracked.class_id}"


def test_unit_motion():
    """the box the tracker reports is the filter's estimate, not the last detection"""

    tracked = build()
    velocity = 10.0

    # a box drifting right at a constant rate
    for frame in range(1, 10):
        tracked.predict()
        shift = np.array([velocity * frame, 0.0, velocity * frame, 0.0])
        tracked.update(BOX + shift, SCORE, 0)

    centre = (tracked.box[0] + tracked.box[2]) / 2
    assert np.isclose(centre, 100 + velocity * 9, atol=2.0), f"the estimate lags the detection: {centre:.1f}"
    assert tracked.box.shape == (4,), "the track reports xyxy"

    # a frame with no detection still advances the estimate; that is what makes a gap survivable
    before = tracked.box[0]
    tracked.predict()
    assert tracked.box[0] > before, "a predicted frame must carry the track forward"
    assert tracked.time_since_update == 1, "a predicted frame counts against the track"


def test_unit_cost():
    """what association scores tracks against detections with"""

    tracked = build()

    # a detection on top of the track costs nothing; one nowhere near it costs everything
    far = BOX + 500.0
    cost = iou_distance([tracked], np.stack([BOX, far]))
    assert cost.shape == (1, 2), f"one row per track, one column per detection, got {cost.shape}"
    assert np.isclose(cost[0, 0], 0.0), f"a perfect overlap must cost 0, got {cost[0, 0]}"
    assert np.isclose(cost[0, 1], 1.0), f"no overlap must cost 1, got {cost[0, 1]}"

    # the cost is measured against the track's estimate, so a track carried through a
    # gap is compared where the filter thinks it is now
    for frame in range(1, 6):
        tracked.predict()
        tracked.update(BOX + np.array([10.0 * frame, 0.0, 10.0 * frame, 0.0]), SCORE, 0)
    tracked.predict()
    assert iou_distance([tracked], BOX[None])[0, 0] > 0.9, "the estimate must have moved off the first box"

    # an empty frame, or no tracks yet, still returns something indexable
    assert iou_distance([tracked], np.empty((0, 4))).shape == (1, 0), "an empty frame keeps the row"
    assert iou_distance([], np.stack([BOX])).shape == (0, 1), "no tracks keeps the column"


def test_unit_associate():
    """turning a cost matrix into matches and leftovers"""

    # the assignment is global: taking the cheapest pair first would lock track 0 onto
    # detection 0 and strand track 1 on a 0.9
    cost = np.array([[0.1, 0.2], [0.1, 0.9]])
    matches, tracks, detections = associate(cost, threshold=0.8)
    assert matches == [(0, 1), (1, 0)], f"the cheapest total wins, got {matches}"
    assert tracks == [] and detections == [], "everything matched"

    # a pairing the assignment makes but the threshold rejects comes back unmatched on both sides
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, tracks, detections = associate(cost, threshold=0.5)
    assert matches == [(0, 0)], f"only the close pairing survives, got {matches}"
    assert tracks == [1], f"got {tracks}"
    assert detections == [1], f"got {detections}"

    # leftovers on either side are reported, so new tracks open and missed ones age
    matches, tracks, detections = associate(np.array([[0.1, 0.9]]), threshold=0.5)
    assert matches == [(0, 0)] and tracks == [] and detections == [1], f"got {matches}, {tracks}, {detections}"

    # an empty frame loses no track, and a first frame matches nothing
    assert associate(np.empty((2, 0)), threshold=0.8) == ([], [0, 1], [])
    assert associate(np.empty((0, 2)), threshold=0.8) == ([], [], [0, 1])


if __name__ == "__main__":
    test_unit_lifecycle()
    test_unit_class()
    test_unit_motion()
    test_unit_cost()
    test_unit_associate()
    print("\nall passed")
