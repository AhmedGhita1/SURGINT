"""
ByteTrack tests
===============

tests the tracking layer: per-frame detections to persistent identities.

coverage:
- lifecycle: tentative, confirmed, lost, confirmed again
- class:     majority vote over matched detections
- motion:    the estimate follows detections and advances without one
- cost:      1 - IoU between track estimates and the frame's boxes
- associate: global assignment and the rejection threshold
- update:    one id per instrument across frames, and the output dtypes
- rescue:    a low scoring box continues a confirmed track, and only a confirmed one
- output:    output_thresh separates staying associated from being reported
- buffer:    a track survives a gap of track_buffer frames
- filter:    weak and degenerate boxes are dropped
- reset:     ids restart at 1
"""

import numpy as np

from surgint.model import Detections
from surgint.runtime.bytetrack import (
    CONFIRM_HITS,
    ByteTrack,
    Track,
    TrackState,
    associate,
    iou_distance,
)
from surgint.runtime.kalman import KalmanFilter

BOX = np.array([90.0, 80.0, 110.0, 120.0])   # 20x40, centred at (100, 100)
SCORE = 0.9


def build(class_id: int = 0) -> Track:
    return Track(7, BOX, SCORE, class_id, KalmanFilter())


def frame(boxes, scores=None, class_ids=None) -> Detections:
    """one frame of detections, in the form the pipeline produces"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    count = len(boxes)
    return Detections(
        boxes,
        np.full(count, SCORE, np.float32) if scores is None else np.asarray(scores, np.float32),
        np.zeros(count, np.int64) if class_ids is None else np.asarray(class_ids, np.int64),
    )


def moved(step: int) -> np.ndarray:
    """the box after step frames of rightward drift"""
    return BOX + np.array([10.0 * step, 0.0, 10.0 * step, 0.0])


def test_lifecycle():
    """the state transitions of a single track"""

    tracked = build()
    assert tracked.track_id == 7, "the id is assigned at construction"
    assert tracked.state is TrackState.TENTATIVE, "one detection gives a tentative track"
    assert tracked.hits == 1, f"got {tracked.hits}"

    # confirmation requires CONFIRM_HITS detections
    for hit in range(2, CONFIRM_HITS):
        tracked.update(BOX, SCORE, 0)
        assert tracked.state is TrackState.TENTATIVE, f"confirmed early at {hit} hits"
    tracked.update(BOX, SCORE, 0)
    assert tracked.state is TrackState.CONFIRMED, f"{CONFIRM_HITS} hits must confirm the track"

    # predict increments time_since_update, update clears it
    tracked.predict()
    tracked.predict()
    assert tracked.time_since_update == 2, f"got {tracked.time_since_update}"
    tracked.update(BOX, SCORE, 0)
    assert tracked.time_since_update == 0, "update must clear time_since_update"

    # a lost track that is matched again returns to confirmed
    tracked.mark_lost()
    assert tracked.state is TrackState.LOST, "mark_lost must set the lost state"
    tracked.update(BOX, SCORE, 0)
    assert tracked.state is TrackState.CONFIRMED, "a rescued track returns to confirmed"
    assert tracked.track_id == 7, "a rescued track keeps its id"


def test_class():
    """the track decides its own label and keeps its confidence associated"""

    tracked = Track(7, BOX, 0.99, 0, KalmanFilter())
    tracked.update(BOX, 0.8, 0)

    # one vote against two does not change the majority
    tracked.update(BOX, 0.6, 1)
    assert tracked.class_id == 0, f"the majority is still 0, got {tracked.class_id}"
    assert np.isclose(tracked.score, 0.99), f"got {tracked.score}"

    # the vote is a count, so a new majority takes over
    tracked.update(BOX, 0.7, 1)
    tracked.update(BOX, 0.65, 1)
    assert tracked.class_id == 1, f"the majority should be 1, got {tracked.class_id}"
    assert np.isclose(tracked.score, 0.7), f"got {tracked.score}"


def test_motion():
    """the reported box is the filter estimate"""

    tracked = build()
    velocity = 10.0

    # a box drifting right at a constant rate
    for frame in range(1, 10):
        tracked.predict()
        shift = np.array([velocity * frame, 0.0, velocity * frame, 0.0])
        tracked.update(BOX + shift, SCORE, 0)

    centre = (tracked.box[0] + tracked.box[2]) / 2
    assert np.isclose(centre, 100 + velocity * 9, atol=2.0), f"centre should be {100 + velocity * 9}, got {centre:.1f}"
    assert tracked.box.shape == (4,), "the track reports four coordinates"

    # predict advances the estimate without a detection
    before = tracked.box[0]
    tracked.predict()
    assert tracked.box[0] > before, "predict must advance the box"
    assert tracked.time_since_update == 1, "predict must increment time_since_update"


def test_cost():
    """the cost matrix association runs on"""

    tracked = build()

    # full overlap costs 0, no overlap costs 1
    far = BOX + 500.0
    cost = iou_distance([tracked], np.stack([BOX, far]))
    assert cost.shape == (1, 2), f"one row per track, one column per detection, got {cost.shape}"
    assert np.isclose(cost[0, 0], 0.0), f"full overlap should cost 0, got {cost[0, 0]}"
    assert np.isclose(cost[0, 1], 1.0), f"no overlap should cost 1, got {cost[0, 1]}"

    # cost is measured from the filter estimate, not the first detection
    for frame in range(1, 6):
        tracked.predict()
        tracked.update(BOX + np.array([10.0 * frame, 0.0, 10.0 * frame, 0.0]), SCORE, 0)
    tracked.predict()
    assert iou_distance([tracked], BOX[None])[0, 0] > 0.9, "the estimate should have moved off the first box"

    # empty inputs keep the matrix two-dimensional
    assert iou_distance([tracked], np.empty((0, 4))).shape == (1, 0), "an empty frame keeps the row"
    assert iou_distance([], np.stack([BOX])).shape == (0, 1), "no tracks keeps the column"


def test_associate():
    """a cost matrix to matches and leftovers"""

    # the assignment minimises the total. the cheapest pair first would give 1.0, not 0.3
    cost = np.array([[0.1, 0.2], [0.1, 0.9]])
    matches, tracks, detections = associate(cost, threshold=0.8)
    assert matches == [(0, 1), (1, 0)], f"expected the 0.3 total, got {matches}"
    assert tracks == [] and detections == [], "everything matched"

    # a pair above the threshold is dropped and reported unmatched on both sides
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, tracks, detections = associate(cost, threshold=0.5)
    assert matches == [(0, 0)], f"expected one match, got {matches}"
    assert tracks == [1], f"got {tracks}"
    assert detections == [1], f"got {detections}"

    # leftovers on either side are reported
    matches, tracks, detections = associate(np.array([[0.1, 0.9]]), threshold=0.5)
    assert matches == [(0, 0)] and tracks == [] and detections == [1], f"got {matches}, {tracks}, {detections}"

    # empty inputs return everything as unmatched
    assert associate(np.empty((2, 0)), threshold=0.8) == ([], [0, 1], [])
    assert associate(np.empty((0, 2)), threshold=0.8) == ([], [], [0, 1])


def test_update():
    """one id per instrument, and the output dtypes"""

    tracker = ByteTrack()

    # nothing is reported before confirmation
    for step in range(CONFIRM_HITS - 1):
        assert len(tracker.update(frame([moved(step)])).boxes) == 0, f"reported at {step + 1} hits"

    tracked = tracker.update(frame([moved(CONFIRM_HITS - 1)]))
    assert tracked.track_ids.tolist() == [1], f"got {tracked.track_ids.tolist()}"

    # the arrays stay aligned and keep their dtypes
    assert tracked.boxes.shape == (1, 4) and tracked.boxes.dtype == np.float32, f"got {tracked.boxes.dtype}"
    assert tracked.scores.dtype == np.float32 and tracked.class_ids.dtype == np.int64
    assert tracked.track_ids.dtype == np.int64, f"got {tracked.track_ids.dtype}"

    # the id holds across later frames
    for step in range(CONFIRM_HITS, CONFIRM_HITS + 5):
        tracked = tracker.update(frame([moved(step)]))
        assert tracked.track_ids.tolist() == [1], f"the id changed at step {step}: {tracked.track_ids.tolist()}"

    # an empty frame reports nothing and keeps the shapes
    empty = tracker.update(frame(np.empty((0, 4))))
    assert empty.boxes.shape == (0, 4) and empty.track_ids.shape == (0,), f"got {empty.boxes.shape}"


def test_rescue():
    """a low scoring box continues a track"""

    # the rescue threshold is stricter than the first pass. it only holds once the
    # filter has learned the motion, which takes about four frames at this speed
    warmup = 6
    tracker = ByteTrack()
    for step in range(warmup):
        tracker.update(frame([moved(step)]))

    # the detector produces a box, at a low score
    weak = tracker.update(frame([moved(warmup)], scores=[0.2]))
    assert weak.track_ids.tolist() == [1], f"the rescue pass should continue track 1, got {weak.track_ids.tolist()}"
    assert tracker.next_id == 2, "no second track should have opened"


def test_rescue_skips_tentative():
    """the second pass ignores tracks that are not confirmed"""

    tracker = ByteTrack()

    # one high scoring detection opens a tentative track
    tracker.update(frame([BOX]))
    assert tracker.tracks[0].state is TrackState.TENTATIVE, "one hit gives a tentative track"

    # low scoring boxes on the same spot must not carry it to confirmation
    tracker.update(frame([BOX], scores=[0.2]))
    assert tracker.tracks == [], "a tentative track must not be rescued on low scores"

    # a lost track is not revived by them either
    tracker = ByteTrack(track_buffer=5)
    for _ in range(CONFIRM_HITS):
        tracker.update(frame([BOX]))
    tracker.update(frame(np.empty((0, 4))))
    assert tracker.tracks[0].state is TrackState.LOST, "the missed track should be lost"

    tracker.update(frame([BOX], scores=[0.2]))
    assert tracker.tracks[0].state is TrackState.LOST, "a lost track must not be revived on low scores"


def test_output_thresh():
    """a rescued track stays associated without being reported"""

    # the filter needs a few frames to learn the motion before the rescue pass holds
    warmup = 6
    tracker = ByteTrack(output_thresh=0.5)
    for step in range(warmup):
        tracker.update(frame([moved(step)]))

    weak = tracker.update(frame([moved(warmup)], scores=[0.2]))
    assert len(weak.boxes) == 0, "a track riding a low scoring box must not be reported"
    assert len(tracker.tracks) == 1, "the track must stay alive"
    assert tracker.tracks[0].time_since_update == 0, "the track was still matched"

    # the same id is reported again once a strong detection returns
    back = tracker.update(frame([moved(warmup + 1)]))
    assert back.track_ids.tolist() == [1], f"expected id 1 back, got {back.track_ids.tolist()}"

    # the default reports whatever it matched
    assert ByteTrack().output_thresh == 0.0, "output_thresh must default to reporting everything"


def test_buffer():
    """a gap within track_buffer, and a gap beyond it"""

    # a stationary instrument, so this measures the buffer and not the motion model
    tracker = ByteTrack(track_buffer=5)
    for _ in range(CONFIRM_HITS):
        tracker.update(frame([BOX]))

    # the track is held through empty frames and not reported
    for _ in range(3):
        assert len(tracker.update(frame(np.empty((0, 4)))).boxes) == 0, "a predicted track must not be reported"

    back = tracker.update(frame([BOX]))
    assert back.track_ids.tolist() == [1], f"expected id 1 within the buffer, got {back.track_ids.tolist()}"

    # past the buffer the track is dropped and the next detection opens a new one
    for _ in range(tracker.track_buffer + 2):
        tracker.update(frame(np.empty((0, 4))))
    assert tracker.tracks == [], "the track should be dropped past the buffer"

    for _ in range(CONFIRM_HITS):
        fresh = tracker.update(frame([BOX]))
    assert fresh.track_ids.tolist() == [2], f"expected id 2, got {fresh.track_ids.tolist()}"


def test_filter():
    """detections dropped before the filter"""

    tracker = ByteTrack(low_thresh=0.1, min_box_area=100.0)

    # below low_thresh there is nothing to associate on
    assert len(tracker.update(frame([BOX], scores=[0.05])).boxes) == 0
    assert tracker.tracks == [], "a box under low_thresh must not open a track"

    # a box clipped to zero area at the frame edge would divide by zero
    tracker.update(frame([[10.0, 10.0, 10.0, 50.0]]))
    assert tracker.tracks == [], "a zero width box must not reach the filter"

    # a box below min_box_area is dropped
    tracker.update(frame([[0.0, 0.0, 5.0, 5.0]]))
    assert tracker.tracks == [], "a box under min_box_area must not open a track"


def test_reset():
    """reset clears the tracks and the id counter"""

    tracker = ByteTrack()
    for step in range(CONFIRM_HITS):
        tracker.update(frame([moved(step)]))
    assert tracker.tracks, "there should be a track to drop"

    tracker.reset()
    assert tracker.tracks == [], "reset must drop every track"

    # the next sequence starts from id 1
    for step in range(CONFIRM_HITS):
        tracked = tracker.update(frame([moved(step)]))
    assert tracked.track_ids.tolist() == [1], f"expected id 1 after reset, got {tracked.track_ids.tolist()}"

