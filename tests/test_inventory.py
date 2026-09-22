"""
Inventory tests
===============

tests the per-session state built on top of the tracker.

coverage:
- accumulate:   an item stays after its track ends
- counts:       distinct tracks per class, and the simultaneous lower bound
- finalize:     track fragments become an immutable class-level estimate
- class:        the latest class a track reports wins
- reset:        state and frame counter cleared between sessions
- task:         detections without track ids are rejected
"""

import numpy as np

from surgint.model import Detections
from surgint.runtime.inventory import Inventory

BOX = [10.0, 10.0, 50.0, 90.0]


def frame(class_ids, track_ids, scores=None):
    """one frame of tracked detections. the boxes do not matter here"""
    count = len(track_ids)
    return Detections(
        np.tile(np.array(BOX, dtype=np.float32), (count, 1)),
        np.full(count, 0.9, np.float32) if scores is None else np.asarray(scores, np.float32),
        np.asarray(class_ids, np.int64),
        np.asarray(track_ids, np.int64),
    )


def test_unit_accumulate():
    """an instrument the camera has moved past is still on the tray"""

    inventory = Inventory()
    for _ in range(3):
        inventory.update(frame([0], [1]))

    # the track ends, and the next frames show a different instrument
    for _ in range(4):
        inventory.update(frame([1], [2]))

    assert set(inventory.items) == {1, 2}, f"got {set(inventory.items)}"

    first = inventory.items[1]
    assert first.frames == 3, f"expected 3, got {first.frames}"
    assert (first.first_seen, first.last_seen) == (0, 2), f"got {first.first_seen}, {first.last_seen}"

    second = inventory.items[2]
    assert (second.first_seen, second.last_seen) == (3, 6), f"got {second.first_seen}, {second.last_seen}"


def test_unit_counts():
    """the upper bound against the lower bound"""

    inventory = Inventory()

    # two scalpels in frame together, then a third track of the same class later
    for _ in range(3):
        inventory.update(frame([0, 0], [1, 2]))
    for _ in range(3):
        inventory.update(frame([0], [3]))

    assert inventory.counts() == {0: 3}, f"three distinct tracks, got {inventory.counts()}"
    assert inventory.simultaneous == {0: 2}, f"two were ever in frame together, got {inventory.simultaneous}"

    # an empty frame changes neither bound
    inventory.update(frame([], []))
    assert inventory.counts() == {0: 3} and inventory.simultaneous == {0: 2}
    assert inventory.frame == 7, f"an empty frame still counts, got {inventory.frame}"


def test_unit_finalize_consolidates_track_fragments():
    """maximum simultaneous observations become the finalized count"""

    inventory = Inventory()

    # Two physical items are seen together. A later identity fragment must not
    # make the finalized inventory claim that a third item existed.
    inventory.update(frame([0, 0], [1, 2], scores=[0.8, 0.9]))
    inventory.update(frame([0], [3], scores=[0.95]))

    finalized = inventory.finalize()

    assert finalized.frames == 2
    assert finalized.counts() == {0: 2}
    assert len(finalized.items) == 1

    item = finalized.items[0]
    assert item.class_id == 0
    assert item.representative_track_id == 3
    assert item.count == 2
    assert item.distinct_tracks == 3
    assert (item.first_seen, item.last_seen) == (0, 1)
    assert item.observation_frames == 3
    assert np.isclose(item.score, 0.95)


def test_unit_finalize_is_immutable_and_detached():
    """later tracking cannot mutate an already finalized result"""

    inventory = Inventory()
    inventory.update(frame([1], [7]))
    finalized = inventory.finalize()

    inventory.update(frame([1], [8]))

    assert finalized.frames == 1
    assert finalized.counts() == {1: 1}
    assert finalized.items[0].distinct_tracks == 1

    try:
        finalized.frames = 2
        assert False, "a finalized inventory must be immutable"
    except AttributeError:
        pass


def test_unit_finalize_empty_session():
    """an empty session produces an empty immutable result"""

    finalized = Inventory().finalize()

    assert finalized.frames == 0
    assert finalized.items == ()
    assert finalized.counts() == {}


def test_unit_class():
    """the track already voted, so the inventory takes its latest answer"""

    inventory = Inventory()
    inventory.update(frame([0], [1]))
    inventory.update(frame([0], [1]))
    inventory.update(frame([1], [1]))

    assert inventory.items[1].class_id == 1, f"got {inventory.items[1].class_id}"
    assert inventory.counts() == {1: 1}, f"got {inventory.counts()}"

    # the score kept is the best the track ever scored
    inventory.update(frame([1], [1], scores=[0.4]))
    assert np.isclose(inventory.items[1].score, 0.9), f"got {inventory.items[1].score}"


def test_unit_reset():
    """sessions are independent"""

    inventory = Inventory()
    for _ in range(3):
        inventory.update(frame([0], [1]))
    assert inventory.items, "there should be an item to drop"

    inventory.reset()
    assert inventory.items == {} and inventory.simultaneous == {}, "reset must clear the state"
    assert inventory.frame == 0, f"the frame counter must rewind, got {inventory.frame}"

    # track id 1 in the next session is a different instrument
    inventory.update(frame([0], [1]))
    assert inventory.items[1].first_seen == 0, f"got {inventory.items[1].first_seen}"


def test_unit_task():
    """an inventory cannot be built from untracked detections"""

    untracked = Detections(
        np.array([BOX], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([0], dtype=np.int64),
    )

    try:
        Inventory().update(untracked)
        assert False, "should raise ValueError without track ids"
    except ValueError as error:
        assert "detection-tracking" in str(error), f"got {error}"


if __name__ == "__main__":
    test_unit_accumulate()
    test_unit_counts()
    test_unit_finalize_consolidates_track_fragments()
    test_unit_finalize_is_immutable_and_detached()
    test_unit_finalize_empty_session()
    test_unit_class()
    test_unit_reset()
    test_unit_task()
    print("\nall passed")
