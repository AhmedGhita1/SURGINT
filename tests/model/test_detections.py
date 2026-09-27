"""Tests for the immutable detection boundary type."""

import re

import numpy as np
import pytest

from surgint.model import Detections


def valid(count: int = 2) -> dict:
    """Return the fields of a well-formed frame as keyword arguments."""
    return {
        "boxes": np.zeros((count, 4), dtype=np.float32),
        "scores": np.zeros(count, dtype=np.float32),
        "class_ids": np.zeros(count, dtype=np.int64),
    }


def test_detections():
    """The boundary type carries model output and optional tracker IDs."""
    boxes = np.zeros((2, 4), dtype=np.float32)
    scores = np.zeros(2, dtype=np.float32)
    class_ids = np.zeros(2, dtype=np.int64)

    detections = Detections(boxes, scores, class_ids)
    assert detections.boxes.shape == (2, 4), "boxes are xyxy in frame pixels"
    assert len(detections.scores) == len(detections.class_ids) == len(detections.boxes)
    assert detections.track_ids is None, "track_ids default to None"

    tracked = Detections(boxes, scores, class_ids, np.array([7, 8], dtype=np.int64))
    assert tracked.track_ids.tolist() == [7, 8], "the same type carries tracks"

    with pytest.raises(Exception):
        detections.boxes = boxes


def test_validation():
    """A frame that cannot be true is refused at the boundary."""
    cases = {
        "boxes must have shape": {"boxes": np.zeros((2, 5), dtype=np.float32)},
        "scores must have shape": {"scores": np.zeros((2, 1), dtype=np.float32)},
        "same detections": {"scores": np.zeros(3, dtype=np.float32)},
        "box coordinate must be finite": {"boxes": np.full((2, 4), np.nan, dtype=np.float32)},
        "score must be finite": {"scores": np.full(2, np.inf, dtype=np.float32)},
        "scores must be in [0, 1]": {"scores": np.full(2, 1.5, dtype=np.float32)},
        "class ids must be non-negative": {"class_ids": np.full(2, -1, dtype=np.int64)},
    }
    for message, override in cases.items():
        with pytest.raises(ValueError, match=re.escape(message)):
            Detections(**{**valid(), **override})

    with pytest.raises(TypeError, match="must be a numpy array"):
        Detections(**{**valid(), "boxes": [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]})


def test_track_ids():
    """Track IDs name distinct instruments in their frame."""
    with pytest.raises(ValueError, match="unique within a frame"):
        Detections(**valid(), track_ids=np.array([7, 7], dtype=np.int64))

    with pytest.raises(ValueError, match="track ids must be non-negative"):
        Detections(**valid(), track_ids=np.array([7, -1], dtype=np.int64))

    with pytest.raises(ValueError, match="same detections"):
        Detections(**valid(), track_ids=np.array([7, 8, 9], dtype=np.int64))


def test_empty_frame():
    """A frame where nothing was detected remains well formed."""
    empty = Detections(**valid(0), track_ids=np.zeros(0, dtype=np.int64))

    assert len(empty.boxes) == 0 and empty.boxes.shape == (0, 4), f"got {empty.boxes.shape}"
    assert len(empty.track_ids) == 0


def test_arrays_are_read_only():
    """Immutability reaches the array contents, not only the fields."""
    detections = Detections(**valid(), track_ids=np.array([7, 8], dtype=np.int64))

    for name in ("boxes", "scores", "class_ids", "track_ids"):
        array = getattr(detections, name)
        assert not array.flags.writeable, f"{name} is still writeable"
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1
