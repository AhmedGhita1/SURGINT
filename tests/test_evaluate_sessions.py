"""
Session evaluation tests
========================

tests the loop that scores the detector and tracker over ordered sessions.

a stub pipeline stands in for the weights, so the boxes and ids are controlled and
the counts are exact. the datasets are plain lists of samples, the shape
SurgintDataset returns when it is built without a transform.

coverage:
- reset:    ids do not carry from one session into the next
- counts:   the aggregate and the per session entries agree
- ordering: frames reach the tracker in index order
- task:     a detection-only pipeline is rejected
"""

import numpy as np

from surgint.engine.evaluator import evaluate_sessions
from surgint.model import Detections

BOX = np.array([10.0, 10.0, 50.0, 90.0])


class StubPipeline:
    """returns a prepared Detections per frame and records the order it saw them"""

    def __init__(self, outputs, tracker=object(), task="detection-tracking"):
        self.outputs = outputs
        self.tracker = tracker
        self.task = task
        self.seen = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict(self, frame, score_threshold):
        self.seen.append(int(frame[0, 0]))
        boxes, track_ids = self.outputs[len(self.seen) - 1]
        return Detections(
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.ones(len(boxes), dtype=np.float32),
            np.zeros(len(boxes), dtype=np.int64),
            np.asarray(track_ids, dtype=np.int64),
        )


def samples(count, marker):
    """a session of `count` frames, each frame tagged with `marker` so order is checkable"""
    return [
        {
            "frame": np.full((4, 4), marker + index, dtype=np.uint8),
            "boxes": BOX.reshape(1, 4),
            "class_ids": np.zeros(1, dtype=np.int64),
            "track_ids": np.array([1], dtype=np.int64),
            "image_id": index,
        }
        for index in range(count)
    ]


def test_unit_reset():
    """the tracker is reset on every session boundary"""

    pipeline = StubPipeline([([BOX], [1])] * 6)
    sessions = {"session_000": samples(3, 0), "session_001": samples(3, 100)}

    result = evaluate_sessions(pipeline, sessions)

    assert pipeline.resets == 2, f"expected one reset per session, got {pipeline.resets}"
    assert result["sequences"] == 2, f"got {result['sequences']}"

    # id 1 is reused in both sessions and must not read as a switch
    assert result["id_switches"] == 0, f"expected 0, got {result['id_switches']}"
    assert result["MOTA"] == 1.0, f"expected 1.0, got {result['MOTA']}"


def test_unit_counts():
    """the aggregate and the per session entries describe the same frames"""

    # session_000 is clean, session_001 misses its second frame
    outputs = [([BOX], [1])] * 3 + [([BOX], [1]), ([], []), ([BOX], [1])]
    sessions = {"session_000": samples(3, 0), "session_001": samples(3, 100)}

    result = evaluate_sessions(StubPipeline(outputs), sessions)

    assert set(result["per_session"]) == {"session_000", "session_001"}, "one entry per session"
    assert result["per_session"]["session_000"]["MOTA"] == 1.0, "the clean session scores 1.0"
    assert np.isclose(result["per_session"]["session_001"]["MOTA"], 1 - 1 / 3), "one miss in three"

    # the aggregate pools the counts, so one miss in six frames
    assert result["gt"] == 6 and result["fn"] == 1, f"got gt {result['gt']}, fn {result['fn']}"
    assert np.isclose(result["MOTA"], 1 - 1 / 6), f"expected {1 - 1 / 6}, got {result['MOTA']}"


def test_unit_ordering():
    """the tracker is stateful, so the frames have to arrive in index order"""

    pipeline = StubPipeline([([BOX], [1])] * 5)
    evaluate_sessions(pipeline, {"session_000": samples(5, 0)})

    assert pipeline.seen == [0, 1, 2, 3, 4], f"frames arrived out of order: {pipeline.seen}"


def test_unit_task():
    """a pipeline with no tracker cannot score identities"""

    pipeline = StubPipeline([([BOX], [1])], tracker=None, task="detection-only")

    try:
        evaluate_sessions(pipeline, {"session_000": samples(1, 0)})
        assert False, "should raise ValueError without a tracker"
    except ValueError as error:
        assert "detection-only" in str(error), f"got {error}"


if __name__ == "__main__":
    test_unit_reset()
    test_unit_counts()
    test_unit_ordering()
    test_unit_task()
    print("\nall passed")
