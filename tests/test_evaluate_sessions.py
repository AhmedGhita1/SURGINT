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
- classes:  the ground truth class reaches the metric
- threshold: the iou threshold handed in is the one used
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
        output = self.outputs[len(self.seen) - 1]
        boxes, track_ids = output[0], output[1]
        class_ids = output[2] if len(output) > 2 else np.zeros(len(boxes), dtype=np.int64)
        return Detections(
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.ones(len(boxes), dtype=np.float32),
            np.asarray(class_ids, dtype=np.int64).reshape(-1),
            np.asarray(track_ids, dtype=np.int64),
        )


def samples(count, marker, class_id=0):
    """a session of `count` frames, each frame tagged with `marker` so order is checkable"""
    return [
        {
            "frame": np.full((4, 4), marker + index, dtype=np.uint8),
            "boxes": BOX.reshape(1, 4),
            "class_ids": np.full(1, class_id, dtype=np.int64),
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


def test_unit_classes_reach_the_metric():
    """a tracked box with the wrong instrument lowers class accuracy, not MOTA"""

    # ground truth is class 3, the stub predicts class 1 on the same box
    wrong = np.array([1], dtype=np.int64)
    pipeline = StubPipeline([([BOX], [1], wrong)] * 3)

    result = evaluate_sessions(pipeline, {"session_000": samples(3, 0, class_id=3)})

    assert result["MOTA"] == 1.0, f"the box was located every frame, got {result['MOTA']}"
    assert result["class_accuracy"] == 0.0, f"no pair agrees, got {result['class_accuracy']}"
    assert result["class_correct"] == 0, f"got {result['class_correct']}"


def test_unit_threshold_is_forwarded():
    """the threshold given to evaluate_sessions is the one the counts are scored at"""

    # the prediction overlaps the ground truth at iou 0.538
    offset = BOX + np.array([13.0, 0.0, 13.0, 0.0])
    outputs = [([offset], [1])] * 2

    lenient = evaluate_sessions(StubPipeline(outputs), {"session_000": samples(2, 0)}, 0.5)
    strict = evaluate_sessions(StubPipeline(outputs), {"session_000": samples(2, 0)}, 0.9)

    assert lenient["fn"] == 0, f"0.538 clears 0.5, got fn {lenient['fn']}"
    assert strict["fn"] == 2, f"0.538 fails 0.9, got fn {strict['fn']}"


if __name__ == "__main__":
    test_unit_reset()
    test_unit_counts()
    test_unit_ordering()
    test_unit_task()
    test_unit_classes_reach_the_metric()
    test_unit_threshold_is_forwarded()
    print("\nall passed")
