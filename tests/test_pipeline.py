"""
Pipeline tests
==============

tests the runtime path: a frame in, Detections out.

a stub detector stands in for the weights, so the boxes are controlled and the
coordinate round trip can be checked exactly.

coverage:
- task:       the key is validated, track_ids stay empty for detection-only
- predict:    Detections in frame pixels, aligned arrays, threshold
- checkpoint: from_checkpoint takes its geometry from meta.yaml
- tracking:   track_ids under detection-tracking, None under detection-only
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from surgint.model import Detections
from surgint.model.detector import Detector
from surgint.model.transform import Transform
from surgint.runtime.bytetrack import CONFIRM_HITS
from surgint.runtime.pipeline import Pipeline

CHECKPOINT = "PekingU/rtdetr_r18vd_coco_o365"
INPUT_SIZE = [320, 192]
FRAME_HEIGHT, FRAME_WIDTH = 180, 320
CLASSES = 2


class StubDetector:
    """one query per frame, at a fixed normalized box and class"""

    def __init__(self, boxes: np.ndarray, class_id: int = 0, score: float = 0.9):
        self.boxes = boxes
        self.class_id = class_id
        self.score = score

    def predict(self, pixel_values):
        assert pixel_values.dim() == 4, "the model always takes a batch"
        assert len(pixel_values) == 1, "a frame is a batch of one"

        logits = torch.full((1, len(self.boxes), CLASSES), -10.0)
        logits[:, :, self.class_id] = torch.logit(torch.tensor(self.score))
        return logits, torch.tensor(self.boxes, dtype=torch.float32).unsqueeze(0)


def blank() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def test_unit_task():
    """the key the pipeline is built with"""

    transform = Transform(INPUT_SIZE)
    detector = StubDetector(np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32))

    assert Pipeline(detector, transform, "detection-only").task == "detection-only"
    assert Pipeline(detector, transform, "detection-tracking").task == "detection-tracking"

    try:
        Pipeline(detector, transform, "segmentation")
        assert False, "should raise for an unknown task"
    except ValueError:
        pass


def test_unit_predict():
    """a frame in, Detections out"""

    transform = Transform(INPUT_SIZE)

    # a frame box put through the transform, then handed back as the model output
    frame_box = np.array([[40.0, 60.0, 100.0, 110.0]], dtype=np.float32)
    normalized = transform(blank(), frame_box, np.array([0], dtype=np.int64))["boxes"]

    pipeline = Pipeline(StubDetector(normalized), transform, "detection-only")
    result = pipeline.predict(blank(), score_threshold=0.3)

    assert isinstance(result, Detections)
    assert len(result.boxes) == len(result.scores) == len(result.class_ids) == 1

    # the box comes back where it started, in frame pixels
    assert np.allclose(result.boxes, frame_box, atol=1e-2), f"got {result.boxes}"
    assert result.boxes.dtype == np.float32

    # detection-only leaves the tracker's field empty
    assert result.track_ids is None, "detection-only must not invent track ids"

    # the threshold filters, so nothing survives above the query's score
    assert len(pipeline.predict(blank(), score_threshold=0.99).boxes) == 0

    # an empty result keeps the shapes a consumer expects
    empty = pipeline.predict(blank(), score_threshold=0.99)
    assert empty.boxes.shape == (0, 4)
    assert empty.scores.shape == (0,) and empty.class_ids.shape == (0,)


def test_unit_tracking():
    """track_ids across frames, and reset between sequences"""

    transform = Transform(INPUT_SIZE)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)

    # detection-only leaves track_ids None. reset is a no-op and must not raise.
    plain = Pipeline(StubDetector(boxes), transform, "detection-only")
    assert plain.tracker is None, "detection-only must not build a tracker"
    assert plain.predict(blank(), 0.3).track_ids is None, "detection-only must not track"
    plain.reset()

    tracked = Pipeline(StubDetector(boxes), transform, "detection-tracking")

    # nothing is reported before confirmation
    for hit in range(CONFIRM_HITS - 1):
        assert len(tracked.predict(blank(), 0.3).boxes) == 0, f"reported at {hit + 1} hits"

    result = tracked.predict(blank(), 0.3)
    assert result.track_ids.tolist() == [1], f"got {result.track_ids.tolist()}"
    assert len(result.boxes) == len(result.track_ids), "the arrays must stay aligned"

    # the id holds while the instrument is present
    for _ in range(3):
        assert tracked.predict(blank(), 0.3).track_ids.tolist() == [1], "the id must not change"

    # reset clears the tracks and the id counter
    tracked.reset()
    for _ in range(CONFIRM_HITS):
        result = tracked.predict(blank(), 0.3)
    assert result.track_ids.tolist() == [1], f"expected id 1 after reset, got {result.track_ids.tolist()}"


def test_unit_from_checkpoint():
    """the geometry comes from meta.yaml, not from a config"""

    detector = Detector.from_pretrained(CHECKPOINT, {0: "scalpel", 1: "scissors"})
    geometry = {
        "input_size": INPUT_SIZE,
        "pad_color": 0,
        "rescale_factor": 1.0,
        "source": CHECKPOINT,
    }

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best"
        detector.save_checkpoint(path, geometry)

        pipeline = Pipeline.from_checkpoint(path, task="detection-tracking", device="cpu")

        # every geometry value came off the checkpoint, none of them defaults
        assert [pipeline.transform.width, pipeline.transform.height] == INPUT_SIZE
        assert pipeline.transform.pad_color == 0, "pad color fell back to the default"
        assert pipeline.transform.rescale_factor == 1.0, "rescale factor fell back to the default"
        assert pipeline.task == "detection-tracking"
        assert pipeline.detector.meta["labels"] == ["scalpel", "scissors"]

        # it still runs on a frame
        assert isinstance(pipeline.predict(blank(), 0.5), Detections)

        # weights without meta.yaml are not a checkpoint
        bare = Path(directory) / "bare"
        detector.model.save_pretrained(bare)
        try:
            Pipeline.from_checkpoint(bare, device="cpu")
            assert False, "should raise for a checkpoint without meta.yaml"
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    test_unit_task()
    test_unit_predict()
    test_unit_tracking()
    test_unit_from_checkpoint()
    print("\nall passed")
