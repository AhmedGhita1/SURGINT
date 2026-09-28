import tempfile
from pathlib import Path

import numpy as np
import pytest

from surgint.model import Detections
from surgint.model.detector import Detector
from surgint.runtime.pipeline import Pipeline

CHECKPOINT = "PekingU/rtdetr_r18vd_coco_o365"
INPUT_SIZE = [320, 192]

pytestmark = pytest.mark.model


def blank() -> np.ndarray:
    return np.zeros((180, 320, 3), dtype=np.uint8)


def test_from_checkpoint():
    """The runtime geometry comes from the model manifest, not a separate config."""
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

        assert [pipeline.transform.width, pipeline.transform.height] == INPUT_SIZE
        assert pipeline.transform.pad_color == 0
        assert pipeline.transform.rescale_factor == 1.0
        assert pipeline.task == "detection-tracking"
        assert pipeline.detector.meta["labels"] == ["scalpel", "scissors"]
        assert isinstance(pipeline.predict(blank(), 0.5), Detections)

        bare = Path(directory) / "bare"
        detector.model.save_pretrained(bare)
        try:
            Pipeline.from_checkpoint(bare, device="cpu")
            assert False, "should raise for an artifact without a manifest"
        except FileNotFoundError:
            pass
