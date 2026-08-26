from functools import lru_cache

import numpy as np
import torch

from surgint.config import Config
from surgint.inference.detector import DetectionResult, InferencePipeline

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


@lru_cache(maxsize=1)
def build_pipeline() -> InferencePipeline:
    config = Config()
    return InferencePipeline(config.checkpoint, config.input_size, device="cpu")


def test_predict_returns_the_contract():
    result = build_pipeline().predict(FRAME, 0.0)

    print(f"detections: {len(result.boxes)}")
    assert isinstance(result, DetectionResult)
    assert not isinstance(result.boxes, torch.Tensor)
    assert result.boxes.dtype == np.float32
    assert len(result.boxes) == len(result.scores) == len(result.class_ids)


def test_predict_clips_boxes_to_the_frame():
    height, width = FRAME.shape[:2]

    boxes = build_pipeline().predict(FRAME, 0.0).boxes

    print(f"x range [{boxes[:, 0::2].min():.1f}, {boxes[:, 0::2].max():.1f}] within [0, {width}]")
    print(f"y range [{boxes[:, 1::2].min():.1f}, {boxes[:, 1::2].max():.1f}] within [0, {height}]")
    assert boxes[:, 0::2].min() >= 0 and boxes[:, 0::2].max() <= width
    assert boxes[:, 1::2].min() >= 0 and boxes[:, 1::2].max() <= height


def test_predict_handles_an_empty_frame():
    result = build_pipeline().predict(FRAME, 0.99)

    print(f"at threshold 0.99: boxes {result.boxes.shape}")
    assert result.boxes.shape == (0, 4)


def test_unit_pipeline():
    for test in [
        test_predict_returns_the_contract,
        test_predict_clips_boxes_to_the_frame,
        test_predict_handles_an_empty_frame,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_pipeline()
    print("\nall passed")
