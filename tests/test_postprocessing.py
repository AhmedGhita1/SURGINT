# unit tests:
#   box conversion, 
#   one class per query, 
#   score filter and order, 
#   empty frames, 
#   batching

import numpy as np
import pytest
import torch

from surgint.detection.postprocessing import decode

WIDTH, HEIGHT = 1024, 576


def logits_from(scores: list[list[float]]) -> torch.Tensor:
    """probabilities to the raw logits decode expects"""
    return torch.logit(torch.tensor([scores], dtype=torch.float32))


def test_decode_converts_to_canvas_pixels():
    # one query centered at (0.5, 0.5), half the canvas wide and tall
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])

    boxes, _, _ = decode(logits_from([[0.9]]), pred_boxes, WIDTH, HEIGHT, 0.5)[0]

    print(f"cxcywh {pred_boxes[0, 0].tolist()}: xyxy {boxes[0].tolist()} on {WIDTH}x{HEIGHT}")
    assert boxes.tolist() == [[256.0, 144.0, 768.0, 432.0]]
    assert boxes.dtype == np.float32


def test_decode_keeps_one_class_per_query():
    # a single query scoring high on two classes must not become two detections
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])

    boxes, scores, class_ids = decode(logits_from([[0.7, 0.9]]), pred_boxes, WIDTH, HEIGHT, 0.5)[0]

    print(f"query scores [0.7, 0.9]: {len(boxes)} detection, class {class_ids.tolist()}, score {scores[0]:.2f}")
    assert len(boxes) == 1
    assert class_ids.tolist() == [1]
    assert scores[0] == pytest.approx(0.9, abs=1e-6)


def test_decode_filters_by_score_and_sorts():
    pred_boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]])
    logits = logits_from([[0.4], [0.95], [0.6]])

    _, scores, _ = decode(logits, pred_boxes, WIDTH, HEIGHT, 0.5)[0]

    print(f"queries [0.4, 0.95, 0.6] at threshold 0.5: {[f'{s:.2f}' for s in scores]}")
    assert len(scores) == 2
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)


def test_decode_handles_no_detections():
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])

    boxes, scores, class_ids = decode(logits_from([[0.1]]), pred_boxes, WIDTH, HEIGHT, 0.5)[0]

    # an empty frame must keep the shapes the pipeline expects
    print(f"nothing above threshold: boxes {boxes.shape}, scores {scores.shape}, ids {class_ids.shape}")
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert class_ids.shape == (0,)


def test_decode_returns_one_entry_per_image():
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]], [[0.5, 0.5, 0.1, 0.1]]])
    logits = torch.logit(torch.tensor([[[0.9]], [[0.1]]], dtype=torch.float32))

    detections = decode(logits, pred_boxes, WIDTH, HEIGHT, 0.5)

    print(f"batch of 2: {[len(boxes) for boxes, _, _ in detections]} detections per image")
    assert len(detections) == 2
    assert len(detections[0][0]) == 1
    assert len(detections[1][0]) == 0


def test_unit_postprocessing():
    for test in [
        test_decode_converts_to_canvas_pixels,
        test_decode_keeps_one_class_per_query,
        test_decode_filters_by_score_and_sorts,
        test_decode_handles_no_detections,
        test_decode_returns_one_entry_per_image,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_postprocessing()
    print("\nall passed")
