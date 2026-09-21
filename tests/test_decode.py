"""
Decode tests
============

tests the conversion from raw model output to a per-image detection list.

coverage:
- classes:  one class per query, never two detections from one box
- filter:   score threshold and descending order
- boxes:    passed through as the normalized cxcywh the model emits
- shapes:   empty frames, one entry per image in the batch
- top k:    max_detections keeps the highest scoring queries
"""

import numpy as np
import torch

from surgint.model.decode import decode


def logits_from(scores: list[list[float]]) -> torch.Tensor:
    """probabilities to the raw logits decode expects"""
    return torch.logit(torch.tensor([scores], dtype=torch.float32))


def test_unit_decode():
    """raw logits and boxes to a scored, sorted detection list"""

    # decode does no geometry; the model already emits normalized cxcywh
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    boxes, _, _ = decode(logits_from([[0.9]]), pred_boxes, 0.5)[0]
    assert boxes.tolist() == [[0.5, 0.5, 0.5, 0.5]], f"boxes were altered: {boxes.tolist()}"
    assert boxes.dtype == np.float32, "boxes must be float32"

    # a single query scoring high on two classes must not become two detections;
    # duplicate boxes under different labels would open two tracks for one instrument
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    boxes, scores, class_ids = decode(logits_from([[0.7, 0.9]]), pred_boxes, 0.5)[0]
    assert len(boxes) == 1, "one query must yield at most one detection"
    assert class_ids.tolist() == [1], "the highest scoring class wins"
    assert np.allclose(scores[0], 0.9, atol=1e-6), f"got {scores[0]}"

    # queries below the threshold are dropped, the rest come back highest first
    pred_boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]])
    _, scores, _ = decode(logits_from([[0.4], [0.95], [0.6]]), pred_boxes, 0.5)[0]
    assert len(scores) == 2, "only two queries clear a threshold of 0.5"
    assert scores.tolist() == sorted(scores.tolist(), reverse=True), "detections must be sorted"

    # an empty frame must keep the shapes the pipeline expects
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])
    boxes, scores, class_ids = decode(logits_from([[0.1]]), pred_boxes, 0.5)[0]
    assert boxes.shape == (0, 4), "empty boxes must stay (0, 4)"
    assert scores.shape == (0,), "empty scores must stay (0,)"
    assert class_ids.shape == (0,), "empty class ids must stay (0,)"

    # one entry per image, so a batch does not merge detections across frames
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]], [[0.5, 0.5, 0.1, 0.1]]])
    logits = torch.logit(torch.tensor([[[0.9]], [[0.1]]], dtype=torch.float32))
    detections = decode(logits, pred_boxes, 0.5)
    assert len(detections) == 2, "one entry per image in the batch"
    assert len(detections[0][0]) == 1, "first image has one detection"
    assert len(detections[1][0]) == 0, "second image has none"


if __name__ == "__main__":
    test_unit_decode()
    print("\nall passed")


def test_unit_decode_max_detections():
    """max_detections caps the list at the highest scoring queries"""

    pred_boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]])
    logits = logits_from([[0.9], [0.5], [0.7]])

    # uncapped, every query over the threshold survives, highest score first
    _, scores, _ = decode(logits, pred_boxes, 0.1)[0]
    assert np.allclose(scores, [0.9, 0.7, 0.5], atol=1e-6), f"got {scores.tolist()}"

    # the cap keeps the top k of that order
    boxes, scores, _ = decode(logits, pred_boxes, 0.1, max_detections=2)[0]
    assert len(scores) == 2, f"expected 2 detections, got {len(scores)}"
    assert np.allclose(scores, [0.9, 0.7], atol=1e-6), f"got {scores.tolist()}"
    expected = [[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]]
    assert np.allclose(boxes, expected, atol=1e-6), f"boxes must follow the scores, got {boxes.tolist()}"

    # a cap above the count changes nothing
    assert len(decode(logits, pred_boxes, 0.1, max_detections=99)[0][1]) == 3
