"""
Box geometry tests
==================

tests the overlap measure shared by evaluation and tracking.

coverage:
- iou:    identical, disjoint, and partial overlap
- shapes: every pairing in the matrix, and empty inputs
"""

import numpy as np

from surgint.model.boxes import iou_matrix, nms

BOX = np.array([[10.0, 10.0, 20.0, 20.0]])


def test_iou():
    """pairwise overlap between two sets of boxes"""

    # a box against itself scores 1
    assert iou_matrix(BOX, BOX)[0, 0] == 1.0, f"got {iou_matrix(BOX, BOX)[0, 0]}"

    # no overlap scores 0
    far = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(BOX, far)[0, 0] == 0.0, f"got {iou_matrix(BOX, far)[0, 0]}"

    # shifted right by 5, so intersection is 50 and union is 150
    shifted = np.array([[15.0, 10.0, 25.0, 20.0]])
    assert iou_matrix(BOX, shifted)[0, 0] == 50 / 150, f"got {iou_matrix(BOX, shifted)[0, 0]}"

    # one row per box, one column per other
    ious = iou_matrix(np.repeat(BOX, 3, axis=0), np.concatenate([BOX, far, shifted]))
    assert ious.shape == (3, 3), f"got {ious.shape}"
    assert np.allclose(ious[0], [1.0, 0.0, 50 / 150]), f"got {ious[0].tolist()}"

    # an empty side keeps the matrix two-dimensional
    empty = np.empty((0, 4))
    assert iou_matrix(empty, BOX).shape == (0, 1), f"got {iou_matrix(empty, BOX).shape}"
    assert iou_matrix(BOX, empty).shape == (1, 0), f"got {iou_matrix(BOX, empty).shape}"



def test_nms():
    """greedy suppression of boxes covering the same object"""

    # two boxes on one object, one elsewhere
    boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                      [1.0, 1.0, 11.0, 11.0],
                      [100.0, 100.0, 110.0, 110.0]])
    scores = np.array([0.6, 0.9, 0.7])

    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2], f"the weaker overlapping box must go, got {keep.tolist()}"

    # a threshold above the pair's iou keeps both
    assert sorted(nms(boxes, scores, 0.95).tolist()) == [0, 1, 2], "nothing overlaps that far"

    # indices come back highest score first
    assert nms(boxes, scores, 0.5)[0] == 1, "the highest scoring box is kept first"

    # suppression is class agnostic, so it runs on boxes alone
    assert nms(boxes[:1], scores[:1], 0.5).tolist() == [0], "a single box survives"
    assert nms(np.empty((0, 4)), np.empty(0), 0.5).tolist() == [], "an empty frame keeps nothing"

