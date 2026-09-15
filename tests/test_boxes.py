"""
Box geometry tests
==================

tests the overlap measure shared by evaluation and tracking.

coverage:
- iou:    identical, disjoint, and partial overlap
- shapes: every pairing in the matrix, and empty inputs
"""

import numpy as np

from surgint.model.boxes import iou_matrix

BOX = np.array([[10.0, 10.0, 20.0, 20.0]])


def test_unit_iou():
    """pairwise overlap between two sets of boxes"""

    # a box against itself is the only way to score 1
    assert iou_matrix(BOX, BOX)[0, 0] == 1.0, f"got {iou_matrix(BOX, BOX)[0, 0]}"

    # no overlap is zero, not a small number
    far = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(BOX, far)[0, 0] == 0.0, f"got {iou_matrix(BOX, far)[0, 0]}"

    # shifted right by 5, so intersection is 50 and union is 150
    shifted = np.array([[15.0, 10.0, 25.0, 20.0]])
    assert iou_matrix(BOX, shifted)[0, 0] == 50 / 150, f"got {iou_matrix(BOX, shifted)[0, 0]}"

    # one row per box, one column per other, so association can index it directly
    ious = iou_matrix(np.repeat(BOX, 3, axis=0), np.concatenate([BOX, far, shifted]))
    assert ious.shape == (3, 3), f"got {ious.shape}"
    assert np.allclose(ious[0], [1.0, 0.0, 50 / 150]), f"got {ious[0].tolist()}"

    # an empty side keeps the matrix shape rather than collapsing it
    empty = np.empty((0, 4))
    assert iou_matrix(empty, BOX).shape == (0, 1), f"got {iou_matrix(empty, BOX).shape}"
    assert iou_matrix(BOX, empty).shape == (1, 0), f"got {iou_matrix(BOX, empty).shape}"


if __name__ == "__main__":
    test_unit_iou()
    print("\nall passed")
