import numpy as np

from surgint.evaluation.metrics import count_matches, iou_matrix

BOX = np.array([[10.0, 10.0, 20.0, 20.0]])


def test_iou_of_identical_boxes_is_one():
    print(f"identical: {iou_matrix(BOX, BOX)[0, 0]:.2f}")
    assert iou_matrix(BOX, BOX)[0, 0] == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    far = np.array([[100.0, 100.0, 110.0, 110.0]])

    print(f"disjoint: {iou_matrix(BOX, far)[0, 0]:.2f}")
    assert iou_matrix(BOX, far)[0, 0] == 0.0


def test_iou_of_half_overlapping_boxes():
    # shifted right by 5, so intersection is 50 and union is 150
    shifted = np.array([[15.0, 10.0, 25.0, 20.0]])

    print(f"half overlap: {iou_matrix(BOX, shifted)[0, 0]:.4f}")
    assert iou_matrix(BOX, shifted)[0, 0] == 50 / 150


def test_ground_truth_as_predictions_matches_everything():
    truth = np.array([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 90.0, 90.0], [0.0, 0.0, 5.0, 5.0]])

    matches = count_matches(truth, truth)

    print(f"gt as predictions: {matches}/{len(truth)}")
    assert matches == len(truth)


def test_two_predictions_cannot_claim_one_box():
    duplicated = np.repeat(BOX, 2, axis=0)

    matches = count_matches(duplicated, BOX)

    print(f"2 predictions on 1 box: {matches} match")
    assert matches == 1


def test_loose_overlap_does_not_match():
    # iou 50/150 is below the 0.5 threshold
    shifted = np.array([[15.0, 10.0, 25.0, 20.0]])

    assert count_matches(shifted, BOX) == 0


def test_empty_inputs_match_nothing():
    empty = np.empty((0, 4))

    assert count_matches(empty, BOX) == 0
    assert count_matches(BOX, empty) == 0


def test_unit_recall():
    for test in [
        test_iou_of_identical_boxes_is_one,
        test_iou_of_disjoint_boxes_is_zero,
        test_iou_of_half_overlapping_boxes,
        test_ground_truth_as_predictions_matches_everything,
        test_two_predictions_cannot_claim_one_box,
        test_loose_overlap_does_not_match,
        test_empty_inputs_match_nothing,
    ]:
        print(f"\n{test.__name__}")
        test()


if __name__ == "__main__":
    test_unit_recall()
    print("\nall passed")
