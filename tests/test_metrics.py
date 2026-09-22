import numpy as np

from surgint.evaluation.metrics import count_matches

BOX = np.array([[10.0, 10.0, 20.0, 20.0]])


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
