"""
MOT metric tests
================

tests the scoring of tracked sequences against ground truth with track ids.

a sequence is an ordered list of (gt_boxes, gt_ids, boxes, track_ids), one entry per frame. 

coverage:
- perfect:   ground truth as the prediction scores 1.0
- misses:    a dropped instrument is a false negative
- ghosts:    an extra box is a false positive
- switches:  an id change on a correctly located box
- aggregate: sequences are summed, not averaged
- empty:     no ground truth and no predictions
- one to one: a prediction is matched to at most one gt, and counts stay non negative
- classes:   a mislabeled track is scored apart from geometry and identity
- threshold: the configured iou threshold is the one applied
"""

import numpy as np
import pytest

from surgint.evaluation.mot import mot_counts, mot_evaluate

BOX_A = [10.0, 10.0, 50.0, 90.0]
BOX_B = [200.0, 10.0, 240.0, 90.0]


def sequence(frames):
    """
    one frame record per entry, from plain lists.

    entries are (gt_boxes, gt_ids, boxes, track_ids) with every class left at 0, or
    (gt_boxes, gt_ids, gt_classes, boxes, track_ids, class_ids) to set the classes.
    """
    records = []
    for frame in frames:
        if len(frame) == 4:
            gt_boxes, gt_ids, boxes, track_ids = frame
            gt_classes = [0] * len(gt_ids)
            class_ids = [0] * len(track_ids)
        else:
            gt_boxes, gt_ids, gt_classes, boxes, track_ids, class_ids = frame

        records.append((
            np.array(gt_boxes, dtype=float).reshape(-1, 4), gt_ids, gt_classes,
            np.array(boxes, dtype=float).reshape(-1, 4), track_ids, class_ids,
        ))
    return records


def test_unit_perfect():
    """the ground truth scored against itself"""

    frames = sequence([([BOX_A, BOX_B], [1, 2], [BOX_A, BOX_B], [7, 8]) for _ in range(5)])
    result = mot_evaluate(mot_counts(frames))

    # ids need not equal the ground truth ids, only be consistent with them
    assert result["MOTA"] == 1.0, f"expected 1.0, got {result['MOTA']}"
    assert result["IDF1"] == 1.0, f"expected 1.0, got {result['IDF1']}"
    assert result["id_switches"] == 0, f"expected 0, got {result['id_switches']}"
    assert result["gt"] == 10 and result["fp"] == 0 and result["fn"] == 0


def test_unit_misses():
    """an instrument the detector did not find"""

    frames = sequence(
        [([BOX_A, BOX_B], [1, 2], [BOX_A], [7]) for _ in range(4)]
    )
    result = mot_evaluate(mot_counts(frames))

    # 8 gt boxes, 4 of them never matched
    assert result["fn"] == 4, f"expected 4, got {result['fn']}"
    assert result["fp"] == 0, f"expected 0, got {result['fp']}"
    assert np.isclose(result["MOTA"], 1 - 4 / 8), f"got {result['MOTA']}"


def test_unit_ghosts():
    """a box with no instrument under it"""

    frames = sequence([([BOX_A], [1], [BOX_A, BOX_B], [7, 8]) for _ in range(4)])
    result = mot_evaluate(mot_counts(frames))

    assert result["fp"] == 4, f"expected 4, got {result['fp']}"
    assert result["fn"] == 0, f"expected 0, got {result['fn']}"

    # MOTA is normalised by the ground truth count, not the prediction count
    assert np.isclose(result["MOTA"], 1 - 4 / 4), f"got {result['MOTA']}"


def test_unit_switches():
    """the box stays right and the identity changes"""

    frames = sequence(
        [([BOX_A], [1], [BOX_A], [7]) for _ in range(3)]
        + [([BOX_A], [1], [BOX_A], [9]) for _ in range(3)]
    )
    result = mot_evaluate(mot_counts(frames))

    # every box is located correctly, so detection is perfect and only the id moved
    assert result["fp"] == 0 and result["fn"] == 0, "the boxes were all correct"
    assert result["id_switches"] == 1, f"expected 1, got {result['id_switches']}"
    assert np.isclose(result["MOTA"], 1 - 1 / 6), f"got {result['MOTA']}"

    # IDF1 keeps only the longer of the two id spans, so half the frames are lost
    assert np.isclose(result["IDF1"], 0.5), f"expected 0.5, got {result['IDF1']}"


def test_unit_aggregate():
    """sixteen sessions make one number"""

    clean = mot_counts(sequence([([BOX_A], [1], [BOX_A], [7]) for _ in range(9)]))
    missed = mot_counts(sequence([([BOX_A], [1], np.empty((0, 4)), []) for _ in range(1)]))

    result = mot_evaluate([clean, missed])

    # one miss in ten frames, not the average of 1.0 and 0.0
    assert result["sequences"] == 2, f"got {result['sequences']}"
    assert result["gt"] == 10 and result["fn"] == 1
    assert np.isclose(result["MOTA"], 0.9), f"expected 0.9, got {result['MOTA']}"

    # ids are per sequence, so the tracker is reset between them and never penalised
    assert result["id_switches"] == 0, f"expected 0, got {result['id_switches']}"


def test_unit_empty():
    """frames with nothing in them"""

    empty = np.empty((0, 4))
    result = mot_evaluate(mot_counts(sequence([(empty, [], empty, []) for _ in range(3)])))

    assert result["MOTA"] == 0.0 and result["IDF1"] == 0.0, "no ground truth scores 0"
    assert result["gt"] == 0 and result["fp"] == 0 and result["fn"] == 0

    # a frame of pure ghosts still counts them
    ghosts = mot_evaluate(mot_counts(sequence([(empty, [], [BOX_A], [7])])))
    assert ghosts["fp"] == 1, f"expected 1, got {ghosts['fp']}"


# two gt boxes overlapping one prediction above the threshold. reaching this needs both
# gt ids to already point at the same track, which happens when they are matched in
# different frames while the other is absent
SHARED = [0.0, 0.0, 10.0, 10.0]
OVERLAPPING = [1.0, 0.0, 11.0, 10.0]


def test_unit_one_prediction_is_not_counted_twice():
    """one prediction overlapping two gt boxes can satisfy only one of them"""

    frames = sequence([
        ([SHARED], [1], [SHARED], [5]),
        ([SHARED], [2], [SHARED], [5]),
        ([SHARED, OVERLAPPING], [1, 2], [SHARED], [5]),
    ])
    counts = mot_counts(frames)

    # three predictions over three frames, each fully matched, so nothing is spare
    assert counts["predictions"] == 3, f"got {counts['predictions']}"
    assert counts["fp"] == 0, f"a prediction was credited twice: fp {counts['fp']}"
    assert counts["gt"] - counts["fn"] <= counts["predictions"], "more matches than predictions"


def test_unit_shared_track_does_not_go_negative():
    """the sequence that produced a negative false positive count"""

    frames = sequence([
        ([SHARED], [1], [SHARED], [5]),                  # gt 1 takes track 5
        ([SHARED], [2], [SHARED], [5]),                  # gt 2 takes track 5, gt 1 absent
        ([SHARED, OVERLAPPING], [1, 2], [SHARED], [5]),  # both present, one prediction
    ])
    result = mot_evaluate(mot_counts(frames))

    assert result["fp"] == 0, f"one prediction, all of it matched, got fp {result['fp']}"
    assert result["fn"] == 1, f"the second gt is unmatched, got fn {result['fn']}"
    assert result["MOTA"] <= 1.0, f"MOTA above 1.0: {result['MOTA']}"


def test_unit_counts_never_go_negative():
    """matching stays one to one across random frames, so no count can go below zero"""

    rng = np.random.default_rng(0)
    for _ in range(50):
        frames = []
        for _ in range(6):
            # boxes drawn tight together, so many pairs clear the threshold at once
            gt_count, prediction_count = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            gt_boxes = [[x := float(rng.integers(0, 4)), 0.0, x + 10.0, 10.0] for _ in range(gt_count)]
            boxes = [[x := float(rng.integers(0, 4)), 0.0, x + 10.0, 10.0] for _ in range(prediction_count)]
            # ids are unique within a frame, as they are in any real annotation
            gt_ids = list(rng.choice(range(1, 6), gt_count, replace=False))
            track_ids = list(rng.choice(range(1, 6), prediction_count, replace=False))
            frames.append((gt_boxes, gt_ids, boxes, track_ids))

        counts = mot_counts(sequence(frames))
        assert counts["fp"] >= 0 and counts["fn"] >= 0, f"negative count: {counts}"
        assert counts["idtp"] <= min(counts["gt"], counts["predictions"]), f"idtp too large: {counts}"
        # a match consumes one box on each side, so it cannot exceed either total
        matched = counts["gt"] - counts["fn"]
        assert 0 <= matched <= min(counts["gt"], counts["predictions"]), f"bad match count: {counts}"
        assert counts["class_correct"] <= matched, f"class_correct above matched: {counts}"


def test_unit_repeated_ids_are_refused():
    """an id naming two objects in one frame has no one-to-one matching to find"""

    with pytest.raises(ValueError, match="ground truth ids repeat"):
        mot_counts(sequence([([SHARED, OVERLAPPING], [1, 1], [SHARED], [5])]))

    with pytest.raises(ValueError, match="track ids repeat"):
        mot_counts(sequence([([SHARED], [1], [SHARED, OVERLAPPING], [5, 5])]))


def test_unit_class_accuracy_is_scored_apart_from_geometry():
    """a track on the right box with the wrong instrument keeps MOTA at 1.0"""

    # gt class 3 every frame, the tracker calls it class 1 in two of four frames
    frames = sequence([
        ([BOX_A], [1], [3], [BOX_A], [7], [3]),
        ([BOX_A], [1], [3], [BOX_A], [7], [1]),
        ([BOX_A], [1], [3], [BOX_A], [7], [1]),
        ([BOX_A], [1], [3], [BOX_A], [7], [3]),
    ])
    result = mot_evaluate(mot_counts(frames))

    # geometry and identity are perfect, so the class error must not reach MOTA or IDF1
    assert result["MOTA"] == 1.0, f"expected 1.0, got {result['MOTA']}"
    assert result["IDF1"] == 1.0, f"expected 1.0, got {result['IDF1']}"
    assert result["fp"] == 0 and result["fn"] == 0, "every box was located"

    assert result["class_correct"] == 2, f"expected 2, got {result['class_correct']}"
    assert np.isclose(result["class_accuracy"], 0.5), f"expected 0.5, got {result['class_accuracy']}"


def test_unit_class_accuracy_is_one_when_labels_agree():
    """the same sequence with the right instrument scores both dimensions perfectly"""

    frames = sequence([([BOX_A, BOX_B], [1, 2], [3, 5], [BOX_A, BOX_B], [7, 8], [3, 5])] * 4)
    result = mot_evaluate(mot_counts(frames))

    assert result["MOTA"] == 1.0 and result["class_accuracy"] == 1.0
    assert result["class_correct"] == 8, f"expected 8, got {result['class_correct']}"


def test_unit_class_accuracy_counts_only_matched_pairs():
    """an unmatched box has no pair to agree with, so it leaves the accuracy alone"""

    # one located and correctly labeled instrument, one never detected
    frames = sequence([([BOX_A, BOX_B], [1, 2], [3, 5], [BOX_A], [7], [3])] * 3)
    result = mot_evaluate(mot_counts(frames))

    assert result["fn"] == 3, f"expected 3, got {result['fn']}"
    assert result["class_accuracy"] == 1.0, f"matched pairs all agree, got {result['class_accuracy']}"


def test_unit_every_box_needs_a_class():
    """a frame with classes missing is refused instead of scored"""

    frames = [(np.array([BOX_A, BOX_B]), [1, 2], [3], np.array([BOX_A]), [7], [3])]

    with pytest.raises(ValueError, match="every box needs a class"):
        mot_counts(frames)


def test_unit_iou_threshold_is_applied():
    """the threshold passed in decides what counts as located"""

    # iou is (10 - 3) / (10 + 3) = 0.538: above 0.5, below 0.9
    offset = [3.0, 0.0, 13.0, 10.0]
    frames = sequence([([[0.0, 0.0, 10.0, 10.0]], [1], [offset], [7])])

    lenient = mot_evaluate(mot_counts(frames, iou_threshold=0.5))
    strict = mot_evaluate(mot_counts(frames, iou_threshold=0.9))

    assert lenient["fn"] == 0 and lenient["fp"] == 0, f"0.538 clears 0.5: {lenient}"
    assert strict["fn"] == 1 and strict["fp"] == 1, f"0.538 fails 0.9: {strict}"


if __name__ == "__main__":
    test_unit_perfect()
    test_unit_misses()
    test_unit_ghosts()
    test_unit_switches()
    test_unit_aggregate()
    test_unit_empty()
    test_unit_one_prediction_is_not_counted_twice()
    test_unit_shared_track_does_not_go_negative()
    test_unit_counts_never_go_negative()
    test_unit_repeated_ids_are_refused()
    test_unit_class_accuracy_is_scored_apart_from_geometry()
    test_unit_class_accuracy_is_one_when_labels_agree()
    test_unit_class_accuracy_counts_only_matched_pairs()
    test_unit_every_box_needs_a_class()
    test_unit_iou_threshold_is_applied()
    print("\nall passed")
