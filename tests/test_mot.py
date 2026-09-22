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
"""

import numpy as np

from surgint.evaluation.mot import mot_counts, mot_evaluate

BOX_A = [10.0, 10.0, 50.0, 90.0]
BOX_B = [200.0, 10.0, 240.0, 90.0]


def sequence(frames):
    """(gt_boxes, gt_ids, boxes, track_ids) per frame, from plain lists"""
    return [
        (np.array(gt_boxes, dtype=float).reshape(-1, 4), gt_ids,
         np.array(boxes, dtype=float).reshape(-1, 4), track_ids)
        for gt_boxes, gt_ids, boxes, track_ids in frames
    ]


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


if __name__ == "__main__":
    test_unit_perfect()
    test_unit_misses()
    test_unit_ghosts()
    test_unit_switches()
    test_unit_aggregate()
    test_unit_empty()
    print("\nall passed")
