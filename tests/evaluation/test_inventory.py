"""
Inventory metric tests
======================

The metric counts physical ground-truth instances once per session and compares them
with the finalized predicted class counts.
"""

import numpy as np
import pytest

from surgint.evaluation.inventory import inventory_counts, inventory_evaluate


def frame(instance_uids, class_ids):
    return np.asarray(instance_uids, dtype=str), np.asarray(class_ids, dtype=np.int64)


def test_perfect_inventory():
    frames = [
        frame(["scalpel-1", "gauze-1"], [0, 1]),
        frame(["scalpel-1", "gauze-1"], [0, 1]),
    ]

    result = inventory_evaluate(inventory_counts(frames, {0: 1, 1: 1}, [0, 1]))

    assert result["inventory_absolute_error"] == 0
    assert result["class_exact_match_rate"] == 1.0
    assert result["session_exact_match_rate"] == 1.0


def test_wrong_classes_do_not_cancel_total_error():
    frames = [frame(["scalpel-1", "gauze-1"], [0, 1])]

    result = inventory_evaluate(inventory_counts(frames, {0: 0, 1: 2}, [0, 1]))

    assert result["inventory_ground_truth"] == result["inventory_predictions"] == 2
    assert result["inventory_error"] == 0
    assert result["inventory_absolute_error"] == 2
    assert result["class_exact_match_rate"] == 0.0
    assert result["session_exact_match_rate"] == 0.0


def test_session_errors_do_not_cancel_when_pooled():
    over = inventory_counts([frame(["item-1"], [0])], {0: 2}, [0])
    under = inventory_counts([frame(["item-2"], [0])], {0: 0}, [0])

    result = inventory_evaluate([over, under])

    assert result["inventory_error"] == 0
    assert result["inventory_absolute_error"] == 2
    assert result["inventory_mean_absolute_error"] == 1.0
    assert result["class_exact_match_rate"] == 0.0
    assert result["inventory_per_class"][0]["absolute_error"] == 2


def test_fixed_vocabulary_scores_absent_classes():
    result = inventory_evaluate(
        inventory_counts([frame(["item-1"], [0])], {0: 1}, [0, 1])
    )

    assert result["inventory_per_class"][1]["ground_truth"] == 0
    assert result["inventory_per_class"][1]["predictions"] == 0
    assert result["inventory_per_class"][1]["exact_match_rate"] == 1.0
    assert result["class_exact_match_rate"] == 1.0


def test_instance_uid_must_keep_one_class():
    frames = [frame(["item-1"], [0]), frame(["item-1"], [1])]

    with pytest.raises(ValueError, match="maps to classes"):
        inventory_counts(frames, {}, [0, 1])


def test_sessions_must_share_the_class_vocabulary():
    first = inventory_counts([], {}, [0])
    second = inventory_counts([], {}, [0, 1])

    with pytest.raises(ValueError, match="same class ids"):
        inventory_evaluate([first, second])
