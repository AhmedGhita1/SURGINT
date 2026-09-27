import numpy as np
import pytest

from surgint.decision import ItemContextOverride, SessionContext
from surgint.model import Detections
from surgint.runtime.decision_support import InventoryDecisionSupport
from surgint.runtime.session import DecisionSupportSession

from .factories import tracked_frame


def session():
    support = InventoryDecisionSupport(
        labels=["scalpel", "gauze"],
        perception_version="perception-v1",
    )
    return DecisionSupportSession(support)


def context():
    return SessionContext(
        workflow_stage="post-procedure-clearing",
        use_state="unused",
        contamination_state="not-regulated",
    )


def test_session_exposes_decisions_only_after_finalization():
    runtime = session()
    runtime.update(tracked_frame([0, 0, 1], [1, 2, 3], [0.9, 0.8, 0.95]))
    runtime.update(tracked_frame([0], [4], [0.97]))

    assert runtime.frame_count == 2
    assert not runtime.is_finalized
    with pytest.raises(RuntimeError, match="not been finalized"):
        runtime.result

    result = runtime.finalize(
        context(),
        overrides={0: ItemContextOverride(lifecycle="reusable")},
    )

    assert runtime.is_finalized
    assert runtime.result is result
    assert result.frames == 2

    by_class = result.by_class()
    assert by_class[0].inventory_item.count == 2
    assert by_class[0].inventory_item.distinct_tracks == 3
    assert by_class[0].decision.action == "secure-transport-to-reprocessing"
    assert by_class[1].decision.action == "facility-general-waste"


def test_finalized_session_rejects_updates_and_second_finalization():
    runtime = session()
    runtime.update(tracked_frame([0], [1], [0.9]))
    runtime.finalize(
        context(),
        overrides={0: ItemContextOverride(lifecycle="reusable")},
    )

    with pytest.raises(RuntimeError, match="already been finalized"):
        runtime.update(tracked_frame([0], [2], [0.9]))
    with pytest.raises(RuntimeError, match="already been finalized"):
        runtime.finalize(context())


def test_failed_finalization_leaves_session_open_for_context_correction():
    runtime = session()
    runtime.update(tracked_frame([0], [1], [0.9]))

    with pytest.raises(ValueError, match=r"absent class ids \[1\]"):
        runtime.finalize(
            context(),
            overrides={1: ItemContextOverride(lifecycle="reusable")},
        )

    assert not runtime.is_finalized
    runtime.update(tracked_frame([0], [2], [0.92]))
    result = runtime.finalize(
        context(),
        overrides={0: ItemContextOverride(lifecycle="reusable")},
    )

    assert result.frames == 2


def test_session_rejects_untracked_detections_without_closing():
    runtime = session()
    detections = Detections(
        boxes=np.zeros((1, 4), dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        class_ids=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="detection-tracking"):
        runtime.update(detections)

    assert runtime.frame_count == 0
    assert not runtime.is_finalized
