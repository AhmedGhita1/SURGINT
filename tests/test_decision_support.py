import numpy as np
import pytest

from surgint.decision import ItemContext, ItemContextOverride, SessionContext
from surgint.model import Detections
from surgint.runtime.decision_support import InventoryDecisionSupport
from surgint.runtime.inventory import Inventory


def tracked_frame(class_ids, track_ids, scores):
    count = len(track_ids)
    return Detections(
        boxes=np.zeros((count, 4), dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        track_ids=np.asarray(track_ids, dtype=np.int64),
    )


def post_procedure(**overrides):
    values = {"workflow_stage": "post-procedure-clearing"}
    values.update(overrides)
    return ItemContext(**values)


def test_tracked_detection_resolves_through_inventory_and_decision_support():
    inventory = Inventory()
    inventory.update(tracked_frame([0], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="detector-v1+tracker-v1",
    )

    decisions = support.resolve_inventory(
        inventory,
        post_procedure(lifecycle="reusable"),
    )

    decision = decisions[7]
    assert decision.track_id == 7
    assert decision.label == "scalpel"
    assert decision.confidence == pytest.approx(0.91)
    assert decision.outcome == "recommendation"
    assert decision.action == "secure-transport-to-reprocessing"
    assert decision.perception_version == "detector-v1+tracker-v1"
    assert decision.ontology_version.endswith("/3.0.0")
    assert decision.policy_version == "2.0.0"


def test_inventory_items_can_use_track_specific_context():
    inventory = Inventory()
    inventory.update(tracked_frame([0, 1], [7, 8], [0.91, 0.88]))
    support = InventoryDecisionSupport(
        labels=["scalpel", "gauze"],
        perception_version="perception-v1",
    )

    decisions = support.resolve_inventory(
        inventory,
        {
            7: post_procedure(lifecycle="single-use"),
            8: post_procedure(
                use_state="unused",
                contamination_state="not-regulated",
            ),
        },
    )

    assert decisions[7].action == "approved-sharps-stream"
    assert decisions[8].action == "facility-general-waste"


def test_inventory_label_not_supported_by_ontology_is_an_outcome():
    inventory = Inventory()
    inventory.update(tracked_frame([0], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["unknown-category"],
        perception_version="perception-v1",
    )

    decision = support.resolve_inventory(inventory, post_procedure())[7]

    assert decision.outcome == "unsupported_item"
    assert decision.action is None


def test_missing_class_id_in_label_map_is_a_configuration_error():
    inventory = Inventory()
    inventory.update(tracked_frame([2], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="perception-v1",
    )

    with pytest.raises(ValueError, match="class_id 2 has no perception label"):
        support.resolve_inventory(inventory, post_procedure())


def test_every_inventory_item_requires_context_when_contexts_are_per_track():
    inventory = Inventory()
    inventory.update(tracked_frame([0, 0], [7, 8], [0.91, 0.88]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="perception-v1",
    )

    with pytest.raises(ValueError, match=r"track ids \[8\]"):
        support.resolve_inventory(inventory, {7: post_procedure()})


def test_label_map_rejects_duplicate_perception_labels():
    with pytest.raises(ValueError, match="duplicates"):
        InventoryDecisionSupport(
            labels=["scalpel", "scalpel"],
            perception_version="perception-v1",
        )


def test_final_inventory_resolves_once_per_class_after_fragmentation():
    inventory = Inventory()
    inventory.update(tracked_frame([0, 0], [7, 8], [0.91, 0.88]))
    inventory.update(tracked_frame([0], [9], [0.95]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="perception-v1",
    )

    result = support.resolve_final_inventory(
        inventory.finalize(),
        post_procedure(lifecycle="reusable"),
    )

    assert result.frames == 2
    assert len(result.items) == 1
    assert set(result.by_class()) == {0}

    resolved = result.by_class()[0]
    assert resolved.inventory_item.count == 2
    assert resolved.inventory_item.distinct_tracks == 3
    assert resolved.decision.track_id == 9
    assert resolved.decision.outcome == "recommendation"
    assert resolved.decision.action == "secure-transport-to-reprocessing"


def test_final_inventory_decision_result_is_immutable():
    inventory = Inventory()
    inventory.update(tracked_frame([0], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="perception-v1",
    )

    result = support.resolve_final_inventory(
        inventory.finalize(),
        post_procedure(lifecycle="reusable"),
    )

    with pytest.raises(AttributeError):
        result.frames = 2
    with pytest.raises(AttributeError):
        result.items[0].decision = None


def test_final_inventory_composes_session_and_class_context():
    inventory = Inventory()
    inventory.update(tracked_frame([0, 1], [7, 8], [0.91, 0.88]))
    support = InventoryDecisionSupport(
        labels=["scalpel", "gauze"],
        perception_version="perception-v1",
    )
    session = SessionContext(
        workflow_stage="post-procedure-clearing",
        use_state="unused",
        contamination_state="not-regulated",
    )

    result = support.resolve_final_inventory(
        inventory.finalize(),
        session,
        overrides={0: ItemContextOverride(lifecycle="reusable")},
    ).by_class()

    assert result[0].decision.action == "secure-transport-to-reprocessing"
    assert result[1].decision.action == "facility-general-waste"


def test_final_inventory_reports_lifecycle_conflict_for_human_review():
    inventory = Inventory()
    inventory.update(tracked_frame([0], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["gauze"],
        perception_version="perception-v1",
    )

    result = support.resolve_final_inventory(
        inventory.finalize(),
        SessionContext(workflow_stage="post-procedure-clearing"),
        overrides={0: ItemContextOverride(lifecycle="reusable")},
    ).by_class()[0]

    assert result.decision.outcome == "human_review"
    assert "conflicts" in result.decision.reason


def test_final_inventory_rejects_override_for_absent_class():
    inventory = Inventory()
    inventory.update(tracked_frame([0], [7], [0.91]))
    support = InventoryDecisionSupport(
        labels=["scalpel"],
        perception_version="perception-v1",
    )

    with pytest.raises(ValueError, match=r"absent class ids \[1\]"):
        support.resolve_final_inventory(
            inventory.finalize(),
            SessionContext(workflow_stage="post-procedure-clearing"),
            overrides={1: ItemContextOverride(lifecycle="reusable")},
        )
