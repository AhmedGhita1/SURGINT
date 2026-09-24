from dataclasses import FrozenInstanceError

import pytest

from surgint.decision import (
    Decision,
    ItemContext,
    ItemContextOverride,
    Observation,
    SessionContext,
)


def test_observation_is_validated_and_immutable():
    observation = Observation(
        label="scalpel",
        confidence=0.91,
        track_id=4,
        perception_version="detector-v1+tracker-v1",
    )

    assert observation.label == "scalpel"
    with pytest.raises(FrozenInstanceError):
        observation.label = "forceps"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_observation_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        Observation("scalpel", confidence, 4, "perception-v1")


def test_item_context_preserves_unknown_values_as_none():
    context = ItemContext(workflow_stage="post-procedure-clearing")

    assert context.use_state is None
    assert context.contamination_state is None
    assert context.lifecycle is None


def test_item_context_rejects_unknown_controlled_values():
    with pytest.raises(ValueError, match="use_state"):
        ItemContext(
            workflow_stage="post-procedure-clearing",
            use_state="maybe",
        )


def test_item_context_has_no_needle_specific_input():
    with pytest.raises(TypeError, match="needle_attached"):
        ItemContext(
            workflow_stage="post-procedure-clearing",
            needle_attached=True,
        )


def test_session_context_composes_class_specific_facts():
    session = SessionContext(
        workflow_stage="post-procedure-clearing",
        use_state="unused",
        contamination_state="not-regulated",
    )

    context = session.for_item(
        ItemContextOverride(
            lifecycle="reusable",
            product_id="catalog-7",
        )
    )

    assert context.workflow_stage == "post-procedure-clearing"
    assert context.use_state == "unused"
    assert context.contamination_state == "not-regulated"
    assert context.lifecycle == "reusable"
    assert context.product_id == "catalog-7"


def test_session_context_rejects_invalid_override_type():
    session = SessionContext(workflow_stage="post-procedure-clearing")

    with pytest.raises(TypeError, match="ItemContextOverride"):
        session.for_item(ItemContext(workflow_stage="post-procedure-clearing"))


def test_recommendation_requires_an_action_and_matched_rule():
    with pytest.raises(ValueError, match="requires an action and matched_rule"):
        _decision(outcome="recommendation", action=None, matched_rule=None)


def test_non_recommendation_cannot_contain_an_action():
    with pytest.raises(ValueError, match="only a recommendation"):
        _decision(outcome="missing_policy", action="some-action")


def test_missing_info_requires_named_missing_fields():
    with pytest.raises(ValueError, match="requires missing_fields"):
        _decision(outcome="missing_info", missing_fields=())


def test_valid_recommendation_carries_provenance():
    decision = _decision(
        outcome="recommendation",
        action="approved-sharps-stream",
        matched_rule="disposable-sharp",
    )

    assert decision.ontology_version.endswith("/3.0.0")
    assert decision.policy_version == "2.0.0"
    assert decision.perception_version == "perception-v1"


def _decision(**overrides):
    values = {
        "label": "scalpel",
        "confidence": 0.91,
        "track_id": 4,
        "workflow_stage": "post-procedure-clearing",
        "decision_intent": "next-handling-action",
        "outcome": "missing_policy",
        "concept_iri": "http://example.test/SURGINT#scalpel",
        "lifecycle": None,
        "sharp_hazard": "sharp",
        "roles": ("cutting",),
        "action": None,
        "matched_rule": None,
        "missing_fields": (),
        "reason": "test decision",
        "perception_version": "perception-v1",
        "ontology_version": "http://example.test/SURGINT/3.0.0",
        "policy_id": "test-policy",
        "policy_version": "2.0.0",
    }
    values.update(overrides)
    return Decision(**values)
