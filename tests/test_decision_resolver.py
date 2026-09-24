import logging

import pytest

from surgint.decision import DecisionResolver, ItemContext, Observation


@pytest.fixture(scope="module")
def resolver():
    return DecisionResolver()


def observation(label="scalpel", confidence=0.91):
    return Observation(
        label=label,
        confidence=confidence,
        track_id=4,
        perception_version="perception-v1",
    )


def context(**overrides):
    values = {"workflow_stage": "post-procedure-clearing"}
    values.update(overrides)
    return ItemContext(**values)


def test_resolver_combines_context_and_ontology_into_a_recommendation(resolver):
    decision = resolver.resolve(
        observation("scalpel"),
        context(lifecycle="reusable"),
    )

    assert decision.outcome == "recommendation"
    assert decision.lifecycle == "reusable"
    assert decision.sharp_hazard == "sharp"
    assert decision.roles == ("cutting",)
    assert decision.action == "secure-transport-to-reprocessing"
    assert decision.matched_rule == "reusable-sharp"


def test_resolver_reports_information_required_by_policy(resolver):
    decision = resolver.resolve(observation("scalpel"), context())

    assert decision.outcome == "missing_info"
    assert decision.missing_fields == ("lifecycle",)
    assert decision.action is None


def test_syringe_is_resolved_as_a_single_use_sharp(resolver):
    decision = resolver.resolve(observation("syringe"), context())

    assert decision.outcome == "recommendation"
    assert decision.lifecycle == "single-use"
    assert decision.sharp_hazard == "sharp"
    assert decision.action == "approved-sharps-stream"


def test_resolver_reports_uncovered_known_facts(resolver):
    decision = resolver.resolve(observation("tray"), context())

    assert decision.outcome == "missing_policy"
    assert decision.lifecycle == "reusable"
    assert decision.sharp_hazard == "non-sharp"


def test_unsupported_item_outranks_low_confidence(resolver):
    decision = resolver.resolve(observation("unknown", confidence=0.01), context())

    assert decision.outcome == "unsupported_item"
    assert decision.concept_iri is None


def test_low_confidence_does_not_extract_or_act_on_facts(resolver):
    decision = resolver.resolve(observation("scalpel", confidence=0.1), context())

    assert decision.outcome == "low_confidence"
    assert decision.concept_iri.endswith("#scalpel")
    assert decision.lifecycle is None
    assert decision.sharp_hazard is None
    assert decision.action is None


def test_context_and_ontology_lifecycle_conflict_requires_review(resolver):
    decision = resolver.resolve(
        observation("gauze"),
        context(lifecycle="reusable"),
    )

    assert decision.outcome == "human_review"
    assert "conflicts" in decision.reason
    assert decision.action is None


def test_recommendation_carries_all_artifact_versions(resolver):
    decision = resolver.resolve(
        observation("scalpel"),
        context(lifecycle="single-use"),
    )

    assert decision.perception_version == "perception-v1"
    assert decision.ontology_version.endswith("/3.0.0")
    assert decision.policy_id == "surgint-demo-handling"
    assert decision.policy_version == "2.0.0"


def test_internal_failure_is_logged_and_fails_closed(monkeypatch, caplog):
    resolver = DecisionResolver()

    def broken(*args, **kwargs):
        raise RuntimeError("ontology lookup failed")

    monkeypatch.setattr("surgint.decision.resolver.ontology_facts", broken)
    with caplog.at_level(logging.ERROR):
        decision = resolver.resolve(observation("scalpel"), context())

    assert decision.outcome == "human_review"
    assert decision.reason == "decision support failed"
    assert "ontology lookup failed" in caplog.text


def test_resolver_rejects_invalid_threshold_configuration():
    with pytest.raises(ValueError, match="low_confidence"):
        DecisionResolver(low_confidence=float("nan"))
