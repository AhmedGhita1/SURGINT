import pytest

from surgint.policy import evaluate, load, parse


def test_policy_returns_a_recommendation_for_one_complete_match():
    policy = load()

    result = evaluate(
        policy,
        {
            "workflow_stage": "post-procedure-clearing",
            "lifecycle": "reusable",
            "intrinsic_sharp_hazard": "sharp",
        },
    )

    assert result.outcome == "recommendation"
    assert result.action == "secure-transport-to-reprocessing"
    assert result.matched_rule == "reusable-sharp"
    assert result.missing_fields == ()


def test_policy_reports_information_required_by_a_possible_rule():
    policy = load()

    result = evaluate(
        policy,
        {
            "workflow_stage": "post-procedure-clearing",
            "lifecycle": "reusable",
        },
    )

    assert result.outcome == "missing_info"
    assert result.action is None
    assert result.matched_rule is None
    assert result.missing_fields == ("intrinsic_sharp_hazard",)


def test_policy_reports_no_rule_for_known_uncovered_facts():
    policy = load()

    result = evaluate(
        policy,
        {
            "workflow_stage": "pre-procedure-setup",
            "lifecycle": "reusable",
            "intrinsic_sharp_hazard": "sharp",
        },
    )

    assert result.outcome == "missing_policy"
    assert result.action is None
    assert result.missing_fields == ()


def test_policy_treats_none_as_unknown():
    policy = load()

    result = evaluate(
        policy,
        {
            "workflow_stage": "post-procedure-clearing",
            "lifecycle": "reusable",
            "intrinsic_sharp_hazard": None,
        },
    )

    assert result.outcome == "missing_info"
    assert result.missing_fields == ("intrinsic_sharp_hazard",)


def test_policy_rejects_overlapping_rules():
    policy = parse(
        {
            "schema_version": 1,
            "policy_id": "overlapping-policy",
            "version": "1.0.0",
            "status": "test-only",
            "rules": [
                {
                    "id": "reusable",
                    "when": {"lifecycle": "reusable"},
                    "action": "first",
                    "explanation": "First rule.",
                },
                {
                    "id": "sharp",
                    "when": {"intrinsic_sharp_hazard": "sharp"},
                    "action": "second",
                    "explanation": "Second rule.",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="can overlap"):
        evaluate(policy, {})


def test_policy_rejects_invalid_fact_values():
    policy = load()

    with pytest.raises(ValueError, match="non-empty string or None"):
        evaluate(policy, {"lifecycle": False})
