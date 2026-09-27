import pytest

from surgint.policy import load, parse


def test_packaged_policy_loads_with_pinned_identity():
    policy = load()

    assert policy.schema_version == 1
    assert policy.id == "surgint-demo-handling"
    assert policy.version == "2.1.0"
    assert policy.status == "demonstration-only"
    assert [rule.id for rule in policy.rules] == [
        "reusable-sharp",
        "reusable-nonsharp",
        "disposable-sharp",
        "unused-disposable-nonsharp",
    ]


def test_policy_conditions_are_immutable():
    policy = load()

    with pytest.raises(TypeError):
        policy.rules[0].conditions["lifecycle"] = "single-use"


def test_policy_rejects_an_unknown_schema_version():
    document = _document()
    document["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported policy schema version"):
        parse(document)


def test_policy_rejects_duplicate_rule_ids():
    document = _document()
    document["rules"].append(dict(document["rules"][0]))

    with pytest.raises(ValueError, match="rule ids must be unique"):
        parse(document)


def test_policy_rejects_unexpected_fields():
    document = _document()
    document["rules"][0]["typo"] = "value"

    with pytest.raises(ValueError, match=r"unexpected=\['typo'\]"):
        parse(document)


def _document():
    return {
        "schema_version": 1,
        "policy_id": "test-policy",
        "version": "1.0.0",
        "status": "test-only",
        "rules": [
            {
                "id": "test-rule",
                "when": {"lifecycle": "reusable"},
                "action": "test-action",
                "explanation": "Used by a loader test.",
            }
        ],
    }
