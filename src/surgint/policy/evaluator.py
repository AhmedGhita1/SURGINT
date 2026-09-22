from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from surgint.policy.loader import Policy, PolicyRule

OUTCOMES = ("recommendation", "missing_info", "missing_policy")


@dataclass(frozen=True)
class Evaluation:
    outcome: str
    action: Optional[str]
    matched_rule: Optional[str]
    missing_fields: tuple[str, ...]
    reason: str


def evaluate(policy: Policy, facts: Mapping[str, Optional[str]]) -> Evaluation:
    """Evaluate known facts against a structurally unambiguous policy."""

    known = _known_facts(facts)
    _reject_overlapping_rules(policy.rules)

    matches = []
    candidates = []
    for rule in policy.rules:
        if _conflicts(rule, known):
            continue

        missing = tuple(sorted(set(rule.conditions) - set(known)))
        if missing:
            candidates.append((rule, missing))
        else:
            matches.append(rule)

    if len(matches) > 1:
        ids = sorted(rule.id for rule in matches)
        raise ValueError(f"multiple policy rules matched: {ids}")

    if matches:
        rule = matches[0]
        return Evaluation(
            outcome="recommendation",
            action=rule.action,
            matched_rule=rule.id,
            missing_fields=(),
            reason=rule.explanation,
        )

    if candidates:
        missing_fields = tuple(
            sorted({field for _, missing in candidates for field in missing})
        )
        return Evaluation(
            outcome="missing_info",
            action=None,
            matched_rule=None,
            missing_fields=missing_fields,
            reason=f"policy requires values for: {', '.join(missing_fields)}",
        )

    return Evaluation(
        outcome="missing_policy",
        action=None,
        matched_rule=None,
        missing_fields=(),
        reason="no policy rule covers the known facts",
    )


def _known_facts(facts: Mapping[str, Optional[str]]) -> dict[str, str]:
    if not isinstance(facts, Mapping):
        raise TypeError("facts must be a mapping")

    known = {}
    for key, value in facts.items():
        if not isinstance(key, str) or not key:
            raise ValueError("fact names must be non-empty strings")
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"fact {key!r} must be a non-empty string or None")
        known[key] = value
    return known


def _conflicts(rule: PolicyRule, known: Mapping[str, str]) -> bool:
    return any(
        key in known and known[key] != expected
        for key, expected in rule.conditions.items()
    )


def _reject_overlapping_rules(rules: tuple[PolicyRule, ...]) -> None:
    for index, left in enumerate(rules):
        for right in rules[index + 1 :]:
            shared = set(left.conditions) & set(right.conditions)
            incompatible = any(
                left.conditions[key] != right.conditions[key] for key in shared
            )
            if not incompatible:
                raise ValueError(
                    f"policy rules {left.id!r} and {right.id!r} can overlap"
                )
