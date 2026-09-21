from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import as_file, files
from types import MappingProxyType

import yaml

POLICY_FILE = "handling.yaml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicyRule:
    id: str
    conditions: Mapping[str, str]
    action: str
    explanation: str


@dataclass(frozen=True)
class Policy:
    schema_version: int
    id: str
    version: str
    status: str
    rules: tuple[PolicyRule, ...]


def load() -> Policy:
    """Load and validate the packaged handling policy."""

    resource = files("surgint.policy") / POLICY_FILE
    with as_file(resource) as path:
        if not path.is_file():
            raise FileNotFoundError(
                f"{POLICY_FILE} is not in the surgint.policy package"
            )
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)

    return parse(document)


def parse(document) -> Policy:
    """Validate a decoded policy document and return its typed representation."""

    root = _mapping(document, "policy")
    _keys(
        root,
        required={"schema_version", "policy_id", "version", "status", "rules"},
        optional=set(),
        location="policy",
    )

    schema_version = root["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported policy schema version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    policy_id = _text(root["policy_id"], "policy.policy_id")
    version = _text(root["version"], "policy.version")
    status = _text(root["status"], "policy.status")

    raw_rules = root["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("policy.rules must be a non-empty list")

    rules = tuple(_rule(raw_rule, index) for index, raw_rule in enumerate(raw_rules))
    ids = [rule.id for rule in rules]
    duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
    if duplicates:
        raise ValueError(f"policy rule ids must be unique; duplicates={duplicates}")

    return Policy(
        schema_version=schema_version,
        id=policy_id,
        version=version,
        status=status,
        rules=rules,
    )


def _rule(document, index: int) -> PolicyRule:
    location = f"policy.rules[{index}]"
    rule = _mapping(document, location)
    _keys(
        rule,
        required={"id", "when", "action", "explanation"},
        optional=set(),
        location=location,
    )

    conditions = _mapping(rule["when"], f"{location}.when")
    if not conditions:
        raise ValueError(f"{location}.when must not be empty")

    normalized_conditions = {
        _text(key, f"{location}.when key"): _text(
            value, f"{location}.when.{key}"
        )
        for key, value in conditions.items()
    }
    return PolicyRule(
        id=_text(rule["id"], f"{location}.id"),
        conditions=MappingProxyType(normalized_conditions),
        action=_text(rule["action"], f"{location}.action"),
        explanation=_text(rule["explanation"], f"{location}.explanation"),
    )


def _mapping(value, location: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _text(value, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _keys(
    mapping: Mapping,
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required - keys)
    unexpected = sorted(keys - required - optional)
    if missing or unexpected:
        raise ValueError(
            f"{location} has invalid keys; "
            f"missing={missing}, unexpected={unexpected}"
        )
