from dataclasses import dataclass
from math import isfinite
from typing import Optional

OUTCOMES = (
    "recommendation",
    "low_confidence",
    "missing_info",
    "missing_policy",
    "unsupported_item",
    "invalid_input",
    "human_review",
)

WORKFLOW_STAGES = (
    "pre-procedure-setup",
    "in-procedure",
    "post-procedure-clearing",
)
USE_STATES = ("unused", "used")
CONTAMINATION_STATES = (
    "not-regulated",
    "potentially-infectious",
    "chemical",
    "cytotoxic",
    "radioactive",
)
LIFECYCLES = ("reusable", "single-use")


@dataclass(frozen=True)
class Observation:
    label: str
    confidence: float
    track_id: int
    perception_version: str

    def __post_init__(self) -> None:
        _text(self.label, "label")
        _confidence(self.confidence)
        _track_id(self.track_id)
        _text(self.perception_version, "perception_version")


@dataclass(frozen=True)
class ItemContext:
    workflow_stage: str
    decision_intent: str = "next-handling-action"
    use_state: Optional[str] = None
    contamination_state: Optional[str] = None
    product_id: Optional[str] = None
    lifecycle: Optional[str] = None

    def __post_init__(self) -> None:
        _choice(self.workflow_stage, WORKFLOW_STAGES, "workflow_stage")
        _text(self.decision_intent, "decision_intent")
        _optional_choice(self.use_state, USE_STATES, "use_state")
        _optional_choice(
            self.contamination_state,
            CONTAMINATION_STATES,
            "contamination_state",
        )
        if self.product_id is not None:
            _text(self.product_id, "product_id")
        _optional_choice(self.lifecycle, LIFECYCLES, "lifecycle")


@dataclass(frozen=True)
class ItemContextOverride:
    """Facts that may differ between finalized inventory classes."""

    product_id: Optional[str] = None
    lifecycle: Optional[str] = None

    def __post_init__(self) -> None:
        if self.product_id is not None:
            _text(self.product_id, "product_id")
        _optional_choice(self.lifecycle, LIFECYCLES, "lifecycle")


@dataclass(frozen=True)
class SessionContext:
    """Decision facts shared by every finalized class in one session."""

    workflow_stage: str
    decision_intent: str = "next-handling-action"
    use_state: Optional[str] = None
    contamination_state: Optional[str] = None

    def __post_init__(self) -> None:
        _choice(self.workflow_stage, WORKFLOW_STAGES, "workflow_stage")
        _text(self.decision_intent, "decision_intent")
        _optional_choice(self.use_state, USE_STATES, "use_state")
        _optional_choice(
            self.contamination_state,
            CONTAMINATION_STATES,
            "contamination_state",
        )

    def for_item(
        self,
        override: Optional[ItemContextOverride] = None,
    ) -> ItemContext:
        """Compose shared session facts with one class-specific override."""

        if override is None:
            override = ItemContextOverride()
        elif not isinstance(override, ItemContextOverride):
            raise TypeError("override must be an ItemContextOverride")

        return ItemContext(
            workflow_stage=self.workflow_stage,
            decision_intent=self.decision_intent,
            use_state=self.use_state,
            contamination_state=self.contamination_state,
            product_id=override.product_id,
            lifecycle=override.lifecycle,
        )


@dataclass(frozen=True)
class Decision:
    label: str
    confidence: float
    track_id: int
    workflow_stage: str
    decision_intent: str
    outcome: str
    concept_iri: Optional[str]
    lifecycle: Optional[str]
    sharp_hazard: Optional[str]
    roles: tuple[str, ...]
    action: Optional[str]
    matched_rule: Optional[str]
    missing_fields: tuple[str, ...]
    reason: str
    perception_version: str
    ontology_version: str
    policy_id: str
    policy_version: str

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown decision outcome {self.outcome!r}")

        if self.outcome != "invalid_input":
            _text(self.label, "label")
            _confidence(self.confidence)
            _track_id(self.track_id)

        _choice(self.workflow_stage, WORKFLOW_STAGES, "workflow_stage")
        _text(self.decision_intent, "decision_intent")
        _optional_text(self.concept_iri, "concept_iri")
        _optional_choice(self.lifecycle, LIFECYCLES, "lifecycle")
        _optional_text(self.sharp_hazard, "sharp_hazard")
        _text_tuple(self.roles, "roles")
        _optional_text(self.action, "action")
        _optional_text(self.matched_rule, "matched_rule")
        _text_tuple(self.missing_fields, "missing_fields")
        _text(self.reason, "reason")
        _text(self.perception_version, "perception_version")
        _text(self.ontology_version, "ontology_version")
        _text(self.policy_id, "policy_id")
        _text(self.policy_version, "policy_version")

        if self.outcome == "recommendation":
            if self.action is None or self.matched_rule is None:
                raise ValueError(
                    "a recommendation requires an action and matched_rule"
                )
            if self.missing_fields:
                raise ValueError("a recommendation cannot have missing_fields")
        elif self.action is not None:
            raise ValueError("only a recommendation can contain an action")

        if self.outcome == "missing_info" and not self.missing_fields:
            raise ValueError("missing_info requires missing_fields")


def _confidence(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be a finite number in [0, 1]")


def _track_id(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("track_id must be a non-negative integer")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_text(value: Optional[str], name: str) -> None:
    if value is not None:
        _text(value, name)


def _choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")


def _optional_choice(
    value: Optional[str], choices: tuple[str, ...], name: str
) -> None:
    if value is not None:
        _choice(value, choices, name)


def _text_tuple(value: tuple[str, ...], name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    for item in value:
        _text(item, f"{name} item")
