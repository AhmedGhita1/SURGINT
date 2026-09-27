import logging
from math import isfinite
from typing import Optional

from surgint.decision.models import Decision, ItemContext, Observation
from surgint.ontology import (
    ConceptFacts,
    VERSION_IRI,
    concepts,
    facts as ontology_facts,
    load as load_ontology,
    validate as validate_ontology,
)
from surgint.policy import Evaluation, Policy, evaluate, load as load_policy

LOW_CONFIDENCE = 0.5

logger = logging.getLogger(__name__)


class DecisionResolver:
    """Combine perception, ontology knowledge, context, and handling policy."""

    def __init__(
        self,
        low_confidence: float = LOW_CONFIDENCE,
        policy: Optional[Policy] = None,
    ):
        if (
            isinstance(low_confidence, bool)
            or not isinstance(low_confidence, (int, float))
            or not isfinite(low_confidence)
            or not 0.0 <= low_confidence <= 1.0
        ):
            raise ValueError("low_confidence must be a finite number in [0, 1]")

        self.world, self.ontology = load_ontology()
        validate_ontology(self.ontology)
        self.concepts = concepts(self.ontology)
        self.policy = policy if policy is not None else load_policy()
        self.low_confidence = float(low_confidence)

    def resolve(self, observation: Observation, context: ItemContext) -> Decision:
        """Resolve one typed observation to one explained outcome."""

        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if not isinstance(context, ItemContext):
            raise TypeError("context must be an ItemContext")

        concept = self.concepts.get(observation.label)
        if concept is None:
            return self._decision(
                observation,
                context,
                outcome="unsupported_item",
                reason=f"{observation.label} is not in the ontology catalog",
            )

        if observation.confidence < self.low_confidence:
            return self._decision(
                observation,
                context,
                outcome="low_confidence",
                reason=(
                    f"score {observation.confidence:.2f} is below "
                    f"{self.low_confidence:.2f}"
                ),
                concept_iri=concept.iri,
            )

        try:
            known = ontology_facts(self.ontology, concept)
            lifecycle, conflict = _lifecycle(known.lifecycle, context.lifecycle)
            if conflict is not None:
                return self._decision(
                    observation,
                    context,
                    outcome="human_review",
                    reason=conflict,
                    known=known,
                )

            policy_facts = _policy_facts(context, known, lifecycle)
            evaluation = evaluate(self.policy, policy_facts)
            return self._from_evaluation(
                observation,
                context,
                known,
                lifecycle,
                evaluation,
            )
        except Exception:
            logger.exception(
                "decision support failed for label=%r track_id=%s",
                observation.label,
                observation.track_id,
            )
            return self._decision(
                observation,
                context,
                outcome="human_review",
                reason="decision support failed",
                concept_iri=concept.iri,
            )

    def _from_evaluation(
        self,
        observation: Observation,
        context: ItemContext,
        known: ConceptFacts,
        lifecycle: Optional[str],
        evaluation: Evaluation,
    ) -> Decision:
        return self._decision(
            observation,
            context,
            outcome=evaluation.outcome,
            reason=evaluation.reason,
            known=known,
            lifecycle=lifecycle,
            action=evaluation.action,
            matched_rule=evaluation.matched_rule,
            missing_fields=evaluation.missing_fields,
        )

    def _decision(
        self,
        observation: Observation,
        context: ItemContext,
        *,
        outcome: str,
        reason: str,
        known: Optional[ConceptFacts] = None,
        concept_iri: Optional[str] = None,
        lifecycle: Optional[str] = None,
        action: Optional[str] = None,
        matched_rule: Optional[str] = None,
        missing_fields: tuple[str, ...] = (),
    ) -> Decision:
        return Decision(
            label=observation.label,
            confidence=observation.confidence,
            track_id=observation.track_id,
            workflow_stage=context.workflow_stage,
            decision_intent=context.decision_intent,
            outcome=outcome,
            concept_iri=known.concept_iri if known is not None else concept_iri,
            lifecycle=(
                lifecycle
                if lifecycle is not None
                else known.lifecycle if known is not None else None
            ),
            sharp_hazard=(
                known.sharp_hazard if known is not None else None
            ),
            roles=known.roles if known is not None else (),
            action=action,
            matched_rule=matched_rule,
            missing_fields=missing_fields,
            reason=reason,
            perception_version=observation.perception_version,
            ontology_version=VERSION_IRI,
            policy_id=self.policy.id,
            policy_version=self.policy.version,
        )


def _lifecycle(
    ontology_value: Optional[str], context_value: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    if (
        ontology_value is not None
        and context_value is not None
        and ontology_value != context_value
    ):
        return None, (
            f"context lifecycle {context_value!r} conflicts with "
            f"ontology lifecycle {ontology_value!r}"
        )
    return context_value if context_value is not None else ontology_value, None


def _policy_facts(
    context: ItemContext,
    known: ConceptFacts,
    lifecycle: Optional[str],
) -> dict[str, Optional[str]]:
    return {
        "workflow_stage": context.workflow_stage,
        "decision_intent": context.decision_intent,
        "lifecycle": lifecycle,
        "sharp_hazard": known.sharp_hazard,
        "use_state": context.use_state,
        "contamination_state": context.contamination_state,
        "product_id": context.product_id,
    }
