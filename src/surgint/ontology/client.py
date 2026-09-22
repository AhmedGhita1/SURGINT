from dataclasses import dataclass
from typing import Optional

from surgint.ontology.loader import load, version_iri
from surgint.ontology.mapping import concepts, facts, unmapped

# every item resolves to exactly one of these
OUTCOMES = (
    "recommendation",
    "low_confidence",
    "missing_info",
    "missing_policy",
    "unsupported_item",
    "human_review",
)

# the approved handling route per (disposal, hazard). both have to be known before a
# route can be chosen: a single-use item goes to a sharps bin or to general waste
# depending on the hazard, and the two are not interchangeable.
ROUTES = {
    ("reusable", "none"): "reprocessing",
    ("reusable", "sharp"): "reprocessing, sharps transport",
    ("single-use", "none"): "general single-use waste",
    ("single-use", "sharp"): "sharps waste",
}

# below this the detector was not sure enough to act on
LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Decision:
    label: str
    outcome: str
    concept: Optional[str]
    disposal: Optional[str]
    hazard: Optional[str]
    route: Optional[str]
    reason: str
    ontology_version: str


class Ontology:
    """resolves an inventory item to a handling route."""

    def __init__(self, low_confidence: float = LOW_CONFIDENCE):
        self.world, self.ontology = load()
        self.version = version_iri(self.world)
        self.concepts = concepts(self.ontology)
        self.low_confidence = low_confidence

    def unmapped(self, labels) -> list:
        """Return the labels that do not map to an ontology class."""
        return unmapped(labels, self.concepts)

    def resolve(self, label: str, confidence: float) -> Decision:
        """Resolve one label and confidence to a single outcome."""
        try:
            return self._resolve(label, confidence)
        except Exception as error:
            # nothing resolved cleanly, so a person decides
            return self._decision(label, "human_review", None, str(error))

    def _resolve(self, label: str, confidence: float) -> Decision:
        concept = self.concepts.get(label)
        if concept is None:
            return self._decision(
                label, "unsupported_item", None, f"{label} is not in the catalog"
            )

        if confidence < self.low_confidence:
            return self._decision(
                label,
                "low_confidence",
                concept,
                f"score {confidence:.2f} is below {self.low_confidence:.2f}",
            )

        known = facts(self.ontology, concept)
        disposal, hazard = known["disposal"], known["hazard"]

        # unasserted means unknown at this label's resolution, never none or false
        if disposal is None or hazard is None:
            missing = "disposal status" if disposal is None else "hazard class"
            return self._decision(
                label,
                "missing_info",
                concept,
                f"{label} does not resolve a {missing}",
                disposal,
                hazard,
            )

        route = ROUTES.get((disposal, hazard))
        if route is None:
            return self._decision(
                label,
                "missing_policy",
                concept,
                f"no approved route for {disposal} and {hazard}",
                disposal,
                hazard,
            )

        return self._decision(
            label,
            "recommendation",
            concept,
            f"{label} is {disposal}, hazard {hazard}",
            disposal,
            hazard,
            route,
        )

    def _decision(
        self,
        label: str,
        outcome: str,
        concept,
        reason: str,
        disposal: Optional[str] = None,
        hazard: Optional[str] = None,
        route: Optional[str] = None,
    ) -> Decision:
        return Decision(
            label=label,
            outcome=outcome,
            concept=concept.name if concept is not None else None,
            disposal=disposal,
            hazard=hazard,
            route=route,
            reason=reason,
            ontology_version=self.version,
        )
