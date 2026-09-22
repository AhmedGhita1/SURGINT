from dataclasses import dataclass
from typing import Optional

from owlready2 import Restriction, VALUE


@dataclass(frozen=True)
class ConceptFacts:
    concept_iri: str
    lifecycle: Optional[str]
    intrinsic_sharp_hazard: Optional[str]
    roles: tuple[str, ...]


def concepts(ontology) -> dict:
    """Map exact perception labels to their ontology classes."""

    mapping = {}
    for concept in ontology.TrayItem.subclasses():
        labels = [str(label) for label in concept.perceptionLabel]
        if len(labels) != 1:
            raise ValueError(
                f"{concept.name} must have exactly one perception label; "
                f"found {labels}"
            )

        label = labels[0]
        if label in mapping:
            raise ValueError(f"perception label {label!r} is not unique")
        mapping[label] = concept
    return mapping


def facts(ontology, concept) -> ConceptFacts:
    """Extract directly asserted category facts from an ontology class."""

    lifecycle = _single(
        _asserted_values(concept, ontology.hasLifecycleDesignation),
        concept,
        "hasLifecycleDesignation",
    )
    intrinsic_sharp_hazard = _single(
        _asserted_values(concept, ontology.hasIntrinsicSharpHazard),
        concept,
        "hasIntrinsicSharpHazard",
    )
    roles = tuple(
        sorted(
            value.name
            for value in _asserted_values(concept, ontology.hasFunctionalRole)
        )
    )

    return ConceptFacts(
        concept_iri=concept.iri,
        lifecycle=lifecycle,
        intrinsic_sharp_hazard=intrinsic_sharp_hazard,
        roles=roles,
    )


def unmapped(labels, mapping: dict) -> list:
    """Return perception labels with no ontology concept."""

    return [label for label in labels if label not in mapping]


def _asserted_values(concept, prop) -> list:
    return [
        restriction.value
        for restriction in concept.is_a
        if isinstance(restriction, Restriction)
        and restriction.type == VALUE
        and restriction.property == prop
    ]


def _single(values: list, concept, property_name: str) -> Optional[str]:
    if len(values) > 1:
        raise ValueError(
            f"{concept.name} asserts multiple values for {property_name}"
        )
    return values[0].name if values else None
