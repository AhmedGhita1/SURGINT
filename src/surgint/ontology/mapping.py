from owlready2 import VALUE, Restriction

# the properties a detector class can carry. unasserted means unknown at this
# label's resolution, not absent. see the ontology README.
PROPERTIES = ("hasDisposalStatus", "hasHazardClass", "hasFunctionalRole")


def concepts(ontology) -> dict:
    """Detector labels mapped to ontology classes."""

    mapping = {}
    for concept in ontology.classes():
        label = concept.label.first()
        if label is None:
            raise ValueError(f"{concept.name} carries no rdfs:label")
        if str(label) in mapping:
            raise ValueError(f"two classes share the label {label!r}")
        mapping[str(label)] = concept
    return mapping


def asserted(concept, prop) -> list:
    """Return the values the class asserts for this property."""

    return [
        restriction.value
        for restriction in concept.is_a
        if isinstance(restriction, Restriction)
        and restriction.type == VALUE
        and restriction.property == prop
    ]


def facts(ontology, concept) -> dict:
    """Return the facts asserted by the class."""

    values = {
        name: [individual.name for individual in asserted(concept, ontology[name])]
        for name in PROPERTIES
    }
    return {
        "disposal": _single(values["hasDisposalStatus"], concept, "hasDisposalStatus"),
        "hazard": _single(values["hasHazardClass"], concept, "hasHazardClass"),
        "roles": values["hasFunctionalRole"],
    }


def _single(values: list, concept, name: str):
    """Return the single value of a functional property, or None if unasserted."""

    if len(values) > 1:
        raise ValueError(f"{concept.name} asserts {len(values)} values for {name}")
    return values[0] if values else None


def unmapped(labels, mapping: dict) -> list:
    """Return the labels that do not map to an ontology class."""
    return [label for label in labels if label not in mapping]
