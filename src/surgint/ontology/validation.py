from owlready2 import FunctionalProperty, Restriction, VALUE

PERCEPTION_LABELS = frozenset(
    {
        "scalpel",
        "forceps",
        "hemostat",
        "scissors",
        "retractor",
        "dissector",
        "syringe",
        "tray",
        "basin",
        "bowl",
        "gauze",
        "bandage",
        "tube",
    }
)

CONTROLLED_VALUES = {
    "hasDisposalStatus": frozenset({"reusable", "single-use"}),
    "hasHazardClass": frozenset({"sharp", "none"}),
    "hasFunctionalRole": frozenset(
        {
            "cutting",
            "grasping",
            "clamping",
            "tissue-retraction",
            "tissue-dissection",
            "fluid-delivery",
            "instrument-support",
            "fluid-containment",
            "absorption",
            "wound-dressing",
            "fluid-conveyance",
        }
    ),
}

FUNCTIONAL_PROPERTIES = frozenset({"hasDisposalStatus", "hasHazardClass"})


def validate(ontology) -> None:
    """Raise ``ValueError`` when the loaded ontology violates its schema."""

    concepts = _concepts_by_label(ontology)
    loaded_labels = set(concepts)
    if loaded_labels != PERCEPTION_LABELS:
        missing = sorted(PERCEPTION_LABELS - loaded_labels)
        unexpected = sorted(loaded_labels - PERCEPTION_LABELS)
        raise ValueError(
            f"perception labels do not match the catalog; "
            f"missing={missing}, unexpected={unexpected}"
        )

    properties = {}
    for name in CONTROLLED_VALUES:
        prop = ontology[name]
        if prop is None:
            raise ValueError(f"required object property {name} is missing")
        properties[name] = prop

    for name in FUNCTIONAL_PROPERTIES:
        if not issubclass(properties[name], FunctionalProperty):
            raise ValueError(f"{name} must be an owl:FunctionalProperty")

    for label, concept in concepts.items():
        for name, allowed in CONTROLLED_VALUES.items():
            values = _asserted_values(concept, properties[name])
            unexpected = sorted(value.name for value in values if value.name not in allowed)
            if unexpected:
                raise ValueError(f"{label} has invalid {name} values: {unexpected}")
            if name in FUNCTIONAL_PROPERTIES and len(values) > 1:
                raise ValueError(f"{label} asserts multiple values for {name}")


def _concepts_by_label(ontology) -> dict:
    concepts = {}
    for concept in ontology.classes():
        labels = [str(label) for label in concept.label]
        if len(labels) != 1:
            raise ValueError(
                f"{concept.name} must have exactly one perception label; "
                f"found {labels}"
            )
        label = labels[0]
        if label in concepts:
            raise ValueError(f"perception label {label!r} is not unique")
        concepts[label] = concept
    return concepts


def _asserted_values(concept, prop) -> list:
    return [
        restriction.value
        for restriction in concept.is_a
        if isinstance(restriction, Restriction)
        and restriction.type == VALUE
        and restriction.property == prop
    ]
