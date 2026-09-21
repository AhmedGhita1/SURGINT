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
    "hasLifecycleDesignation": frozenset({"reusable", "single-use"}),
    "hasIntrinsicSharpHazard": frozenset({"sharp", "no-intrinsic-sharp"}),
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

FUNCTIONAL_PROPERTIES = frozenset(
    {"hasLifecycleDesignation", "hasIntrinsicSharpHazard"}
)

SCHEMA_CLASSES = frozenset(
    {"TrayItem", "LifecycleDesignation", "IntrinsicSharpHazard", "FunctionalRole"}
)

PROPERTY_TYPES = {
    "hasLifecycleDesignation": "LifecycleDesignation",
    "hasIntrinsicSharpHazard": "IntrinsicSharpHazard",
    "hasFunctionalRole": "FunctionalRole",
}


def validate(ontology) -> None:
    """Raise ``ValueError`` when the loaded ontology violates its schema."""

    loaded_classes = {concept.name for concept in ontology.classes()}
    expected_classes = PERCEPTION_LABELS | SCHEMA_CLASSES
    if loaded_classes != expected_classes:
        missing = sorted(expected_classes - loaded_classes)
        unexpected = sorted(loaded_classes - expected_classes)
        raise ValueError(
            f"ontology classes do not match the schema; "
            f"missing={missing}, unexpected={unexpected}"
        )

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

        if prop.domain != [ontology.TrayItem]:
            raise ValueError(f"{name} must have domain TrayItem")
        expected_range = ontology[PROPERTY_TYPES[name]]
        if prop.range != [expected_range]:
            raise ValueError(f"{name} must have range {expected_range.name}")

    for name in FUNCTIONAL_PROPERTIES:
        if not issubclass(properties[name], FunctionalProperty):
            raise ValueError(f"{name} must be an owl:FunctionalProperty")

    for label, concept in concepts.items():
        if ontology.TrayItem not in concept.is_a:
            raise ValueError(f"{label} must be a direct subclass of TrayItem")
        for name, allowed in CONTROLLED_VALUES.items():
            values = _asserted_values(concept, properties[name])
            unexpected = sorted(value.name for value in values if value.name not in allowed)
            if unexpected:
                raise ValueError(f"{label} has invalid {name} values: {unexpected}")
            if name in FUNCTIONAL_PROPERTIES and len(values) > 1:
                raise ValueError(f"{label} asserts multiple values for {name}")


def _concepts_by_label(ontology) -> dict:
    concepts = {}
    for name in PERCEPTION_LABELS:
        concept = ontology[name]
        labels = [str(label) for label in concept.perceptionLabel]
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
