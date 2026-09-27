import pytest

from surgint.ontology import concepts, facts, load, unmapped

PERCEPTION_LABELS = {
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


def test_concepts_map_every_perception_label():
    _, ontology = load()

    mapping = concepts(ontology)

    assert set(mapping) == PERCEPTION_LABELS
    assert mapping["scalpel"].name == "scalpel"


def test_mapping_does_not_depend_on_the_display_label():
    _, ontology = load()
    ontology.scalpel.label = ["Scalpel display name"]

    mapping = concepts(ontology)

    assert mapping["scalpel"] is ontology.scalpel


def test_mapping_rejects_duplicate_perception_labels():
    _, ontology = load()
    ontology.forceps.perceptionLabel = ["scalpel"]

    with pytest.raises(ValueError, match="is not unique"):
        concepts(ontology)


def test_facts_extract_asserted_category_knowledge():
    _, ontology = load()

    known = facts(ontology, concepts(ontology)["gauze"])

    assert known.concept_iri.endswith("#gauze")
    assert known.lifecycle == "single-use"
    assert known.sharp_hazard == "non-sharp"
    assert known.roles == ("absorption",)


@pytest.mark.parametrize(
    ("label", "lifecycle", "sharp_hazard", "role"),
    [
        ("hemostat", "reusable", "non-sharp", "clamping"),
        ("scissors", "reusable", "sharp", "cutting"),
        ("retractor", "reusable", "sharp", "tissue-retraction"),
        ("dissector", "reusable", "sharp", "tissue-dissection"),
        ("syringe", "single-use", "sharp", "fluid-delivery"),
        ("tray", "reusable", "non-sharp", "instrument-support"),
        ("basin", "reusable", "non-sharp", "fluid-containment"),
        ("bowl", "reusable", "non-sharp", "fluid-containment"),
        ("gauze", "single-use", "non-sharp", "absorption"),
        ("tube", "single-use", "non-sharp", "fluid-conveyance"),
    ],
)
def test_facts_match_the_current_deployment_profile(
    label,
    lifecycle,
    sharp_hazard,
    role,
):
    _, ontology = load()

    known = facts(ontology, concepts(ontology)[label])

    assert known.lifecycle == lifecycle
    assert known.sharp_hazard == sharp_hazard
    assert known.roles == (role,)


def test_facts_preserve_unasserted_values_as_none():
    _, ontology = load()
    mapping = concepts(ontology)

    assert facts(ontology, mapping["scalpel"]).lifecycle is None
    assert facts(ontology, mapping["forceps"]).sharp_hazard is None


def test_facts_reject_multiple_functional_values():
    _, ontology = load()
    ontology.gauze.is_a.append(
        ontology.hasSharpHazard.value(ontology.sharp)
    )

    with pytest.raises(ValueError, match="multiple values for hasSharpHazard"):
        facts(ontology, ontology.gauze)


def test_unmapped_preserves_input_order():
    _, ontology = load()
    mapping = concepts(ontology)

    assert unmapped(["unknown-a", "gauze", "unknown-b"], mapping) == [
        "unknown-a",
        "unknown-b",
    ]
