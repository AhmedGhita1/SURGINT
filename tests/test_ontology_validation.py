import pytest

from surgint.ontology import load, validate


def test_packaged_ontology_passes_schema_validation():
    _, ontology = load()

    validate(ontology)


def test_validation_rejects_a_missing_perception_label():
    _, ontology = load()
    ontology.scalpel.label = []

    with pytest.raises(ValueError, match="exactly one perception label"):
        validate(ontology)


def test_validation_rejects_a_duplicate_perception_label():
    _, ontology = load()
    ontology.forceps.label = ["scalpel"]

    with pytest.raises(ValueError, match="is not unique"):
        validate(ontology)


def test_validation_rejects_an_invalid_controlled_value():
    _, ontology = load()
    ontology.forceps.is_a.append(
        ontology.hasHazardClass.value(ontology.cutting)
    )

    with pytest.raises(ValueError, match="invalid hasHazardClass"):
        validate(ontology)


def test_validation_rejects_multiple_functional_values():
    _, ontology = load()
    ontology.gauze.is_a.append(ontology.hasHazardClass.value(ontology.sharp))

    with pytest.raises(ValueError, match="multiple values for hasHazardClass"):
        validate(ontology)
