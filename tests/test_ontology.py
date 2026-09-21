"""
Ontology tests
==============

tests the decision-support boundary: a label and a score in, one outcome out.

the ontology file ships with the package, so these run against the real artifact.
no detector is involved.

coverage:
- loading:  the package data loads and its version iri is the pinned one
- mapping:  every class joins by rdfs:label, the facts match what is asserted
- outcomes: each of the six is reachable, and the resolution order holds
"""

import pytest

from surgint.ontology import OUTCOMES, Ontology
from surgint.ontology.loader import VERSION_IRI, load
from surgint.ontology.mapping import concepts, facts

CLASSES = 13
TAXONOMY = [
    "scalpel", "forceps", "hemostat", "scissors", "retractor", "dissector",
    "syringe", "tray", "basin", "bowl", "gauze", "bandage", "tube",
]


@pytest.fixture(scope="module")
def ontology():
    return Ontology()


def test_load_pins_the_version():
    world, onto = load()
    assert len(list(onto.classes())) == CLASSES
    assert len(concepts(onto)) == CLASSES

    from surgint.ontology.loader import version_iri

    assert version_iri(world) == VERSION_IRI


def test_every_label_joins(ontology):
    """the taxonomy and the ontology are matched by name and must stay in step"""
    assert ontology.unmapped(TAXONOMY) == []
    assert sorted(ontology.concepts) == sorted(TAXONOMY)


def test_facts_read_the_restrictions():
    """the facts are ObjectHasValue restrictions on the class, not instance assertions"""
    _, onto = load()
    mapping = concepts(onto)

    assert facts(onto, mapping["gauze"]) == {
        "disposal": "single-use",
        "hazard": "none",
        "roles": ["absorption"],
    }
    # unasserted is unknown, never none or false
    assert facts(onto, mapping["scalpel"])["disposal"] is None
    assert facts(onto, mapping["scalpel"])["hazard"] == "sharp"


def test_recommendation(ontology):
    """both properties known and the pair has an approved route"""
    decision = ontology.resolve("gauze", 0.9)

    assert decision.outcome == "recommendation"
    assert decision.disposal == "single-use"
    assert decision.hazard == "none"
    assert decision.route == "general single-use waste"
    assert decision.ontology_version == VERSION_IRI


def test_missing_info(ontology):
    """the concept is known, a property it needs is unasserted"""
    # the label does not separate a reusable handle from a single-use blade
    assert ontology.resolve("scalpel", 0.9).outcome == "missing_info"
    # single-use is asserted, but sharps bin against general waste needs the hazard
    syringe = ontology.resolve("syringe", 0.9)
    assert syringe.outcome == "missing_info"
    assert syringe.disposal == "single-use"
    assert syringe.route is None


def test_low_confidence(ontology):
    decision = ontology.resolve("gauze", 0.1)

    assert decision.outcome == "low_confidence"
    # the concept resolved, so it is reported; the facts are not acted on
    assert decision.concept is not None
    assert decision.route is None


def test_unsupported_item(ontology):
    """a label the ontology has no concept for. the version skew outcome"""
    decision = ontology.resolve("clamp", 0.9)

    assert decision.outcome == "unsupported_item"
    assert decision.concept is None
    assert ontology.unmapped(["gauze", "clamp"]) == ["clamp"]


def test_unsupported_outranks_low_confidence(ontology):
    """a gap in the ontology is not a confidence problem, so it is checked first"""
    assert ontology.resolve("clamp", 0.01).outcome == "unsupported_item"


def test_missing_policy(ontology, monkeypatch):
    """both properties known, no approved route for the pair"""
    monkeypatch.setattr("surgint.ontology.client.ROUTES", {})
    decision = ontology.resolve("gauze", 0.9)

    assert decision.outcome == "missing_policy"
    assert decision.disposal == "single-use"
    assert decision.route is None


def test_human_review(ontology, monkeypatch):
    """nothing resolved cleanly, so a person decides"""
    def broken(*args, **kwargs):
        raise RuntimeError("reasoner unavailable")

    monkeypatch.setattr("surgint.ontology.client.facts", broken)
    decision = ontology.resolve("gauze", 0.9)

    assert decision.outcome == "human_review"
    assert "reasoner unavailable" in decision.reason


def test_taxonomy_resolves_to_a_known_outcome(ontology):
    """no label may crash or fall through"""
    for label in TAXONOMY:
        assert ontology.resolve(label, 0.9).outcome in OUTCOMES
