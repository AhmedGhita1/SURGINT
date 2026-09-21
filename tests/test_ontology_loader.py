from surgint.ontology import ONTOLOGY_IRI, VERSION_IRI, load, version_iri


def test_packaged_ontology_loads_with_pinned_identity():
    world, ontology = load()

    assert world is not None
    assert ontology.base_iri.rstrip("#") == ONTOLOGY_IRI
    assert version_iri(world) == VERSION_IRI


def test_packaged_ontology_contains_the_perception_categories():
    _, ontology = load()

    concepts = {
        str(concept.perceptionLabel.first()): concept.name
        for concept in ontology.classes()
        if concept.perceptionLabel
    }
    assert concepts == {
        "scalpel": "scalpel",
        "forceps": "forceps",
        "hemostat": "hemostat",
        "scissors": "scissors",
        "retractor": "retractor",
        "dissector": "dissector",
        "syringe": "syringe",
        "tray": "tray",
        "basin": "basin",
        "bowl": "bowl",
        "gauze": "gauze",
        "bandage": "bandage",
        "tube": "tube",
    }


def test_packaged_ontology_contains_the_schema_classes():
    _, ontology = load()

    assert {concept.name for concept in ontology.classes()} == {
        "TrayItem",
        "LifecycleDesignation",
        "IntrinsicSharpHazard",
        "FunctionalRole",
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
