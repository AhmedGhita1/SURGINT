from importlib.resources import as_file, files

from owlready2 import World

ONTOLOGY_FILE = "SURGINT.owl"
ONTOLOGY_IRI = "http://www.semanticweb.org/ahmedghita/ontologies/2026/SURGINT"
VERSION_IRI = f"{ONTOLOGY_IRI}/3.0.0"
VERSION_QUERY = """
SELECT ?version {
    ?ontology <http://www.w3.org/2002/07/owl#versionIRI> ?version
}
"""


def version_iri(world: World) -> str:
    """Return the single version IRI asserted in the loaded world."""

    versions = [str(row[0]) for row in world.sparql(VERSION_QUERY)]
    if len(versions) != 1:
        raise ValueError(f"expected one ontology version, loaded {versions}")
    return versions[0]


def load() -> tuple[World, object]:
    """Load the packaged ontology and verify its identity and version."""

    resource = files("surgint.ontology") / ONTOLOGY_FILE
    with as_file(resource) as path:
        if not path.is_file():
            raise FileNotFoundError(
                f"{ONTOLOGY_FILE} is not in the surgint.ontology package"
            )

        world = World()
        ontology = world.get_ontology(str(path)).load()

    if ontology.base_iri.rstrip("#") != ONTOLOGY_IRI:
        raise ValueError(
            f"expected ontology {ONTOLOGY_IRI}, loaded {ontology.base_iri}"
        )

    loaded_version = version_iri(world)
    if loaded_version != VERSION_IRI:
        raise ValueError(
            f"expected ontology version {VERSION_IRI}, loaded {loaded_version}"
        )

    return world, ontology
