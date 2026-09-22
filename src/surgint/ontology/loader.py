from contextlib import ExitStack
from importlib.resources import as_file, files

from owlready2 import World

# the rdf/xml serialization. the owx copy is for owl tools, owlready2 reads this one.
ONTOLOGY = "SURGINT.owl"

VERSION_IRI = "http://www.semanticweb.org/ahmedghita/ontologies/2026/SURGINT/1.0.0"

VERSION_QUERY = "SELECT ?version { ?ontology <http://www.w3.org/2002/07/owl#versionIRI> ?version }"


def version_iri(world: World) -> str:
    """Return the version IRI asserted by the loaded ontology."""
    rows = list(world.sparql(VERSION_QUERY))
    if len(rows) != 1:
        raise ValueError(f"expected one versionIRI, found {len(rows)}")
    return str(rows[0][0])


def load(name: str = ONTOLOGY) -> tuple[World, object]:
    """Load the packaged ontology and return the world and the ontology."""
    # the file has to outlive this call, so the stack is never closed. it is a real
    # path in every install this project supports; as_file only copies out of a zip
    # import, which nothing here uses.
    path = ExitStack().enter_context(as_file(files("surgint.ontology") / name))
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not in the surgint.ontology package")

    world = World()
    # a plain path, not a file:// uri. owlready2 mishandles the windows form.
    ontology = world.get_ontology(str(path)).load()

    version = version_iri(world)
    if version != VERSION_IRI:
        raise ValueError(f"expected ontology {VERSION_IRI}, loaded {version}")

    return world, ontology
