"""
Packaging tests
===============

tests the distribution metadata, which is the contract for `pip install surgint`.

nothing in the repository exercises that contract: ci and the notebooks satisfy
dependencies from requirements.txt before the package is built, so an undeclared
import or an unpackaged resource stays invisible until someone installs the
package on its own.

coverage:
- imports:  every third-party module the source imports is a declared dependency
- resource: the ontology the loader reads is matched by a package-data pattern
"""

import ast
import re
import sys
import tomllib
from fnmatch import fnmatch
from importlib.metadata import packages_distributions
from pathlib import Path

from surgint.ontology.loader import ONTOLOGY

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "surgint"
PYPROJECT = ROOT / "pyproject.toml"


def canonical(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def metadata() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def top_level_imports(path: Path) -> set:
    """Return the top-level module names imported by one source file.

    Walks the whole tree rather than the module header, so imports deferred
    inside a function are covered too.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        # a relative import resolves inside the package and needs no declaration
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def declared_distributions() -> set:
    """Return every distribution named by the project, required or optional."""
    project = metadata()["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    # strip the extras and the version specifier: "imageio[ffmpeg]==2.36.0" -> "imageio"
    return {canonical(re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0]) for line in requirements}


def candidates(module: str, index: dict) -> set:
    """Return the distribution names a top-level module could be declared under."""
    installed = index.get(module)
    if installed:
        return {canonical(name) for name in installed}

    # the module is not installed here, so its distribution cannot be looked up.
    # falling back to the import name keeps an undeclared module failing rather
    # than passing unchecked.
    return {canonical(module)}


def test_source_imports_are_declared():
    """every third-party import in src/ resolves to a declared dependency"""
    index = packages_distributions()
    declared = declared_distributions()

    undeclared = {}
    for path in sorted(SOURCE.rglob("*.py")):
        for module in sorted(top_level_imports(path)):
            if module == "surgint" or module in sys.stdlib_module_names:
                continue
            if not candidates(module, index) & declared:
                undeclared.setdefault(module, []).append(str(path.relative_to(ROOT)))

    report = "; ".join(f"{module} in {', '.join(sites)}" for module, sites in sorted(undeclared.items()))
    assert not undeclared, f"undeclared third-party imports: {report}"


def test_loaded_ontology_is_packaged():
    """the serialization the loader reads is included as package data"""
    patterns = metadata()["tool"]["setuptools"]["package-data"]["surgint.ontology"]

    assert any(fnmatch(ONTOLOGY, pattern) for pattern in patterns), (
        f"{ONTOLOGY} matches no package-data pattern in {patterns}"
    )
