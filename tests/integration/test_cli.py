"""
CLI tests
=========

tests that every command line entry point still loads and parses its arguments.

these catch the failures a unit test cannot see: an import that broke when a module
moved, or an argparse definition that stopped accepting its own defaults. argparse is
driven in process, so no model is loaded and no data is read.

coverage:
- help: every entry point builds its parser and prints usage
"""

import importlib
import sys

import pytest

COMMANDS = (
    ("train", "pipelines.train"),
    ("evaluate", "pipelines.evaluate"),
    ("infer", "surgint.cli.infer"),
    ("render", "surgint.cli.render"),
)


@pytest.mark.parametrize(("command", "module_name"), COMMANDS)
def test_help_builds_the_parser(command, module_name, monkeypatch, capsys):
    """--help reaches argparse, so the module, its imports and its defaults are sound"""
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), f"{command} has no main()"

    monkeypatch.setattr(sys, "argv", [command, "--help"])
    with pytest.raises(SystemExit) as exit:
        module.main()

    assert exit.value.code == 0, f"--help exited {exit.value.code}"
    assert "usage" in capsys.readouterr().out.lower(), "argparse printed no usage text"
