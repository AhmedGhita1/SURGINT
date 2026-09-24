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

COMMANDS = ("train", "evaluate", "infer", "render")


@pytest.mark.parametrize("command", COMMANDS)
def test_help_builds_the_parser(command, monkeypatch, capsys):
    """--help reaches argparse, so the module, its imports and its defaults are sound"""
    module = importlib.import_module(f"surgint.cli.{command}")
    assert callable(getattr(module, "main", None)), f"{command} has no main()"

    monkeypatch.setattr(sys, "argv", [command, "--help"])
    with pytest.raises(SystemExit) as exit:
        module.main()

    assert exit.value.code == 0, f"--help exited {exit.value.code}"
    assert "usage" in capsys.readouterr().out.lower(), "argparse printed no usage text"
