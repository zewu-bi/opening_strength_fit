from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

from opening_strength_fit.commands.project_contracts import collect_errors

ROOT = Path(__file__).resolve().parents[1]


def test_project_contracts_are_satisfied() -> None:
    assert collect_errors() == []


def test_project_entrypoints_are_importable() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entrypoints = pyproject["project"]["scripts"]

    for command, target in entrypoints.items():
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)

        assert command.startswith("osf-")
        assert callable(getattr(module, attr))


def test_experiment_registry_is_aligned() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "opening_strength_fit.cli.audit_experiments"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
