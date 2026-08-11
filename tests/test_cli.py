from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from opening_strength_fit import cli

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_command_dispatches_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeEntryPoint:
        name = "osf-build-raw-source-cache"

        def load(self):
            return lambda: calls.append(sys.argv.copy())

    entrypoint = FakeEntryPoint()
    monkeypatch.setattr(cli, "_console_entrypoints", lambda: {entrypoint.name: entrypoint})

    cli.main(["data", "build-raw", "--config", "run.toml"])

    assert calls == [["osf-build-raw-source-cache", "--config", "run.toml"]]


def test_legacy_suffix_dispatches_to_existing_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeEntryPoint:
        def load(self):
            return lambda: calls.append(sys.argv.copy())

    monkeypatch.setattr(
        cli,
        "_console_entrypoints",
        lambda: {"osf-run-gap-risk-attribution": FakeEntryPoint()},
    )

    cli.main(["run-gap-risk-attribution", "--help"])

    assert calls == [["osf-run-gap-risk-attribution", "--help"]]


def test_unknown_command_shows_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_console_entrypoints", lambda: {})

    with pytest.raises(SystemExit, match="unknown osf command"):
        cli.main(["not-a-command"])


def test_aliases_only_target_declared_compatibility_commands() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    commands = set(pyproject["project"]["scripts"])

    assert set(cli.COMMAND_ALIASES.values()) <= commands
