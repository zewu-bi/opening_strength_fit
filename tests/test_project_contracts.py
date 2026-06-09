from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

from opening_strength_fit.commands.experiment_audit import (
    RunRecord,
    metrics_status,
    summarize_values,
)
from opening_strength_fit.commands.project_contracts import collect_errors

ROOT = Path(__file__).resolve().parents[1]


def _audit_record(*, kind: str, status: str) -> RunRecord:
    return RunRecord(
        run_id="test_run",
        config_path=Path("experiments/runs/test_run.toml"),
        kind=kind,
        model="lightgbm",
        status=status,
        selection_mode="cross_section",
        tick_path="",
        pvc_dir="/mnt/output/test_run",
        local_dir="output/legacy/analysis/test_run",
    )


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


def test_experiment_audit_metrics_status_distinguishes_artifact_runs() -> None:
    artifact = _audit_record(kind="labeled_cache", status="completed")
    completed_training = _audit_record(kind="experiment", status="completed")
    queued_training = _audit_record(kind="experiment", status="queued")

    assert metrics_status(artifact, has_metrics=False) == "n/a"
    assert metrics_status(artifact, has_metrics=True) == "unexpected"
    assert metrics_status(completed_training, has_metrics=False) == "missing"
    assert metrics_status(queued_training, has_metrics=False) == "pending"


def test_experiment_audit_summary_prioritizes_actionable_counts() -> None:
    records = [
        {"status": "completed", "metrics": "yes"},
        {"status": "queued", "metrics": "pending"},
        {"status": "completed", "metrics": "n/a"},
        {"status": "running", "metrics": "missing"},
    ]

    assert (
        summarize_values(
            records,
            "status",
            ("queued", "running", "completed", "canceled", "superseded"),
        )
        == "queued=1, running=1, completed=2"
    )
    assert (
        summarize_values(
            records,
            "metrics",
            ("missing", "pending", "unexpected", "yes", "n/a"),
        )
        == "missing=1, pending=1, yes=1, n/a=1"
    )
