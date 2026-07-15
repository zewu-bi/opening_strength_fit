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
        missing_run_fields=(),
    )


def _run_audit_fixture(
    tmp_path: Path,
    *,
    run_id: str,
    run_toml: str,
    with_job: bool = False,
    require_metrics: bool = False,
    metric_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    runs_dir = tmp_path / "runs"
    jobs_dir = tmp_path / "jobs"
    metrics_dir = tmp_path / "metrics"
    runs_dir.mkdir(parents=True)
    jobs_dir.mkdir()
    metrics_dir.mkdir()
    (runs_dir / f"{run_id}.toml").write_text(run_toml, encoding="utf-8")
    if with_job:
        (jobs_dir / f"{run_id}_job.yaml").write_text("kind: Job\n", encoding="utf-8")
    if metric_text is not None:
        (metrics_dir / f"{run_id}_metrics_by_year.csv").write_text(metric_text, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "opening_strength_fit.cli.audit_experiments",
        "--runs-dir",
        str(runs_dir),
        "--jobs-dir",
        str(jobs_dir),
        "--metrics-dir",
        str(metrics_dir),
    ]
    if require_metrics:
        command.append("--require-metrics")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
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
    local_artifact = _audit_record(kind="realistic_acceptance", status="completed")
    completed_training = _audit_record(kind="experiment", status="completed")
    queued_training = _audit_record(kind="experiment", status="queued")

    assert metrics_status(artifact, has_metrics=False) == "n/a"
    assert metrics_status(artifact, has_metrics=True) == "unexpected"
    assert metrics_status(local_artifact, has_metrics=False) == "n/a"
    assert metrics_status(completed_training, has_metrics=False) == "missing"
    assert metrics_status(queued_training, has_metrics=False) == "pending"


def test_experiment_audit_rejects_unknown_run_status(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="invalid_status",
        run_toml="""\
[run]
id = "invalid_status"
kind = "labeled_cache"
description = "invalid status fixture"
status = "submitted"
""",
        with_job=True,
    )

    assert result.returncode == 1
    assert "alignment_errors:" in result.stdout
    assert "unknown status='submitted'" in result.stdout


def test_experiment_audit_allows_local_realistic_acceptance_run(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="local_replay",
        run_toml="""\
[run]
id = "local_replay"
kind = "realistic_acceptance"
description = "local replay fixture"
status = "completed"
""",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "local_replay" in result.stdout
    assert "local" in result.stdout


def test_experiment_audit_only_requires_ignored_metrics_in_strict_mode(tmp_path: Path) -> None:
    run_toml = """\
[run]
id = "completed_training"
kind = "experiment"
description = "completed training fixture"
status = "completed"
"""
    optional = _run_audit_fixture(
        tmp_path / "optional",
        run_id="completed_training",
        run_toml=run_toml,
        with_job=True,
    )
    strict = _run_audit_fixture(
        tmp_path / "strict",
        run_id="completed_training",
        run_toml=run_toml,
        with_job=True,
        require_metrics=True,
    )

    assert optional.returncode == 0, optional.stdout + optional.stderr
    assert "metrics_requirement: optional" in optional.stdout
    assert strict.returncode == 1
    assert "metrics_requirement: strict" in strict.stdout
    assert "completed_training: missing metrics csv" in strict.stdout


def test_experiment_audit_requires_explicit_run_fields(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="missing_kind",
        run_toml="""\
[run]
id = "missing_kind"
description = "missing kind fixture"
status = "completed"
""",
        with_job=True,
    )

    assert result.returncode == 1
    assert "missing_kind: missing required [run] fields kind" in result.stdout


def test_experiment_audit_strict_requires_exploration_metrics(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="exploration_training",
        run_toml="""\
[run]
id = "exploration_training"
kind = "exploration"
description = "exploration training fixture"
status = "completed"
""",
        with_job=True,
        require_metrics=True,
    )

    assert result.returncode == 1
    assert "exploration_training: missing metrics csv" in result.stdout


def test_experiment_audit_validates_metrics_contents(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="completed_training",
        run_toml="""\
[run]
id = "completed_training"
kind = "experiment"
description = "completed training fixture"
status = "completed"
""",
        with_job=True,
        require_metrics=True,
        metric_text="run_id,test_year,model_name,rows\nother_run,2025,lightgbm,10\n",
    )

    assert result.returncode == 1
    assert "completed_training: metrics csv run_id mismatch on rows 2" in result.stdout


def test_experiment_audit_accepts_valid_strict_training_metrics(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="completed_training",
        run_toml="""\
[run]
id = "completed_training"
kind = "experiment"
description = "completed training fixture"
status = "completed"
""",
        with_job=True,
        require_metrics=True,
        metric_text="run_id,test_year,model_name,rows\ncompleted_training,2025,lightgbm,10\n",
    )

    assert result.returncode == 0, result.stdout + result.stderr


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
