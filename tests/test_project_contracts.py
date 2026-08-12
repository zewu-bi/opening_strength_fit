from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

from opening_strength_fit.commands import project_contracts as project_validation
from opening_strength_fit.commands.experiment_audit import (
    RunRecord,
    collect_jobs,
    collect_runs,
    metrics_status,
    summarize_values,
)
from opening_strength_fit.commands.project_contracts import check_matrix_cases, collect_errors

ROOT = Path(__file__).resolve().parents[1]


def _audit_record(*, kind: str, status: str) -> RunRecord:
    return RunRecord(
        run_id="test_run",
        config_path=Path("experiments/runs/test_run.toml"),
        kind=kind,
        model="lightgbm",
        status=status,
        selection_mode="cross_section",
        pvc_dir="/mnt/output/test_run",
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
        "opening_strength_fit.commands.experiment_audit",
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


def test_indexed_job_can_cover_multiple_run_configs(tmp_path: Path) -> None:
    (tmp_path / "cache_family_job.yaml").write_text(
        'opening-strength-fit/run-ids: "cache_2019,cache_2020"\nkind: Job\n',
        encoding="utf-8",
    )

    assert collect_jobs(tmp_path) == {
        "cache_2019": {"training"},
        "cache_2020": {"training"},
    }


def test_rendered_job_declaration_counts_as_retained_job(tmp_path: Path) -> None:
    run_id = "rendered_run"
    (tmp_path / f"{run_id}.toml").write_text(
        f'''\
[run]
id = "{run_id}"
kind = "experiment"
description = "fixture"
status = "completed"

[k8s]
render_mode = "sharded"
render_sha256 = "{"0" * 64}"
''',
        encoding="utf-8",
    )

    record = collect_runs(tmp_path)[run_id]

    assert record.rendered_job_kinds == frozenset({"sharded_training"})


def test_run_can_declare_plain_and_sharded_rendered_jobs(tmp_path: Path) -> None:
    run_id = "dual_rendered_run"
    (tmp_path / f"{run_id}.toml").write_text(
        f'''\
[run]
id = "{run_id}"
kind = "experiment"
description = "fixture"
status = "completed"

[k8s]
render_mode = "training"
render_sha256 = "{"0" * 64}"

[k8s.sharded]
render_mode = "sharded"
render_sha256 = "{"1" * 64}"
''',
        encoding="utf-8",
    )

    record = collect_runs(tmp_path)[run_id]

    assert record.rendered_job_kinds == frozenset({"training", "sharded_training"})


def test_dependency_profiles_require_locked_direct_dependencies(monkeypatch) -> None:
    read = project_validation.read
    monkeypatch.setattr(
        project_validation,
        "read",
        lambda path: "" if path == "requirements.lock" else read(path),
    )
    errors: list[str] = []

    project_validation.check_dependency_profiles(errors)

    assert any(
        error.startswith("requirements.lock: core dependencies are not pinned:") for error in errors
    )


def test_formal_k8s_jobs_reject_site_packages_source_overlays(monkeypatch) -> None:
    monkeypatch.setattr(
        project_validation,
        "read",
        lambda _path: (
            """\
kind: Job
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      containers:
        - command: [\"osf-train\"]
          volumeMounts:
            - mountPath: /usr/local/lib/python3.11/site-packages/opening_strength_fit/model.py
"""
        ),
    )
    errors: list[str] = []

    project_validation.check_k8s_jobs(
        ["experiments/jobs/example_job.yaml"], errors, incubator_assets=set()
    )

    assert errors == [
        "experiments/jobs/example_job.yaml: formal k8s job must use packaged image code, "
        "not site-packages overlays"
    ]


def test_untracked_experiment_assets_require_incubator_registration(monkeypatch) -> None:
    monkeypatch.setattr(
        project_validation,
        "untracked_project_files",
        lambda: {
            "experiments/jobs/support/registered_job.yaml",
            "experiments/scripts/missing_probe.py",
            "src/opening_strength_fit/local_work.py",
        },
    )
    errors: list[str] = []

    project_validation.check_incubator_coverage(
        {"experiments/jobs/support/registered_job.yaml"}, errors
    )

    assert errors == [
        "experiments/scripts/missing_probe.py: untracked experiment asset is missing from "
        "experiments/incubator.toml"
    ]


def test_project_contract_rejects_duplicate_markdown_documents() -> None:
    errors: list[str] = []

    project_validation.check_source_layout(["experiments/evidence/README.md"], errors)

    assert errors == [
        "experiments/evidence/README.md: duplicate project documentation is forbidden"
    ]


def test_matrix_contract_rejects_cross_window_label_path() -> None:
    errors: list[str] = []
    check_matrix_cases(
        "experiments/runs/example.toml",
        {
            "matrix": {
                "cases": [
                    {
                        "name": "w0931_0940_h1m",
                        "window": "09:31-09:40",
                        "horizon": "1m",
                        "feature_path": "/data/opening_0931_0940_features_350",
                        "label_path": "/data/opening_1001_1010_labels_h1m_v2",
                    }
                ]
            }
        },
        errors,
    )

    assert errors == [
        "experiments/runs/example.toml: matrix case 1 label_path does not match window/horizon"
    ]


def test_project_entrypoints_are_importable() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entrypoints = pyproject["project"]["scripts"]

    for command, target in entrypoints.items():
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)

        assert command == "osf" or command.startswith("osf-")
        assert callable(getattr(module, attr))


def test_experiment_registry_is_aligned() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "opening_strength_fit.commands.experiment_audit"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_experiment_audit_metrics_status_distinguishes_artifact_runs() -> None:
    artifact = _audit_record(kind="labeled_cache", status="completed")
    local_artifact = _audit_record(kind="realistic_acceptance", status="completed")
    local_opening_audit = _audit_record(kind="opening_limit_audit", status="completed")
    completed_training = _audit_record(kind="experiment", status="completed")
    queued_training = _audit_record(kind="experiment", status="queued")

    assert metrics_status(artifact, has_metrics=False) == "n/a"
    assert metrics_status(artifact, has_metrics=True) == "unexpected"
    assert metrics_status(local_artifact, has_metrics=False) == "n/a"
    assert metrics_status(local_opening_audit, has_metrics=False) == "n/a"
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


def test_experiment_audit_requires_inactive_closeout_reason(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="closed_without_reason",
        run_toml="""\
[run]
id = "closed_without_reason"
kind = "labeled_cache"
description = "inactive closeout fixture"
status = "superseded"
""",
        with_job=True,
    )

    assert result.returncode == 1
    assert "missing required [run] fields closed_at, status_reason" in result.stdout


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


def test_experiment_audit_allows_local_comparison_analysis_run(tmp_path: Path) -> None:
    result = _run_audit_fixture(
        tmp_path,
        run_id="local_comparison",
        run_toml="""\
[run]
id = "local_comparison"
kind = "comparison_analysis"
description = "local comparison fixture"
status = "completed"
""",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "local_comparison" in result.stdout
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


def test_experiment_audit_summary_only_suppresses_run_table(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    jobs_dir = tmp_path / "jobs"
    metrics_dir = tmp_path / "metrics"
    runs_dir.mkdir()
    jobs_dir.mkdir()
    metrics_dir.mkdir()
    (runs_dir / "local_analysis.toml").write_text(
        """\
[run]
id = "local_analysis"
kind = "comparison_analysis"
description = "summary fixture"
status = "completed"
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opening_strength_fit.commands.experiment_audit",
            "--runs-dir",
            str(runs_dir),
            "--jobs-dir",
            str(jobs_dir),
            "--metrics-dir",
            str(metrics_dir),
            "--summary-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status_counts: completed=1" in result.stdout
    assert "local_analysis" not in result.stdout
