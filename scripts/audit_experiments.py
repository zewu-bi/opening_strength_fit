from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from opening_strength_fit.config import load_toml


JOB_SUFFIXES = (
    ("_sharded_job.yaml", "sharded_training"),
    ("_job.yaml", "training"),
)
ACTIVE_STATUSES = {"queued", "running"}
COMPLETED_STATUS = "completed"
KNOWN_STATUSES = {*ACTIVE_STATUSES, COMPLETED_STATUS}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_path: Path
    kind: str
    model: str
    status: str
    selection_mode: str
    tick_path: str
    pvc_dir: str
    local_dir: str


def collect_runs(runs_dir: Path) -> dict[str, RunRecord]:
    runs = {}
    for path in sorted(runs_dir.glob("*.toml")):
        config = load_toml(path)
        run_id = str(config.get("run", {}).get("id", path.stem))
        output = config.get("output", {})
        run_section = config.get("run", {})
        data = config.get("data", {})
        model = str(config.get("model", {}).get("name", "ridge"))
        evaluation = config.get("evaluation", {})
        status = str(run_section.get("status", "completed"))
        kind = str(run_section.get("kind", "experiment"))
        pvc_dir = str(output.get("k8s_dir", f"/mnt/output/opening_strength_fit/{run_id}"))
        runs[run_id] = RunRecord(
            run_id=run_id,
            config_path=path,
            kind=kind,
            model=model,
            status=status,
            selection_mode=str(evaluation.get("selection_mode", "symbol_day")),
            tick_path=str(data.get("tick_path", "")),
            pvc_dir=pvc_dir,
            local_dir=str(output.get("local_dir", f"output/local/{run_id}")),
        )
    return runs


def split_job_name(path: Path) -> tuple[str, str] | None:
    for suffix, kind in JOB_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)], kind
    return None


def collect_jobs(jobs_dir: Path) -> dict[str, set[str]]:
    jobs: dict[str, set[str]] = {}
    for path in sorted(jobs_dir.glob("*.yaml")):
        split = split_job_name(path)
        if split is None:
            continue
        run_id, kind = split
        jobs.setdefault(run_id, set()).add(kind)
    return jobs


def collect_metrics(metrics_dir: Path) -> set[str]:
    return {
        path.name.removesuffix("_metrics_by_year.csv")
        for path in metrics_dir.glob("*_metrics_by_year.csv")
    }


def has_training_job(kinds: set[str]) -> bool:
    return bool({"training", "sharded_training"} & kinds)


def is_artifact_run(record: RunRecord) -> bool:
    return record.kind == "feature_audit"


def is_exploration_run(record: RunRecord) -> bool:
    return record.kind == "exploration"


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def print_table(records: list[dict[str, str]]) -> None:
    if not records:
        print("No experiments found.")
        return
    columns = [
        "run_id",
        "status",
        "model",
        "selection",
        "jobs",
        "metrics",
        "pvc_dir",
    ]
    widths = {
        column: max(len(column), *(len(record[column]) for record in records))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for record in records:
        print("  ".join(record[column].ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local experiment config/job/metrics alignment."
    )
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--jobs-dir", default="experiments/jobs")
    parser.add_argument("--metrics-dir", default="experiments/results/metrics")
    args = parser.parse_args()

    runs = collect_runs(Path(args.runs_dir))
    jobs = collect_jobs(Path(args.jobs_dir))
    metrics = collect_metrics(Path(args.metrics_dir))

    records = []
    errors = []
    warnings = []
    for run_id, record in sorted(runs.items()):
        if record.config_path.stem != run_id:
            errors.append(
                f"{run_id}: config filename must match run.id ({record.config_path.name})"
            )
        job_kinds = jobs.get(run_id, set())
        is_cache = is_artifact_run(record)
        is_exploration = is_exploration_run(record)
        has_jobs = has_training_job(job_kinds)
        has_metrics = run_id in metrics
        is_running = record.status in ACTIVE_STATUSES
        is_completed = record.status == COMPLETED_STATUS
        if not has_jobs:
            errors.append(f"{run_id}: missing training job yaml")

        if record.status not in KNOWN_STATUSES:
            warnings.append(
                f"{run_id}: unknown status={record.status!r}; use queued, running, or completed"
            )

        if has_jobs and not has_metrics and is_running and not (
            is_cache or is_exploration
        ):
            warnings.append(
                f"{run_id}: has job yaml but no metrics yet; status={record.status!r} is plausible"
            )
        if has_metrics and not is_completed:
            warnings.append(
                f"{run_id}: metrics exist but status={record.status!r}; update status to completed after confirming results"
            )
        if not has_metrics and is_completed and not (is_cache or is_exploration):
            errors.append(f"{run_id}: missing metrics csv")
        if has_metrics and is_cache:
            errors.append(f"{run_id}: artifact run should not have metrics csv")

        records.append(
            {
                "run_id": run_id,
                "status": record.status,
                "model": record.kind if (is_cache or is_exploration) else record.model,
                "selection": record.selection_mode,
                "jobs": ",".join(sorted(job_kinds)) if job_kinds else "missing",
                "metrics": format_bool(has_metrics),
                "pvc_dir": record.pvc_dir,
            }
        )

    for run_id in sorted(set(jobs) - set(runs)):
        errors.append(f"{run_id}: job yaml has no matching run config")
    for run_id in sorted(metrics - set(runs)):
        errors.append(f"{run_id}: metrics csv has no matching run config")

    print("experiment_alignment:")
    print_table(records)
    if warnings:
        print("\nalignment_warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("\nalignment_errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("\nalignment_ok: yes")


if __name__ == "__main__":
    main()
