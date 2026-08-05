from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from opening_strength_fit.config import load_toml
from opening_strength_fit.pvc_layout import run_output_dir

JOB_SUFFIXES = (
    ("_pool_internal_analysis_job.yaml", "pool_internal_analysis"),
    ("_sharded_job.yaml", "sharded_training"),
    ("_job.yaml", "training"),
)
ACTIVE_STATUSES = {"queued", "running"}
COMPLETED_STATUS = "completed"
INACTIVE_STATUSES = {"canceled", "superseded"}
KNOWN_STATUSES = {*ACTIVE_STATUSES, COMPLETED_STATUS, *INACTIVE_STATUSES}
LOCAL_ONLY_RUN_KINDS = {"comparison_analysis", "opening_limit_audit", "realistic_acceptance"}
METRICS_SUFFIX = "_metrics_by_year.csv"
REQUIRED_RUN_FIELDS = ("id", "kind", "description", "status")
REQUIRED_METRICS_COLUMNS = ("run_id", "test_year", "model_name", "rows")


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
    missing_run_fields: tuple[str, ...]


def collect_runs(runs_dir: Path) -> dict[str, RunRecord]:
    runs = {}
    for path in sorted(runs_dir.glob("*.toml")):
        config = load_toml(path)
        run_section = config.get("run", {})
        missing_fields = tuple(
            field for field in REQUIRED_RUN_FIELDS if not str(run_section.get(field, "")).strip()
        )
        run_id = str(run_section.get("id") or path.stem)
        output = config.get("output", {})
        data = config.get("data", {})
        model = str(config.get("model", {}).get("name", "ridge"))
        evaluation = config.get("evaluation", {})
        status = str(run_section.get("status", ""))
        kind = str(run_section.get("kind", ""))
        pvc_dir = run_output_dir(config, run_id)
        runs[run_id] = RunRecord(
            run_id=run_id,
            config_path=path,
            kind=kind,
            model=model,
            status=status,
            selection_mode=str(evaluation.get("selection_mode", "symbol_day")),
            tick_path=str(data.get("tick_path", "")),
            pvc_dir=pvc_dir,
            local_dir=str(output.get("local_dir", f"output/legacy/analysis/{run_id}")),
            missing_run_fields=missing_fields,
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


def collect_metrics(metrics_dir: Path) -> tuple[set[str], list[str]]:
    run_ids = set()
    errors = []
    for path in sorted(metrics_dir.glob(f"*{METRICS_SUFFIX}")):
        run_id = path.name.removesuffix(METRICS_SUFFIX)
        run_ids.add(run_id)
        try:
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                columns = set(reader.fieldnames or [])
                missing_columns = [
                    column for column in REQUIRED_METRICS_COLUMNS if column not in columns
                ]
                if missing_columns:
                    errors.append(
                        f"{run_id}: metrics csv missing columns {', '.join(missing_columns)}"
                    )
                    continue

                row_count = 0
                mismatched_rows = []
                for row_number, row in enumerate(reader, start=2):
                    row_count += 1
                    if row.get("run_id") != run_id:
                        mismatched_rows.append(row_number)
                if row_count == 0:
                    errors.append(f"{run_id}: metrics csv is empty")
                if mismatched_rows:
                    rows = ", ".join(str(row) for row in mismatched_rows[:5])
                    suffix = "..." if len(mismatched_rows) > 5 else ""
                    errors.append(f"{run_id}: metrics csv run_id mismatch on rows {rows}{suffix}")
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            errors.append(f"{run_id}: metrics csv is unreadable ({exc})")
    return run_ids, errors


def has_training_job(kinds: set[str]) -> bool:
    return bool({"training", "sharded_training"} & kinds)


def is_artifact_run(record: RunRecord) -> bool:
    return record.kind in {
        "capacity_audit",
        "capacity_acceptance",
        "exposure_input",
        "exposure_audit",
        "feature_audit",
        "feature_hygiene",
        "cache_transform",
        "labeled_cache",
        "clickhouse_labeled_cache",
        "next_close_label_cache",
        "opening_limit_audit",
        "target_cache",
        "score_risk_sweep",
        "alpha_conditioned_rolling_validation",
        "ask_level_attribution",
        "execution_context",
        "realistic_acceptance",
        "comparison_analysis",
        "strategy_acceptance",
        "gap_risk_attribution",
        "pool_internal_analysis",
    }


def is_exploration_run(record: RunRecord) -> bool:
    return record.kind == "exploration"


def is_pool_internal_analysis_run(record: RunRecord, kinds: set[str]) -> bool:
    return record.kind == "pool_internal_analysis" and "pool_internal_analysis" in kinds


def requires_job(record: RunRecord) -> bool:
    return record.kind not in LOCAL_ONLY_RUN_KINDS


def jobs_status(record: RunRecord, kinds: set[str]) -> str:
    if kinds:
        return ",".join(sorted(kinds))
    return "missing" if requires_job(record) else "local"


def metrics_status(record: RunRecord, *, has_metrics: bool) -> str:
    if is_artifact_run(record):
        return "unexpected" if has_metrics else "n/a"
    if has_metrics:
        return "yes"
    if record.status in ACTIVE_STATUSES:
        return "pending"
    return "missing"


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
        column: max(len(column), *(len(record[column]) for record in records)) for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for record in records:
        print("  ".join(record[column].ljust(widths[column]) for column in columns))


def summarize_values(records: list[dict[str, str]], column: str, order: tuple[str, ...]) -> str:
    counts = Counter(record[column] for record in records)
    keys = [key for key in order if key in counts]
    keys.extend(sorted(set(counts) - set(keys)))
    return ", ".join(f"{key}={counts[key]}" for key in keys) if keys else "none"


def print_summary(records: list[dict[str, str]], *, require_metrics: bool) -> None:
    print("\naudit_summary:")
    print(f"  metrics_requirement: {'strict' if require_metrics else 'optional'}")
    print(
        "  status_counts: "
        + summarize_values(
            records,
            "status",
            ("queued", "running", "completed", "canceled", "superseded"),
        )
    )
    print(
        "  metrics_counts: "
        + summarize_values(
            records,
            "metrics",
            ("missing", "pending", "unexpected", "yes", "n/a"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit experiment config/job alignment and optional local metrics evidence."
    )
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--jobs-dir", default="experiments/jobs")
    parser.add_argument("--metrics-dir", default="experiments/results/metrics")
    parser.add_argument(
        "--require-metrics",
        action="store_true",
        help="Fail when a completed training run is absent from the local metrics mirror.",
    )
    args = parser.parse_args()

    runs = collect_runs(Path(args.runs_dir))
    jobs = collect_jobs(Path(args.jobs_dir))
    metrics, metric_errors = collect_metrics(Path(args.metrics_dir))

    records = []
    errors = list(metric_errors)
    warnings = []
    for run_id, record in sorted(runs.items()):
        if record.missing_run_fields:
            errors.append(
                f"{run_id}: missing required [run] fields {', '.join(record.missing_run_fields)}"
            )
        if record.config_path.stem != run_id:
            errors.append(
                f"{run_id}: config filename must match run.id ({record.config_path.name})"
            )
        job_kinds = jobs.get(run_id, set())
        is_artifact = is_artifact_run(record)
        is_exploration = is_exploration_run(record)
        is_pool_analysis = is_pool_internal_analysis_run(record, job_kinds)
        has_jobs = has_training_job(job_kinds)
        has_metrics = run_id in metrics
        is_running = record.status in ACTIVE_STATUSES
        is_completed = record.status == COMPLETED_STATUS
        if requires_job(record) and not has_jobs and not is_pool_analysis:
            errors.append(f"{run_id}: missing training job yaml")

        if record.status not in KNOWN_STATUSES:
            errors.append(
                f"{run_id}: unknown status={record.status!r}; use queued, running, "
                "completed, canceled, or superseded"
            )

        if has_jobs and not has_metrics and is_running and not is_artifact:
            warnings.append(
                f"{run_id}: has job yaml but no metrics yet; status={record.status!r} is plausible"
            )
        if has_metrics and not is_completed:
            warnings.append(
                f"{run_id}: metrics exist but status={record.status!r}; update status to completed after confirming results"
            )
        if args.require_metrics and not has_metrics and is_completed and not is_artifact:
            errors.append(f"{run_id}: missing metrics csv")
        if has_metrics and is_artifact:
            errors.append(f"{run_id}: artifact run should not have metrics csv")

        records.append(
            {
                "run_id": run_id,
                "status": record.status,
                "model": record.kind if (is_artifact or is_exploration) else record.model,
                "selection": record.selection_mode,
                "jobs": jobs_status(record, job_kinds),
                "metrics": metrics_status(record, has_metrics=has_metrics),
                "pvc_dir": record.pvc_dir,
            }
        )

    for run_id in sorted(set(jobs) - set(runs)):
        errors.append(f"{run_id}: job yaml has no matching run config")
    for run_id in sorted(metrics - set(runs)):
        errors.append(f"{run_id}: metrics csv has no matching run config")

    print("experiment_alignment:")
    print_table(records)
    print_summary(records, require_metrics=args.require_metrics)
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
