from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s import rendered_job_specs
from opening_strength_fit.pvc_layout import run_output_dir

JOB_SUFFIXES = tuple(
    zip(
        "_pool_internal_analysis_job.yaml _w0931_jobs.yaml _w1001_jobs.yaml "
        "_w1401_jobs.yaml _sharded_job.yaml _job.yaml".split(),
        "pool_internal_analysis sharded_training sharded_training sharded_training "
        "sharded_training training".split(),
        strict=True,
    )
)
PREPARED_STATUS = "prepared"
ACTIVE_STATUSES = set("queued running".split())
COMPLETED_STATUS = "completed"
INACTIVE_STATUSES = set("canceled superseded".split())
KNOWN_STATUSES = {PREPARED_STATUS, *ACTIVE_STATUSES, COMPLETED_STATUS, *INACTIVE_STATUSES}
LOCAL_ONLY_RUN_KINDS = set("comparison_analysis opening_limit_audit realistic_acceptance".split())
TRAINING_JOB_KINDS = set("training sharded_training indexed_builder".split())
ARTIFACT_RUN_KINDS = set(
    "alpha_conditioned_rolling_validation ask_level_attribution cache_transform "
    "capacity_acceptance capacity_audit clickhouse_labeled_cache comparison_analysis "
    "execution_context exposure_audit exposure_input feature_audit feature_hygiene "
    "gap_risk_attribution labeled_cache long_horizon_label_split long_horizon_labels "
    "next_close_label_cache opening_limit_audit pool_internal_analysis raw_source_cache "
    "realistic_acceptance score_risk_sweep short_label_cache strategy_acceptance target_cache "
    "training_feature_dataset".split()
)
METRICS_SUFFIX = "_metrics_by_year.csv"
JOB_RUN_IDS_PATTERN = re.compile(r'opening-strength-fit/run-ids:\s*["\']?([^"\'\n]+)')
REQUIRED_RUN_FIELDS = ("id", "kind", "description", "status")
REQUIRED_INACTIVE_RUN_FIELDS = ("closed_at", "status_reason")
REQUIRED_METRICS_COLUMNS = ("run_id", "test_year", "model_name", "rows")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_path: Path
    kind: str
    model: str
    status: str
    selection_mode: str
    pvc_dir: str
    missing_run_fields: tuple[str, ...]
    rendered_job_kinds: frozenset[str] = frozenset()


def collect_runs(runs_dir: Path) -> dict[str, RunRecord]:
    runs = {}
    grouped_rendered_jobs: dict[str, set[str]] = {}
    for path in sorted(runs_dir.glob("*.toml")):
        config = load_toml(path)
        run_section = config.get("run", {})
        status = str(run_section.get("status", ""))
        required_fields = (
            (*REQUIRED_RUN_FIELDS, *REQUIRED_INACTIVE_RUN_FIELDS)
            if status in INACTIVE_STATUSES
            else REQUIRED_RUN_FIELDS
        )
        missing_fields = tuple(
            field for field in required_fields if not str(run_section.get(field, "")).strip()
        )
        run_id = str(run_section.get("id") or path.stem)
        try:
            rendered_kinds = frozenset(spec.kind for spec in rendered_job_specs(config))
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        k8s = config.get("k8s", {})
        indexed_template = str(k8s.get("indexed_run_id_template", "") or "")
        if indexed_template:
            for year in k8s.get("years", []):
                grouped_run_id = indexed_template.format(year=int(year))
                grouped_rendered_jobs.setdefault(grouped_run_id, set()).update(rendered_kinds)
        runs[run_id] = RunRecord(
            run_id=run_id,
            config_path=path,
            kind=str(run_section.get("kind", "")),
            model=str(config.get("model", {}).get("name", "ridge")),
            status=status,
            selection_mode=str(config.get("evaluation", {}).get("selection_mode", "symbol_day")),
            pvc_dir=run_output_dir(config, run_id),
            missing_run_fields=missing_fields,
            rendered_job_kinds=rendered_kinds,
        )
    if missing_grouped_runs := sorted(set(grouped_rendered_jobs) - set(runs)):
        missing = ", ".join(missing_grouped_runs)
        raise SystemExit(f"indexed renderer references missing run config(s): {missing}")
    for grouped_run_id, kinds in grouped_rendered_jobs.items():
        record = runs[grouped_run_id]
        runs[grouped_run_id] = replace(
            record,
            rendered_job_kinds=record.rendered_job_kinds | frozenset(kinds),
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
        match = JOB_RUN_IDS_PATTERN.search(path.read_text(encoding="utf-8"))
        run_ids = (
            tuple(item.strip() for item in match.group(1).split(",") if item.strip())
            if match
            else (run_id,)
        )
        for declared_run_id in run_ids:
            jobs.setdefault(declared_run_id, set()).add(kind)
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

                rows = list(reader)
                mismatched_rows = [
                    row_number
                    for row_number, row in enumerate(rows, start=2)
                    if row.get("run_id") != run_id
                ]
                if not rows:
                    errors.append(f"{run_id}: metrics csv is empty")
                if mismatched_rows:
                    rows = ", ".join(str(row) for row in mismatched_rows[:5])
                    suffix = "..." if len(mismatched_rows) > 5 else ""
                    errors.append(f"{run_id}: metrics csv run_id mismatch on rows {rows}{suffix}")
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            errors.append(f"{run_id}: metrics csv is unreadable ({exc})")
    return run_ids, errors


def is_artifact_run(record: RunRecord) -> bool:
    return record.kind in ARTIFACT_RUN_KINDS


def requires_job(record: RunRecord) -> bool:
    return record.status != PREPARED_STATUS and record.kind not in LOCAL_ONLY_RUN_KINDS


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
    columns = "run_id status model selection jobs metrics pvc_dir".split()
    widths = {
        column: max(len(column), *(len(record[column]) for record in records)) for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
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
    summaries = (
        (
            "status",
            ("prepared", "queued", "running", "completed", "canceled", "superseded"),
        ),
        ("metrics", ("missing", "pending", "unexpected", "yes", "n/a")),
    )
    for column, order in summaries:
        print(f"  {column}_counts: {summarize_values(records, column, order)}")


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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress the per-run table while retaining validation, warnings, and counts.",
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
        job_kinds = jobs.get(run_id, set()) | set(record.rendered_job_kinds)
        is_artifact = is_artifact_run(record)
        is_pool_analysis = (
            record.kind == "pool_internal_analysis" and "pool_internal_analysis" in job_kinds
        )
        has_jobs = bool(TRAINING_JOB_KINDS & job_kinds)
        has_metrics = run_id in metrics
        is_running = record.status in ACTIVE_STATUSES
        is_completed = record.status == COMPLETED_STATUS
        if requires_job(record) and not has_jobs and not is_pool_analysis:
            errors.append(f"{run_id}: missing training job yaml")

        if record.status not in KNOWN_STATUSES:
            errors.append(
                f"{run_id}: unknown status={record.status!r}; use queued, running, "
                "completed, canceled, superseded, or prepared"
            )

        if has_jobs and not has_metrics and is_running and not is_artifact:
            warnings.append(
                f"{run_id}: has job yaml but no metrics yet; status={record.status!r} is plausible"
            )
        if has_metrics and is_running:
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
                "model": record.kind
                if (is_artifact or record.kind == "exploration")
                else record.model,
                "selection": record.selection_mode,
                "jobs": jobs_status(record, job_kinds),
                "metrics": metrics_status(record, has_metrics=has_metrics),
                "pvc_dir": record.pvc_dir,
            }
        )

    for run_id in sorted(set(jobs) - set(runs)):
        errors.append(f"{run_id}: job yaml has no matching run config")
    orphan_metrics = metrics - set(runs)
    if orphan_metrics:
        warnings.append(
            f"{len(orphan_metrics)} archived metrics set(s) have no retained run config"
        )

    print("experiment_alignment:")
    if not args.summary_only:
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
