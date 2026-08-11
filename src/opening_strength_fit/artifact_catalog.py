from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from opening_strength_fit.k8s import RunSpec

SCORE_RISK_ARTIFACTS = (
    "score_risk_summary.csv",
    "score_risk_minute_summary.csv",
    "score_risk_group_metrics.csv",
    "score_risk_trace.json",
)
ROLLING_VALIDATION_ARTIFACTS = (
    "rolling_summary.csv",
    "rolling_month_summary.csv",
    "rolling_group_metrics.csv",
    "rolling_trace.json",
)
ROLLING_VALIDATION_SHARD_ARTIFACTS = (
    "rolling_group_metrics.csv",
    "rolling_month_summary.csv",
    "rolling_summary.csv",
    "rolling_trace.json",
)
FEATURE_AUDIT_ARTIFACTS = (
    "feature_audit_metrics.csv",
    "feature_audit_permutation.csv",
    "feature_importance.csv",
    "feature_group_importance.csv",
    "feature_audit_trace.json",
)
FEATURE_AUDIT_COMBINED_CSVS = (
    "feature_audit_metrics.csv",
    "feature_audit_permutation.csv",
    "feature_importance.csv",
    "feature_group_importance.csv",
)
FEATURE_HYGIENE_ARTIFACTS = (
    "feature_hygiene.csv",
    "feature_correlation_pairs.csv",
    "feature_correlation_clusters.csv",
    "feature_prune_candidates.csv",
    "feature_keep_list.txt",
    "feature_drop_list.txt",
    "feature_hygiene_trace.json",
)
CAPACITY_AUDIT_ARTIFACTS = (
    "capacity_audit_selected.csv",
    "capacity_audit_group_metrics.csv",
    "capacity_audit_daily_summary.csv",
    "capacity_audit_month_summary.csv",
    "capacity_audit_summary.csv",
    "capacity_audit_trace.json",
)
CAPACITY_ACCEPTANCE_ARTIFACTS = (
    "capacity_acceptance_daily_summary.csv",
    "capacity_acceptance_summary.csv",
    "capacity_acceptance_trace.json",
)
STRATEGY_ACCEPTANCE_ARTIFACTS = (
    "strategy_acceptance_summary.csv",
    "strategy_acceptance_daily.csv",
    "strategy_acceptance_group_metrics.csv",
    "strategy_acceptance_capacity_summary.csv",
    "strategy_acceptance_overlap_summary.csv",
    "strategy_acceptance_overlap_daily.csv",
    "strategy_acceptance_overlap_adjacent.csv",
    "strategy_acceptance_tail_summary.csv",
    "strategy_acceptance_tail_monthly.csv",
    "strategy_acceptance_tail_concentration.csv",
    "strategy_acceptance_bootstrap.csv",
    "strategy_acceptance_leave_one_out.csv",
    "strategy_acceptance_trace.json",
    "_SUCCESS",
)
EXPOSURE_AUDIT_ARTIFACTS = (
    "exposure_audit_group_metrics.csv",
    "exposure_audit_month_summary.csv",
    "exposure_audit_summary.csv",
    "exposure_audit_category_summary.csv",
    "exposure_audit_industry_group_metrics.csv",
    "exposure_audit_industry_month_summary.csv",
    "exposure_audit_industry_summary.csv",
    "exposure_audit_daily_concentration.csv",
    "exposure_audit_concentration_summary.csv",
    "exposure_audit_trace.json",
)
GAP_ATTRIBUTION_ARTIFACTS = (
    "gap_attribution_outcomes_by_month.csv",
    "gap_attribution_outcomes_overall.csv",
    "gap_attribution_feature_exposure_overall.csv",
    "gap_attribution_penalized_feature_delta.csv",
    "gap_attribution_residual_penalized_vs_kept.csv",
    "gap_attribution_trace.json",
)
PRESENTATION_CORE_ARCHIVE_PROFILE = "presentation_core"
PRESENTATION_CORE_REPORT_KEYS = {
    "short_excess_rank_ic_plot_data",
    "short_excess_rank_ic_figure",
    "next_excess_rank_ic_plot_data",
    "next_excess_rank_ic_figure",
    "daily_cumulative_plot_data",
    "daily_cumulative_figure",
    "daily_cumulative_trace",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_artifact(source: Path, destination: Path) -> None:
    require_file(source)
    if source.suffix.lower() != ".svg":
        shutil.copyfile(source, destination)
        return
    destination.write_text(
        "\n".join(line.rstrip() for line in source.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )


def copy_csv_columns(source: Path, destination: Path, columns: tuple[str, ...]) -> int:
    require_file(source)
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        missing = [column for column in columns if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{source}: missing columns: {missing}")
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            rows = 0
            for row in reader:
                writer.writerow({column: row[column] for column in columns})
                rows += 1
    return rows


def copy_artifact_specs(
    root: Path,
    destination: Path,
    specs: tuple[tuple[Path, str], ...],
) -> dict[str, str]:
    sources = {}
    for relative_source, output_name in specs:
        copy_artifact(root / relative_source, destination / output_name)
        sources[output_name] = relative_source.as_posix()
    return sources


def artifact_file_manifest(destination: Path, sources: dict[str, str]) -> dict[str, object]:
    return {
        name: {
            "source": sources[name],
            "sha256": file_sha256(destination / name),
            "bytes": (destination / name).stat().st_size,
        }
        for name in sorted(sources)
    }


def record_named_artifacts(
    *,
    output_dir: Path,
    records_dir: Path,
    record_prefix: str,
    names: tuple[str, ...],
) -> list[Path]:
    archive_dir = records_dir / "backtests" / record_prefix
    copied = []
    for name in names:
        source = output_dir / name
        if source.exists():
            destination = archive_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def record_requested_artifacts(
    *,
    output_dir: Path,
    records_dir: str,
    record_prefix: str,
    names: tuple[str, ...],
) -> list[Path]:
    if not records_dir:
        return []
    return record_named_artifacts(
        output_dir=output_dir,
        records_dir=Path(records_dir),
        record_prefix=record_prefix,
        names=names,
    )


def is_score_risk_sweep(spec: RunSpec) -> bool:
    return spec.kind == "score_risk_sweep"


def is_rolling_validation(spec: RunSpec) -> bool:
    return spec.kind == "alpha_conditioned_rolling_validation"


def is_gap_attribution(spec: RunSpec) -> bool:
    return spec.kind == "gap_risk_attribution"


def is_capacity_audit(spec: RunSpec) -> bool:
    return spec.kind == "capacity_audit"


def is_capacity_acceptance(spec: RunSpec) -> bool:
    return spec.kind == "capacity_acceptance"


def is_exposure_audit(spec: RunSpec) -> bool:
    return spec.kind == "exposure_audit"


def is_feature_audit(spec: RunSpec) -> bool:
    return spec.kind == "feature_audit"


def is_feature_hygiene(spec: RunSpec) -> bool:
    return spec.kind == "feature_hygiene"


def is_pool_internal_analysis(spec: RunSpec) -> bool:
    return spec.pool_internal_analysis_enabled


def is_non_standard_artifact_run(spec: RunSpec) -> bool:
    return (
        is_score_risk_sweep(spec)
        or is_rolling_validation(spec)
        or is_gap_attribution(spec)
        or is_capacity_acceptance(spec)
        or is_capacity_audit(spec)
        or spec.kind == "strategy_acceptance"
        or is_exposure_audit(spec)
        or is_feature_audit(spec)
        or is_feature_hygiene(spec)
    )
