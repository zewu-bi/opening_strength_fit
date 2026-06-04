from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import (
    month_periods,
    write_artifact_fetch_trace,
    write_json,
)
from opening_strength_fit.commands.alpha_conditioned_rolling_validation import (
    summarize_group_metrics as summarize_rolling_group_metrics,
)
from opening_strength_fit.commands.artifact_sync_remote import fetch_remote_file_if_exists
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
GAP_ATTRIBUTION_ARTIFACTS = (
    "gap_attribution_outcomes_by_month.csv",
    "gap_attribution_outcomes_overall.csv",
    "gap_attribution_feature_exposure_overall.csv",
    "gap_attribution_penalized_feature_delta.csv",
    "gap_attribution_residual_penalized_vs_kept.csv",
    "gap_attribution_trace.json",
)


def is_score_risk_sweep(spec: RunSpec) -> bool:
    return spec.kind == "score_risk_sweep"


def is_rolling_validation(spec: RunSpec) -> bool:
    return spec.kind == "alpha_conditioned_rolling_validation"


def is_gap_attribution(spec: RunSpec) -> bool:
    return spec.kind == "gap_risk_attribution"


def is_non_standard_artifact_run(spec: RunSpec) -> bool:
    return is_score_risk_sweep(spec) or is_rolling_validation(spec) or is_gap_attribution(spec)


def pull_artifact_set(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
    artifact_names: tuple[str, ...],
) -> tuple[Path, list[Path], list[str]]:
    output_dir = output_root / spec.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []
    missing: list[str] = []
    for name in artifact_names:
        remote_path = f"{spec.pvc_dir}/{name}"
        local_path = output_dir / name
        if fetch_remote_file_if_exists(hfcli, spec, pod_name, remote_path, local_path):
            pulled.append(local_path)
        else:
            missing.append(name)
    return output_dir, pulled, missing


def record_artifact_fetch(
    spec: RunSpec,
    output_dir: Path,
    pulled: list[Path],
    missing: list[str],
) -> Path:
    return write_artifact_fetch_trace(
        output_dir,
        fetched_at_utc=datetime.now(UTC).isoformat(),
        run_id=spec.run_id,
        namespace=spec.namespace,
        pvc_dir=spec.pvc_dir,
        files=pulled,
        missing=missing,
    )


def pull_score_risk_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        SCORE_RISK_ARTIFACTS,
    )
    if not pulled:
        raise SystemExit(f"{spec.run_id}: no score-risk artifacts found under {spec.pvc_dir}")
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


def pull_rolling_validation_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        ROLLING_VALIDATION_ARTIFACTS,
    )

    if (output_dir / "rolling_summary.csv").exists():
        record_artifact_fetch(spec, output_dir, pulled, missing)
        return pulled

    shard_paths = pull_rolling_validation_shards(
        hfcli,
        spec,
        pod_name,
        output_dir,
    )
    if shard_paths:
        pulled.extend(shard_paths)
    if not pulled:
        raise SystemExit(
            f"{spec.run_id}: no rolling-validation artifacts found under {spec.pvc_dir}"
        )
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


def pull_gap_attribution_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        GAP_ATTRIBUTION_ARTIFACTS,
    )
    if not pulled:
        raise SystemExit(f"{spec.run_id}: no gap-attribution artifacts found under {spec.pvc_dir}")
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


def combine_rolling_validation_shards(
    output_dir: Path,
    *,
    months: list[str],
    missing_months: list[str],
) -> list[Path]:
    group_frames = []
    for month in months:
        path = output_dir / f"month_{month}" / "rolling_group_metrics.csv"
        if path.exists():
            group_frames.append(pd.read_csv(path))
    if not group_frames:
        return []

    group_metrics = pd.concat(group_frames, ignore_index=True)
    group_metrics["risk_model"] = group_metrics["risk_model"].fillna("").astype(str)
    month_summary, summary = summarize_rolling_group_metrics(group_metrics)

    group_path = output_dir / "rolling_group_metrics.csv"
    month_path = output_dir / "rolling_month_summary.csv"
    summary_path = output_dir / "rolling_summary.csv"
    trace_path = output_dir / "rolling_trace.json"
    group_metrics.to_csv(group_path, index=False)
    month_summary.to_csv(month_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_json(
        trace_path,
        {
            "combined_at_utc": datetime.now(UTC).isoformat(),
            "months": months,
            "missing_months": missing_months,
            "outputs": {
                "group_metrics": str(group_path),
                "month_summary": str(month_path),
                "summary": str(summary_path),
            },
        },
        ensure_ascii=True,
    )
    return [group_path, month_path, summary_path, trace_path]


def pull_rolling_validation_shards(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    if not spec.test_start_month or not spec.test_end_month:
        return []

    pulled: list[Path] = []
    missing_months: list[str] = []
    months = month_periods(spec.test_start_month, spec.test_end_month)
    for month in months:
        shard_dir = output_dir / f"month_{month}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        found = False
        for name in ROLLING_VALIDATION_SHARD_ARTIFACTS:
            remote_path = f"{spec.pvc_dir}/month_{month}/{name}"
            local_path = shard_dir / name
            if fetch_remote_file_if_exists(hfcli, spec, pod_name, remote_path, local_path):
                pulled.append(local_path)
                found = True
        if not found:
            missing_months.append(month)

    combined = combine_rolling_validation_shards(
        output_dir,
        months=months,
        missing_months=missing_months,
    )
    return [*pulled, *combined]


def record_artifact_file(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def record_lightweight_artifacts(
    spec: RunSpec,
    artifacts_root: Path,
    records_dir: Path,
) -> list[Path]:
    output_dir = artifacts_root / spec.run_id
    backtests_dir = records_dir / "backtests"
    if is_score_risk_sweep(spec):
        records = [
            (
                output_dir / "score_risk_summary.csv",
                backtests_dir / f"{spec.run_id}_summary.csv",
            ),
        ]
    elif is_rolling_validation(spec):
        records = [
            (
                output_dir / "rolling_summary.csv",
                backtests_dir / f"{spec.run_id}_summary.csv",
            ),
            (
                output_dir / "rolling_month_summary.csv",
                backtests_dir / f"{spec.run_id}_month_summary.csv",
            ),
            (
                output_dir / "rolling_trace.json",
                backtests_dir / f"{spec.run_id}_trace.json",
            ),
        ]
    elif is_gap_attribution(spec):
        records = [
            (
                output_dir / "gap_attribution_outcomes_by_month.csv",
                backtests_dir / f"{spec.run_id}_outcomes_by_month.csv",
            ),
            (
                output_dir / "gap_attribution_outcomes_overall.csv",
                backtests_dir / f"{spec.run_id}_outcomes_overall.csv",
            ),
            (
                output_dir / "gap_attribution_feature_exposure_overall.csv",
                backtests_dir / f"{spec.run_id}_feature_exposure_overall.csv",
            ),
            (
                output_dir / "gap_attribution_penalized_feature_delta.csv",
                backtests_dir / f"{spec.run_id}_penalized_feature_delta.csv",
            ),
            (
                output_dir / "gap_attribution_residual_penalized_vs_kept.csv",
                backtests_dir / f"{spec.run_id}_residual_penalized_vs_kept.csv",
            ),
            (
                output_dir / "gap_attribution_trace.json",
                backtests_dir / f"{spec.run_id}_trace.json",
            ),
        ]
    else:
        return []

    return [
        path
        for source, destination in records
        if (path := record_artifact_file(source, destination)) is not None
    ]
