from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import (
    month_window_periods,
    write_artifact_fetch_trace,
    write_json,
)
from opening_strength_fit.artifact_catalog import (
    CAPACITY_ACCEPTANCE_ARTIFACTS,
    CAPACITY_AUDIT_ARTIFACTS,
    EXPOSURE_AUDIT_ARTIFACTS,
    FEATURE_AUDIT_ARTIFACTS,
    FEATURE_AUDIT_COMBINED_CSVS,
    FEATURE_HYGIENE_ARTIFACTS,
    GAP_ATTRIBUTION_ARTIFACTS,
    PRESENTATION_CORE_ARCHIVE_PROFILE,
    PRESENTATION_CORE_REPORT_KEYS,
    ROLLING_VALIDATION_ARTIFACTS,
    ROLLING_VALIDATION_SHARD_ARTIFACTS,
    SCORE_RISK_ARTIFACTS,
    STRATEGY_ACCEPTANCE_ARTIFACTS,
    is_capacity_acceptance,
    is_capacity_audit,
    is_exposure_audit,
    is_feature_audit,
    is_feature_hygiene,
    is_gap_attribution,
    is_pool_internal_analysis,
    is_rolling_validation,
    is_score_risk_sweep,
)
from opening_strength_fit.commands.artifact_sync_remote import (
    fetch_remote_directory_if_exists,
    fetch_remote_file_if_exists,
)
from opening_strength_fit.k8s import RunSpec
from opening_strength_fit.pool_internal_artifacts import record_pool_internal_outputs
from opening_strength_fit.pvc_layout import rolling_shard_dir_candidates
from opening_strength_fit.score_variant_eval import (
    summarize_group_metrics as summarize_rolling_group_metrics,
)

DEFAULT_ARTIFACTS_ROOT = Path("output/artifacts")


def local_artifact_dir(spec: RunSpec, output_root: Path | None) -> Path:
    if output_root is None and spec.local_dir:
        return Path(spec.local_dir)
    root = DEFAULT_ARTIFACTS_ROOT if output_root is None else output_root
    return root / spec.run_id


def pull_artifact_set(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
    artifact_names: tuple[str, ...],
) -> tuple[Path, list[Path], list[str]]:
    output_dir = local_artifact_dir(spec, output_root)
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


def pull_required_artifact_set(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
    artifact_names: tuple[str, ...],
    artifact_label: str,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli, spec, pod_name, output_root, artifact_names
    )
    if not pulled:
        raise SystemExit(f"{spec.run_id}: no {artifact_label} artifacts found under {spec.pvc_dir}")
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


def pull_score_risk_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli, spec, pod_name, output_root, SCORE_RISK_ARTIFACTS, "score-risk"
    )


def pull_rolling_validation_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
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
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli, spec, pod_name, output_root, GAP_ATTRIBUTION_ARTIFACTS, "gap-attribution"
    )


def pull_capacity_audit_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli, spec, pod_name, output_root, CAPACITY_AUDIT_ARTIFACTS, "capacity-audit"
    )


def pull_capacity_acceptance_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        CAPACITY_ACCEPTANCE_ARTIFACTS,
        "capacity-acceptance",
    )


def pull_exposure_audit_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli, spec, pod_name, output_root, EXPOSURE_AUDIT_ARTIFACTS, "exposure-audit"
    )


def pull_feature_audit_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        FEATURE_AUDIT_ARTIFACTS,
    )
    if (output_dir / "feature_audit_metrics.csv").exists():
        record_artifact_fetch(spec, output_dir, pulled, missing)
        return pulled

    shard_paths = pull_feature_audit_shards(hfcli, spec, pod_name, output_dir)
    if shard_paths:
        pulled.extend(shard_paths)
    if not pulled:
        raise SystemExit(f"{spec.run_id}: no feature-audit artifacts found under {spec.pvc_dir}")
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


def pull_feature_hygiene_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    return pull_required_artifact_set(
        hfcli, spec, pod_name, output_root, FEATURE_HYGIENE_ARTIFACTS, "feature-hygiene"
    )


def pull_pool_internal_analysis_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    if not spec.pool_internal_analysis_enabled:
        return []
    output_dir = local_artifact_dir(spec, output_root)
    if not fetch_remote_directory_if_exists(
        hfcli,
        spec,
        pod_name,
        spec.pool_internal_analysis_dir,
        output_dir,
    ):
        raise SystemExit(
            f"{spec.run_id}: no pool-internal analysis artifacts found under "
            f"{spec.pool_internal_analysis_dir}"
        )
    _prune_pool_internal_artifacts_for_archive_profile(spec, output_dir)
    pulled = sorted(path for path in output_dir.rglob("*") if path.is_file())
    record_artifact_fetch(spec, output_dir, pulled, [])
    return pulled


def _prune_pool_internal_artifacts_for_archive_profile(
    spec: RunSpec,
    output_dir: Path,
) -> None:
    if spec.pool_internal_archive_profile != PRESENTATION_CORE_ARCHIVE_PROFILE:
        return
    trace_path = output_dir / "pool_internal_trace.json"
    if not trace_path.exists():
        return
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    report_plots = payload.get("report_plots", {})
    if not isinstance(report_plots, dict):
        return
    payload["report_plots"] = {
        key: value for key, value in report_plots.items() if key in PRESENTATION_CORE_REPORT_KEYS
    }
    keep_dirs = set()
    for value in payload["report_plots"].values():
        parts = Path(str(value)).parts
        if "reports" in parts:
            index = parts.index("reports")
            if index + 1 < len(parts):
                keep_dirs.add(parts[index + 1])
    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        for path in reports_dir.iterdir():
            if path.is_dir() and path.name not in keep_dirs:
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
    write_json(trace_path, payload, ensure_ascii=True)


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


def _pull_month_shards(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
    *,
    artifact_names: tuple[str, ...],
    combine_shards: Callable[..., list[Path]],
) -> list[Path]:
    if not spec.test_start_month or not spec.test_end_month:
        return []

    pulled: list[Path] = []
    missing_months: list[str] = []
    windows = month_window_periods(
        spec.test_start_month,
        spec.test_end_month,
        test_months=spec.test_months,
        stride_months=spec.test_stride_months,
    )
    labels = [start if start == end else f"{start}_{end}" for start, end in windows]
    for (start_month, end_month), label in zip(windows, labels, strict=True):
        shard_dir = output_dir / f"month_{start_month}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        found = False
        for name in artifact_names:
            local_path = shard_dir / name
            fetched = any(
                fetch_remote_file_if_exists(
                    hfcli,
                    spec,
                    pod_name,
                    f"{spec.pvc_dir}/{remote_dir}/{name}",
                    local_path,
                )
                for remote_dir in rolling_shard_dir_candidates(
                    start_month,
                    end_month,
                    preferred_layout=spec.output_layout,
                )
            )
            if fetched:
                pulled.append(local_path)
                found = True
        if not found:
            missing_months.append(label)

    combined = combine_shards(
        output_dir,
        months=[start for start, _ in windows],
        missing_months=missing_months,
    )
    return [*pulled, *combined]


def pull_rolling_validation_shards(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    return _pull_month_shards(
        hfcli,
        spec,
        pod_name,
        output_dir,
        artifact_names=ROLLING_VALIDATION_SHARD_ARTIFACTS,
        combine_shards=combine_rolling_validation_shards,
    )


def combine_feature_audit_shards(
    output_dir: Path,
    *,
    months: list[str],
    missing_months: list[str],
) -> list[Path]:
    combined_paths: list[Path] = []
    outputs: dict[str, str] = {}
    for name in FEATURE_AUDIT_COMBINED_CSVS:
        frames = []
        for month in months:
            path = output_dir / f"month_{month}" / name
            if path.exists():
                frame = pd.read_csv(path)
                if "test_month" not in frame.columns:
                    frame.insert(0, "test_month", month)
                frames.append(frame)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        destination = output_dir / name
        combined.to_csv(destination, index=False)
        combined_paths.append(destination)
        outputs[name.removesuffix(".csv")] = str(destination)

    trace_path = output_dir / "feature_audit_trace.json"
    write_json(
        trace_path,
        {
            "combined_at_utc": datetime.now(UTC).isoformat(),
            "months": months,
            "missing_months": missing_months,
            "outputs": outputs,
        },
        ensure_ascii=True,
    )
    combined_paths.append(trace_path)
    return combined_paths


def pull_feature_audit_shards(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    return _pull_month_shards(
        hfcli,
        spec,
        pod_name,
        output_dir,
        artifact_names=FEATURE_AUDIT_ARTIFACTS,
        combine_shards=combine_feature_audit_shards,
    )


def record_artifact_file(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def record_lightweight_artifacts(
    spec: RunSpec,
    artifacts_root: Path | None,
    records_dir: Path,
) -> list[Path]:
    output_dir = local_artifact_dir(spec, artifacts_root)
    backtests_dir = records_dir / "backtests"
    if is_pool_internal_analysis(spec):
        report_plots = _local_pool_internal_report_plots(
            spec,
            output_dir,
        )
        record_prefix = spec.pool_internal_record_prefix or spec.run_id
        return record_pool_internal_outputs(
            output_dir=output_dir,
            records_dir=records_dir,
            record_prefix=record_prefix,
            report_plots=report_plots,
            record_subdir=record_prefix,
        )
    if is_score_risk_sweep(spec):
        records = [
            (
                output_dir / "score_risk_summary.csv",
                backtests_dir / f"{spec.run_id}_summary.csv",
            ),
        ]
    elif is_rolling_validation(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [
            (
                output_dir / "rolling_summary.csv",
                archive_dir / "summary.csv",
            ),
            (
                output_dir / "rolling_month_summary.csv",
                archive_dir / "month_summary.csv",
            ),
            (
                output_dir / "rolling_trace.json",
                archive_dir / "trace.json",
            ),
        ]
    elif is_gap_attribution(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [
            (
                output_dir / "gap_attribution_outcomes_by_month.csv",
                archive_dir / "outcomes_by_month.csv",
            ),
            (
                output_dir / "gap_attribution_outcomes_overall.csv",
                archive_dir / "outcomes_overall.csv",
            ),
            (
                output_dir / "gap_attribution_feature_exposure_overall.csv",
                archive_dir / "feature_exposure_overall.csv",
            ),
            (
                output_dir / "gap_attribution_penalized_feature_delta.csv",
                archive_dir / "penalized_feature_delta.csv",
            ),
            (
                output_dir / "gap_attribution_residual_penalized_vs_kept.csv",
                archive_dir / "residual_penalized_vs_kept.csv",
            ),
            (
                output_dir / "gap_attribution_trace.json",
                archive_dir / "trace.json",
            ),
        ]
    elif is_exposure_audit(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [(output_dir / name, archive_dir / name) for name in EXPOSURE_AUDIT_ARTIFACTS]
    elif is_capacity_audit(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [(output_dir / name, archive_dir / name) for name in CAPACITY_AUDIT_ARTIFACTS]
    elif is_capacity_acceptance(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [
            (output_dir / name, archive_dir / name) for name in CAPACITY_ACCEPTANCE_ARTIFACTS
        ]
    elif spec.kind == "strategy_acceptance":
        archive_dir = backtests_dir / spec.run_id
        names = (STRATEGY_ACCEPTANCE_ARTIFACTS[i] for i in (0, 3, 4, 7, 10, 11, 12, 13))
        records = [(output_dir / name, archive_dir / name) for name in names]
    elif is_feature_audit(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [
            (
                output_dir / "feature_audit_metrics.csv",
                archive_dir / "feature_audit_metrics.csv",
            ),
            (
                output_dir / "feature_audit_permutation.csv",
                archive_dir / "feature_audit_permutation.csv",
            ),
            (
                output_dir / "feature_importance.csv",
                archive_dir / "feature_importance.csv",
            ),
            (
                output_dir / "feature_group_importance.csv",
                archive_dir / "feature_group_importance.csv",
            ),
            (
                output_dir / "feature_audit_trace.json",
                archive_dir / "feature_audit_trace.json",
            ),
        ]
    elif is_feature_hygiene(spec):
        archive_dir = backtests_dir / spec.run_id
        records = [(output_dir / name, archive_dir / name) for name in FEATURE_HYGIENE_ARTIFACTS]
    else:
        return []

    return [
        path
        for source, destination in records
        if (path := record_artifact_file(source, destination)) is not None
    ]


def _local_pool_internal_report_plots(spec: RunSpec, output_dir: Path) -> dict[str, str]:
    trace_path = output_dir / "pool_internal_trace.json"
    if not trace_path.exists():
        return {}
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    remote_plots = payload.get("report_plots", {})
    if not isinstance(remote_plots, dict):
        return {}
    local_plots: dict[str, str] = {}
    remote_root = spec.pool_internal_analysis_dir.rstrip("/")
    for key, raw_path in remote_plots.items():
        remote_path = str(raw_path)
        if remote_path.startswith(f"{remote_root}/"):
            local_path = output_dir / remote_path[len(remote_root) + 1 :]
        elif remote_path == remote_root:
            local_path = output_dir
        else:
            matches = sorted(output_dir.rglob(Path(remote_path).name))
            local_path = matches[0] if matches else output_dir / Path(remote_path).name
        if local_path.exists():
            local_plots[str(key)] = str(local_path)
    return local_plots
