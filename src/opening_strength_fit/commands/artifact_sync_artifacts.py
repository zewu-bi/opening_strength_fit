from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import (
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
    record_named_artifacts,
)
from opening_strength_fit.commands.artifact_sync_metrics import run_shards
from opening_strength_fit.commands.artifact_sync_remote import (
    fetch_remote_directory_if_exists,
    fetch_remote_file_if_exists,
)
from opening_strength_fit.k8s import RunSpec
from opening_strength_fit.pool_internal_artifacts import record_pool_internal_outputs
from opening_strength_fit.score_variant_eval import (
    summarize_group_metrics as summarize_rolling_group_metrics,
)

DEFAULT_ARTIFACTS_ROOT = Path("output/artifacts")
ARCHIVE_ARTIFACTS_BY_KIND = {
    "capacity_acceptance": CAPACITY_ACCEPTANCE_ARTIFACTS,
    "capacity_audit": CAPACITY_AUDIT_ARTIFACTS,
    "exposure_audit": EXPOSURE_AUDIT_ARTIFACTS,
    "feature_audit": FEATURE_AUDIT_ARTIFACTS,
    "feature_hygiene": FEATURE_HYGIENE_ARTIFACTS,
    "strategy_acceptance": tuple(
        STRATEGY_ACCEPTANCE_ARTIFACTS[index] for index in (0, 3, 4, 7, 10, 11, 12, 13)
    ),
}
ARCHIVE_RENAMES_BY_KIND = {
    "score_risk_sweep": (("score_risk_summary.csv", "{run_id}_summary.csv"),),
    "alpha_conditioned_rolling_validation": (
        ("rolling_summary.csv", "summary.csv"),
        ("rolling_month_summary.csv", "month_summary.csv"),
        ("rolling_trace.json", "trace.json"),
    ),
    "gap_risk_attribution": (
        ("gap_attribution_outcomes_by_month.csv", "outcomes_by_month.csv"),
        ("gap_attribution_outcomes_overall.csv", "outcomes_overall.csv"),
        ("gap_attribution_feature_exposure_overall.csv", "feature_exposure_overall.csv"),
        ("gap_attribution_penalized_feature_delta.csv", "penalized_feature_delta.csv"),
        ("gap_attribution_residual_penalized_vs_kept.csv", "residual_penalized_vs_kept.csv"),
        ("gap_attribution_trace.json", "trace.json"),
    ),
}


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


def artifact_puller(
    artifact_names: tuple[str, ...], artifact_label: str
) -> Callable[..., list[Path]]:
    return partial(
        pull_required_artifact_set,
        artifact_names=artifact_names,
        artifact_label=artifact_label,
    )


pull_score_risk_artifacts = artifact_puller(SCORE_RISK_ARTIFACTS, "score-risk")
pull_gap_attribution_artifacts = artifact_puller(GAP_ATTRIBUTION_ARTIFACTS, "gap-attribution")
pull_capacity_audit_artifacts = artifact_puller(CAPACITY_AUDIT_ARTIFACTS, "capacity-audit")
pull_capacity_acceptance_artifacts = artifact_puller(
    CAPACITY_ACCEPTANCE_ARTIFACTS, "capacity-acceptance"
)
pull_exposure_audit_artifacts = artifact_puller(EXPOSURE_AUDIT_ARTIFACTS, "exposure-audit")
pull_feature_hygiene_artifacts = artifact_puller(FEATURE_HYGIENE_ARTIFACTS, "feature-hygiene")
pull_strategy_acceptance_artifacts = artifact_puller(
    STRATEGY_ACCEPTANCE_ARTIFACTS, "strategy-acceptance"
)


def _pull_shardable_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
    *,
    artifact_names: tuple[str, ...],
    combined_name: str,
    pull_shards: Callable[..., list[Path]],
    artifact_label: str,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        artifact_names,
    )
    if (output_dir / combined_name).exists():
        record_artifact_fetch(spec, output_dir, pulled, missing)
        return pulled
    pulled.extend(pull_shards(hfcli, spec, pod_name, output_dir))
    if not pulled:
        raise SystemExit(f"{spec.run_id}: no {artifact_label} artifacts found under {spec.pvc_dir}")
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


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
    keep_dirs = {
        parts[index + 1]
        for value in payload["report_plots"].values()
        if "reports" in (parts := Path(str(value)).parts)
        if (index := parts.index("reports")) + 1 < len(parts)
    }
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
    shard_kind, _, shards = run_shards(spec)
    if shard_kind != "monthly":
        return []
    for _, label, shard_dirs in shards:
        start_month = label.split("_", 1)[0]
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
                for remote_dir in shard_dirs
            )
            if fetched:
                pulled.append(local_path)
                found = True
        if not found:
            missing_months.append(label)

    combined = combine_shards(
        output_dir,
        months=[label.split("_", 1)[0] for _, label, _ in shards],
        missing_months=missing_months,
    )
    return [*pulled, *combined]


pull_rolling_validation_shards = partial(
    _pull_month_shards,
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


pull_feature_audit_shards = partial(
    _pull_month_shards,
    artifact_names=FEATURE_AUDIT_ARTIFACTS,
    combine_shards=combine_feature_audit_shards,
)


pull_rolling_validation_artifacts = partial(
    _pull_shardable_artifacts,
    artifact_names=ROLLING_VALIDATION_ARTIFACTS,
    combined_name="rolling_summary.csv",
    pull_shards=pull_rolling_validation_shards,
    artifact_label="rolling-validation",
)
pull_feature_audit_artifacts = partial(
    _pull_shardable_artifacts,
    artifact_names=FEATURE_AUDIT_ARTIFACTS,
    combined_name="feature_audit_metrics.csv",
    pull_shards=pull_feature_audit_shards,
    artifact_label="feature-audit",
)
ARTIFACT_PULLERS = {
    "alpha_conditioned_rolling_validation": pull_rolling_validation_artifacts,
    "capacity_acceptance": pull_capacity_acceptance_artifacts,
    "capacity_audit": pull_capacity_audit_artifacts,
    "exposure_audit": pull_exposure_audit_artifacts,
    "feature_audit": pull_feature_audit_artifacts,
    "feature_hygiene": pull_feature_hygiene_artifacts,
    "gap_risk_attribution": pull_gap_attribution_artifacts,
    "score_risk_sweep": pull_score_risk_artifacts,
    "strategy_acceptance": pull_strategy_acceptance_artifacts,
}


def record_lightweight_artifacts(
    spec: RunSpec,
    artifacts_root: Path | None,
    records_dir: Path,
) -> list[Path]:
    output_dir = local_artifact_dir(spec, artifacts_root)
    if spec.pool_internal_analysis_enabled:
        record_prefix = spec.pool_internal_record_prefix or spec.run_id
        return record_pool_internal_outputs(
            output_dir=output_dir,
            records_dir=records_dir,
            record_prefix=record_prefix,
            report_plots=_local_pool_internal_report_plots(spec, output_dir),
            record_subdir=record_prefix,
        )
    names = ARCHIVE_ARTIFACTS_BY_KIND.get(spec.kind, ())
    renames = ARCHIVE_RENAMES_BY_KIND.get(spec.kind)
    if not names and not renames:
        return []
    return record_named_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix="" if spec.kind == "score_risk_sweep" else spec.run_id,
        names=names,
        renames=tuple(
            (name, destination.format(run_id=spec.run_id)) for name, destination in renames
        )
        if renames
        else (),
    )


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
