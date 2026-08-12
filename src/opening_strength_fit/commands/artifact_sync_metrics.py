from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import month_window_periods, write_json
from opening_strength_fit.commands.artifact_sync_remote import (
    fetch_binary_file,
    fetch_remote_file_if_exists,
    remote_file_exists,
)
from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s import RunSpec
from opening_strength_fit.pvc_layout import (
    rolling_shard_dir_candidates,
    yearly_shard_dir_candidates,
)
from opening_strength_fit.reports import metrics_by_year_from_windows

METRICS_SUFFIX = "_metrics_by_year.csv"
MONTHLY_METRICS_SUFFIX = "_metrics_by_month.csv"
METRIC_FILES = (
    ("metrics_by_year.csv", METRICS_SUFFIX),
    ("metrics_by_month.csv", MONTHLY_METRICS_SUFFIX),
)
DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR = (
    "/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1"
)


def run_shards(
    spec: RunSpec,
) -> tuple[str, str, list[tuple[object, str, tuple[str, ...]]]]:
    if spec.test_start_month and spec.test_end_month:
        shards = []
        for start_month, end_month in month_window_periods(
            spec.test_start_month,
            spec.test_end_month,
            test_months=spec.test_months,
            stride_months=spec.test_stride_months,
        ):
            label = start_month if start_month == end_month else f"{start_month}_{end_month}"
            shards.append(
                (
                    label,
                    label,
                    rolling_shard_dir_candidates(
                        start_month,
                        end_month,
                        preferred_layout=spec.output_layout,
                    ),
                )
            )
        return "monthly", "month", shards
    if spec.test_start_year > 0 and spec.test_end_year > 0:
        return (
            "yearly",
            "year",
            [
                (
                    year,
                    str(year),
                    yearly_shard_dir_candidates(year, preferred_layout=spec.output_layout),
                )
                for year in range(spec.test_start_year, spec.test_end_year + 1)
            ],
        )
    return "", "", []


def pull_root_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    pulled: list[Path] = []
    for remote_name, suffix in METRIC_FILES:
        local_path = output_dir / f"{spec.run_id}{suffix}"
        if fetch_remote_file_if_exists(
            hfcli, spec, pod_name, f"{spec.pvc_dir}/{remote_name}", local_path
        ):
            pulled.append(local_path)
        elif not pulled:
            return []
    return pulled


def combine_metric_frames(
    frames: list[pd.DataFrame],
    *,
    monthly: bool,
    run_id: str,
    output_dir: Path,
) -> list[Path]:
    if not frames:
        raise SystemExit(f"{run_id}: no shard metric CSVs found")
    pulled: list[Path] = []
    metrics = pd.concat(frames, ignore_index=True)
    if monthly:
        sort_columns = [
            column for column in ("test_year", "test_month") if column in metrics.columns
        ]
        if sort_columns:
            metrics = metrics.sort_values(sort_columns)
        month_path = output_dir / f"{run_id}_metrics_by_month.csv"
        metrics.to_csv(month_path, index=False)
        pulled.append(month_path)
        metrics = metrics_by_year_from_windows(metrics)
    elif "test_year" in metrics.columns:
        metrics = metrics.sort_values("test_year")
    year_path = output_dir / f"{run_id}_metrics_by_year.csv"
    metrics.to_csv(year_path, index=False)
    return [year_path, *pulled]


def pull_shard_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
    *,
    allow_partial: bool = False,
    raw_root: Path | None = None,
) -> list[Path]:
    raw_dir = (raw_root or output_dir) / spec.run_id / "raw_metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    shard_kind, period_name, shards = run_shards(spec)
    if not shards:
        raise SystemExit(f"{spec.run_id}: no root metrics found and config has no shard date range")
    for _, label, shard_dirs in shards:
        local_path = raw_dir / f"metrics_by_year_{label}.csv"
        fetched = any(
            fetch_remote_file_if_exists(
                hfcli,
                spec,
                pod_name,
                f"{spec.pvc_dir}/{shard_dir}/metrics_by_year.csv",
                local_path,
            )
            for shard_dir in shard_dirs
        )
        if fetched:
            frames.append(pd.read_csv(local_path))
        else:
            missing.append(label)
    if missing:
        if allow_partial and frames:
            print(f"  {spec.run_id}: partial metrics; missing {period_name}s {missing}")
        else:
            raise SystemExit(f"{spec.run_id}: missing shard metrics for {period_name}s {missing}")
    return combine_metric_frames(
        frames,
        monthly=shard_kind == "monthly",
        run_id=spec.run_id,
        output_dir=output_dir,
    )


def pull_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
    *,
    allow_partial: bool = False,
    raw_root: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled = pull_root_metrics(hfcli, spec, pod_name, output_dir)
    if pulled:
        return pulled
    return pull_shard_metrics(
        hfcli,
        spec,
        pod_name,
        output_dir,
        allow_partial=allow_partial,
        raw_root=raw_root,
    )


def summarize_predictions(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    date_index = pd.to_datetime(frame["date"], errors="coerce")
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "dates": int(date_index.nunique()),
        "symbols": int(frame["symbol"].astype(str).nunique()),
        "date_min": str(date_index.min().date()),
        "date_max": str(date_index.max().date()),
    }
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        summary["timestamp_min"] = str(timestamp.min())
        summary["timestamp_max"] = str(timestamp.max())
    return summary


def combine_prediction_files(raw_dir: Path, output_path: Path) -> dict[str, object]:
    frames = []
    file_rows = {}
    for path in sorted(raw_dir.glob("predictions_*.parquet")):
        frame = pd.read_parquet(path)
        frames.append(frame)
        file_rows[path.name] = int(len(frame))
    if not frames:
        raise SystemExit(f"no local prediction parquet files found in {raw_dir}")
    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [column for column in ["date", "symbol", "timestamp"] if column in combined]
    if sort_cols:
        combined = combined.sort_values(sort_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return summarize_predictions(output_path) | {"file_rows": file_rows}


def fetch_predictions(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
    *,
    allow_partial: bool = False,
) -> Path:
    output_dir = output_root / spec.run_id
    raw_dir = output_dir / "raw"
    combined_path = output_dir / "predictions_all.parquet"
    trace_path = output_dir / "fetch_trace.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched_files = []
    combined_stats: dict[str, object]
    combined_remote_path = f"{spec.pvc_dir}/predictions_all.parquet"
    single_remote_path = f"{spec.pvc_dir}/predictions.parquet"
    for kind, remote_path in (
        ("combined", combined_remote_path),
        ("single", single_remote_path),
    ):
        if not remote_file_exists(hfcli, spec.namespace, pod_name, remote_path):
            continue
        print(f"fetching {remote_path} -> {combined_path}")
        fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, combined_path)
        fetched_files.append(
            {"kind": kind, "remote_path": remote_path, "local_path": str(combined_path)}
        )
        combined_stats = summarize_predictions(combined_path)
        break
    else:
        shard_kind, period_name, shards = run_shards(spec)
        if not shards:
            raise SystemExit(
                f"{spec.run_id}: no combined/single predictions found and config has no test date range"
            )
        missing: list[str] = []
        for period, label, shard_dirs in shards:
            remote_candidates = [
                f"{spec.pvc_dir}/{shard_dir}/{name}"
                for shard_dir in shard_dirs
                for name in ("predictions.parquet", f"predictions_{label}.parquet")
            ]
            remote_path = next(
                (
                    candidate
                    for candidate in remote_candidates
                    if remote_file_exists(hfcli, spec.namespace, pod_name, candidate)
                ),
                "",
            )
            if not remote_path:
                if allow_partial:
                    missing.append(label)
                    continue
                raise SystemExit(f"no prediction parquet found for {shard_kind} shard {label}")
            local_path = raw_dir / f"predictions_{label}.parquet"
            print(f"fetching {remote_path} -> {local_path}")
            fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
            fetched_files.append(
                {
                    "kind": shard_kind,
                    period_name: period,
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                }
            )
        if missing:
            print(f"  {spec.run_id}: partial predictions; missing {period_name}s {missing}")
        if allow_partial and not fetched_files:
            raise SystemExit(f"{spec.run_id}: no completed {shard_kind} prediction shards found")
        combined_stats = combine_prediction_files(raw_dir, combined_path)

    trace = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "run_id": spec.run_id,
        "namespace": spec.namespace,
        "pvc_dir": spec.pvc_dir,
        "local_output_dir": str(output_dir),
        "combined_output": str(combined_path),
        "files": fetched_files,
        "combined_stats": combined_stats,
    }
    write_json(trace_path, trace, ensure_ascii=True)
    return combined_path


def next_close_label_years(spec: RunSpec) -> list[int]:
    if spec.test_start_month and spec.test_end_month:
        start_year = int(spec.test_start_month.split("-", 1)[0])
        end_year = int(spec.test_end_month.split("-", 1)[0])
    else:
        start_year = spec.test_start_year
        end_year = spec.test_end_year
    if start_year <= 0 or end_year <= 0:
        return []
    return list(range(start_year, end_year + 1))


def next_close_label_output_dir(output_root: Path, years: list[int]) -> Path:
    if not years:
        raise ValueError("years must not be empty")
    label = str(years[0]) if len(years) == 1 else f"{years[0]}_{years[-1]}"
    return output_root / f"next_close_labels_{label}"


def pull_next_close_labels(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
    *,
    label_pvc_dir: str = DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR,
) -> list[Path]:
    years = next_close_label_years(spec)
    if not years:
        raise SystemExit(f"{spec.run_id}: config has no test years for next-close labels")
    output_dir = next_close_label_output_dir(output_root, years)
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []
    missing: list[int] = []
    for year in years:
        name = f"opening_{year}_next_close_labels_v1.parquet"
        local_path = output_dir / name
        if fetch_remote_file_if_exists(
            hfcli,
            spec,
            pod_name,
            f"{label_pvc_dir.rstrip('/')}/{name}",
            local_path,
        ):
            pulled.append(local_path)
        else:
            missing.append(year)
    if missing:
        raise SystemExit(f"{spec.run_id}: missing next-close labels for years {missing}")
    return pulled


def collect_run_statuses(runs_dir: Path) -> dict[str, str]:
    statuses = {}
    for path in sorted(runs_dir.glob("*.toml")):
        run = load_toml(path).get("run", {})
        statuses[str(run.get("id", path.stem))] = str(run.get("status", "completed"))
    return statuses


def record_metrics(run_id: str, metrics_dir: Path, records_dir: Path) -> list[Path]:
    records: list[Path] = []
    for _, suffix in METRIC_FILES:
        source = metrics_dir / f"{run_id}{suffix}"
        if not source.exists():
            continue
        destination = records_dir / "metrics" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        records.append(destination)
    return records
