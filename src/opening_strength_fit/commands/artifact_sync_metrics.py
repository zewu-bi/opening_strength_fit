from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import month_periods, write_json
from opening_strength_fit.commands.artifact_sync_remote import (
    fetch_binary_file,
    fetch_remote_file_if_exists,
    remote_file_exists,
)
from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s import RunSpec
from opening_strength_fit.reports import metrics_by_year_from_windows

METRICS_SUFFIX = "_metrics_by_year.csv"


def pull_root_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    pulled: list[Path] = []
    year_path = output_dir / f"{spec.run_id}_metrics_by_year.csv"
    if not fetch_remote_file_if_exists(
        hfcli,
        spec,
        pod_name,
        f"{spec.pvc_dir}/metrics_by_year.csv",
        year_path,
    ):
        return pulled
    pulled.append(year_path)
    month_path = output_dir / f"{spec.run_id}_metrics_by_month.csv"
    if fetch_remote_file_if_exists(
        hfcli,
        spec,
        pod_name,
        f"{spec.pvc_dir}/metrics_by_month.csv",
        month_path,
    ):
        pulled.append(month_path)
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
) -> list[Path]:
    raw_dir = output_dir / spec.run_id / "raw_metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    if spec.test_start_month and spec.test_end_month:
        for month in month_periods(spec.test_start_month, spec.test_end_month):
            remote_path = f"{spec.pvc_dir}/month_{month}/metrics_by_year.csv"
            local_path = raw_dir / f"metrics_by_year_{month}.csv"
            if fetch_remote_file_if_exists(hfcli, spec, pod_name, remote_path, local_path):
                frames.append(pd.read_csv(local_path))
            else:
                missing.append(month)
        if missing:
            if allow_partial and frames:
                print(f"  {spec.run_id}: partial metrics; missing months {missing}")
                return combine_metric_frames(
                    frames,
                    monthly=True,
                    run_id=spec.run_id,
                    output_dir=output_dir,
                )
            raise SystemExit(f"{spec.run_id}: missing shard metrics for months {missing}")
        return combine_metric_frames(
            frames,
            monthly=True,
            run_id=spec.run_id,
            output_dir=output_dir,
        )
    if spec.test_start_year > 0 and spec.test_end_year > 0:
        for year in range(spec.test_start_year, spec.test_end_year + 1):
            remote_path = f"{spec.pvc_dir}/year_{year}/metrics_by_year.csv"
            local_path = raw_dir / f"metrics_by_year_{year}.csv"
            if fetch_remote_file_if_exists(hfcli, spec, pod_name, remote_path, local_path):
                frames.append(pd.read_csv(local_path))
            else:
                missing.append(str(year))
        if missing:
            if allow_partial and frames:
                print(f"  {spec.run_id}: partial metrics; missing years {missing}")
                return combine_metric_frames(
                    frames,
                    monthly=False,
                    run_id=spec.run_id,
                    output_dir=output_dir,
                )
            raise SystemExit(f"{spec.run_id}: missing shard metrics for years {missing}")
        return combine_metric_frames(
            frames,
            monthly=False,
            run_id=spec.run_id,
            output_dir=output_dir,
        )
    raise SystemExit(f"{spec.run_id}: no root metrics found and config has no shard date range")


def pull_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
    *,
    allow_partial: bool = False,
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
    if remote_file_exists(hfcli, spec.namespace, pod_name, combined_remote_path):
        print(f"fetching {combined_remote_path} -> {combined_path}")
        fetch_binary_file(hfcli, spec.namespace, pod_name, combined_remote_path, combined_path)
        fetched_files.append(
            {
                "kind": "combined",
                "remote_path": combined_remote_path,
                "local_path": str(combined_path),
            }
        )
        combined_stats = summarize_predictions(combined_path)
    elif remote_file_exists(hfcli, spec.namespace, pod_name, single_remote_path):
        print(f"fetching {single_remote_path} -> {combined_path}")
        fetch_binary_file(hfcli, spec.namespace, pod_name, single_remote_path, combined_path)
        fetched_files.append(
            {"kind": "single", "remote_path": single_remote_path, "local_path": str(combined_path)}
        )
        combined_stats = summarize_predictions(combined_path)
    elif spec.test_start_month and spec.test_end_month:
        missing_months: list[str] = []
        for month in month_periods(spec.test_start_month, spec.test_end_month):
            remote_candidates = [
                f"{spec.pvc_dir}/month_{month}/predictions_{month}.parquet",
                f"{spec.pvc_dir}/month_{month}/predictions.parquet",
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
                    missing_months.append(month)
                    continue
                raise SystemExit(f"no prediction parquet found for monthly shard {month}")
            local_path = raw_dir / f"predictions_{month}.parquet"
            print(f"fetching {remote_path} -> {local_path}")
            fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
            fetched_files.append(
                {
                    "kind": "monthly",
                    "month": month,
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                }
            )
        if missing_months:
            print(f"  {spec.run_id}: partial predictions; missing months {missing_months}")
        if allow_partial and not fetched_files:
            raise SystemExit(f"{spec.run_id}: no completed monthly prediction shards found")
        combined_stats = combine_prediction_files(raw_dir, combined_path)
    elif spec.test_start_year > 0 and spec.test_end_year > 0:
        missing_years: list[str] = []
        for year in range(spec.test_start_year, spec.test_end_year + 1):
            remote_candidates = [
                f"{spec.pvc_dir}/year_{year}/predictions_{year}.parquet",
                f"{spec.pvc_dir}/year_{year}/predictions.parquet",
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
                    missing_years.append(str(year))
                    continue
                raise SystemExit(f"no prediction parquet found for yearly shard {year}")
            local_path = raw_dir / f"predictions_{year}.parquet"
            print(f"fetching {remote_path} -> {local_path}")
            fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
            fetched_files.append(
                {
                    "kind": "yearly",
                    "year": year,
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                }
            )
        if missing_years:
            print(f"  {spec.run_id}: partial predictions; missing years {missing_years}")
        if allow_partial and not fetched_files:
            raise SystemExit(f"{spec.run_id}: no completed yearly prediction shards found")
        combined_stats = combine_prediction_files(raw_dir, combined_path)
    else:
        raise SystemExit(
            f"{spec.run_id}: no combined/single predictions found and config has no test date range"
        )

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


def collect_run_statuses(runs_dir: Path) -> dict[str, str]:
    statuses = {}
    for path in sorted(runs_dir.glob("*.toml")):
        config = load_toml(path)
        run_id = str(config.get("run", {}).get("id", path.stem))
        statuses[run_id] = str(config.get("run", {}).get("status", "completed"))
    return statuses


def record_metrics(run_id: str, metrics_dir: Path, records_dir: Path) -> Path | None:
    source = metrics_dir / f"{run_id}{METRICS_SUFFIX}"
    if not source.exists():
        return None
    destination = records_dir / "metrics" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
