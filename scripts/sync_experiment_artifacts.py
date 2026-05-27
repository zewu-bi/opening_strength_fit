from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s import DEFAULT_IMAGE, RunSpec
from opening_strength_fit.k8s import command_succeeds, delete_temp_pod, ensure_temp_pod
from opening_strength_fit.k8s import load_run_spec, run_command
from opening_strength_fit.training import _metrics_by_year_from_windows


METRICS_SUFFIX = "_metrics_by_year.csv"


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=/pvc/path")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.rstrip("/")
    if not label or not path:
        raise argparse.ArgumentTypeError("--run needs a non-empty label and path")
    return label, path


def ad_hoc_run_spec(label: str, pvc_dir: str, namespace: str) -> RunSpec:
    return RunSpec(
        run_id=label,
        pvc_dir=pvc_dir,
        namespace=namespace,
        pvc="bizewu-private-data",
        mount_path="/mnt/output",
        pull_secret="highfort",
        image=DEFAULT_IMAGE,
        test_start_year=0,
        test_end_year=0,
    )


def validate_specs(specs: list[RunSpec]) -> None:
    if not specs:
        return
    first = specs[0]
    for spec in specs[1:]:
        if (
            spec.namespace != first.namespace
            or spec.pvc != first.pvc
            or spec.mount_path != first.mount_path
            or spec.pull_secret != first.pull_secret
            or spec.image != first.image
        ):
            raise SystemExit(
                "All synced runs must share namespace, pvc, mount_path, "
                "image_pull_secret, and helper image."
            )


def remote_file_exists(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
) -> bool:
    return command_succeeds(
        [
            hfcli,
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod_name,
            "--",
            "/bin/sh",
            "-lc",
            f"test -f '{remote_path}'",
        ]
    )


def fetch_binary_file(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        hfcli,
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "--",
        "cat",
        remote_path,
    ]
    with local_path.open("wb") as file:
        run_command(command, stdout=file)


def fetch_remote_file_if_exists(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    remote_path: str,
    local_path: Path,
) -> bool:
    if not remote_file_exists(hfcli, spec.namespace, pod_name, remote_path):
        return False
    print(f"fetching {remote_path} -> {local_path}")
    fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
    return True


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
        metrics = _metrics_by_year_from_windows(metrics)
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
            raise SystemExit(f"{spec.run_id}: missing shard metrics for years {missing}")
        return combine_metric_frames(
            frames,
            monthly=False,
            run_id=spec.run_id,
            output_dir=output_dir,
        )
    raise SystemExit(
        f"{spec.run_id}: no root metrics found and config has no shard date range"
    )


def pull_metrics(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled = pull_root_metrics(hfcli, spec, pod_name, output_dir)
    if pulled:
        return pulled
    return pull_shard_metrics(hfcli, spec, pod_name, output_dir)


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


def month_periods(start_month: str, end_month: str) -> list[str]:
    return [str(month) for month in pd.period_range(start_month, end_month, freq="M")]


def fetch_predictions(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path,
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
            {"kind": "combined", "remote_path": combined_remote_path, "local_path": str(combined_path)}
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
                raise SystemExit(f"no prediction parquet found for monthly shard {month}")
            local_path = raw_dir / f"predictions_{month}.parquet"
            print(f"fetching {remote_path} -> {local_path}")
            fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
            fetched_files.append(
                {"kind": "monthly", "month": month, "remote_path": remote_path, "local_path": str(local_path)}
            )
        combined_stats = combine_prediction_files(raw_dir, combined_path)
    elif spec.test_start_year > 0 and spec.test_end_year > 0:
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
                raise SystemExit(f"no prediction parquet found for yearly shard {year}")
            local_path = raw_dir / f"predictions_{year}.parquet"
            print(f"fetching {remote_path} -> {local_path}")
            fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
            fetched_files.append(
                {"kind": "yearly", "year": year, "remote_path": remote_path, "local_path": str(local_path)}
            )
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
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync K8s PVC metrics/predictions and archive lightweight records."
    )
    parser.add_argument("--config", action="append", type=load_run_spec)
    parser.add_argument("--run", action="append", type=parse_run)
    parser.add_argument("--namespace", default="bizewu")
    parser.add_argument("--hfcli", default="hfcli")
    parser.add_argument("--pod", default="")
    parser.add_argument("--timeout", default="300s")
    parser.add_argument("--metrics-dir", default="output/k8s/metrics")
    parser.add_argument("--predictions-root", default="output/predictions")
    parser.add_argument("--records-dir", default="experiments/results")
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--metrics", action="store_true", help="Fetch metrics CSVs.")
    parser.add_argument("--predictions", action="store_true", help="Fetch prediction parquet.")
    parser.add_argument("--record", action="store_true", help="Archive fetched metrics.")
    parser.add_argument("--all", action="store_true", help="Fetch metrics, fetch predictions, and archive metrics.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs: list[RunSpec] = []
    if args.config:
        specs.extend(args.config)
    if args.run:
        specs.extend(ad_hoc_run_spec(label, pvc_dir, args.namespace) for label, pvc_dir in args.run)
    if not specs:
        raise SystemExit("pass at least one --config or --run")
    validate_specs(specs)

    no_action = not (args.metrics or args.predictions or args.record or args.all)
    fetch_metrics_flag = args.all or args.metrics or no_action
    fetch_predictions_flag = args.all or args.predictions or no_action
    record_flag = args.all or args.record or no_action

    metrics_dir = Path(args.metrics_dir)
    predictions_root = Path(args.predictions_root)
    records_dir = Path(args.records_dir)
    needs_pod = fetch_metrics_flag or fetch_predictions_flag
    pod_name = args.pod
    created_temp_pod = False
    if needs_pod and not pod_name:
        pod_name = ensure_temp_pod(
            args.hfcli,
            specs[0],
            args.timeout,
            "opening-strength-sync-artifacts",
            dry_run=args.dry_run,
        )
        created_temp_pod = not args.dry_run

    if args.dry_run:
        print("sync_plan:")
        print(f"  pod: {pod_name or '<none>'}")
        print(f"  metrics: {fetch_metrics_flag}")
        print(f"  predictions: {fetch_predictions_flag}")
        print(f"  record: {record_flag}")
        for spec in specs:
            print(f"  {spec.run_id}: {spec.pvc_dir}")
        return

    statuses = collect_run_statuses(Path(args.runs_dir))
    try:
        if fetch_metrics_flag:
            print("pulled_metrics:")
            for spec in specs:
                paths = pull_metrics(args.hfcli, spec, pod_name, metrics_dir)
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
        if fetch_predictions_flag:
            print("pulled_predictions:")
            for spec in specs:
                path = fetch_predictions(args.hfcli, spec, pod_name, predictions_root)
                print(f"  {spec.run_id}: {path}")
    finally:
        if created_temp_pod:
            delete_temp_pod(args.hfcli, specs[0].namespace, pod_name)

    if record_flag:
        print("recorded_metrics:")
        for spec in specs:
            path = record_metrics(spec.run_id, metrics_dir, records_dir)
            status = statuses.get(spec.run_id, "")
            if path:
                print(f"  {spec.run_id}: {path}")
                if status and status != "completed":
                    print(f"  {spec.run_id}: config status is {status!r}; confirm before final archive")
            else:
                print(f"  {spec.run_id}: missing {metrics_dir / f'{spec.run_id}{METRICS_SUFFIX}'}")


if __name__ == "__main__":
    main()
