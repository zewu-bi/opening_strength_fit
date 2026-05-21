import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.k8s import RunSpec
from opening_strength_fit.k8s import command_succeeds, delete_temp_pod, ensure_temp_pod
from opening_strength_fit.k8s import load_run_spec, run_command


def parse_config(path: str) -> RunSpec:
    return load_run_spec(path)


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


def combine_yearly_predictions(raw_dir: Path, output_path: Path) -> dict[str, object]:
    return combine_prediction_files(raw_dir, output_path)


def month_periods(start_month: str, end_month: str) -> list[str]:
    return [str(month) for month in pd.period_range(start_month, end_month, freq="M")]


def summarize_predictions(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    date_index = pd.to_datetime(frame["date"], errors="coerce")
    summary = {
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch opening-strength prediction parquet files from the k8s PVC into "
            "a local traceable folder."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/runs/gbm_opening_1y_next_month.toml",
        help="Run config that defines run.id, output.k8s_dir, namespace, and date range.",
    )
    parser.add_argument("--hfcli", default="hfcli")
    parser.add_argument(
        "--pod",
        default="",
        help="Optional existing pod with the output PVC mounted.",
    )
    parser.add_argument(
        "--timeout",
        default="300s",
        help="kubectl wait timeout for a temporary pull pod.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/backtest/gbm_opening_1y_next_month",
        help="Local output directory for raw parquet files, combined predictions, and traces.",
    )
    args = parser.parse_args()

    spec = parse_config(args.config)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    combined_path = output_dir / "predictions_all.parquet"
    trace_path = output_dir / "fetch_trace.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    pod_name = args.pod
    created_temp_pod = False
    if not pod_name:
        pod_name = ensure_temp_pod(
            args.hfcli,
            spec,
            args.timeout,
            "opening-strength-fetch-predictions",
        )
        created_temp_pod = True

    fetched_files = []
    try:
        combined_remote_path = f"{spec.pvc_dir}/predictions_all.parquet"
        single_remote_path = f"{spec.pvc_dir}/predictions.parquet"
        if remote_file_exists(args.hfcli, spec.namespace, pod_name, combined_remote_path):
            print(f"fetching {combined_remote_path} -> {combined_path}")
            fetch_binary_file(
                args.hfcli,
                spec.namespace,
                pod_name,
                combined_remote_path,
                combined_path,
            )
            fetched_files.append(
                {
                    "kind": "combined",
                    "remote_path": combined_remote_path,
                    "local_path": str(combined_path),
                }
            )
            combined_stats = summarize_predictions(combined_path)
        elif remote_file_exists(args.hfcli, spec.namespace, pod_name, single_remote_path):
            print(f"fetching {single_remote_path} -> {combined_path}")
            fetch_binary_file(
                args.hfcli,
                spec.namespace,
                pod_name,
                single_remote_path,
                combined_path,
            )
            fetched_files.append(
                {
                    "kind": "single",
                    "remote_path": single_remote_path,
                    "local_path": str(combined_path),
                }
            )
            combined_stats = summarize_predictions(combined_path)
        else:
            if spec.test_start_month and spec.test_end_month:
                for month in month_periods(spec.test_start_month, spec.test_end_month):
                    remote_candidates = [
                        f"{spec.pvc_dir}/month_{month}/predictions_{month}.parquet",
                        f"{spec.pvc_dir}/month_{month}/predictions.parquet",
                    ]
                    remote_path = next(
                        (
                            candidate
                            for candidate in remote_candidates
                            if remote_file_exists(
                                args.hfcli,
                                spec.namespace,
                                pod_name,
                                candidate,
                            )
                        ),
                        "",
                    )
                    if not remote_path:
                        raise SystemExit(
                            f"no prediction parquet found for monthly shard {month}"
                        )
                    local_path = raw_dir / f"predictions_{month}.parquet"
                    print(f"fetching {remote_path} -> {local_path}")
                    fetch_binary_file(
                        args.hfcli,
                        spec.namespace,
                        pod_name,
                        remote_path,
                        local_path,
                    )
                    fetched_files.append(
                        {
                            "kind": "monthly",
                            "month": month,
                            "remote_path": remote_path,
                            "local_path": str(local_path),
                        }
                    )
                combined_stats = combine_prediction_files(raw_dir, combined_path)
            elif spec.test_start_year <= 0 or spec.test_end_year <= 0:
                raise SystemExit(
                    "no combined/single predictions found and config has no test date range"
                )
            else:
                for year in range(spec.test_start_year, spec.test_end_year + 1):
                    remote_candidates = [
                        f"{spec.pvc_dir}/year_{year}/predictions_{year}.parquet",
                        f"{spec.pvc_dir}/year_{year}/predictions.parquet",
                    ]
                    remote_path = next(
                        (
                            candidate
                            for candidate in remote_candidates
                            if remote_file_exists(
                                args.hfcli,
                                spec.namespace,
                                pod_name,
                                candidate,
                            )
                        ),
                        "",
                    )
                    if not remote_path:
                        raise SystemExit(
                            f"no prediction parquet found for yearly shard {year}"
                        )
                    local_path = raw_dir / f"predictions_{year}.parquet"
                    print(f"fetching {remote_path} -> {local_path}")
                    fetch_binary_file(
                        args.hfcli,
                        spec.namespace,
                        pod_name,
                        remote_path,
                        local_path,
                    )
                    fetched_files.append(
                        {
                            "kind": "yearly",
                            "year": year,
                            "remote_path": remote_path,
                            "local_path": str(local_path),
                        }
                    )
                combined_stats = combine_yearly_predictions(raw_dir, combined_path)
    finally:
        if created_temp_pod:
            delete_temp_pod(args.hfcli, spec.namespace, pod_name)
    trace = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "config": args.config,
        "run_id": spec.run_id,
        "namespace": spec.namespace,
        "pvc_dir": spec.pvc_dir,
        "local_output_dir": str(output_dir),
        "combined_output": str(combined_path),
        "files": fetched_files,
        "combined_stats": combined_stats,
    }
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    print("fetch_predictions_complete:")
    print(f"  combined_output: {combined_path}")
    print(f"  trace: {trace_path}")
    print(f"  rows: {combined_stats['rows']}")
    print(f"  dates: {combined_stats['dates']}")
    print(f"  symbols: {combined_stats['symbols']}")


if __name__ == "__main__":
    main()
