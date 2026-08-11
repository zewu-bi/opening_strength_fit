from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.feature_utils import finite_numeric
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.io.json import json_safe as json_safe
from opening_strength_fit.io.json import write_json as write_json
from opening_strength_fit.labels import finite_numeric_series, normalize_return_label_frame
from opening_strength_fit.schema import DECISION_KEY_COLUMNS

KEY_COLUMNS = DECISION_KEY_COLUMNS
NEXT_CLOSE_LABEL_COL = "alpha_return_next_close"
NEXT_CLOSE_CACHE_NAME = "clickhouse_next_close_labels.parquet"


def newey_west_mean_se(values: pd.Series, lag: int = 5) -> tuple[float, float]:
    array = finite_numeric(values).dropna().to_numpy(dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    demeaned = array - array.mean()
    count = len(array)
    long_run_variance = float(np.dot(demeaned, demeaned) / count)
    for offset in range(1, min(int(lag), count - 1) + 1):
        covariance = float(np.dot(demeaned[offset:], demeaned[:-offset]) / count)
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    return float(array.mean()), float(standard_error)


def newey_west_mean_ci(values: pd.Series, lag: int = 5) -> tuple[float, float]:
    mean, standard_error = newey_west_mean_se(values, lag=lag)
    delta = 1.96 * standard_error
    return mean - delta, mean + delta


def equal_weighted_period_means(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
    period_column: str,
    value_columns: Sequence[str],
    count_name: str = "",
) -> pd.DataFrame:
    groups = list(by)
    values = list(value_columns)
    summary = (
        frame.groupby([*groups, period_column], sort=False)[values]
        .mean()
        .groupby(groups, sort=False)
        .mean()
        .reset_index()
    )
    if not count_name:
        return summary
    counts = frame.groupby(groups, sort=False).size().rename(count_name).reset_index()
    return summary.merge(counts, on=groups, validate="one_to_one")


def write_analysis_result(
    output_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metrics_filename: str,
    summary_filename: str,
    trace: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / metrics_filename
    if metrics_path.suffix == ".parquet":
        metrics.to_parquet(metrics_path, index=False)
    else:
        metrics.to_csv(metrics_path, index=False)
    summary.to_csv(output_dir / summary_filename, index=False)
    write_json(output_dir / "trace.json", trace, ensure_ascii=False, sort_keys=True)
    print("SUMMARY_CSV")
    print(summary.to_csv(index=False).strip())
    print("TRACE_JSON=" + json.dumps(trace, ensure_ascii=False, sort_keys=True))


def clock_range(start: str, end: str) -> list[str]:
    start_ts = pd.Timestamp(f"2000-01-01 {start}")
    end_ts = pd.Timestamp(f"2000-01-01 {end}")
    if end_ts < start_ts:
        raise SystemExit("--end-clock must be >= --start-clock")
    return [
        timestamp.strftime("%H:%M") for timestamp in pd.date_range(start_ts, end_ts, freq="min")
    ]


def month_periods(start_month: str, end_month: str) -> list[str]:
    return [str(month) for month in pd.period_range(start_month, end_month, freq="M")]


def month_window_periods(
    start_month: str,
    end_month: str,
    *,
    test_months: int = 1,
    stride_months: int | None = None,
) -> list[tuple[str, str]]:
    test_months = int(test_months)
    stride_months = test_months if stride_months is None else int(stride_months)
    if test_months < 1:
        raise SystemExit("test_months must be >= 1")
    if stride_months < 1:
        raise SystemExit("stride_months must be >= 1")

    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    windows: list[tuple[str, str]] = []
    current = start
    while current <= end:
        window_end = current + test_months - 1
        if window_end > end:
            break
        windows.append((str(current), str(window_end)))
        current += stride_months
    return windows


def normalize_next_close_labels(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = KEY_COLUMNS,
    label_col: str = NEXT_CLOSE_LABEL_COL,
) -> pd.DataFrame:
    return normalize_return_label_frame(
        frame,
        key_columns=key_columns,
        label_col=label_col,
    )


def load_next_close_label_file(
    path: str | Path,
    *,
    key_columns: Sequence[str] = KEY_COLUMNS,
    label_col: str = NEXT_CLOSE_LABEL_COL,
) -> pd.DataFrame:
    required = [*key_columns, label_col]
    return normalize_next_close_labels(
        read_frame(path, columns=list(required)),
        key_columns=key_columns,
        label_col=label_col,
    )


def load_or_fetch_next_close_labels(
    base: pd.DataFrame,
    *,
    output_dir: Path,
    label_input: str | Path | None = None,
    fetch_labels: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    missing_message: str = "next-close labels not found",
    cache_name: str = NEXT_CLOSE_CACHE_NAME,
    key_columns: Sequence[str] = KEY_COLUMNS,
    label_col: str = NEXT_CLOSE_LABEL_COL,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_path = output_dir / cache_name
    input_path = Path(label_input) if label_input else None
    if input_path and input_path.exists():
        labels = load_next_close_label_file(
            input_path,
            key_columns=key_columns,
            label_col=label_col,
        )
        write_frame(labels, cached_path)
        return labels
    if cached_path.exists():
        return load_next_close_label_file(
            cached_path,
            key_columns=key_columns,
            label_col=label_col,
        )
    if fetch_labels is None:
        raise SystemExit(missing_message)
    labels = normalize_next_close_labels(
        fetch_labels(base.copy()),
        key_columns=key_columns,
        label_col=label_col,
    )
    write_frame(labels, cached_path)
    return labels


def finite_mean(series: pd.Series) -> float:
    values = finite_numeric_series(series).dropna()
    return float(values.mean()) if len(values) else float("nan")


def positive_rate(series: pd.Series) -> float:
    values = finite_numeric_series(series).dropna()
    return float((values > 0).mean()) if len(values) else float("nan")


def positive_count(series: pd.Series) -> int:
    values = finite_numeric_series(series).dropna()
    return int((values > 0).sum())


def selection_return_stats(
    full_group: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    label_col: str,
    prefix: str,
    scale: float = 10_000.0,
) -> dict[str, float]:
    all_mean = finite_mean(full_group[label_col])
    top_mean = finite_mean(selected[label_col]) if len(selected) else float("nan")
    return {
        f"{prefix}_all_mean_bps": all_mean * scale,
        f"{prefix}_top_mean_bps": top_mean * scale,
        f"{prefix}_top_excess_bps": (top_mean - all_mean) * scale,
        f"{prefix}_top_win_rate": positive_rate(selected[label_col])
        if len(selected)
        else float("nan"),
    }


def write_artifact_fetch_trace(
    output_dir: Path,
    *,
    fetched_at_utc: str,
    run_id: str,
    namespace: str,
    pvc_dir: str,
    files: Sequence[Path],
    missing: Sequence[str],
) -> Path:
    trace_path = output_dir / "artifact_fetch_trace.json"
    write_json(
        trace_path,
        {
            "fetched_at_utc": fetched_at_utc,
            "run_id": run_id,
            "namespace": namespace,
            "pvc_dir": pvc_dir,
            "local_output_dir": str(output_dir),
            "files": [str(path) for path in files],
            "missing": list(missing),
        },
        ensure_ascii=True,
    )
    return trace_path
