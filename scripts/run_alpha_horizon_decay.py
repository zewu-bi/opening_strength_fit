from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    get_tick_client,
    validate_table_name,
)
from opening_strength_fit.evaluation import (
    resolve_group_cols,
    score_bucket_returns,
    top_score_trades,
)
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns


DEFAULT_OUTPUT_ROOT = "output/reports/opening_alpha_horizon_decay_delay2_cohort_avg"
DEFAULT_RUNS = (
    (
        "Universe",
        "output/predictions/lgbm_opening_1y_next_month_delay2/predictions_all.parquet",
    ),
    (
        "Strong",
        "output/predictions/lgbm_opening_1y_next_month_strong_delay2/predictions_all.parquet",
    ),
)
DEFAULT_HORIZONS = (
    "1m",
    "2m",
    "5m",
    "10m",
    "close",
    "next_close",
)
DEFAULT_GROUP_COLS = ("date", "decision_target_timestamp")
DEFAULT_CLOSE_OFFSET_US = 54_000_000_000
DEFAULT_CLOSE_LOOKBACK_SECONDS = 1_800
DEFAULT_DECISION_MAX_LAG_SECONDS = 5
DEFAULT_TIMED_TARGET_END_TIME = "09:40:00"
RUN_COLORS = {
    "Universe": "#1f77b4",
    "Strong": "#d17a22",
}
HORIZON_LABEL_CANDIDATES = (
    "alpha_return_{horizon}",
    "label_{horizon}",
    "gross_label_{horizon}",
    "return_{horizon}",
    "{horizon}_label",
    "{horizon}_return",
)
TIME_HORIZON_SUFFIXES = {"s", "m", "h"}


@dataclass(frozen=True)
class RunInput:
    label: str
    path: Path


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    label: str
    seconds: int | None = None


def parse_run(value: str) -> RunInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return RunInput(label=label, path=Path(raw_path))


def parse_label_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--horizon-label must be formatted as horizon=column"
        )
    horizon, column = value.split("=", 1)
    horizon = normalize_horizon_name(horizon)
    column = column.strip()
    if not horizon or not column:
        raise argparse.ArgumentTypeError("horizon and column cannot be empty")
    return horizon, column


def normalize_horizon_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def parse_seconds_horizon(value: str) -> int | None:
    horizon = normalize_horizon_name(value)
    if horizon in {"close", "next_open", "next_close"}:
        return None
    aliases = {
        "30sec": "30s",
        "60sec": "60s",
        "5min": "5m",
    }
    horizon = aliases.get(horizon, horizon)
    if len(horizon) < 2 or horizon[-1] not in TIME_HORIZON_SUFFIXES:
        raise argparse.ArgumentTypeError(
            f"unknown horizon {value!r}; use Ns, Nm, Nh, close, next_open, or next_close"
        )
    amount = int(horizon[:-1])
    unit = horizon[-1]
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    return amount * multiplier


def horizon_specs(values: Iterable[str]) -> list[HorizonSpec]:
    specs = []
    for value in values:
        name = normalize_horizon_name(value)
        name = {"30sec": "30s", "60sec": "60s", "5min": "5m"}.get(name, name)
        seconds = parse_seconds_horizon(name)
        label = name.replace("_", " ")
        specs.append(HorizonSpec(name=name, label=label, seconds=seconds))
    return specs


def parse_clock_values(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    clocks: set[str] = set()
    for value in values:
        for part in str(value).replace(",", " ").split():
            timestamp = pd.Timestamp(f"2000-01-01 {part}")
            clocks.add(timestamp.strftime("%H:%M:%S"))
    return clocks


def optional_clock_seconds(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    token = str(value).strip().lower()
    if token in {"none", "null", "off", "false"}:
        return None
    timestamp = pd.Timestamp(f"2000-01-01 {value}")
    return int(timestamp.hour) * 3_600 + int(timestamp.minute) * 60 + int(timestamp.second)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure opening predictor alpha decay across forward horizons. "
            "The default runs are the delay2 LightGBM universe and strong branches."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help=(
            "Prediction input formatted as label=path. Defaults to local delay2 "
            "Universe and Strong prediction files."
        ),
    )
    parser.add_argument(
        "--label-input",
        default="",
        help=(
            "Optional horizon-label parquet/csv keyed by date, symbol, and "
            "decision_target_timestamp. Use for precomputed close/T+1 labels."
        ),
    )
    parser.add_argument(
        "--tick-input",
        default="",
        help=(
            "Optional raw tick parquet/csv context used to compute horizon labels. "
            "It must include cumulative volume/turnover for intraday VWAP labels "
            "and a price column for close/next open/next close labels."
        ),
    )
    parser.add_argument(
        "--horizon",
        action="append",
        help=(
            "Horizon to evaluate. Defaults to 1m, 2m, 5m, 10m, close, "
            "and next_close."
        ),
    )
    parser.add_argument(
        "--horizon-label",
        action="append",
        type=parse_label_mapping,
        help=(
            "Explicit mapping from horizon to an existing label column, for "
            "example --horizon-label 5m=alpha_return_5m."
        ),
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument(
        "--group-col",
        action="append",
        help=(
            "Grouping column for cross-section selection and IC. Defaults to "
            "date and decision_target_timestamp."
        ),
    )
    parser.add_argument(
        "--decision-time",
        action="append",
        help=(
            "Optional decision clock filter. Can be repeated or comma-separated, "
            "for example --decision-time 09:30:00,09:32:00."
        ),
    )
    parser.add_argument(
        "--sell-window-seconds",
        type=int,
        default=60,
        help="VWAP exit window width for intraday horizons.",
    )
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--sample-context",
        default="",
        help=(
            "Optional delay2 labeled cache or prediction parquet used as the "
            "opening-window price context. Defaults to the union of --run inputs."
        ),
    )
    parser.add_argument(
        "--no-sampled-intraday",
        action="store_true",
        help="Do not derive 1m..10m labels from sampled opening decision rows.",
    )
    parser.add_argument(
        "--sampled-exit-price-col",
        default="mid_price",
        help=(
            "Exit price column for opening-window sampled decay. The default "
            "mid_price avoids mixing 60s VWAP labels with minute sampled exits."
        ),
    )
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--turnover-col", default="turnover")
    parser.add_argument("--volume-unit-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--price-col",
        default="auto",
        help=(
            "Price column for close/next open/next close exits. auto uses "
            "mid_price, then last_price, then bid/ask midpoint."
        ),
    )
    parser.add_argument("--open-time", default="09:30:00")
    parser.add_argument("--close-time", default="15:00:00")
    parser.add_argument(
        "--timed-target-end-time",
        default=DEFAULT_TIMED_TARGET_END_TIME,
        help=(
            "Latest target clock for timed intraday horizons. Defaults to 09:40:00 "
            "so cache and ClickHouse runs compare the same opening-window targets. "
            "Use none to disable."
        ),
    )
    parser.add_argument(
        "--clickhouse-close-labels",
        action="store_true",
        help=(
            "Fetch same-day close and next-trading-day close prices from "
            "ClickHouse for requested close horizons."
        ),
    )
    parser.add_argument(
        "--clickhouse-intraday-labels",
        action="store_true",
        help=(
            "Fetch opening-window target-minute bid/ask mid prices from "
            "ClickHouse for timed horizons such as 1m, 2m, 5m, and 10m."
        ),
    )
    parser.add_argument("--clickhouse-host", default=None)
    parser.add_argument("--clickhouse-port", type=int, default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-table", default=None)
    parser.add_argument(
        "--clickhouse-close-offset-us",
        type=int,
        default=DEFAULT_CLOSE_OFFSET_US,
    )
    parser.add_argument(
        "--clickhouse-close-lookback-seconds",
        type=int,
        default=DEFAULT_CLOSE_LOOKBACK_SECONDS,
    )
    parser.add_argument(
        "--clickhouse-calendar-days-after",
        type=int,
        default=14,
        help="Calendar-day padding after the last sample date for next_close.",
    )
    parser.add_argument(
        "--clickhouse-decision-max-lag-seconds",
        type=int,
        default=DEFAULT_DECISION_MAX_LAG_SECONDS,
        help="Maximum target-minute sampling lag when deriving intraday labels from ClickHouse.",
    )
    parser.add_argument(
        "--max-future-gap-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional as-of tolerance for intraday VWAP endpoints. 0 disables "
            "the tolerance."
        ),
    )
    parser.add_argument(
        "--max-price-gap-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional as-of tolerance for close/next open/next close price exits. "
            "0 disables the tolerance."
        ),
    )
    parser.add_argument(
        "--allow-missing-horizons",
        action="store_true",
        help="Continue when a requested horizon label is unavailable.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for summary CSVs, labels, figures, and trace.",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def available_columns(path: Path) -> list[str] | None:
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    target = path
    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            return None
        target = parquet_files[0]
    if target.suffix.lower() != ".parquet":
        return None
    return list(pq.ParquetFile(target).schema_arrow.names)


def read_selected_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return read_frame(path)
    available = available_columns(path)
    selected = [column for column in columns if available is None or column in available]
    if not selected:
        return read_frame(path)
    return read_frame(path, columns=selected)


def normalize_frame_times(frame: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(frame)
    if {"date", "symbol", "timestamp"}.issubset(out.columns):
        out = ensure_timestamp_columns(out)
    for column in (
        "timestamp",
        "decision_target_timestamp",
        "entry_timestamp",
    ):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def load_prediction(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction input for {label} does not exist: {path}")
    frame = normalize_frame_times(read_frame(path))
    required = {"date", "symbol", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"prediction input for {label} missing columns: {missing}")
    frame["branch"] = label
    return frame


def filter_decision_times(frame: pd.DataFrame, clocks: set[str]) -> pd.DataFrame:
    if not clocks:
        return frame
    if "decision_time" in frame.columns:
        values = frame["decision_time"].astype(str)
        return frame.loc[values.isin(clocks)].copy()
    timestamp_col = (
        "decision_target_timestamp"
        if "decision_target_timestamp" in frame.columns
        else "timestamp"
    )
    if timestamp_col not in frame.columns:
        raise SystemExit("--decision-time requires decision_time or timestamp columns")
    values = pd.to_datetime(frame[timestamp_col], errors="coerce").dt.strftime("%H:%M:%S")
    return frame.loc[values.isin(clocks)].copy()


def label_column_name(horizon: str) -> str:
    return f"alpha_return_{horizon}"


def explicit_label_map(values: list[tuple[str, str]] | None) -> dict[str, str]:
    return dict(values or [])


def label_candidates(horizon: str) -> list[str]:
    candidates = [
        template.format(horizon=horizon)
        for template in HORIZON_LABEL_CANDIDATES
    ]
    if horizon == "60s":
        candidates.extend(["label", "gross_label"])
    return list(dict.fromkeys(candidates))


def find_existing_label_column(
    frame: pd.DataFrame,
    horizon: str,
    explicit: dict[str, str],
) -> str | None:
    if horizon in explicit:
        column = explicit[horizon]
        if column not in frame.columns:
            raise SystemExit(
                f"explicit horizon label {horizon}={column} is missing from input"
            )
        return column
    for column in label_candidates(horizon):
        if column in frame.columns:
            return column
    return None


def merge_label_input(
    predictions: pd.DataFrame,
    label_input: Path,
    horizons: list[HorizonSpec],
    explicit: dict[str, str],
) -> pd.DataFrame:
    available = available_columns(label_input)
    key_cols = key_columns_for_merge(predictions)
    candidate_cols = []
    for spec in horizons:
        candidate_cols.extend(label_candidates(spec.name))
        if spec.name in explicit:
            candidate_cols.append(explicit[spec.name])
    if available is not None:
        read_cols = [column for column in [*key_cols, *candidate_cols] if column in available]
        missing_keys = sorted(set(key_cols) - set(read_cols))
        if missing_keys:
            raise SystemExit(
                f"label input {label_input} is missing merge keys: {missing_keys}"
            )
    else:
        read_cols = None
    labels = normalize_frame_times(read_selected_frame(label_input, read_cols))
    label_cols = []
    for spec in horizons:
        column = find_existing_label_column(labels, spec.name, explicit)
        if column is not None:
            label_cols.append(column)
    label_cols = list(dict.fromkeys(label_cols))
    if not label_cols:
        return predictions
    labels = labels[[*key_cols, *label_cols]].drop_duplicates(key_cols)
    rename = {}
    for spec in horizons:
        column = find_existing_label_column(labels, spec.name, explicit)
        if column and column != label_column_name(spec.name):
            rename[column] = label_column_name(spec.name)
    labels = labels.rename(columns=rename)
    merged = predictions.merge(labels, on=key_cols, how="left", suffixes=("", "_labelctx"))
    for spec in horizons:
        column = label_column_name(spec.name)
        labelctx_column = f"{column}_labelctx"
        if labelctx_column in merged.columns:
            merged[column] = pd.to_numeric(
                merged[labelctx_column],
                errors="coerce",
            ).combine_first(pd.to_numeric(merged.get(column), errors="coerce"))
            merged = merged.drop(columns=[labelctx_column])
    return merged


def key_columns_for_merge(frame: pd.DataFrame) -> list[str]:
    for time_col in ("decision_target_timestamp", "timestamp"):
        if time_col in frame.columns:
            return ["date", "symbol", time_col]
    raise SystemExit("inputs need decision_target_timestamp or timestamp merge key")


def build_base_samples(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "symbol", "entry_timestamp", "buy_price"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(
            f"tick-input horizon labels require prediction columns: {missing}"
        )
    key_cols = key_columns_for_merge(frame)
    cols = list(dict.fromkeys([*key_cols, "entry_timestamp", "buy_price"]))
    samples = frame[cols].drop_duplicates(key_cols).copy()
    samples["entry_timestamp"] = pd.to_datetime(
        samples["entry_timestamp"],
        errors="coerce",
    )
    samples["buy_price"] = pd.to_numeric(samples["buy_price"], errors="coerce")
    return samples


def load_ticks_for_horizons(
    tick_input: Path,
    *,
    volume_col: str,
    turnover_col: str,
    price_col: str,
) -> pd.DataFrame:
    requested = {
        "date",
        "symbol",
        "timestamp",
        "time",
        "exch_time_offset_us",
        volume_col,
        turnover_col,
        "mid_price",
        "last_price",
        "ask_price_1",
        "bid_price_1",
    }
    if price_col != "auto":
        requested.add(price_col)
    available = available_columns(tick_input)
    columns = None if available is None else [column for column in requested if column in available]
    ticks = normalize_frame_times(read_selected_frame(tick_input, columns))
    missing = [column for column in ("date", "symbol", "timestamp") if column not in ticks]
    if missing:
        raise SystemExit(f"tick input missing required columns: {missing}")
    return ticks.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)


def resolve_price_column(ticks: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, str]:
    if price_col != "auto":
        if price_col not in ticks.columns:
            raise SystemExit(f"price column {price_col!r} not found in tick input")
        return ticks, price_col
    if "mid_price" in ticks.columns:
        return ticks, "mid_price"
    if {"ask_price_1", "bid_price_1"}.issubset(ticks.columns):
        ticks = ticks.copy()
        ask = pd.to_numeric(ticks["ask_price_1"], errors="coerce")
        bid = pd.to_numeric(ticks["bid_price_1"], errors="coerce")
        ticks["mid_price"] = np.where((ask > 0) & (bid > 0), (ask + bid) / 2.0, np.nan)
        return ticks, "mid_price"
    if "last_price" in ticks.columns:
        return ticks, "last_price"
    raise SystemExit(
        "tick input needs mid_price, last_price, or bid/ask level 1 for price exits"
    )


def future_vwap_labels(
    samples: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    horizon: HorizonSpec,
    volume_col: str,
    turnover_col: str,
    volume_unit_multiplier: float,
    sell_window_seconds: int,
    fee_bps: float,
    max_gap_seconds: float | None,
) -> pd.Series:
    if horizon.seconds is None:
        raise ValueError("future_vwap_labels requires a timed horizon")
    for column in (volume_col, turnover_col):
        if column not in ticks.columns:
            raise SystemExit(f"tick input missing required column for VWAP: {column}")

    out = pd.Series(np.nan, index=samples.index, dtype="float64")
    tolerance = pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds else None
    tick_groups = {
        key_tuple(key): group.sort_values("timestamp")
        for key, group in ticks.groupby(["date", "symbol"], sort=False, observed=True)
    }

    for key, group in samples.groupby(["date", "symbol"], sort=False, observed=True):
        right = tick_groups.get(key_tuple(key))
        if right is None:
            continue
        sample_group = group.dropna(subset=["entry_timestamp", "buy_price"])
        if sample_group.empty:
            continue
        right_frame = (
            right[["timestamp", volume_col, turnover_col]]
            .dropna(subset=["timestamp"])
            .rename(columns={"timestamp": "_future_ts"})
            .sort_values("_future_ts")
        )
        if right_frame.empty:
            continue
        left_start = pd.DataFrame(
            {
                "_row": sample_group.index.to_numpy(),
                "_target_ts": sample_group["entry_timestamp"]
                + pd.to_timedelta(horizon.seconds, unit="s"),
            }
        ).sort_values("_target_ts")
        left_end = pd.DataFrame(
            {
                "_row": sample_group.index.to_numpy(),
                "_target_ts": sample_group["entry_timestamp"]
                + pd.to_timedelta(horizon.seconds + sell_window_seconds, unit="s"),
            }
        ).sort_values("_target_ts")
        start = pd.merge_asof(
            left_start,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_row")
        end = pd.merge_asof(
            left_end,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_row")
        common = start.index.intersection(end.index)
        if common.empty:
            continue
        start_volume = pd.to_numeric(start.loc[common, volume_col], errors="coerce")
        end_volume = pd.to_numeric(end.loc[common, volume_col], errors="coerce")
        start_turnover = pd.to_numeric(
            start.loc[common, turnover_col],
            errors="coerce",
        )
        end_turnover = pd.to_numeric(end.loc[common, turnover_col], errors="coerce")
        sell_volume = end_volume - start_volume
        sell_turnover = end_turnover - start_turnover
        denominator = sell_volume * float(volume_unit_multiplier)
        sell_vwap = sell_turnover / denominator.replace(0, np.nan)
        buy_price = pd.to_numeric(samples.loc[common, "buy_price"], errors="coerce")
        label = sell_vwap / buy_price - 1.0 - float(fee_bps) / 10_000.0
        valid = sell_volume.gt(0) & sell_turnover.gt(0) & buy_price.gt(0)
        out.loc[common[valid.to_numpy()]] = label.loc[valid].to_numpy()

    return out


def key_tuple(key: object) -> tuple[object, ...]:
    return key if isinstance(key, tuple) else (key,)


def next_date_map(ticks: pd.DataFrame) -> dict[str, str]:
    dates = sorted(ticks["date"].astype(str).dropna().unique())
    return {date: dates[index + 1] for index, date in enumerate(dates[:-1])}


def price_exit_labels(
    samples: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    horizon: HorizonSpec,
    price_col: str,
    open_time: str,
    close_time: str,
    fee_bps: float,
    max_gap_seconds: float | None,
) -> pd.Series:
    out = pd.Series(np.nan, index=samples.index, dtype="float64")
    tolerance = pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds else None

    work = samples.dropna(subset=["buy_price"]).copy()
    if horizon.name == "close":
        work["_target_date"] = work["date"].astype(str)
        work["_target_time"] = close_time
        direction = "backward"
    elif horizon.name in {"next_open", "next_close"}:
        next_dates = next_date_map(ticks)
        work["_target_date"] = work["date"].astype(str).map(next_dates)
        work["_target_time"] = open_time if horizon.name == "next_open" else close_time
        direction = "forward" if horizon.name == "next_open" else "backward"
    else:
        raise ValueError(f"unsupported price horizon: {horizon.name}")

    work = work.dropna(subset=["_target_date"])
    if work.empty:
        return out
    work["_target_ts"] = pd.to_datetime(
        work["_target_date"].astype(str) + " " + work["_target_time"].astype(str),
        errors="coerce",
    )
    tick_groups = {
        key_tuple(key): group.sort_values("timestamp")
        for key, group in ticks.groupby(["date", "symbol"], sort=False, observed=True)
    }

    for key, group in work.groupby(["_target_date", "symbol"], sort=False, observed=True):
        right = tick_groups.get(key_tuple(key))
        if right is None:
            continue
        right_frame = (
            right[["timestamp", price_col]]
            .dropna(subset=["timestamp", price_col])
            .rename(columns={"timestamp": "_future_ts"})
            .sort_values("_future_ts")
        )
        if right_frame.empty:
            continue
        left = pd.DataFrame(
            {
                "_row": group.index.to_numpy(),
                "_target_ts": group["_target_ts"].to_numpy(),
            }
        ).sort_values("_target_ts")
        merged = pd.merge_asof(
            left,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction=direction,
            tolerance=tolerance,
        ).set_index("_row")
        if merged.empty:
            continue
        exit_price = pd.to_numeric(merged[price_col], errors="coerce")
        buy_price = pd.to_numeric(samples.loc[merged.index, "buy_price"], errors="coerce")
        label = exit_price / buy_price - 1.0 - float(fee_bps) / 10_000.0
        valid = exit_price.gt(0) & buy_price.gt(0)
        out.loc[merged.index[valid.to_numpy()]] = label.loc[valid].to_numpy()

    return out


def compute_tick_horizon_labels(
    predictions: pd.DataFrame,
    tick_input: Path,
    horizons: list[HorizonSpec],
    *,
    volume_col: str,
    turnover_col: str,
    volume_unit_multiplier: float,
    sell_window_seconds: int,
    fee_bps: float,
    price_col: str,
    open_time: str,
    close_time: str,
    max_future_gap_seconds: float | None,
    max_price_gap_seconds: float | None,
) -> pd.DataFrame:
    samples = build_base_samples(predictions)
    ticks = load_ticks_for_horizons(
        tick_input,
        volume_col=volume_col,
        turnover_col=turnover_col,
        price_col=price_col,
    )
    ticks, resolved_price_col = resolve_price_column(ticks, price_col)
    labels = samples[key_columns_for_merge(samples)].copy()
    for spec in horizons:
        column = label_column_name(spec.name)
        if spec.seconds is not None:
            labels[column] = future_vwap_labels(
                samples,
                ticks,
                horizon=spec,
                volume_col=volume_col,
                turnover_col=turnover_col,
                volume_unit_multiplier=volume_unit_multiplier,
                sell_window_seconds=sell_window_seconds,
                fee_bps=fee_bps,
                max_gap_seconds=max_future_gap_seconds,
            )
        else:
            labels[column] = price_exit_labels(
                samples,
                ticks,
                horizon=spec,
                price_col=resolved_price_col,
                open_time=open_time,
                close_time=close_time,
                fee_bps=fee_bps,
                max_gap_seconds=max_price_gap_seconds,
            )
    return labels


def attach_available_prediction_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonSpec],
    explicit: dict[str, str],
) -> pd.DataFrame:
    out = predictions.copy()
    for spec in horizons:
        target = label_column_name(spec.name)
        if target in out.columns:
            continue
        source = find_existing_label_column(out, spec.name, explicit)
        if source is not None:
            out[target] = pd.to_numeric(out[source], errors="coerce")
    return out


def load_sample_context(
    predictions: pd.DataFrame,
    sample_context: str,
    *,
    exit_price_col: str,
) -> pd.DataFrame:
    key_cols = ["date", "symbol", "decision_target_timestamp"]
    required_columns = [*key_cols, exit_price_col]
    columns = [*key_cols, exit_price_col]
    if exit_price_col == "mid_price":
        columns.extend(["ask_price_1", "bid_price_1"])
    columns = list(dict.fromkeys(columns))
    if sample_context:
        path = Path(sample_context)
        available = available_columns(path)
        read_cols = columns if available is None else [col for col in columns if col in available]
        missing = sorted(set(required_columns) - set(read_cols))
        if missing:
            raise SystemExit(f"sample context is missing required columns: {missing}")
        context = normalize_frame_times(read_selected_frame(path, read_cols))
    else:
        missing = [col for col in required_columns if col not in predictions.columns]
        if missing:
            raise SystemExit(
                "sampled intraday decay needs --sample-context or prediction "
                f"columns: {missing}"
            )
        context = predictions[[col for col in columns if col in predictions.columns]].copy()
    context["date"] = context["date"].astype(str)
    context["symbol"] = context["symbol"].astype(str)
    context["decision_target_timestamp"] = pd.to_datetime(
        context["decision_target_timestamp"],
        errors="coerce",
    )
    if exit_price_col == "mid_price" and {"ask_price_1", "bid_price_1"}.issubset(
        context.columns
    ):
        ask = pd.to_numeric(context["ask_price_1"], errors="coerce")
        bid = pd.to_numeric(context["bid_price_1"], errors="coerce")
        context[exit_price_col] = np.where((ask > 0) & (bid > 0), (ask + bid) / 2.0, np.nan)
    context[exit_price_col] = pd.to_numeric(context[exit_price_col], errors="coerce")
    return (
        context.dropna(subset=["decision_target_timestamp", exit_price_col])
        .sort_values(key_cols)
        .drop_duplicates(key_cols)
        .reset_index(drop=True)
    )


def compute_sampled_intraday_labels(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    horizons: list[HorizonSpec],
    *,
    exit_price_col: str,
    fee_bps: float,
    target_end_seconds: int | None,
) -> pd.DataFrame:
    timed_horizons = [spec for spec in horizons if spec.seconds is not None]
    if not timed_horizons:
        return pd.DataFrame(columns=key_columns_for_merge(predictions))
    key_cols = key_columns_for_merge(predictions)
    if "decision_target_timestamp" not in key_cols:
        return pd.DataFrame(columns=key_cols)
    required = [*key_cols, "buy_price"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise SystemExit(f"sampled intraday labels require prediction columns: {missing}")

    base = predictions[required].copy()
    base["_row"] = np.arange(len(base), dtype="int64")
    base["date"] = base["date"].astype(str)
    base["symbol"] = base["symbol"].astype(str)
    base["decision_target_timestamp"] = pd.to_datetime(
        base["decision_target_timestamp"],
        errors="coerce",
    )
    base["buy_price"] = pd.to_numeric(base["buy_price"], errors="coerce")

    output = predictions[key_cols].copy()
    output["_row"] = np.arange(len(output), dtype="int64")
    right = context.rename(
        columns={
            "decision_target_timestamp": "_target_ts",
            exit_price_col: "_exit_price",
        }
    )[["date", "symbol", "_target_ts", "_exit_price"]]

    for spec in timed_horizons:
        target_col = label_column_name(spec.name)
        left = base[["_row", "date", "symbol", "decision_target_timestamp", "buy_price"]].copy()
        left["_target_ts"] = left["decision_target_timestamp"] + pd.to_timedelta(
            int(spec.seconds),
            unit="s",
        )
        if target_end_seconds is not None:
            target_seconds = (
                left["_target_ts"].dt.hour.astype("int64") * 3_600
                + left["_target_ts"].dt.minute.astype("int64") * 60
                + left["_target_ts"].dt.second.astype("int64")
            )
            left = left.loc[target_seconds <= int(target_end_seconds)].copy()
        merged = left.merge(
            right,
            on=["date", "symbol", "_target_ts"],
            how="left",
            sort=False,
        )
        exit_price = pd.to_numeric(merged["_exit_price"], errors="coerce")
        buy_price = pd.to_numeric(merged["buy_price"], errors="coerce")
        label = exit_price / buy_price - 1.0 - float(fee_bps) / 10_000.0
        label = label.where((exit_price > 0) & (buy_price > 0))
        aligned = pd.Series(label.to_numpy(), index=merged["_row"].to_numpy())
        output[target_col] = output["_row"].map(aligned)

    return output.drop(columns=["_row"]).drop_duplicates(key_cols)


def clickhouse_setting(value, env_name: str, default):
    if value not in (None, ""):
        return value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    return default


def query_trading_dates(
    client,
    *,
    table: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    table = validate_table_name(table)
    sql = f"""select distinct TradingDay as date
from {table}
where TradingDay >= {{start_date:String}}
  and TradingDay <= {{end_date:String}}
order by TradingDay"""
    frame = client.query_df(
        sql,
        parameters={"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return []
    return [str(value) for value in frame["date"].dropna().astype(str)]


def target_offset_us(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    hour = ts.dt.hour.astype("int64")
    minute = ts.dt.minute.astype("int64")
    second = ts.dt.second.astype("int64")
    microsecond = ts.dt.microsecond.astype("int64")
    return (
        (hour * 3_600 + minute * 60 + second) * 1_000_000
        + microsecond
    ).astype("Int64")


def query_intraday_mid_prices(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
    target_offsets: list[int],
    max_lag_seconds: int,
) -> pd.DataFrame:
    if not dates or not symbols or not target_offsets:
        return pd.DataFrame(
            columns=["date", "symbol", "target_offset_us", "exit_mid_price"]
        )
    table = validate_table_name(table)
    max_lag_us = int(max_lag_seconds) * 1_000_000
    min_offset = int(min(target_offsets))
    max_offset = int(max(target_offsets) + max_lag_us)
    sql = f"""select
    TradingDay as date,
    Symbol as symbol,
    target_offset_us,
    argMin((AskPrice1 + BidPrice1) / 2.0, ExchTimeOffsetUs) as exit_mid_price,
    min(ExchTimeOffsetUs) as matched_offset_us
from (
    select
        TradingDay,
        Symbol,
        ExchTimeOffsetUs,
        AskPrice1,
        BidPrice1,
        arrayJoin({{target_offsets:Array(UInt64)}}) as target_offset_us
    from {table}
    where TradingDay in {{dates:Array(String)}}
      and Symbol in {{symbols:Array(String)}}
      and ExchTimeOffsetUs >= {{min_offset_us:UInt64}}
      and ExchTimeOffsetUs <= {{max_offset_us:UInt64}}
      and AskPrice1 > 0
      and BidPrice1 > 0
)
where 1
  and ExchTimeOffsetUs >= target_offset_us
  and ExchTimeOffsetUs <= target_offset_us + {{max_lag_us:UInt64}}
group by TradingDay, Symbol, target_offset_us"""
    frame = client.query_df(
        sql,
        parameters={
            "dates": dates,
            "symbols": symbols,
            "target_offsets": [int(value) for value in target_offsets],
            "min_offset_us": min_offset,
            "max_offset_us": max_offset,
            "max_lag_us": max_lag_us,
        },
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "symbol", "target_offset_us", "exit_mid_price"]
        )
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["target_offset_us"] = pd.to_numeric(
        frame["target_offset_us"],
        errors="coerce",
    ).astype("Int64")
    frame["exit_mid_price"] = pd.to_numeric(
        frame["exit_mid_price"],
        errors="coerce",
    )
    return frame.dropna(subset=["target_offset_us", "exit_mid_price"])


def compute_clickhouse_intraday_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonSpec],
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    decision_max_lag_seconds: int,
    fee_bps: float,
    target_end_seconds: int | None,
) -> pd.DataFrame:
    timed_horizons = [spec for spec in horizons if spec.seconds is not None]
    key_cols = key_columns_for_merge(predictions)
    output = predictions[key_cols].copy()
    if not timed_horizons:
        return output
    if not username or not password:
        raise SystemExit(
            "ClickHouse intraday labels need credentials. Set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD or pass --clickhouse-user/--clickhouse-password."
        )
    if "decision_target_timestamp" not in predictions.columns:
        raise SystemExit("ClickHouse intraday labels require decision_target_timestamp")
    if "buy_price" not in predictions.columns:
        raise SystemExit("ClickHouse intraday labels require prediction column: buy_price")

    sample = predictions[[*key_cols, "buy_price"]].copy()
    sample["date"] = sample["date"].astype(str)
    sample["symbol"] = sample["symbol"].astype(str)
    sample["decision_target_timestamp"] = pd.to_datetime(
        sample["decision_target_timestamp"],
        errors="coerce",
    )
    sample["buy_price"] = pd.to_numeric(sample["buy_price"], errors="coerce")
    sample["_row"] = np.arange(len(sample), dtype="int64")
    sample = sample.dropna(subset=["decision_target_timestamp", "buy_price"])
    if sample.empty:
        return output.drop_duplicates(key_cols)

    target_frames = []
    for spec in timed_horizons:
        targets = sample[["_row", "date", "symbol", "decision_target_timestamp", "buy_price"]].copy()
        targets["horizon"] = spec.name
        targets["target_timestamp"] = targets["decision_target_timestamp"] + pd.to_timedelta(
            int(spec.seconds),
            unit="s",
        )
        if target_end_seconds is not None:
            target_seconds = (
                targets["target_timestamp"].dt.hour.astype("int64") * 3_600
                + targets["target_timestamp"].dt.minute.astype("int64") * 60
                + targets["target_timestamp"].dt.second.astype("int64")
            )
            targets = targets.loc[target_seconds <= int(target_end_seconds)].copy()
        targets["target_offset_us"] = target_offset_us(targets["target_timestamp"])
        target_frames.append(targets)
    target_frame = pd.concat(target_frames, ignore_index=True)
    target_frame = target_frame.dropna(subset=["target_offset_us"])
    if target_frame.empty:
        return output.drop_duplicates(key_cols)

    dates = sorted(target_frame["date"].dropna().astype(str).unique())
    symbols = sorted(target_frame["symbol"].dropna().astype(str).unique())
    offsets = sorted(int(value) for value in target_frame["target_offset_us"].dropna().unique())
    client = get_tick_client(
        host=host,
        port=int(port),
        username=username,
        password=password,
    )
    price_frame = query_intraday_mid_prices(
        client,
        table=table,
        dates=dates,
        symbols=symbols,
        target_offsets=offsets,
        max_lag_seconds=decision_max_lag_seconds,
    )
    if price_frame.empty:
        return output.drop_duplicates(key_cols)

    merged = target_frame.merge(
        price_frame[["date", "symbol", "target_offset_us", "exit_mid_price"]],
        on=["date", "symbol", "target_offset_us"],
        how="left",
    )
    merged["label"] = (
        pd.to_numeric(merged["exit_mid_price"], errors="coerce")
        / pd.to_numeric(merged["buy_price"], errors="coerce")
        - 1.0
        - float(fee_bps) / 10_000.0
    )
    labels_wide = merged.pivot_table(
        index="_row",
        columns="horizon",
        values="label",
        aggfunc="first",
    )
    output["_row"] = np.arange(len(output), dtype="int64")
    for spec in timed_horizons:
        if spec.name in labels_wide.columns:
            output[label_column_name(spec.name)] = output["_row"].map(
                labels_wide[spec.name]
            )
    return output.drop(columns=["_row"]).drop_duplicates(key_cols)


def query_close_prices(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
    close_offset_us: int,
    close_lookback_seconds: int,
) -> pd.DataFrame:
    if not dates or not symbols:
        return pd.DataFrame(columns=["date", "symbol", "close_price"])
    table = validate_table_name(table)
    start_offset = max(0, int(close_offset_us) - int(close_lookback_seconds) * 1_000_000)
    sql = f"""select
    TradingDay as date,
    Symbol as symbol,
    argMax(LastPrice, ExchTimeOffsetUs) as close_price
from {table}
where TradingDay in {{dates:Array(String)}}
  and Symbol in {{symbols:Array(String)}}
  and ExchTimeOffsetUs >= {{start_offset_us:UInt64}}
  and ExchTimeOffsetUs <= {{close_offset_us:UInt64}}
  and LastPrice > 0
group by TradingDay, Symbol"""
    frame = client.query_df(
        sql,
        parameters={
            "dates": dates,
            "symbols": symbols,
            "start_offset_us": int(start_offset),
            "close_offset_us": int(close_offset_us),
        },
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "close_price"])
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["close_price"] = pd.to_numeric(frame["close_price"], errors="coerce")
    return frame.dropna(subset=["close_price"])


def compute_clickhouse_close_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonSpec],
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    close_offset_us: int,
    close_lookback_seconds: int,
    calendar_days_after: int,
    fee_bps: float,
) -> pd.DataFrame:
    close_horizons = {spec.name for spec in horizons if spec.name in {"close", "next_close"}}
    key_cols = key_columns_for_merge(predictions)
    output = predictions[key_cols].copy()
    if not close_horizons:
        return output
    if not username or not password:
        raise SystemExit(
            "ClickHouse close labels need credentials. Set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD or pass --clickhouse-user/--clickhouse-password."
        )
    if "buy_price" not in predictions.columns:
        raise SystemExit("close labels require prediction column: buy_price")

    sample = predictions[["date", "symbol", "buy_price", *key_cols[2:]]].copy()
    sample["date"] = sample["date"].astype(str)
    sample["symbol"] = sample["symbol"].astype(str)
    sample["buy_price"] = pd.to_numeric(sample["buy_price"], errors="coerce")
    unique_dates = sorted(sample["date"].dropna().unique())
    unique_symbols = sorted(sample["symbol"].dropna().unique())
    if not unique_dates or not unique_symbols:
        return output

    start_date = str(pd.Timestamp(unique_dates[0]).date())
    end_date = str(
        (
            pd.Timestamp(unique_dates[-1]) + pd.Timedelta(days=int(calendar_days_after))
        ).date()
    )
    client = get_tick_client(
        host=host,
        port=int(port),
        username=username,
        password=password,
    )
    trading_dates = query_trading_dates(
        client,
        table=table,
        start_date=start_date,
        end_date=end_date,
    )
    needed_dates = [date for date in trading_dates if date >= start_date]
    close_prices = query_close_prices(
        client,
        table=table,
        dates=needed_dates,
        symbols=unique_symbols,
        close_offset_us=close_offset_us,
        close_lookback_seconds=close_lookback_seconds,
    )
    if close_prices.empty:
        return output
    next_date = {
        date: trading_dates[index + 1]
        for index, date in enumerate(trading_dates[:-1])
        if date in set(unique_dates)
    }
    close_by_key = close_prices.set_index(["date", "symbol"])["close_price"]
    sample_index = pd.MultiIndex.from_frame(sample[["date", "symbol"]])

    if "close" in close_horizons:
        close_price = close_by_key.reindex(sample_index).to_numpy()
        label = close_price / sample["buy_price"].to_numpy(dtype="float64") - 1.0
        output[label_column_name("close")] = label - float(fee_bps) / 10_000.0

    if "next_close" in close_horizons:
        next_keys = pd.MultiIndex.from_arrays(
            [
                sample["date"].map(next_date).astype("object"),
                sample["symbol"].astype(str),
            ],
            names=["date", "symbol"],
        )
        next_close_price = close_by_key.reindex(next_keys).to_numpy()
        label = next_close_price / sample["buy_price"].to_numpy(dtype="float64") - 1.0
        output[label_column_name("next_close")] = label - float(fee_bps) / 10_000.0

    return output.drop_duplicates(key_cols)


def numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def attach_horizon_labels(
    predictions: pd.DataFrame,
    *,
    horizons: list[HorizonSpec],
    label_input: str,
    tick_input: str,
    sample_context: str,
    sampled_intraday: bool,
    sampled_exit_price_col: str,
    clickhouse_intraday_labels: bool,
    clickhouse_close_labels: bool,
    clickhouse_host: str,
    clickhouse_port: int,
    clickhouse_user: str,
    clickhouse_password: str,
    clickhouse_table: str,
    clickhouse_close_offset_us: int,
    clickhouse_close_lookback_seconds: int,
    clickhouse_calendar_days_after: int,
    clickhouse_decision_max_lag_seconds: int,
    timed_target_end_seconds: int | None,
    explicit: dict[str, str],
    volume_col: str,
    turnover_col: str,
    volume_unit_multiplier: float,
    sell_window_seconds: int,
    fee_bps: float,
    price_col: str,
    open_time: str,
    close_time: str,
    max_future_gap_seconds: float | None,
    max_price_gap_seconds: float | None,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    trace: dict[str, object] = {}
    out = attach_available_prediction_labels(predictions, horizons, explicit)

    if label_input:
        out = merge_label_input(out, Path(label_input), horizons, explicit)
        out = attach_available_prediction_labels(out, horizons, explicit)
        trace["label_input"] = label_input

    if sampled_intraday:
        context = load_sample_context(
            out,
            sample_context,
            exit_price_col=sampled_exit_price_col,
        )
        labels = compute_sampled_intraday_labels(
            out,
            context,
            horizons,
            exit_price_col=sampled_exit_price_col,
            fee_bps=fee_bps,
            target_end_seconds=timed_target_end_seconds,
        )
        label_path = output_root / "sampled_intraday_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = out.merge(
            labels,
            on=key_columns_for_merge(out),
            how="left",
            suffixes=("", "_samplectx"),
        )
        for spec in horizons:
            column = label_column_name(spec.name)
            sample_column = f"{column}_samplectx"
            if sample_column in out.columns:
                out[column] = numeric_or_nan(out, column).combine_first(
                    pd.to_numeric(out[sample_column], errors="coerce")
                )
                out = out.drop(columns=[sample_column])
        trace["sample_context"] = sample_context or "run_inputs"
        trace["sampled_exit_price_col"] = sampled_exit_price_col
        trace["sampled_intraday_labels"] = str(label_path)

    if clickhouse_intraday_labels:
        labels = compute_clickhouse_intraday_labels(
            out,
            horizons,
            host=clickhouse_host,
            port=clickhouse_port,
            username=clickhouse_user,
            password=clickhouse_password,
            table=clickhouse_table,
            decision_max_lag_seconds=clickhouse_decision_max_lag_seconds,
            fee_bps=fee_bps,
            target_end_seconds=timed_target_end_seconds,
        )
        label_path = output_root / "clickhouse_intraday_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = out.merge(
            labels,
            on=key_columns_for_merge(out),
            how="left",
            suffixes=("", "_chctx"),
        )
        for spec in horizons:
            column = label_column_name(spec.name)
            ch_column = f"{column}_chctx"
            if ch_column in out.columns:
                out[column] = numeric_or_nan(out, column).combine_first(
                    pd.to_numeric(out[ch_column], errors="coerce")
                )
                out = out.drop(columns=[ch_column])
        trace["clickhouse_intraday_labels"] = str(label_path)
        trace["clickhouse_decision_max_lag_seconds"] = clickhouse_decision_max_lag_seconds

    if tick_input:
        labels = compute_tick_horizon_labels(
            out,
            Path(tick_input),
            horizons,
            volume_col=volume_col,
            turnover_col=turnover_col,
            volume_unit_multiplier=volume_unit_multiplier,
            sell_window_seconds=sell_window_seconds,
            fee_bps=fee_bps,
            price_col=price_col,
            open_time=open_time,
            close_time=close_time,
            max_future_gap_seconds=max_future_gap_seconds,
            max_price_gap_seconds=max_price_gap_seconds,
        )
        label_path = output_root / "alpha_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = out.merge(
            labels,
            on=key_columns_for_merge(out),
            how="left",
            suffixes=("", "_tickctx"),
        )
        for spec in horizons:
            column = label_column_name(spec.name)
            tick_column = f"{column}_tickctx"
            if tick_column in out.columns:
                out[column] = numeric_or_nan(out, column).combine_first(
                    pd.to_numeric(out[tick_column], errors="coerce")
                )
                out = out.drop(columns=[tick_column])
        trace["tick_input"] = tick_input
        trace["horizon_labels"] = str(label_path)

    if clickhouse_close_labels:
        labels = compute_clickhouse_close_labels(
            out,
            horizons,
            host=clickhouse_host,
            port=clickhouse_port,
            username=clickhouse_user,
            password=clickhouse_password,
            table=clickhouse_table,
            close_offset_us=clickhouse_close_offset_us,
            close_lookback_seconds=clickhouse_close_lookback_seconds,
            calendar_days_after=clickhouse_calendar_days_after,
            fee_bps=fee_bps,
        )
        label_path = output_root / "clickhouse_close_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = out.merge(
            labels,
            on=key_columns_for_merge(out),
            how="left",
            suffixes=("", "_closectx"),
        )
        for horizon in ("close", "next_close"):
            column = label_column_name(horizon)
            close_column = f"{column}_closectx"
            if close_column in out.columns:
                out[column] = numeric_or_nan(out, column).combine_first(
                    pd.to_numeric(out[close_column], errors="coerce")
                )
                out = out.drop(columns=[close_column])
        trace["clickhouse_close_labels"] = str(label_path)
        trace["clickhouse_table"] = clickhouse_table

    return out, trace


def rank_ic_by_group(
    frame: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    group_cols: tuple[str, ...],
) -> tuple[float, float, float, int]:
    resolved = resolve_group_cols(frame, group_cols)
    if not resolved:
        valid = frame[[label_col, score_col]].dropna()
        if len(valid) < 2:
            return float("nan"), float("nan"), float("nan"), 0
        corr = valid[label_col].rank(method="average").corr(
            valid[score_col].rank(method="average")
        )
        return float(corr), float("nan"), float("nan"), 1
    values = []
    for _, group in frame.groupby(list(resolved), sort=False, observed=True):
        valid = group[[label_col, score_col]].dropna()
        if len(valid) < 3:
            continue
        if valid[label_col].nunique() < 2 or valid[score_col].nunique() < 2:
            continue
        corr = valid[label_col].rank(method="average").corr(
            valid[score_col].rank(method="average")
        )
        if pd.notna(corr):
            values.append(float(corr))
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    mean = float(series.mean())
    ir = mean / std if std and np.isfinite(std) else float("nan")
    return mean, std, ir, int(len(series))


def select_bottom_score_trades(
    frame: pd.DataFrame,
    *,
    top_n: int,
    label_col: str,
    score_col: str,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    work = frame.copy()
    work["_inverse_prediction"] = -pd.to_numeric(work[score_col], errors="coerce")
    return top_score_trades(
        work,
        top_n=top_n,
        label_col=label_col,
        score_col="_inverse_prediction",
        group_cols=group_cols,
    ).drop(columns=["_inverse_prediction"], errors="ignore")


def group_return_sem(
    trades: pd.DataFrame,
    *,
    label_col: str,
    group_cols: tuple[str, ...],
) -> float:
    resolved = resolve_group_cols(trades, group_cols)
    if not resolved or trades.empty:
        return float("nan")
    group_means = trades.groupby(list(resolved), observed=True)[label_col].mean()
    if len(group_means) < 2:
        return float("nan")
    return float(group_means.std(ddof=1) / np.sqrt(len(group_means)))


def summarize_horizon(
    frame: pd.DataFrame,
    *,
    branch: str,
    horizon: HorizonSpec,
    label_col: str,
    score_col: str,
    top_n: int,
    group_cols: tuple[str, ...],
) -> dict[str, object]:
    valid = frame.loc[
        pd.to_numeric(frame[label_col], errors="coerce").notna()
        & pd.to_numeric(frame[score_col], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return {
            "branch": branch,
            "horizon": horizon.name,
            "horizon_label": horizon.label,
            "horizon_seconds": horizon.seconds,
            "label_col": label_col,
            "rows": 0,
        }
    valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce")
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    resolved_groups = resolve_group_cols(valid, group_cols)
    top = top_score_trades(
        valid,
        top_n=top_n,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    bottom = select_bottom_score_trades(
        valid,
        top_n=top_n,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    rank_ic_mean, rank_ic_std, rank_ic_ir, rank_ic_groups = rank_ic_by_group(
        valid,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    top_mean = float(top[label_col].mean()) if not top.empty else float("nan")
    bottom_mean = float(bottom[label_col].mean()) if not bottom.empty else float("nan")
    return {
        "branch": branch,
        "horizon": horizon.name,
        "horizon_label": horizon.label,
        "horizon_seconds": horizon.seconds,
        "label_col": label_col,
        "rows": int(len(valid)),
        "dates": int(valid["date"].nunique()) if "date" in valid else 0,
        "symbols": int(valid["symbol"].nunique()) if "symbol" in valid else 0,
        "groups": int(valid.groupby(list(resolved_groups)).ngroups)
        if resolved_groups
        else 1,
        "group_cols": ",".join(resolved_groups) if resolved_groups else "global",
        "top_n": int(top_n),
        "top_trades": int(len(top)),
        "top_groups": int(top.groupby(list(resolved_groups)).ngroups)
        if resolved_groups and not top.empty
        else (1 if not top.empty else 0),
        "mean_alpha_return": top_mean,
        "mean_alpha_return_bps": top_mean * 10_000.0,
        "mean_alpha_return_sem": group_return_sem(
            top,
            label_col=label_col,
            group_cols=group_cols,
        ),
        "mean_alpha_return_bps_sem": group_return_sem(
            top,
            label_col=label_col,
            group_cols=group_cols,
        )
        * 10_000.0,
        "median_alpha_return": float(top[label_col].median())
        if not top.empty
        else float("nan"),
        "top_win_rate": float((top[label_col] > 0).mean())
        if not top.empty
        else float("nan"),
        "all_mean_return": float(valid[label_col].mean()),
        "all_mean_return_bps": float(valid[label_col].mean() * 10_000.0),
        "bottom_mean_return": bottom_mean,
        "bottom_mean_return_bps": bottom_mean * 10_000.0,
        "top_bottom_spread": top_mean - bottom_mean,
        "top_bottom_spread_bps": (top_mean - bottom_mean) * 10_000.0,
        "group_rank_ic_mean": rank_ic_mean,
        "group_rank_ic_std": rank_ic_std,
        "group_rank_ic_ir": rank_ic_ir,
        "group_rank_ic_groups": rank_ic_groups,
    }


def build_summary_tables(
    predictions: pd.DataFrame,
    *,
    horizons: list[HorizonSpec],
    top_n: int,
    score_bins: int,
    group_cols: tuple[str, ...],
    allow_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_rows = []
    bucket_frames = []
    missing: list[str] = []
    for branch, branch_frame in predictions.groupby("branch", sort=False):
        for spec in horizons:
            label_col = label_column_name(spec.name)
            if label_col not in branch_frame.columns:
                missing.append(f"{branch}:{spec.name}")
                continue
            non_null = pd.to_numeric(branch_frame[label_col], errors="coerce").notna()
            if not non_null.any():
                missing.append(f"{branch}:{spec.name}")
                continue
            summary_rows.append(
                summarize_horizon(
                    branch_frame,
                    branch=branch,
                    horizon=spec,
                    label_col=label_col,
                    score_col="prediction",
                    top_n=top_n,
                    group_cols=group_cols,
                )
            )
            buckets = score_bucket_returns(
                branch_frame,
                bins=score_bins,
                label_col=label_col,
                score_col="prediction",
                group_cols=group_cols,
            )
            buckets.insert(0, "horizon", spec.name)
            buckets.insert(0, "branch", branch)
            bucket_frames.append(buckets)

    if missing and not allow_missing:
        raise SystemExit(
            "missing requested horizon labels: "
            + ", ".join(missing)
            + ". Provide --tick-input, --label-input/--horizon-label, or use "
            "--allow-missing-horizons for partial output."
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = add_decay_retention(summary, horizons)
    buckets = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
    return summary, buckets, missing


def add_decay_retention(summary: pd.DataFrame, horizons: list[HorizonSpec]) -> pd.DataFrame:
    out = summary.copy()
    horizon_order = {spec.name: index for index, spec in enumerate(horizons)}
    branch_order = {
        branch: index for index, branch in enumerate(out["branch"].drop_duplicates())
    }
    out["horizon_order"] = out["horizon"].map(horizon_order)
    out["branch_order"] = out["branch"].map(branch_order)
    out = out.sort_values(["branch_order", "horizon_order"])
    out["retention_vs_first"] = np.nan
    out["retention_vs_60s"] = np.nan
    for branch, branch_frame in out.groupby("branch", sort=False):
        first = branch_frame["mean_alpha_return_bps"].dropna()
        first_value = first.iloc[0] if not first.empty else np.nan
        sixty = branch_frame.loc[
            branch_frame["horizon"] == "60s",
            "mean_alpha_return_bps",
        ].dropna()
        sixty_value = sixty.iloc[0] if not sixty.empty else np.nan
        branch_index = branch_frame.index
        if pd.notna(first_value) and first_value != 0:
            out.loc[branch_index, "retention_vs_first"] = (
                out.loc[branch_index, "mean_alpha_return_bps"] / first_value
            )
        if pd.notna(sixty_value) and sixty_value != 0:
            out.loc[branch_index, "retention_vs_60s"] = (
                out.loc[branch_index, "mean_alpha_return_bps"] / sixty_value
            )
    return out


def plot_mean_alpha_return(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    if summary.empty:
        return
    sort_cols = ["horizon_order", "branch_order"] if "branch_order" in summary else ["horizon_order", "branch"]
    table = summary.sort_values(sort_cols)
    horizons = list(table["horizon"].drop_duplicates())
    branch_sort_cols = ["branch_order"] if "branch_order" in table else ["branch"]
    branches = list(
        table.sort_values(branch_sort_cols)["branch"].drop_duplicates()
    )
    x = np.arange(len(horizons))
    width = 0.78 / max(1, len(branches))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, branch in enumerate(branches):
        branch_table = table.loc[table["branch"] == branch].set_index("horizon")
        values = [branch_table.loc[h, "mean_alpha_return_bps"] if h in branch_table.index else np.nan for h in horizons]
        errors = [
            branch_table.loc[h, "mean_alpha_return_bps_sem"] if h in branch_table.index else np.nan
            for h in horizons
        ]
        offset = (idx - (len(branches) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=branch,
            color=RUN_COLORS.get(branch, None),
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            va = "bottom" if value >= 0 else "top"
            y = value + (1.2 if value >= 0 else -1.2)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.1f}",
                ha="center",
                va=va,
                fontsize=8,
            )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([h.replace("_", "\n") for h in horizons])
    ax.set_ylabel("Mean alpha return (bps)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rank_ic(summary: pd.DataFrame, output_path: Path, *, title: str) -> None:
    if summary.empty or "group_rank_ic_mean" not in summary:
        return
    sort_cols = ["horizon_order", "branch_order"] if "branch_order" in summary else ["horizon_order", "branch"]
    table = summary.sort_values(sort_cols)
    horizons = list(table["horizon"].drop_duplicates())
    branch_sort_cols = ["branch_order"] if "branch_order" in table else ["branch"]
    branches = list(table.sort_values(branch_sort_cols)["branch"].drop_duplicates())
    x = np.arange(len(horizons), dtype="float64")
    width = 0.78 / max(1, len(branches))

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for idx, branch in enumerate(branches):
        branch_table = table.loc[table["branch"] == branch].set_index("horizon")
        values = [
            float(branch_table.loc[horizon, "group_rank_ic_mean"])
            if horizon in branch_table.index
            else np.nan
            for horizon in horizons
        ]
        offset = (idx - (len(branches) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=branch,
            color=RUN_COLORS.get(branch, None),
            alpha=0.9,
        )
        for bar, value in zip(bars, values, strict=True):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.004 if value >= 0 else -0.004),
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([horizon.replace("_", "\n") for horizon in horizons])
    ax.set_ylabel("Group rank IC mean")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    runs = args.run or [RunInput(label, Path(path)) for label, path in DEFAULT_RUNS]
    horizons = horizon_specs(args.horizon or DEFAULT_HORIZONS)
    explicit = explicit_label_map(args.horizon_label)
    decision_times = parse_clock_values(args.decision_time)
    group_cols = tuple(args.group_col or DEFAULT_GROUP_COLS)
    max_future_gap = args.max_future_gap_seconds or None
    max_price_gap = args.max_price_gap_seconds or None
    timed_target_end_seconds = optional_clock_seconds(args.timed_target_end_time)

    prediction_frames = [
        filter_decision_times(load_prediction(run.path, run.label), decision_times)
        for run in runs
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    clickhouse_host = str(
        clickhouse_setting(
            args.clickhouse_host,
            "CLICKHOUSE_HOST",
            DEFAULT_CLICKHOUSE_TICK_HOST,
        )
    )
    clickhouse_port = int(
        clickhouse_setting(
            args.clickhouse_port,
            "CLICKHOUSE_PORT",
            DEFAULT_CLICKHOUSE_TICK_PORT,
        )
    )
    clickhouse_user = str(
        clickhouse_setting(args.clickhouse_user, "CLICKHOUSE_USER", "") or ""
    )
    clickhouse_password = str(
        clickhouse_setting(args.clickhouse_password, "CLICKHOUSE_PASSWORD", "") or ""
    )
    clickhouse_table = str(
        clickhouse_setting(
            args.clickhouse_table,
            "CLICKHOUSE_TICK_TABLE",
            DEFAULT_CLICKHOUSE_TICK_TABLE,
        )
    )
    predictions, attach_trace = attach_horizon_labels(
        predictions,
        horizons=horizons,
        label_input=args.label_input,
        tick_input=args.tick_input,
        sample_context=args.sample_context,
        sampled_intraday=not args.no_sampled_intraday,
        sampled_exit_price_col=args.sampled_exit_price_col,
        clickhouse_intraday_labels=args.clickhouse_intraday_labels,
        clickhouse_close_labels=args.clickhouse_close_labels,
        clickhouse_host=clickhouse_host,
        clickhouse_port=clickhouse_port,
        clickhouse_user=clickhouse_user,
        clickhouse_password=clickhouse_password,
        clickhouse_table=clickhouse_table,
        clickhouse_close_offset_us=args.clickhouse_close_offset_us,
        clickhouse_close_lookback_seconds=args.clickhouse_close_lookback_seconds,
        clickhouse_calendar_days_after=args.clickhouse_calendar_days_after,
        clickhouse_decision_max_lag_seconds=args.clickhouse_decision_max_lag_seconds,
        timed_target_end_seconds=timed_target_end_seconds,
        explicit=explicit,
        volume_col=args.volume_col,
        turnover_col=args.turnover_col,
        volume_unit_multiplier=args.volume_unit_multiplier,
        sell_window_seconds=args.sell_window_seconds,
        fee_bps=args.fee_bps,
        price_col=args.price_col,
        open_time=args.open_time,
        close_time=args.close_time,
        max_future_gap_seconds=max_future_gap,
        max_price_gap_seconds=max_price_gap,
        output_root=output_root,
    )
    summary, buckets, missing = build_summary_tables(
        predictions,
        horizons=horizons,
        top_n=args.top_n,
        score_bins=args.score_bins,
        group_cols=group_cols,
        allow_missing=args.allow_missing_horizons,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "alpha_horizon_decay_summary.csv"
    buckets_path = output_root / "alpha_horizon_decay_buckets.csv"
    trace_path = output_root / "alpha_horizon_decay_trace.json"
    summary.to_csv(summary_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    mean_plot_path = output_root / "alpha_horizon_decay_mean_return.png"
    rank_ic_plot_path = output_root / "alpha_horizon_decay_rank_ic.png"
    plot_mean_alpha_return(
        summary,
        mean_plot_path,
        title=f"Delay2 opening score horizon decay, Top{args.top_n}",
    )
    plot_rank_ic(
        summary,
        rank_ic_plot_path,
        title="Delay2 opening score rank IC by horizon",
    )
    trace = {
        "runs": [{"label": run.label, "path": str(run.path)} for run in runs],
        "horizons": [spec.__dict__ for spec in horizons],
        "decision_times": sorted(decision_times),
        "group_cols": group_cols,
        "top_n": args.top_n,
        "score_bins": args.score_bins,
        "sampled_intraday": not args.no_sampled_intraday,
        "sampled_exit_price_col": args.sampled_exit_price_col,
        "timed_target_end_time": args.timed_target_end_time,
        "clickhouse_intraday_labels": bool(args.clickhouse_intraday_labels),
        "clickhouse_close_labels": bool(args.clickhouse_close_labels),
        "missing_horizons": missing,
        "summary": str(summary_path),
        "buckets": str(buckets_path),
        "mean_return_plot": str(mean_plot_path),
        "rank_ic_plot": str(rank_ic_plot_path),
        **attach_trace,
    }
    write_json(trace_path, trace)
    print(
        json.dumps(
            json_ready(
                {
                    "summary": summary_path,
                    "buckets": buckets_path,
                    "mean_return_plot": mean_plot_path,
                    "rank_ic_plot": rank_ic_plot_path,
                    "trace": trace_path,
                    "missing_horizons": missing,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
