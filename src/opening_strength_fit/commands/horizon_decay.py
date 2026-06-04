from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.commands.horizon_clickhouse_labels import (
    clickhouse_setting,
    compute_clickhouse_close_labels,
    compute_clickhouse_intraday_labels,
)
from opening_strength_fit.commands.horizon_clickhouse_labels import (
    query_close_prices as query_close_prices,
)
from opening_strength_fit.commands.horizon_local_labels import (
    attach_available_prediction_labels,
    compute_sampled_intraday_labels,
    compute_tick_horizon_labels,
    explicit_label_map,
    filter_decision_times,
    key_columns_for_merge,
    label_column_name,
    load_prediction,
    load_sample_context,
    merge_label_input,
)
from opening_strength_fit.commands.horizon_reporting import (
    build_summary_tables,
    plot_mean_alpha_return,
    plot_rank_ic,
)
from opening_strength_fit.io import write_frame

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
        raise argparse.ArgumentTypeError("--horizon-label must be formatted as horizon=column")
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
        help=("Horizon to evaluate. Defaults to 1m, 2m, 5m, 10m, close, and next_close."),
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
        help=("Optional as-of tolerance for intraday VWAP endpoints. 0 disables the tolerance."),
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
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        filter_decision_times(load_prediction(run.path, run.label), decision_times) for run in runs
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
    clickhouse_user = str(clickhouse_setting(args.clickhouse_user, "CLICKHOUSE_USER", "") or "")
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
