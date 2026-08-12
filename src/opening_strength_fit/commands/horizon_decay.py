from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    DEFAULT_CLOSE_LOOKBACK_SECONDS,
    DEFAULT_CLOSE_OFFSET_US,
    clickhouse_setting,
    compute_clickhouse_close_labels,
    compute_clickhouse_intraday_labels,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    query_close_prices as query_close_prices,
)
from opening_strength_fit.horizon_local_labels import (
    attach_available_prediction_labels,
    compute_sampled_intraday_labels,
    compute_tick_horizon_labels,
    explicit_label_map,
    filter_decision_times,
    load_prediction,
    load_sample_context,
    merge_label_input,
)
from opening_strength_fit.horizon_reporting import (
    build_summary_tables,
    plot_mean_alpha_return,
    plot_rank_ic,
)
from opening_strength_fit.horizons import (
    HorizonSpec,
    horizon_specs,
    key_columns_for_merge,
    label_column_name,
    normalize_horizon_name,
)
from opening_strength_fit.io import json_safe, write_frame, write_json

DEFAULT_OUTPUT_ROOT = "output/legacy/reports/opening_alpha_horizon_decay_delay2_cohort_avg"
DEFAULT_RUNS = (
    (
        "Universe",
        "output/legacy/predictions/lgbm_opening_1y_next_month_delay2/predictions_all.parquet",
    ),
    (
        "Strong",
        "output/legacy/predictions/lgbm_opening_1y_next_month_strong_delay2/predictions_all.parquet",
    ),
)
DEFAULT_HORIZONS = tuple("1m 2m 5m 10m close next_close".split())
DEFAULT_GROUP_COLS = ("date", "decision_target_timestamp")
DEFAULT_DECISION_MAX_LAG_SECONDS = 5
DEFAULT_TIMED_TARGET_END_TIME = "09:40:00"


@dataclass(frozen=True)
class RunInput:
    label: str
    path: Path


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
    add = parser.add_argument
    for option, kwargs in (
        (
            "run",
            {
                "action": "append",
                "type": parse_run,
                "help": "Prediction input formatted as label=path. Defaults to local delay2 Universe and Strong prediction files.",
            },
        ),
        (
            "label-input",
            {
                "default": "",
                "help": "Optional horizon-label parquet/csv keyed by date, symbol, and decision_target_timestamp. Use for precomputed close/T+1 labels.",
            },
        ),
        (
            "tick-input",
            {
                "default": "",
                "help": "Optional raw tick parquet/csv context used to compute horizon labels. It must include cumulative volume/turnover for intraday VWAP labels and a price column for close/next open/next close labels.",
            },
        ),
        (
            "horizon",
            {
                "action": "append",
                "help": "Horizon to evaluate. Defaults to 1m, 2m, 5m, 10m, close, and next_close.",
            },
        ),
        (
            "horizon-label",
            {
                "action": "append",
                "type": parse_label_mapping,
                "help": "Explicit mapping from horizon to an existing label column, for example --horizon-label 5m=alpha_return_5m.",
            },
        ),
    ):
        add(f"--{option}", **kwargs)
    add("--top-n", type=int, default=20)
    add("--score-bins", type=int, default=10)
    add(
        "--group-col",
        action="append",
        help="Grouping column for cross-section selection and IC. Defaults to date and decision_target_timestamp.",
    )
    add(
        "--decision-time",
        action="append",
        help="Optional decision clock filter. Can be repeated or comma-separated, for example --decision-time 09:30:00,09:32:00.",
    )
    add(
        "--sell-window-seconds",
        type=int,
        default=60,
        help="VWAP exit window width for intraday horizons.",
    )
    add("--fee-bps", type=float, default=0.0)
    add(
        "--sample-context",
        default="",
        help="Optional delay2 labeled cache or prediction parquet used as the opening-window price context. Defaults to the union of --run inputs.",
    )
    add(
        "--no-sampled-intraday",
        action="store_true",
        help="Do not derive 1m..10m labels from sampled opening decision rows.",
    )
    add(
        "--sampled-exit-price-col",
        default="mid_price",
        help="Exit price column for opening-window sampled decay. The default mid_price avoids mixing 60s VWAP labels with minute sampled exits.",
    )
    add("--volume-col", default="volume")
    add("--turnover-col", default="turnover")
    add("--volume-unit-multiplier", type=float, default=1.0)
    add(
        "--price-col",
        default="auto",
        help="Price column for close/next open/next close exits. auto uses mid_price, then last_price, then bid/ask midpoint.",
    )
    add("--open-time", default="09:30:00")
    add("--close-time", default="15:00:00")
    add(
        "--timed-target-end-time",
        default=DEFAULT_TIMED_TARGET_END_TIME,
        help="Latest target clock for timed intraday horizons. Defaults to 09:40:00 so cache and ClickHouse runs compare the same opening-window targets. Use none to disable.",
    )
    add(
        "--clickhouse-close-labels",
        action="store_true",
        help="Fetch same-day close and next-trading-day close prices from ClickHouse for requested close horizons.",
    )
    add(
        "--clickhouse-intraday-labels",
        action="store_true",
        help="Fetch opening-window target-minute bid/ask mid prices from ClickHouse for timed horizons such as 1m, 2m, 5m, and 10m.",
    )
    add("--clickhouse-host", default=None)
    add("--clickhouse-port", type=int, default=None)
    add("--clickhouse-user", default=None)
    add("--clickhouse-password", default=None)
    add("--clickhouse-table", default=None)
    add("--clickhouse-close-offset-us", type=int, default=DEFAULT_CLOSE_OFFSET_US)
    add("--clickhouse-close-lookback-seconds", type=int, default=DEFAULT_CLOSE_LOOKBACK_SECONDS)
    add(
        "--clickhouse-calendar-days-after",
        type=int,
        default=14,
        help="Calendar-day padding after the last sample date for next_close.",
    )
    add(
        "--clickhouse-decision-max-lag-seconds",
        type=int,
        default=DEFAULT_DECISION_MAX_LAG_SECONDS,
        help="Maximum target-minute sampling lag when deriving intraday labels from ClickHouse.",
    )
    add(
        "--max-future-gap-seconds",
        type=float,
        default=0.0,
        help="Optional as-of tolerance for intraday VWAP endpoints. 0 disables the tolerance.",
    )
    add(
        "--max-price-gap-seconds",
        type=float,
        default=0.0,
        help="Optional as-of tolerance for close/next open/next close price exits. 0 disables the tolerance.",
    )
    add(
        "--allow-missing-horizons",
        action="store_true",
        help="Continue when a requested horizon label is unavailable.",
    )
    add(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for summary CSVs, labels, figures, and trace.",
    )
    return parser.parse_args()


def _merge_preferred_horizon_labels(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: list[HorizonSpec] | tuple[str, ...],
    suffix: str,
) -> pd.DataFrame:
    out = frame.merge(
        labels,
        on=key_columns_for_merge(frame),
        how="left",
        suffixes=("", f"_{suffix}"),
    )
    for horizon in horizons:
        column = label_column_name(horizon.name if isinstance(horizon, HorizonSpec) else horizon)
        incoming = f"{column}_{suffix}"
        if incoming in out.columns:
            current = (
                pd.to_numeric(out[column], errors="coerce")
                if column in out.columns
                else pd.Series(float("nan"), index=out.index, dtype="float64")
            )
            out[column] = current.combine_first(pd.to_numeric(out.pop(incoming), errors="coerce"))
    return out


def attach_horizon_labels(
    predictions: pd.DataFrame,
    *,
    horizons: list[HorizonSpec],
    args: argparse.Namespace,
    timed_target_end_seconds: int | None,
    explicit: dict[str, str],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    clickhouse = {
        "host": str(
            clickhouse_setting(
                args.clickhouse_host, "CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST
            )
        ),
        "port": int(
            clickhouse_setting(
                args.clickhouse_port, "CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT
            )
        ),
        "username": str(clickhouse_setting(args.clickhouse_user, "CLICKHOUSE_USER", "") or ""),
        "password": str(
            clickhouse_setting(args.clickhouse_password, "CLICKHOUSE_PASSWORD", "") or ""
        ),
        "table": str(
            clickhouse_setting(
                args.clickhouse_table, "CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE
            )
        ),
    }
    trace: dict[str, object] = {}
    out = attach_available_prediction_labels(predictions, horizons, explicit)

    if args.label_input:
        out = merge_label_input(out, Path(args.label_input), horizons, explicit)
        out = attach_available_prediction_labels(out, horizons, explicit)
        trace["label_input"] = args.label_input

    if not args.no_sampled_intraday:
        context = load_sample_context(
            out,
            args.sample_context,
            exit_price_col=args.sampled_exit_price_col,
        )
        labels = compute_sampled_intraday_labels(
            out,
            context,
            horizons,
            exit_price_col=args.sampled_exit_price_col,
            fee_bps=args.fee_bps,
            target_end_seconds=timed_target_end_seconds,
        )
        label_path = output_root / "sampled_intraday_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = _merge_preferred_horizon_labels(out, labels, horizons, "samplectx")
        trace["sample_context"] = args.sample_context or "run_inputs"
        trace["sampled_exit_price_col"] = args.sampled_exit_price_col
        trace["sampled_intraday_labels"] = str(label_path)

    if args.clickhouse_intraday_labels:
        labels = compute_clickhouse_intraday_labels(
            out,
            horizons,
            **clickhouse,
            decision_max_lag_seconds=args.clickhouse_decision_max_lag_seconds,
            fee_bps=args.fee_bps,
            target_end_seconds=timed_target_end_seconds,
        )
        label_path = output_root / "clickhouse_intraday_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = _merge_preferred_horizon_labels(out, labels, horizons, "chctx")
        trace["clickhouse_intraday_labels"] = str(label_path)
        trace["clickhouse_decision_max_lag_seconds"] = args.clickhouse_decision_max_lag_seconds

    if args.tick_input:
        labels = compute_tick_horizon_labels(
            out,
            Path(args.tick_input),
            horizons,
            volume_col=args.volume_col,
            turnover_col=args.turnover_col,
            volume_unit_multiplier=args.volume_unit_multiplier,
            sell_window_seconds=args.sell_window_seconds,
            fee_bps=args.fee_bps,
            price_col=args.price_col,
            open_time=args.open_time,
            close_time=args.close_time,
            max_future_gap_seconds=args.max_future_gap_seconds or None,
            max_price_gap_seconds=args.max_price_gap_seconds or None,
        )
        label_path = output_root / "alpha_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = _merge_preferred_horizon_labels(out, labels, horizons, "tickctx")
        trace["tick_input"] = args.tick_input
        trace["horizon_labels"] = str(label_path)

    if args.clickhouse_close_labels:
        labels = compute_clickhouse_close_labels(
            out,
            horizons,
            **clickhouse,
            close_offset_us=args.clickhouse_close_offset_us,
            close_lookback_seconds=args.clickhouse_close_lookback_seconds,
            calendar_days_after=args.clickhouse_calendar_days_after,
            fee_bps=args.fee_bps,
        )
        label_path = output_root / "clickhouse_close_horizon_labels.parquet"
        write_frame(labels, label_path)
        out = _merge_preferred_horizon_labels(out, labels, ("close", "next_close"), "closectx")
        trace["clickhouse_close_labels"] = str(label_path)
        trace["clickhouse_table"] = clickhouse["table"]

    return out, trace


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    runs = args.run or [RunInput(label, Path(path)) for label, path in DEFAULT_RUNS]
    horizons = horizon_specs(args.horizon or DEFAULT_HORIZONS)
    explicit = explicit_label_map(args.horizon_label)
    decision_times = parse_clock_values(args.decision_time)
    group_cols = tuple(args.group_col or DEFAULT_GROUP_COLS)
    timed_target_end_seconds = optional_clock_seconds(args.timed_target_end_time)

    prediction_frames = [
        filter_decision_times(load_prediction(run.path, run.label), decision_times) for run in runs
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions, attach_trace = attach_horizon_labels(
        predictions,
        horizons=horizons,
        args=args,
        timed_target_end_seconds=timed_target_end_seconds,
        explicit=explicit,
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
    write_json(trace_path, trace, sort_keys=True)
    print(
        json.dumps(
            json_safe(
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
