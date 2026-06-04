from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    DEFAULT_TICK_END_OFFSET_US,
    DEFAULT_TICK_START_OFFSET_US,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_window,
)
from opening_strength_fit.config import (
    config_bool,
    config_list,
    config_str,
    config_value,
    load_toml,
)
from opening_strength_fit.dataset import load_ticks
from opening_strength_fit.io import write_frame
from opening_strength_fit.model import feature_columns
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, parse_clock_times
from opening_strength_fit.schema import EXCHANGE_OFFSET_US_COL, available_depth_levels
from opening_strength_fit.training_labeled import build_labeled_frame_from_config
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

DEFAULT_DATES = ("2021-09-22", "2021-09-23")
DEFAULT_TRADABLE_STATUSES = ("T0", "20", "TRADE")
TICK_PREVIEW_COLUMNS = (
    "date",
    "symbol",
    "time",
    "status",
    "last_price",
    "volume",
    "turnover",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
)
LABELED_PREVIEW_COLUMNS = (
    "date",
    "symbol",
    "decision_time",
    "timestamp",
    "status",
    "ask_price_1",
    "spread_bps",
    "depth_imbalance_10",
    "turnover_diff_30t",
    "return_10t",
    "preopen_turnover",
    "entry_timestamp",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "buy_price",
    "sell_vwap",
    "label",
)


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        parts = shlex.split(line, comments=True)
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _status(ok: bool, *, warn: bool = False) -> str:
    if ok:
        return "ok"
    return "warn" if warn else "fail"


def _check(status: str, detail: str) -> str:
    return f"{status}: {detail}"


def _compact_pass(check: str) -> str:
    return "ok" if check.startswith("ok:") else check


def _clock(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%H:%M:%S")


def _int_or_none(value) -> int | None:
    return None if value in (None, "") else int(value)


def _sort_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("date", "symbol", "timestamp") if column in ticks.columns]
    if sort_columns:
        return ticks.sort_values(sort_columns).reset_index(drop=True)
    return ticks.reset_index(drop=True)


def load_source_ticks(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        return load_ticks(args.input)

    if not args.no_env_file:
        load_env_file(args.env_file)
        args.user = args.user or os.getenv("CLICKHOUSE_USER")
        args.password = args.password or os.getenv("CLICKHOUSE_PASSWORD")
        args.host = os.getenv("CLICKHOUSE_HOST", args.host)
        args.port = _env_int("CLICKHOUSE_PORT", args.port)
        args.table = os.getenv("CLICKHOUSE_TICK_TABLE", args.table)

    if not args.user or not args.password:
        raise SystemExit(
            "missing ClickHouse credentials: set CLICKHOUSE_USER/CLICKHOUSE_PASSWORD, "
            "source .env, or pass --user/--password"
        )

    client = get_tick_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )
    frames = []
    for symbol in args.symbol:
        for trading_day in args.date:
            ticks = query_tick_window(
                client,
                symbol=symbol,
                trading_day=trading_day,
                table=args.table,
                start_offset_us=args.start_offset_us,
                end_offset_us=args.end_offset_us,
            )
            ticks = normalize_clickhouse_ticks(ticks)
            frames.append(ticks)
            if args.source_window_summary:
                print_mapping(
                    f"source_window[{trading_day},{symbol}]",
                    dataset_summary(ticks),
                )

    if not frames:
        raise SystemExit("no source windows requested")
    return _sort_ticks(pd.concat(frames, ignore_index=True))


def _opening_window_check(
    ticks: pd.DataFrame,
    requested_dates: list[str],
) -> str:
    if ticks.empty or "date" not in ticks.columns:
        return _check("fail", "no rows")
    dates_with_rows = set(ticks["date"].astype(str).unique())
    missing = sorted(set(requested_dates) - dates_with_rows)
    by_date = ticks.groupby("date")["timestamp"].agg(["size", "min", "max"])
    status = _status(not missing)
    row_min = int(by_date["size"].min()) if len(by_date) else 0
    row_max = int(by_date["size"].max()) if len(by_date) else 0
    time_min = by_date["min"].min().strftime("%H:%M:%S") if len(by_date) else ""
    time_max = by_date["max"].max().strftime("%H:%M:%S") if len(by_date) else ""
    detail = (
        f"{len(dates_with_rows)}/{len(requested_dates)} dates; "
        f"rows/day={row_min:,}-{row_max:,}; window={time_min}->{time_max}"
    )
    if missing:
        detail += f"; missing={missing}"
    return _check(status, detail)


def _symbol_filter_check(
    *,
    ticks: pd.DataFrame,
    labeled: pd.DataFrame,
    config: dict,
) -> str:
    if not config_bool(config, "universe", "enabled", True):
        return _check("skip", "universe.enabled=false")
    regex = config_str(
        config,
        "universe",
        "symbol_regex",
        DEFAULT_A_SHARE_SYMBOL_REGEX,
    )
    pattern = re.compile(regex)
    source_symbols = sorted(ticks["symbol"].astype(str).str.upper().unique())
    labeled_symbols = sorted(labeled["symbol"].astype(str).str.upper().unique())
    source_bad = [symbol for symbol in source_symbols if not pattern.fullmatch(symbol)]
    labeled_bad = [symbol for symbol in labeled_symbols if not pattern.fullmatch(symbol)]
    status = _status(not labeled_bad, warn=bool(source_bad))
    if not source_bad and not labeled_bad:
        return _check(status, f"{len(labeled_symbols)} labeled symbols match A-share regex")
    return _check(
        status,
        f"source_bad={source_bad[:5] or '<none>'}; labeled_bad={labeled_bad[:5] or '<none>'}",
    )


def _cumulative_check(ticks: pd.DataFrame) -> str:
    missing = [column for column in ("volume", "turnover") if column not in ticks.columns]
    if missing:
        return _check("fail", f"missing columns: {missing}")
    sort_columns = [
        column
        for column in ("date", "symbol", "timestamp", EXCHANGE_OFFSET_US_COL)
        if column in ticks.columns
    ]
    work = ticks.sort_values(sort_columns).copy()
    group = work.groupby(["date", "symbol"], sort=False)
    volume_diff = group["volume"].diff()
    turnover_diff = group["turnover"].diff()
    volume_decreases = int((volume_diff < 0).sum())
    turnover_decreases = int((turnover_diff < -1e-7).sum())
    status = _status(volume_decreases == 0 and turnover_decreases == 0)
    if volume_decreases == 0 and turnover_decreases == 0:
        return _check(status, "Volume/Turnover monotonic per date x symbol")
    return _check(
        status,
        f"volume_decreases={volume_decreases:,}; turnover_decreases={turnover_decreases:,}",
    )


def _timestamp_order_check(ticks: pd.DataFrame) -> str:
    if ticks.empty or "timestamp" not in ticks.columns:
        return _check("fail", "missing timestamp")
    sort_columns = ["date", "symbol", "timestamp"]
    work = ticks.sort_values(sort_columns).copy()
    group = work.groupby(["date", "symbol"], sort=False)
    timestamp_backwards = int((group["timestamp"].diff() < pd.Timedelta(0)).sum())
    offset_backwards = 0
    offset_mismatch = 0
    if EXCHANGE_OFFSET_US_COL in work.columns:
        offset_diff = group[EXCHANGE_OFFSET_US_COL].diff()
        offset_backwards = int((offset_diff < 0).sum())
        expected = pd.to_datetime(work["date"].astype(str)) + pd.to_timedelta(
            work[EXCHANGE_OFFSET_US_COL].astype("int64"), unit="us"
        )
        offset_mismatch = int((expected != work["timestamp"]).sum())
    status = _status(timestamp_backwards == 0 and offset_backwards == 0 and offset_mismatch == 0)
    if timestamp_backwards == 0 and offset_backwards == 0 and offset_mismatch == 0:
        return _check(status, "timestamp sorted and equals TradingDay + ExchTimeOffsetUs")
    return _check(
        status,
        f"timestamp_backwards={timestamp_backwards:,}; "
        f"offset_backwards={offset_backwards:,}; offset_mismatch={offset_mismatch:,}",
    )


def _ask1_execution_check(
    labeled: pd.DataFrame,
    config: dict,
    tradable_statuses: list[str],
) -> str:
    required = {"ask_price_1", "ask_volume_1", "bid_price_1", "status"}
    missing = sorted(required - set(labeled.columns))
    if missing:
        return _check("fail", f"missing columns: {missing}")
    allowed = {status.upper() for status in tradable_statuses}
    status_upper = labeled["status"].astype(str).str.upper()
    decision_rows = len(labeled)
    tradable = status_upper.isin(allowed)
    ask_price_ok = pd.to_numeric(labeled["ask_price_1"], errors="coerce").gt(0)
    ask_volume_ok = pd.to_numeric(labeled["ask_volume_1"], errors="coerce").gt(0)
    bid_price = pd.to_numeric(labeled["bid_price_1"], errors="coerce")
    ask_price = pd.to_numeric(labeled["ask_price_1"], errors="coerce")
    spread_ok = ask_price.ge(bid_price) | bid_price.le(0) | bid_price.isna()
    entry_tick_delay = _int_or_none(config_value(config, "labels", "entry_tick_delay", 0)) or 0
    buy_price_ok = True
    if "buy_price" in labeled.columns:
        buy_price = pd.to_numeric(labeled["buy_price"], errors="coerce")
        if entry_tick_delay == 0:
            buy_price_ok = bool(np.isclose(buy_price, ask_price, equal_nan=True).all())
        else:
            buy_price_ok = bool(buy_price.gt(0).all())
    entry_ok = pd.Series(True, index=labeled.index)
    if entry_tick_delay > 0:
        if "entry_timestamp" not in labeled.columns or "entry_delay_seconds" not in labeled.columns:
            entry_ok = pd.Series(False, index=labeled.index)
        else:
            delay = pd.to_numeric(labeled["entry_delay_seconds"], errors="coerce")
            entry_ok = labeled["entry_timestamp"].notna() & delay.ge(0)
        max_entry_gap = _int_or_none(config_value(config, "labels", "entry_max_gap_seconds", None))
        if max_entry_gap is not None:
            if "entry_max_tick_gap_seconds" not in labeled.columns:
                entry_ok &= False
            else:
                entry_gap = pd.to_numeric(
                    labeled["entry_max_tick_gap_seconds"],
                    errors="coerce",
                )
                entry_ok &= entry_gap.le(max_entry_gap)
        if "entry_status" in labeled.columns:
            entry_ok &= labeled["entry_status"].astype(str).str.upper().isin(allowed)
    ok = bool(
        decision_rows
        and tradable.all()
        and ask_price_ok.all()
        and ask_volume_ok.all()
        and spread_ok.all()
        and buy_price_ok
        and entry_ok.all()
    )
    if ok:
        if entry_tick_delay:
            return _check(
                "ok",
                f"{decision_rows:,} rows use ask1 at entry_tick_delay={entry_tick_delay}",
            )
        return _check("ok", f"{decision_rows:,} decision rows use ask1 as buy_price")
    return _check(
        _status(ok),
        f"rows={decision_rows:,}; status_ok={int(tradable.sum()):,}; "
        f"ask1_ok={int(ask_price_ok.sum()):,}; askvol1_ok={int(ask_volume_ok.sum()):,}; "
        f"spread_ok={int(spread_ok.sum()):,}; "
        f"buy_price_ok={buy_price_ok}; entry_ok={int(entry_ok.sum()):,}; "
        f"entry_tick_delay={entry_tick_delay}",
    )


def _decision_point_check(
    labeled: pd.DataFrame,
    config: dict,
    tradable_statuses: list[str],
) -> str:
    if labeled.empty:
        return _check("fail", "no sampled decision rows")
    allowed = {status.upper() for status in tradable_statuses}
    status_ok = (
        labeled["status"].astype(str).str.upper().isin(allowed)
        if "status" in labeled.columns
        else pd.Series(False, index=labeled.index)
    )
    decision_times = parse_clock_times(
        config_value(config, "sample", "decision_times", DEFAULT_DECISION_TIMES)
    )
    decision_ok = (
        labeled["decision_time"].astype(str).isin(decision_times)
        if "decision_time" in labeled.columns
        else pd.Series(False, index=labeled.index)
    )
    max_lag = _int_or_none(config_value(config, "sample", "decision_max_lag_seconds", 5))
    lag_ok = pd.Series(True, index=labeled.index)
    max_observed_lag = None
    if "decision_lag_seconds" in labeled.columns and max_lag is not None:
        lag = pd.to_numeric(labeled["decision_lag_seconds"], errors="coerce")
        lag_ok = lag.ge(0) & lag.le(max_lag)
        max_observed_lag = float(lag.max()) if len(lag) else None
    if "decision_target_timestamp" in labeled.columns:
        time_clock = _clock(labeled["decision_target_timestamp"])
    elif "decision_time" in labeled.columns:
        time_clock = labeled["decision_time"].astype(str)
    else:
        time_clock = _clock(labeled["timestamp"])
    in_window = time_clock.ge("09:30:00") & time_clock.le("09:40:00")
    ok = bool(status_ok.all() and decision_ok.all() and lag_ok.all() and in_window.all())
    if ok:
        return _check(
            "ok",
            f"{len(labeled):,} rows; max_lag={max_observed_lag}; statuses={sorted(allowed)}",
        )
    return _check(
        _status(ok),
        f"rows={len(labeled):,}; status_ok={int(status_ok.sum()):,}; "
        f"decision_time_ok={int(decision_ok.sum()):,}; "
        f"lag_ok={int(lag_ok.sum()):,}; max_lag={max_observed_lag}; "
        f"window_ok={int(in_window.sum()):,}",
    )


def print_quality_checks(
    *,
    ticks: pd.DataFrame,
    labeled: pd.DataFrame,
    config: dict,
    requested_dates: list[str],
) -> None:
    tradable_statuses = config_list(
        config,
        "filters",
        "tradable_statuses",
        DEFAULT_TRADABLE_STATUSES,
    )
    print_mapping(
        "source_quality_checks",
        {
            "opening_window_daily_data": _compact_pass(
                _opening_window_check(ticks, requested_dates)
            ),
            "symbol_filter": _compact_pass(
                _symbol_filter_check(
                    ticks=ticks,
                    labeled=labeled,
                    config=config,
                )
            ),
            "volume_turnover_cumulative": _compact_pass(_cumulative_check(ticks)),
            "timestamp_exchange_order": _compact_pass(_timestamp_order_check(ticks)),
        },
    )
    print_mapping(
        "sample_quality_checks",
        {
            "ask1_executable_buy_price": _compact_pass(
                _ask1_execution_check(
                    labeled,
                    config,
                    tradable_statuses,
                )
            ),
            "decision_points_tradable": _compact_pass(
                _decision_point_check(
                    labeled,
                    config,
                    tradable_statuses,
                )
            ),
        },
    )


def print_dataset_checks(
    *,
    ticks: pd.DataFrame,
    labeled: pd.DataFrame,
) -> None:
    tick_summary = dataset_summary(ticks)
    labeled_summary = dataset_summary(labeled)
    features = feature_columns(labeled)
    quote_anomalies = quote_anomaly_count(ticks)
    valid_labels = int(labeled.get("valid_label", labeled["label"].notna()).sum())
    label_rows = len(labeled)
    label_nan_rate = (
        float(labeled["label"].isna().mean()) if label_rows and "label" in labeled else 1.0
    )
    feature_nan_columns = pd.Series(dtype="int64")
    feature_nan_count = 0
    if features:
        feature_frame = labeled[features].replace([np.inf, -np.inf], np.nan)
        feature_nan_columns = feature_frame.isna().sum()
        feature_nan_columns = feature_nan_columns[feature_nan_columns > 0].sort_values(
            ascending=False
        )
        feature_nan_count = int(feature_nan_columns.sum())
    valid_mask = (
        labeled["valid_label"].astype(bool)
        if "valid_label" in labeled.columns
        else labeled["label"].notna()
    )
    label_values = (
        pd.to_numeric(labeled.loc[valid_mask, "label"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        if "label" in labeled.columns
        else pd.Series(dtype="float64")
    )

    print_mapping(
        "tick_dataset_check",
        {
            "rows": format_tick_rows(len(ticks), quote_anomalies),
            "columns": len(ticks.columns),
            "date_range": f"{tick_summary.get('date_min')} -> {tick_summary.get('date_max')}",
            "dates": tick_summary.get("n_dates"),
            "symbols": tick_summary.get("n_symbols"),
            "time_range": f"{tick_summary.get('time_min')} -> {tick_summary.get('time_max')}",
            "depth_levels": available_depth_levels(ticks),
        },
    )
    print_mapping(
        "labeled_dataset_check",
        {
            "rows": f"{label_rows:,}",
            "features": len(features),
            "date_range": f"{labeled_summary.get('date_min')} -> {labeled_summary.get('date_max')}",
            "dates": labeled_summary.get("n_dates"),
            "symbols": labeled_summary.get("n_symbols"),
            "time_range": f"{labeled_summary.get('time_min')} -> {labeled_summary.get('time_max')}",
            "label_coverage": (f"{valid_labels:,}/{label_rows:,} (nan_rate={label_nan_rate:.2%})"),
            "feature_nan_values": f"{feature_nan_count:,}",
            "feature_nan_columns": (
                ", ".join(
                    f"{column}={int(count):,}"
                    for column, count in feature_nan_columns.head(8).items()
                )
                if feature_nan_count
                else "<none>"
            ),
        },
    )
    print_mapping(
        "label_distribution_check",
        {
            "rows": len(label_values),
            "mean": float(label_values.mean()) if len(label_values) else float("nan"),
            "std": float(label_values.std()) if len(label_values) else float("nan"),
            "min": float(label_values.min()) if len(label_values) else float("nan"),
            "p05": float(label_values.quantile(0.05)) if len(label_values) else float("nan"),
            "p50": float(label_values.quantile(0.50)) if len(label_values) else float("nan"),
            "p95": float(label_values.quantile(0.95)) if len(label_values) else float("nan"),
            "max": float(label_values.max()) if len(label_values) else float("nan"),
            "win_rate": float((label_values > 0).mean()) if len(label_values) else float("nan"),
        },
    )


def _preview_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, [column for column in columns if column in frame.columns]].replace(
        [np.inf, -np.inf],
        np.nan,
    )


def quote_anomaly_count(ticks: pd.DataFrame) -> int:
    required = {"ask_price_1", "bid_price_1"}
    if ticks.empty or not required.issubset(ticks.columns):
        return 0
    ask1 = pd.to_numeric(ticks["ask_price_1"], errors="coerce")
    bid1 = pd.to_numeric(ticks["bid_price_1"], errors="coerce")
    anomaly = ask1.isna() | bid1.isna() | ask1.le(0) | bid1.le(0)
    return int(anomaly.sum())


def format_tick_rows(rows: int, quote_anomalies: int) -> str:
    if quote_anomalies <= 0:
        return f"{rows:,}"
    normal_rows = max(rows - quote_anomalies, 0)
    return f"{normal_rows:,}+{quote_anomalies:,} (raw quote ask/bid<=0)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or load a small opening tick dataset, check feature/label coverage, "
            "and optionally write the raw tick sample for local smoke training."
        )
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional existing tick parquet/csv. If omitted, fetch ClickHouse windows.",
    )
    parser.add_argument("--symbol", nargs="+", default=["000925.SZ"])
    parser.add_argument("--date", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument(
        "--config",
        default="experiments/runs/gbm_opening_1y_next_month.toml",
        help="Run config used for label/sample settings during inspection.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional parquet/csv path to write the raw tick sample.",
    )
    parser.add_argument(
        "--labeled-output",
        default="",
        help="Optional parquet/csv path to write the labeled debug table.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=0,
        help="Print this many raw tick rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--label-preview-rows",
        type=int,
        default=5,
        help="Print this many labeled rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--nan-preview-rows",
        type=int,
        default=5,
        help="Print this many invalid label rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--source-window-summary",
        action="store_true",
        help="Print one summary per requested date x symbol source window.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Backward-compatible alias for --preview-rows.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument(
        "--host",
        default=os.getenv("CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT),
    )
    parser.add_argument("--user", default=os.getenv("CLICKHOUSE_USER"))
    parser.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD"))
    parser.add_argument(
        "--table",
        default=os.getenv("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
    )
    parser.add_argument("--start-offset-us", type=int, default=DEFAULT_TICK_START_OFFSET_US)
    parser.add_argument("--end-offset-us", type=int, default=DEFAULT_TICK_END_OFFSET_US)
    args = parser.parse_args()
    if args.head is not None:
        args.preview_rows = args.head

    config = load_toml(args.config)
    ticks = load_source_ticks(args)
    labeled = build_labeled_frame_from_config(ticks, config)
    print_quality_checks(
        ticks=ticks,
        labeled=labeled,
        config=config,
        requested_dates=[str(pd.Timestamp(date).date()) for date in args.date],
    )
    print_dataset_checks(ticks=ticks, labeled=labeled)

    if args.preview_rows > 0:
        print("\ntick_preview:")
        print(
            _preview_columns(ticks, TICK_PREVIEW_COLUMNS)
            .head(args.preview_rows)
            .to_string(index=False)
        )

    if args.label_preview_rows > 0:
        print("\nlabeled_preview:")
        print(
            _preview_columns(labeled, LABELED_PREVIEW_COLUMNS)
            .head(args.label_preview_rows)
            .to_string(index=False)
        )

    if args.nan_preview_rows > 0 and "valid_label" in labeled.columns:
        invalid = labeled.loc[~labeled["valid_label"]]
        if not invalid.empty:
            print("\ninvalid_label_preview:")
            cols = [
                column
                for column in ("date", "symbol", "timestamp", "time", "label", "valid_label")
                if column in invalid.columns
            ]
            print(invalid.loc[:, cols].head(args.nan_preview_rows).to_string(index=False))

    if args.output:
        write_frame(ticks, args.output)
        print(f"\nwrote raw ticks: {args.output}")
    if args.labeled_output:
        write_frame(labeled, args.labeled_output)
        print(f"wrote labeled debug table: {args.labeled_output}")


if __name__ == "__main__":
    main()
