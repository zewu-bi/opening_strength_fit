from __future__ import annotations

import argparse
import json
import os
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
    query_tick_day_window,
)
from opening_strength_fit.candidates import filter_opening_candidates
from opening_strength_fit.config import config_value, load_toml, run_id
from opening_strength_fit.dataset import build_labeled_feature_frame, load_ticks
from opening_strength_fit.evaluation import (
    format_group_cols,
    group_cols_for_mode,
    score_bucket_returns,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.io import write_frame
from opening_strength_fit.model import (
    evaluate_prediction_frame,
    fit_gbm_frame,
    fit_ridge_frame,
    predict_frame,
)
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.rolling import (
    annual_rolling_date_splits,
    chronological_date_split,
    monthly_rolling_date_splits,
)
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, parse_clock_times
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.universe import (
    DEFAULT_A_SHARE_SYMBOL_REGEX,
    filter_symbol_universe,
    load_symbol_list,
)


def load_run_config(path: str) -> dict:
    if not path:
        return {}
    return load_toml(path)


def build_training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="", help="Optional TOML run config.")
    parser.add_argument("--input", default=None, help="Tick parquet/csv path override.")
    parser.add_argument(
        "--data-source",
        choices=["auto", "path", "clickhouse"],
        default=None,
        help="Override [data].source. --input always uses a local/path source.",
    )
    parser.add_argument(
        "--input-kind",
        choices=["auto", "raw_ticks", "labeled"],
        default=None,
        help="Whether --input is raw ticks or an already labeled research dataset.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--test-end-year", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-start-month", default=None)
    parser.add_argument("--test-end-month", default=None)
    parser.add_argument(
        "--rolling-annual",
        action="store_true",
        help="Use train <= test_year-1 and test = calendar year splits.",
    )
    parser.add_argument(
        "--rolling-monthly",
        action="store_true",
        help="Use rolling N-month train windows and one calendar month tests.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "rolling_annual", "rolling_monthly"],
        default=None,
        help="Override [window].mode.",
    )
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override [evaluation].top_n for top-score summaries.",
    )
    parser.add_argument(
        "--clickhouse-host",
        default=None,
        help="ClickHouse host override. Defaults to CLICKHOUSE_HOST or config.",
    )
    parser.add_argument(
        "--clickhouse-port",
        type=int,
        default=None,
        help="ClickHouse port override. Defaults to CLICKHOUSE_PORT or config.",
    )
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-table", default=None)
    parser.add_argument("--start-offset-us", type=int, default=None)
    parser.add_argument("--end-offset-us", type=int, default=None)
    return parser


def _int_config(config: dict, section: str, key: str, default: int) -> int:
    return int(config_value(config, section, key, default))


def _float_config(config: dict, section: str, key: str, default: float) -> float:
    return float(config_value(config, section, key, default))


def _str_config(config: dict, section: str, key: str, default: str) -> str:
    return str(config_value(config, section, key, default))


def _bool_config(config: dict, section: str, key: str, default: bool) -> bool:
    value = config_value(config, section, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_int_config(
    config: dict,
    section: str,
    key: str,
    default: int | None = None,
) -> int | None:
    value = config_value(config, section, key, default)
    return None if value in (None, "") else int(value)


def _list_config(
    config: dict,
    section: str,
    key: str,
    default: list[str] | tuple[str, ...],
) -> list[str]:
    value = config_value(config, section, key, default)
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    return [part.strip() for part in parts if part and part.strip()]


def _clock_list_config(
    config: dict,
    section: str,
    key: str,
    default: list[str] | tuple[str, ...],
) -> list[str]:
    return parse_clock_times(config_value(config, section, key, default))


def _float_mapping_config(config: dict, section: str, key: str) -> dict[str, float]:
    value = config_value(config, section, key, {})
    if not value:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"[{section}].{key} must be a table of column = value")
    return {
        str(column): float(threshold)
        for column, threshold in value.items()
        if threshold not in (None, "")
    }


def _apply_candidate_filter_from_config(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if not _bool_config(config, "candidate_filter", "enabled", False):
        return frame
    return filter_opening_candidates(
        frame,
        min_values=_float_mapping_config(config, "candidate_filter", "min"),
        max_values=_float_mapping_config(config, "candidate_filter", "max"),
        rank_min_values=_float_mapping_config(
            config,
            "candidate_filter",
            "rank_min",
        ),
        rank_group_cols=_list_config(
            config,
            "candidate_filter",
            "rank_group_cols",
            ["date", "decision_target_timestamp"],
        ),
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"object is not JSON serializable: {type(value)!r}")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _test_year(value: str) -> int:
    return int(pd.Timestamp(value).year)


def _prediction_r2(predictions: pd.DataFrame) -> float:
    frame = predictions.loc[
        predictions["label"].notna() & predictions["prediction"].notna()
    ]
    if len(frame) < 2:
        return float("nan")
    y = frame["label"].astype("float64").to_numpy()
    y_hat = frame["prediction"].astype("float64").to_numpy()
    total = float(np.square(y - y.mean()).sum())
    if total == 0.0:
        return float("nan")
    residual = float(np.square(y - y_hat).sum())
    return 1.0 - residual / total


def _metrics_row(
    *,
    run_name: str,
    split,
    train_stats: dict[str, int],
    predictions: pd.DataFrame,
    metrics: dict[str, object],
    top_summary: dict[str, object],
    model_name: str,
    alpha: float,
    feature_count: int,
    evaluation_settings: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_name,
        "test_year": _test_year(split.test_start_date),
        "test_month": str(pd.Timestamp(split.test_start_date).to_period("M")),
        "train_start_date": split.train_start_date,
        "train_end_date": split.train_end_date,
        "test_start_date": split.test_start_date,
        "test_end_date": split.test_end_date,
        "train_rows": int(train_stats["rows"]),
        "train_dates": int(train_stats["dates"]),
        "train_symbols": int(train_stats["symbols"]),
        "test_rows": int(len(predictions)),
        "test_dates": int(metrics.get("dates", 0)),
        "test_symbols": int(metrics.get("symbols", 0)),
        "features": int(feature_count),
        "model_name": model_name,
        "alpha": float(alpha),
        "model_test_r2": _prediction_r2(predictions),
        "ic_mode": str(evaluation_settings["ic_mode"]),
        "selection_mode": str(evaluation_settings["selection_mode"]),
        "top_n": int(evaluation_settings["top_n"]),
    }
    row.update(metrics)
    for key, value in top_summary.items():
        row[f"top_score_{key}"] = value
    return row


def _feature_limit(args: argparse.Namespace, config: dict) -> int | None:
    raw = (
        args.feature_limit
        if args.feature_limit is not None
        else _int_config(config, "data", "feature_limit", 0)
    )
    return raw if raw and raw > 0 else None


def build_labeled_frame_from_config(ticks: pd.DataFrame, config: dict) -> pd.DataFrame:
    volume_unit_multiplier = _float_config(
        config,
        "labels",
        "volume_unit_multiplier",
        1.0,
    )
    use_universe = _bool_config(config, "universe", "enabled", True)
    symbols_file = _str_config(config, "universe", "symbols_file", "")
    universe_symbols = load_symbol_list(symbols_file) if symbols_file else None
    sample_mode = _str_config(config, "sample", "mode", "all_ticks")
    max_decision_lag = config_value(
        config,
        "sample",
        "decision_max_lag_seconds",
        5,
    )
    labeled = build_labeled_feature_frame(
        ticks,
        buy_price_col=_str_config(config, "labels", "buy_price_col", "ask_price_1"),
        volume_col=_str_config(config, "labels", "volume_col", "volume"),
        turnover_col=_str_config(config, "labels", "turnover_col", "turnover"),
        hold_seconds=_int_config(config, "labels", "hold_seconds", 60),
        sell_window_seconds=_int_config(config, "labels", "sell_window_seconds", 60),
        volume_unit_multiplier=volume_unit_multiplier,
        fee_bps=_float_config(config, "labels", "fee_bps", 0.0),
        sample_start_time=_str_config(config, "sample", "start_time", "09:30:00"),
        sample_end_time=_str_config(config, "sample", "end_time", "09:40:00"),
        include_preopen=_bool_config(config, "features", "include_preopen", True),
        max_future_gap_seconds=config_value(
            config,
            "labels",
            "max_future_gap_seconds",
            None,
        ),
        tradable_statuses=_list_config(config, "filters", "tradable_statuses", []),
        universe_regex=(
            _str_config(
                config,
                "universe",
                "symbol_regex",
                DEFAULT_A_SHARE_SYMBOL_REGEX,
            )
            if use_universe
            else None
        ),
        universe_symbols=universe_symbols if use_universe else None,
        sample_mode=sample_mode,
        decision_times=_clock_list_config(
            config,
            "sample",
            "decision_times",
            DEFAULT_DECISION_TIMES,
        ),
        decision_max_lag_seconds=(
            None if max_decision_lag in (None, "") else int(max_decision_lag)
        ),
    )
    return _apply_candidate_filter_from_config(labeled, config)


def _input_kind(args: argparse.Namespace, config: dict) -> str:
    return args.input_kind or _str_config(config, "data", "input_kind", "auto")


def _looks_labeled(frame: pd.DataFrame) -> bool:
    return {"date", "symbol", "timestamp", "label"}.issubset(frame.columns)


def _load_training_frame(path: str, args: argparse.Namespace, config: dict) -> pd.DataFrame:
    frame = load_ticks(path)
    kind = _input_kind(args, config)
    if kind == "labeled" or (kind == "auto" and _looks_labeled(frame)):
        labeled = ensure_timestamp_columns(standardize_columns(frame))
        if _bool_config(config, "universe", "enabled", True):
            symbols_file = _str_config(config, "universe", "symbols_file", "")
            labeled = filter_symbol_universe(
                labeled,
                symbol_regex=_str_config(
                    config,
                    "universe",
                    "symbol_regex",
                    DEFAULT_A_SHARE_SYMBOL_REGEX,
                ),
                symbols=load_symbol_list(symbols_file) if symbols_file else None,
            )
        return _apply_candidate_filter_from_config(labeled, config)
    if kind not in {"auto", "raw_ticks"}:
        raise SystemExit(
            f"unknown data.input_kind={kind!r}; expected auto, raw_ticks, or labeled"
        )
    return build_labeled_frame_from_config(frame, config)


def _resolved_data_source(args: argparse.Namespace, config: dict, tick_path: str) -> str:
    if args.input:
        return "path"
    source = args.data_source or _str_config(config, "data", "source", "auto")
    source = source.strip().lower()
    if source == "auto":
        return "path" if tick_path else "clickhouse"
    if source in {"path", "clickhouse"}:
        return source
    raise SystemExit(
        f"unknown data.source={source!r}; expected auto, path, or clickhouse"
    )


def _period_end_date(period: pd.Period) -> str:
    return str(((period + 1).to_timestamp() - pd.Timedelta(days=1)).date())


def _clickhouse_date_bounds(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    explicit_start = config_value(
        config,
        "data",
        "start_date",
        config_value(config, "clickhouse", "start_date", None),
    )
    explicit_end = config_value(
        config,
        "data",
        "end_date",
        config_value(config, "clickhouse", "end_date", None),
    )
    if explicit_start and explicit_end:
        return str(pd.Timestamp(explicit_start).date()), str(pd.Timestamp(explicit_end).date())

    window_mode = _resolved_window_mode(args, config)
    if window_mode == "rolling_monthly":
        train_months = (
            args.train_months
            if args.train_months is not None
            else _int_config(config, "window", "train_months", 12)
        )
        first_test_month = (
            args.test_start_month
            or config_value(config, "window", "test_start_month", None)
        )
        last_test_month = (
            args.test_end_month
            or config_value(config, "window", "test_end_month", None)
        )
        if not first_test_month or not last_test_month:
            raise SystemExit(
                "ClickHouse rolling_monthly source needs [window].test_start_month "
                "and [window].test_end_month, or CLI overrides."
            )
        first_period = pd.Period(first_test_month, freq="M")
        last_period = pd.Period(last_test_month, freq="M")
        start_period = first_period - int(train_months)
        return str(start_period.to_timestamp().date()), _period_end_date(last_period)

    if window_mode == "rolling_annual":
        train_start_year = (
            args.train_start_year
            if args.train_start_year is not None
            else _optional_int_config(config, "window", "train_start_year", None)
        )
        first_test_year = _test_year_from_args(args, config, "start")
        last_test_year = _test_year_from_args(args, config, "end")
        if train_start_year is None or first_test_year is None or last_test_year is None:
            raise SystemExit(
                "ClickHouse rolling_annual source needs train_start_year and "
                "test start/end years."
            )
        return f"{int(train_start_year):04d}-01-01", f"{int(last_test_year):04d}-12-31"

    train_start_date = config_value(
        config,
        "window",
        "train_start_date",
        config_value(config, "data", "train_start_date", None),
    )
    test_start_date = args.test_start_date or config_value(
        config,
        "window",
        "test_start_date",
        None,
    )
    test_end_date = args.test_end_date or config_value(
        config,
        "window",
        "test_end_date",
        None,
    )
    start_date = explicit_start or train_start_date
    end_date = explicit_end or test_end_date
    if not start_date or not end_date:
        raise SystemExit(
            "ClickHouse chronological source needs [data].start_date/[data].end_date "
            "or [window].train_start_date plus test_start_date/test_end_date."
        )
    if test_start_date and str(pd.Timestamp(start_date).date()) >= str(pd.Timestamp(test_start_date).date()):
        raise SystemExit("ClickHouse chronological train source must start before test_start_date")
    return str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date())


def _clickhouse_setting(
    args: argparse.Namespace,
    config: dict,
    arg_name: str,
    config_key: str,
    env_name: str,
    default,
):
    arg_value = getattr(args, arg_name)
    if arg_value not in (None, ""):
        return arg_value
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    return config_value(config, "clickhouse", config_key, default)


def _load_clickhouse_labeled_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    host = str(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_host",
            "host",
            "CLICKHOUSE_HOST",
            DEFAULT_CLICKHOUSE_TICK_HOST,
        )
    )
    port = int(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_port",
            "port",
            "CLICKHOUSE_PORT",
            DEFAULT_CLICKHOUSE_TICK_PORT,
        )
    )
    user = _clickhouse_setting(
        args,
        config,
        "clickhouse_user",
        "user",
        "CLICKHOUSE_USER",
        None,
    )
    password = _clickhouse_setting(
        args,
        config,
        "clickhouse_password",
        "password",
        "CLICKHOUSE_PASSWORD",
        None,
    )
    table = str(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_table",
            "table",
            "CLICKHOUSE_TICK_TABLE",
            DEFAULT_CLICKHOUSE_TICK_TABLE,
        )
    )
    start_offset_us = int(
        args.start_offset_us
        if args.start_offset_us is not None
        else config_value(
            config,
            "clickhouse",
            "start_offset_us",
            DEFAULT_TICK_START_OFFSET_US,
        )
    )
    end_offset_us = int(
        args.end_offset_us
        if args.end_offset_us is not None
        else config_value(
            config,
            "clickhouse",
            "end_offset_us",
            DEFAULT_TICK_END_OFFSET_US,
        )
    )
    if not user or not password:
        raise SystemExit(
            "missing ClickHouse credentials: set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD, pass CLI overrides, or configure a K8s secret."
        )

    start_date, end_date = _clickhouse_date_bounds(args, config)
    dates = [str(date.date()) for date in pd.date_range(start_date, end_date, freq="D")]
    use_universe = _bool_config(config, "universe", "enabled", True)
    symbols_file = _str_config(config, "universe", "symbols_file", "")
    symbols = sorted(load_symbol_list(symbols_file)) if use_universe and symbols_file else None
    symbol_regex = (
        _str_config(config, "universe", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
        if use_universe
        else None
    )
    print_mapping(
        "clickhouse_source",
        {
            "host": host,
            "port": port,
            "table": table,
            "date_start": start_date,
            "date_end": end_date,
            "calendar_days": len(dates),
            "start_offset_us": start_offset_us,
            "end_offset_us": end_offset_us,
            "symbol_regex": symbol_regex or "",
            "symbols": len(symbols) if symbols else 0,
        },
    )

    client = get_tick_client(
        host=host,
        port=port,
        username=str(user),
        password=str(password),
    )
    labeled_parts = []
    for trading_day in dates:
        ticks = query_tick_day_window(
            client,
            trading_day=trading_day,
            table=table,
            start_offset_us=start_offset_us,
            end_offset_us=end_offset_us,
            symbol_regex=symbol_regex,
            symbols=symbols,
        )
        if ticks.empty:
            print(f"skip empty ClickHouse day: {trading_day}")
            continue
        ticks = normalize_clickhouse_ticks(ticks)
        labeled = build_labeled_frame_from_config(ticks, config)
        if labeled.empty:
            print(f"skip empty labeled day: {trading_day}")
            continue
        labeled_parts.append(labeled)
        print_mapping(f"clickhouse_labeled[{trading_day}]", dataset_summary(labeled))

    if not labeled_parts:
        raise SystemExit(
            "ClickHouse source produced no labeled rows; check date range, "
            "symbol universe, and sample/label settings."
        )
    return pd.concat(labeled_parts, ignore_index=True)


def _test_year_from_args(args: argparse.Namespace, config: dict, key: str) -> int | None:
    if key == "start":
        if args.test_start_year is not None:
            return args.test_start_year
        test_date = args.test_start_date or config_value(config, "window", "test_start_date", None)
        explicit = _optional_int_config(config, "window", "test_start_year", None)
    else:
        if args.test_end_year is not None:
            return args.test_end_year
        test_date = args.test_end_date or config_value(config, "window", "test_end_date", None)
        explicit = _optional_int_config(config, "window", "test_end_year", None)
    if explicit is not None:
        return explicit
    return int(pd.Timestamp(test_date).year) if test_date else None


def _date_splits(labeled: pd.DataFrame, args: argparse.Namespace, config: dict):
    window_mode = _resolved_window_mode(args, config)
    if window_mode == "rolling_monthly":
        return monthly_rolling_date_splits(
            labeled,
            train_months=(
                args.train_months
                if args.train_months is not None
                else _int_config(config, "window", "train_months", 12)
            ),
            first_test_month=args.test_start_month
            or config_value(config, "window", "test_start_month", None),
            last_test_month=args.test_end_month
            or config_value(config, "window", "test_end_month", None),
        )
    if window_mode == "rolling_annual":
        return annual_rolling_date_splits(
            labeled,
            train_start_year=(
                args.train_start_year
                if args.train_start_year is not None
                else _optional_int_config(config, "window", "train_start_year", None)
            ),
            first_test_year=_test_year_from_args(args, config, "start"),
            last_test_year=_test_year_from_args(args, config, "end"),
            min_train_years=_int_config(config, "window", "min_train_years", 1),
        )
    return [
        chronological_date_split(
            labeled,
            test_start_date=args.test_start_date
            or config_value(config, "window", "test_start_date", None),
            test_end_date=args.test_end_date
            or config_value(config, "window", "test_end_date", None),
            train_fraction=_float_config(config, "window", "train_fraction", 0.8),
        )
    ]


def _resolved_window_mode(args: argparse.Namespace, config: dict) -> str:
    window_mode = args.split_mode or _str_config(config, "window", "mode", "chronological")
    if args.rolling_monthly:
        return "rolling_monthly"
    if args.rolling_annual:
        return "rolling_annual"
    return window_mode


def _evaluation_settings(config: dict, args: argparse.Namespace) -> dict[str, object]:
    bucket_mode = _str_config(config, "evaluation", "bucket_mode", "daily")
    selection_mode = _str_config(config, "evaluation", "selection_mode", "symbol_day")
    ic_mode = _str_config(config, "evaluation", "ic_mode", bucket_mode)
    bucket_group_cols = group_cols_for_mode(bucket_mode)
    selection_group_cols = group_cols_for_mode(selection_mode)
    ic_group_cols = group_cols_for_mode(ic_mode)
    return {
        "score_bucket_mode": bucket_mode,
        "score_bucket_group_cols": format_group_cols(bucket_group_cols),
        "selection_mode": selection_mode,
        "selection_group_cols": format_group_cols(selection_group_cols),
        "ic_mode": ic_mode,
        "ic_group_cols": format_group_cols(ic_group_cols),
        "top_n": (
            args.top_n
            if args.top_n is not None
            else _int_config(config, "evaluation", "top_n", 20)
        ),
        "score_bins": _int_config(config, "evaluation", "score_bins", 5),
        "_bucket_group_cols": bucket_group_cols,
        "_selection_group_cols": selection_group_cols,
        "_ic_group_cols": ic_group_cols,
    }


def _fit_prediction_model(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
):
    model_name = _str_config(config, "model", "name", "ridge").strip().lower()
    feature_limit = _feature_limit(args, config)
    if model_name == "ridge":
        return fit_ridge_frame(train, alpha=alpha, feature_limit=feature_limit)
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return fit_gbm_frame(
            train,
            feature_limit=feature_limit,
            max_iter=_int_config(config, "model", "max_iter", 100),
            learning_rate=_float_config(config, "model", "learning_rate", 0.05),
            max_leaf_nodes=_int_config(config, "model", "max_leaf_nodes", 31),
            l2_regularization=_float_config(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            random_state=_int_config(config, "model", "random_state", 7),
        )
    raise SystemExit(f"unsupported model.name={model_name!r}; expected ridge or gbm")


def _model_json(config: dict, alpha: float) -> dict[str, object]:
    model_name = _str_config(config, "model", "name", "ridge").strip().lower()
    if model_name == "ridge":
        return {"name": "ridge", "alpha": alpha}
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return {
            "name": "gbm",
            "max_iter": _int_config(config, "model", "max_iter", 100),
            "learning_rate": _float_config(config, "model", "learning_rate", 0.05),
            "max_leaf_nodes": _int_config(config, "model", "max_leaf_nodes", 31),
            "l2_regularization": _float_config(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            "random_state": _int_config(config, "model", "random_state", 7),
        }
    return {"name": model_name}


def _fit_predict_split(
    *,
    labeled: pd.DataFrame,
    split,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
    evaluation_settings: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
    test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()
    model, train_stats = _fit_prediction_model(
        train,
        args=args,
        config=config,
        alpha=alpha,
    )
    predictions = predict_frame(model, test)
    if "valid_label" in predictions.columns:
        predictions = predictions.loc[predictions["valid_label"]].copy()

    metrics = evaluate_prediction_frame(
        predictions,
        group_cols=evaluation_settings["_ic_group_cols"],
    )
    buckets = score_bucket_returns(
        predictions,
        bins=int(evaluation_settings["score_bins"]),
        group_cols=evaluation_settings["_bucket_group_cols"],
    )
    top_trades = top_score_trades(
        predictions,
        top_n=int(evaluation_settings["top_n"]),
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary = summarize_trades(
        top_trades,
        group_cols=evaluation_settings["_selection_group_cols"],
    )

    prediction_year = _test_year(split.test_start_date)
    prediction_period = (
        str(pd.Timestamp(split.test_start_date).to_period("M"))
        if split.test_start_date[:7] == split.test_end_date[:7]
        else str(prediction_year)
    )
    write_frame(predictions, output_dir / f"predictions_{prediction_period}.parquet")
    buckets.to_csv(output_dir / f"score_buckets_{prediction_period}.csv", index=False)
    metrics_row = _metrics_row(
        run_name=run_name,
        split=split,
        train_stats=train_stats,
        predictions=predictions,
        metrics=metrics,
        top_summary=top_summary,
        model_name=model.model_name,
        alpha=alpha,
        feature_count=len(model.features),
        evaluation_settings=evaluation_settings,
    )
    print_mapping(f"train_stats[{prediction_period}]", train_stats)
    print_mapping(f"prediction_metrics[{prediction_period}]", metrics)
    print_mapping(
        f"top_score_summary[{prediction_period},top_n={evaluation_settings['top_n']}]",
        top_summary,
    )
    return predictions, metrics_row, train_stats


def _metrics_by_year_from_windows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or metrics["test_year"].is_unique:
        return metrics.copy()

    rows = []
    weighted_mean_cols = {
        "model_test_r2",
        "overall_ic",
        "overall_rank_ic",
        "group_ic_mean",
        "group_ic_std",
        "group_ic_ir",
        "group_rank_ic_mean",
        "group_rank_ic_std",
        "group_rank_ic_ir",
        "daily_ic_mean",
        "daily_ic_std",
        "daily_ic_ir",
        "daily_rank_ic_mean",
        "daily_rank_ic_std",
        "daily_rank_ic_ir",
        "mean_label",
        "win_rate",
        "top_score_mean_return",
        "top_score_median_return",
        "top_score_win_rate",
        "top_score_return_std",
    }
    sum_cols = {
        "test_rows",
        "test_dates",
        "rows",
        "dates",
        "ic_groups",
        "top_score_trades",
        "top_score_groups",
    }
    max_cols = {"train_rows", "train_dates", "train_symbols", "test_symbols", "symbols"}

    for year, group in metrics.groupby("test_year", sort=True):
        row: dict[str, object] = {
            "run_id": group["run_id"].iloc[0],
            "test_year": int(year),
            "test_month": f"{int(year)}",
            "train_start_date": group["train_start_date"].min(),
            "train_end_date": group["train_end_date"].max(),
            "test_start_date": group["test_start_date"].min(),
            "test_end_date": group["test_end_date"].max(),
        }
        weights = group["test_rows"].astype("float64").clip(lower=1.0)
        for column in group.columns:
            if column in row or column == "test_year":
                continue
            if column in sum_cols and pd.api.types.is_numeric_dtype(group[column]):
                row[column] = group[column].sum()
            elif column in max_cols and pd.api.types.is_numeric_dtype(group[column]):
                row[column] = group[column].max()
            elif column in weighted_mean_cols and pd.api.types.is_numeric_dtype(group[column]):
                valid = group[column].notna()
                row[column] = (
                    float(np.average(group.loc[valid, column], weights=weights.loc[valid]))
                    if valid.any()
                    else float("nan")
                )
            elif pd.api.types.is_numeric_dtype(group[column]):
                row[column] = group[column].iloc[0]
            else:
                row[column] = group[column].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def train_from_args(args: argparse.Namespace) -> None:
    config = load_run_config(args.config)
    tick_path = (
        args.input
        or _str_config(config, "data", "tick_path", "")
        or os.environ.get("OPENING_STRENGTH_TICK_PATH", "")
    )
    data_source = _resolved_data_source(args, config, tick_path)
    if data_source == "path" and not tick_path:
        raise SystemExit(
            "No tick data path supplied. Set [data].tick_path, --input, "
            "OPENING_STRENGTH_TICK_PATH, or use [data].source = \"clickhouse\"."
        )

    run_name = run_id(config, args.config) if args.config else "local_ridge_opening"
    output_dir = Path(
        args.output_dir
        or _str_config(config, "output", "local_dir", f"output/local/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_source == "clickhouse":
        labeled = _load_clickhouse_labeled_frame(args, config)
    else:
        labeled = _load_training_frame(tick_path, args, config)
    print_mapping("dataset", dataset_summary(labeled))

    alpha = (
        args.alpha
        if args.alpha is not None
        else _float_config(config, "model", "alpha", 1.0)
    )
    evaluation_settings = _evaluation_settings(config, args)
    splits = _date_splits(labeled, args, config)
    print_mapping(
        "split_plan",
        {
            "windows": len(splits),
            "first_test": splits[0].test_start_date,
            "last_test": splits[-1].test_end_date,
            "mode": _resolved_window_mode(args, config),
        },
    )

    prediction_frames = []
    metric_rows = []
    train_stats_by_window = {}
    for split in splits:
        predictions, metrics_row, train_stats = _fit_predict_split(
            labeled=labeled,
            split=split,
            run_name=run_name,
            output_dir=output_dir,
            args=args,
            config=config,
            alpha=alpha,
            evaluation_settings=evaluation_settings,
        )
        prediction_frames.append(predictions)
        metric_rows.append(metrics_row)
        train_stats_by_window[str(metrics_row["test_month"])] = train_stats

    combined_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    if not combined_predictions.empty:
        sort_cols = [
            column
            for column in ["date", "symbol", "timestamp", "decision_time"]
            if column in combined_predictions.columns
        ]
        combined_predictions = combined_predictions.sort_values(sort_cols)
    write_frame(combined_predictions, output_dir / "predictions.parquet")

    combined_buckets = score_bucket_returns(
        combined_predictions,
        bins=int(evaluation_settings["score_bins"]),
        group_cols=evaluation_settings["_bucket_group_cols"],
    )
    combined_buckets.to_csv(output_dir / "score_buckets.csv", index=False)

    metrics_by_window = pd.DataFrame(metric_rows)
    metrics_by_year = _metrics_by_year_from_windows(metrics_by_window)
    metrics_by_year.to_csv(output_dir / "metrics_by_year.csv", index=False)
    metrics_by_year.to_parquet(output_dir / "metrics_by_year.parquet", index=False)
    if not metrics_by_window["test_month"].is_unique or len(metrics_by_window) != len(metrics_by_year):
        metrics_by_window.to_csv(output_dir / "metrics_by_month.csv", index=False)
        metrics_by_window.to_parquet(output_dir / "metrics_by_month.parquet", index=False)

    _write_json(
        output_dir / "metrics.json",
        {
            "run_id": run_name,
            "windows": len(splits),
            "train_window": f"{splits[0].train_start_date} -> {splits[-1].train_end_date}",
            "test_window": f"{splits[0].test_start_date} -> {splits[-1].test_end_date}",
            "train_stats_by_window": train_stats_by_window,
            "model": _model_json(config, alpha),
            "evaluation": {
                key: value
                for key, value in evaluation_settings.items()
                if not key.startswith("_")
            },
            "metrics_by_window": metric_rows,
            "metrics_by_year": metrics_by_year.to_dict(orient="records"),
        },
    )

    print_mapping(
        "evaluation_settings",
        {key: value for key, value in evaluation_settings.items() if not key.startswith("_")},
    )
    print(f"\nwrote outputs: {output_dir}")
