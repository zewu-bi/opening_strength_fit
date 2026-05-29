from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

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
from opening_strength_fit.candidates import (
    filter_opening_candidates,
    opening_candidate_mask,
)
from opening_strength_fit.config import (
    config_bool,
    config_clock_list,
    config_float,
    config_float_mapping,
    config_int,
    config_int_tuple,
    config_list,
    config_optional_int,
    config_str,
    config_value,
    load_toml,
    run_id,
)
from opening_strength_fit.dataset import build_labeled_feature_frame, load_ticks
from opening_strength_fit.evaluation import (
    format_group_cols,
    group_cols_for_mode,
    score_bucket_returns,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.features import (
    add_postopen_decision_features,
    add_postopen_v2_decision_features,
)
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.model import (
    evaluate_prediction_frame,
    fit_gbm_frame,
    fit_lightgbm_frame,
    fit_ridge_frame,
    predict_frame,
)
from opening_strength_fit.reports import (
    dataset_summary,
    metrics_by_year_from_windows,
    print_mapping,
)
from opening_strength_fit.rolling import (
    annual_rolling_date_splits,
    chronological_date_split,
    monthly_rolling_date_splits,
)
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, parse_clock_times
from opening_strength_fit.schema import (
    ensure_timestamp_columns,
    normalize_clock_time,
    standardize_columns,
)
from opening_strength_fit.stock_pool import (
    StockPoolConfig,
    add_configured_stock_pool_feature,
    apply_stock_pool_cli_overrides,
    configured_stock_pool_selection_frame,
    filter_configured_stock_pool_train,
    load_configured_stock_pool,
    stock_pool_config_from_mapping,
    stock_pool_evaluation_settings,
    stock_pool_runtime_summary,
)
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
        choices=["auto", "path", "clickhouse", "labeled_pvc"],
        default=None,
        help="Override [data].source. --input always uses a local/path source.",
    )
    parser.add_argument(
        "--labeled-input",
        default=None,
        help="PVC/local labeled parquet/csv path override for data.source=labeled_pvc.",
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
    parser.add_argument(
        "--pool",
        choices=["L", "M", "S", "l", "m", "s"],
        default=None,
        help=(
            "Use mentor stock pool L/M/S as a selection mask. "
            "By default this keeps full-universe training and restricts TopN selection."
        ),
    )
    parser.add_argument(
        "--pool-path",
        default=None,
        help="Explicit stock-pool parquet path, e.g. lml.bzw@ssd/data/pool_S.parquet.",
    )
    parser.add_argument(
        "--pool-date-lag-sessions",
        type=int,
        default=None,
        help="Use the pool from this many prior pool sessions; set 1 for conservative no-lookahead checks.",
    )
    parser.add_argument(
        "--pool-filter-train",
        action="store_true",
        help="Also restrict training rows to the selected stock pool. Default only restricts TopN selection.",
    )
    parser.add_argument(
        "--pool-add-feature",
        action="store_true",
        help="Add stock_pool_member as a model feature. Default only annotates predictions.",
    )
    return parser


def _feature_filters_from_config(config: dict) -> dict[str, tuple[str, ...]]:
    return {
        "include_columns": tuple(
            config_list(config, "features", "include_feature_columns", [])
        ),
        "include_prefixes": tuple(
            config_list(config, "features", "include_feature_prefixes", [])
        ),
        "include_patterns": tuple(
            config_list(config, "features", "include_feature_regexes", [])
        ),
        "drop_columns": tuple(config_list(config, "features", "drop_feature_columns", [])),
        "drop_prefixes": tuple(config_list(config, "features", "drop_feature_prefixes", [])),
        "drop_patterns": tuple(config_list(config, "features", "drop_feature_regexes", [])),
    }


def _drop_features_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    prefixes = tuple(config_list(config, "features", "drop_feature_prefixes", []))
    columns = tuple(config_list(config, "features", "drop_feature_columns", []))
    if not prefixes and not columns:
        return labeled
    drop_columns = [
        column
        for column in labeled.columns
        if column in columns or (prefixes and column.startswith(prefixes))
    ]
    if not drop_columns:
        return labeled
    return labeled.drop(columns=drop_columns)


def _apply_feature_transforms_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if config_bool(config, "features", "include_postopen_decision", False):
        labeled = add_postopen_decision_features(
            labeled,
            windows=config_int_tuple(
                config,
                "features",
                "postopen_decision_windows",
                (1, 3, 5),
            ),
        )
    if config_bool(config, "features", "include_postopen_v2", False):
        labeled = add_postopen_v2_decision_features(
            labeled,
            windows=config_int_tuple(
                config,
                "features",
                "postopen_v2_windows",
                (1, 2, 3, 5),
            ),
            depth_levels=config_int_tuple(
                config,
                "features",
                "postopen_v2_depth_levels",
                (3, 5, 10),
            ),
        )
    labeled = _drop_features_from_config(labeled, config)
    return labeled


def _apply_candidate_filter_from_config(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if not config_bool(config, "candidate_filter", "enabled", False):
        return frame
    return filter_opening_candidates(
        frame,
        min_values=config_float_mapping(config, "candidate_filter", "min"),
        max_values=config_float_mapping(config, "candidate_filter", "max"),
        rank_min_values=config_float_mapping(
            config,
            "candidate_filter",
            "rank_min",
        ),
        rank_max_values=config_float_mapping(
            config,
            "candidate_filter",
            "rank_max",
        ),
        rank_group_cols=config_list(
            config,
            "candidate_filter",
            "rank_group_cols",
            ["date", "decision_target_timestamp"],
        ),
        rank_method=config_str(config, "candidate_filter", "rank_method", "first"),
    )


def _apply_sample_weight_from_config(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if not config_bool(config, "sample_weight", "enabled", False):
        return frame
    output_col = config_str(config, "sample_weight", "output_col", "sample_weight")
    pass_weight = config_float(config, "sample_weight", "pass_weight", 1.0)
    fail_weight = config_float(config, "sample_weight", "fail_weight", 0.25)
    mask = opening_candidate_mask(
        frame,
        min_values=config_float_mapping(config, "sample_weight", "min"),
        max_values=config_float_mapping(config, "sample_weight", "max"),
        rank_min_values=config_float_mapping(config, "sample_weight", "rank_min"),
        rank_max_values=config_float_mapping(config, "sample_weight", "rank_max"),
        rank_group_cols=config_list(
            config,
            "sample_weight",
            "rank_group_cols",
            ["date", "decision_target_timestamp"],
        ),
        rank_method=config_str(config, "sample_weight", "rank_method", "first"),
    )
    out = frame.copy()
    out[output_col] = np.where(mask.to_numpy(), pass_weight, fail_weight)
    return out


def _apply_guard_features_from_config(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if not config_bool(config, "guard_features", "enabled", False):
        return frame
    rank_columns = tuple(config_list(config, "guard_features", "rank_columns", []))
    pass_col = config_str(config, "guard_features", "pass_col", "guard_pass")
    prefix = config_str(config, "guard_features", "prefix", "guard_")
    rank_group_cols = config_list(
        config,
        "guard_features",
        "rank_group_cols",
        ["date", "decision_target_timestamp"],
    )
    rank_method = config_str(config, "guard_features", "rank_method", "average")

    out = frame.copy()
    group_cols = [column for column in rank_group_cols if column in out.columns]
    if rank_columns and not group_cols:
        raise SystemExit("guard feature ranks need at least one available group column")
    for column in rank_columns:
        if column not in out.columns:
            raise SystemExit(f"guard feature missing required column: {column}")
        values = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        out[f"{prefix}{column}_rank_pct"] = values.groupby(
            [out[col] for col in group_cols]
        ).rank(method=rank_method, pct=True)

    if pass_col:
        mask = opening_candidate_mask(
            out,
            min_values=config_float_mapping(config, "guard_features", "min"),
            max_values=config_float_mapping(config, "guard_features", "max"),
            rank_min_values=config_float_mapping(config, "guard_features", "rank_min"),
            rank_max_values=config_float_mapping(config, "guard_features", "rank_max"),
            rank_group_cols=rank_group_cols,
            rank_method=rank_method,
        )
        out[pass_col] = mask.astype("int8")
    return out


def _clock_from_series(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype(str)
        .str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False)
        .fillna("")
    )
    return extracted.map(lambda value: normalize_clock_time(value) if value else "")


def _filter_labeled_sample_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    mode = config_str(config, "sample", "mode", "")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"decision", "decision_point", "decision_points"}:
        return labeled

    decision_times = parse_clock_times(
        config_value(config, "sample", "decision_times", DEFAULT_DECISION_TIMES)
    )
    if not decision_times:
        raise SystemExit("decision point sampling needs at least one decision time")

    if "decision_time" in labeled.columns:
        clock = _clock_from_series(labeled["decision_time"])
    else:
        time_col = (
            "decision_target_timestamp"
            if "decision_target_timestamp" in labeled.columns
            else "timestamp"
        )
        clock = pd.to_datetime(labeled[time_col], errors="coerce").dt.strftime(
            "%H:%M:%S"
        )
    mask = clock.isin(set(decision_times))

    max_lag = config_value(config, "sample", "decision_max_lag_seconds", None)
    if max_lag not in (None, "") and "decision_lag_seconds" in labeled.columns:
        lag = pd.to_numeric(labeled["decision_lag_seconds"], errors="coerce")
        mask &= lag.ge(0.0) & lag.le(float(max_lag))

    return labeled.loc[mask].copy()


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


def _prediction_r2(predictions: pd.DataFrame, *, target_col: str = "label") -> float:
    target_col = target_col if target_col in predictions.columns else "label"
    frame = predictions.loc[
        predictions[target_col].notna() & predictions["prediction"].notna()
    ]
    if len(frame) < 2:
        return float("nan")
    y = frame[target_col].astype("float64").to_numpy()
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
    target_col: str,
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
        "model_target_col": target_col,
        "model_test_r2": _prediction_r2(predictions, target_col=target_col),
        "ic_mode": str(evaluation_settings["ic_mode"]),
        "selection_mode": str(evaluation_settings["selection_mode"]),
        "top_n": int(evaluation_settings["top_n"]),
        "stock_pool_enabled": bool(evaluation_settings.get("stock_pool_enabled", False)),
        "stock_pool_name": str(evaluation_settings.get("stock_pool_name", "")),
        "stock_pool_path": str(evaluation_settings.get("stock_pool_path", "")),
        "stock_pool_date_lag_sessions": int(
            evaluation_settings.get("stock_pool_date_lag_sessions", 0)
        ),
        "stock_pool_filter_train": bool(
            evaluation_settings.get("stock_pool_filter_train", False)
        ),
        "stock_pool_filter_selection": bool(
            evaluation_settings.get("stock_pool_filter_selection", False)
        ),
        "stock_pool_add_feature": bool(
            evaluation_settings.get("stock_pool_add_feature", False)
        ),
    }
    row.update(metrics)
    for key, value in top_summary.items():
        row[f"top_score_{key}"] = value
    return row


def _feature_limit(args: argparse.Namespace, config: dict) -> int | None:
    raw = (
        args.feature_limit
        if args.feature_limit is not None
        else config_int(config, "data", "feature_limit", 0)
    )
    return raw if raw and raw > 0 else None


def build_labeled_frame_from_config(
    ticks: pd.DataFrame,
    config: dict,
    *,
    apply_candidate_filter: bool = True,
) -> pd.DataFrame:
    volume_unit_multiplier = config_float(
        config,
        "labels",
        "volume_unit_multiplier",
        1.0,
    )
    use_universe = config_bool(config, "universe", "enabled", True)
    symbols_file = config_str(config, "universe", "symbols_file", "")
    universe_symbols = load_symbol_list(symbols_file) if symbols_file else None
    sample_mode = config_str(config, "sample", "mode", "all_ticks")
    max_decision_lag = config_value(
        config,
        "sample",
        "decision_max_lag_seconds",
        5,
    )
    labeled = build_labeled_feature_frame(
        ticks,
        buy_price_col=config_str(config, "labels", "buy_price_col", "ask_price_1"),
        volume_col=config_str(config, "labels", "volume_col", "volume"),
        turnover_col=config_str(config, "labels", "turnover_col", "turnover"),
        hold_seconds=config_int(config, "labels", "hold_seconds", 60),
        sell_window_seconds=config_int(config, "labels", "sell_window_seconds", 60),
        volume_unit_multiplier=volume_unit_multiplier,
        fee_bps=config_float(config, "labels", "fee_bps", 0.0),
        entry_tick_delay=config_int(config, "labels", "entry_tick_delay", 0),
        entry_max_gap_seconds=config_value(
            config,
            "labels",
            "entry_max_gap_seconds",
            None,
        ),
        sample_start_time=config_str(config, "sample", "start_time", "09:30:00"),
        sample_end_time=config_str(config, "sample", "end_time", "09:40:00"),
        include_preopen=config_bool(config, "features", "include_preopen", True),
        max_future_gap_seconds=config_value(
            config,
            "labels",
            "max_future_gap_seconds",
            None,
        ),
        tradable_statuses=config_list(config, "filters", "tradable_statuses", []),
        universe_regex=(
            config_str(
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
        decision_times=config_clock_list(
            config,
            "sample",
            "decision_times",
            DEFAULT_DECISION_TIMES,
        ),
        decision_max_lag_seconds=(
            None if max_decision_lag in (None, "") else int(max_decision_lag)
        ),
    )
    labeled = _apply_feature_transforms_from_config(labeled, config)
    if apply_candidate_filter:
        return _apply_candidate_filter_from_config(labeled, config)
    return labeled


def _input_kind(args: argparse.Namespace, config: dict) -> str:
    return args.input_kind or config_str(config, "data", "input_kind", "auto")


def _looks_labeled(frame: pd.DataFrame) -> bool:
    return {"date", "symbol", "timestamp", "label"}.issubset(frame.columns)


def _filter_labeled_frame(labeled: pd.DataFrame, config: dict) -> pd.DataFrame:
    labeled = ensure_timestamp_columns(standardize_columns(labeled))
    if config_bool(config, "universe", "enabled", True):
        symbols_file = config_str(config, "universe", "symbols_file", "")
        labeled = filter_symbol_universe(
            labeled,
            symbol_regex=config_str(
                config,
                "universe",
                "symbol_regex",
                DEFAULT_A_SHARE_SYMBOL_REGEX,
            ),
            symbols=load_symbol_list(symbols_file) if symbols_file else None,
        )
    labeled = _filter_labeled_sample_from_config(labeled, config)
    labeled = _apply_feature_transforms_from_config(labeled, config)
    return _apply_candidate_filter_from_config(labeled, config)


def _load_training_frame(path: str, args: argparse.Namespace, config: dict) -> pd.DataFrame:
    frame = load_ticks(path)
    kind = _input_kind(args, config)
    if kind == "labeled" or (kind == "auto" and _looks_labeled(frame)):
        return _filter_labeled_frame(frame, config)
    if kind not in {"auto", "raw_ticks"}:
        raise SystemExit(
            f"unknown data.input_kind={kind!r}; expected auto, raw_ticks, or labeled"
        )
    return build_labeled_frame_from_config(frame, config)


def _resolved_data_source(args: argparse.Namespace, config: dict, tick_path: str) -> str:
    if args.input:
        return "path"
    if getattr(args, "labeled_input", None):
        return "labeled_pvc"
    source = args.data_source or config_str(config, "data", "source", "auto")
    source = source.strip().lower()
    if source == "auto":
        labeled_path = (
            os.environ.get("OPENING_STRENGTH_LABELED_PATH", "")
            or config_value(config, "data", "labeled_path", "")
        )
        if labeled_path:
            return "labeled_pvc"
        return "path" if tick_path else "clickhouse"
    if source in {"path", "clickhouse", "labeled_pvc"}:
        return source
    raise SystemExit(
        f"unknown data.source={source!r}; expected auto, path, clickhouse, or labeled_pvc"
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
            else config_int(config, "window", "train_months", 12)
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
            else config_optional_int(config, "window", "train_start_year", None)
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


def _cache_path(config: dict) -> Path | None:
    raw = config_value(
        config,
        "cache",
        "labeled_path",
        config_value(config, "cache", "path", ""),
    )
    if raw in (None, ""):
        return None
    return Path(str(raw))


def _labeled_pvc_path(args: argparse.Namespace, config: dict) -> Path:
    raw = (
        args.labeled_input
        or os.environ.get("OPENING_STRENGTH_LABELED_PATH", "")
        or config_value(config, "data", "labeled_path", "")
        or config_value(config, "data", "tick_path", "")
        or config_value(
            config,
            "cache",
            "labeled_path",
            config_value(config, "cache", "path", ""),
        )
    )
    if raw in (None, ""):
        raise SystemExit(
            "No labeled PVC path supplied. Set [data].labeled_path, "
            "[data].tick_path, [cache].labeled_path, --labeled-input, or "
            "OPENING_STRENGTH_LABELED_PATH."
        )
    return Path(str(raw))


def _arg_value(args: argparse.Namespace, name: str, default=None):
    return getattr(args, name, default)


def _rolling_monthly_date_bounds(
    args: argparse.Namespace,
    config: dict,
) -> tuple[str, str] | None:
    train_months = (
        _arg_value(args, "train_months")
        if _arg_value(args, "train_months") is not None
        else config_int(config, "window", "train_months", 12)
    )
    first_test_month = _arg_value(args, "test_start_month") or config_value(
        config,
        "window",
        "test_start_month",
        None,
    )
    last_test_month = _arg_value(args, "test_end_month") or config_value(
        config,
        "window",
        "test_end_month",
        None,
    )
    if not first_test_month or not last_test_month:
        return None
    first_period = pd.Period(first_test_month, freq="M")
    last_period = pd.Period(last_test_month, freq="M")
    start_period = first_period - int(train_months)
    return str(start_period.to_timestamp().date()), _period_end_date(last_period)


def _labeled_pvc_date_filters(
    args: argparse.Namespace,
    config: dict,
) -> tuple[list[tuple[str, str, object]] | None, dict[str, str]]:
    if _resolved_window_mode(args, config) != "rolling_monthly":
        return None, {}
    bounds = _rolling_monthly_date_bounds(args, config)
    if not bounds:
        return None, {}
    start_date, end_date = bounds
    return [
        ("date", ">=", start_date),
        ("date", "<=", end_date),
    ], {"date_start": start_date, "date_end": end_date}


def _cache_lock_done_path(lock_path: Path) -> Path:
    return Path(f"{lock_path}.done")


def _cache_lock_heartbeat_path(lock_path: Path) -> Path:
    return lock_path / "heartbeat"


def _write_cache_lock_heartbeat(lock_path: Path) -> None:
    try:
        lock_path.mkdir(parents=True, exist_ok=True)
        _cache_lock_heartbeat_path(lock_path).write_text(
            json.dumps({"pid": os.getpid(), "time": time.time()}) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _cache_lock_has_fresh_heartbeat(
    lock_path: Path,
    *,
    stale_after_seconds: float,
) -> bool:
    heartbeat_path = _cache_lock_heartbeat_path(lock_path)
    try:
        heartbeat_age = time.time() - heartbeat_path.stat().st_mtime
    except OSError:
        return False
    return heartbeat_age <= float(stale_after_seconds)


class _CacheLockHeartbeat:
    def __init__(self, lock_path: Path, interval_seconds: float = 60.0) -> None:
        self.lock_path = lock_path
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_CacheLockHeartbeat":
        _write_cache_lock_heartbeat(self.lock_path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.interval_seconds))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            _write_cache_lock_heartbeat(self.lock_path)


def _load_labeled_cache(path: Path) -> pd.DataFrame:
    labeled = read_frame(path)
    return ensure_timestamp_columns(standardize_columns(labeled))


def _write_labeled_cache(labeled: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes) or ".parquet"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp{suffix}")
    write_frame(labeled, tmp_path)
    os.replace(tmp_path, path)


def _mark_cache_ready(cache_path: Path, lock_path: Path) -> None:
    _cache_lock_done_path(lock_path).write_text(
        json.dumps(
            {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "time": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _clear_cache_ready(lock_path: Path) -> None:
    try:
        _cache_lock_done_path(lock_path).unlink()
    except FileNotFoundError:
        return


def _acquire_cache_lock(
    lock_path: Path,
    timeout_seconds: float,
    *,
    cache_path: Path | None = None,
    cache_read: bool = True,
    poll_seconds: float = 15.0,
) -> str:
    start = time.monotonic()
    timeout_seconds = float(timeout_seconds)
    heartbeat_stale_after = max(timeout_seconds, float(poll_seconds) * 3.0, 60.0)
    while True:
        if cache_path and cache_read and cache_path.exists():
            return "cache_ready"
        try:
            lock_path.mkdir(parents=True)
            _write_cache_lock_heartbeat(lock_path)
            return "acquired"
        except FileExistsError:
            if (
                _cache_lock_done_path(lock_path).exists()
                and cache_path
                and cache_read
                and cache_path.exists()
            ):
                return "cache_ready"
            if timeout_seconds > 0.0 and (
                time.monotonic() - start
            ) >= timeout_seconds:
                if _cache_lock_has_fresh_heartbeat(
                    lock_path,
                    stale_after_seconds=heartbeat_stale_after,
                ):
                    start = time.monotonic()
                else:
                    return "timeout"
            time.sleep(float(poll_seconds))


def _release_cache_lock(lock_path: Path) -> None:
    try:
        for child in lock_path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
        lock_path.rmdir()
    except FileNotFoundError:
        return


def _build_clickhouse_labeled_frame(
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
    use_universe = config_bool(config, "universe", "enabled", True)
    symbols_file = config_str(config, "universe", "symbols_file", "")
    symbols = sorted(load_symbol_list(symbols_file)) if use_universe and symbols_file else None
    symbol_regex = (
        config_str(config, "universe", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
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
        labeled = build_labeled_frame_from_config(
            ticks,
            config,
            apply_candidate_filter=False,
        )
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


def _load_clickhouse_labeled_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    cache_enabled = config_bool(config, "cache", "enabled", False)
    cache_path = _cache_path(config) if cache_enabled else None
    cache_read = config_bool(config, "cache", "read", True)
    cache_write = config_bool(config, "cache", "write", True)

    if cache_path and cache_read and cache_path.exists():
        print_mapping("labeled_cache", {"action": "read", "path": str(cache_path)})
        return _apply_candidate_filter_from_config(_load_labeled_cache(cache_path), config)

    if not cache_path or not cache_write:
        return _apply_candidate_filter_from_config(
            _build_clickhouse_labeled_frame(args, config),
            config,
        )

    lock_path = Path(f"{cache_path}.lock")
    timeout_seconds = config_int(config, "cache", "lock_timeout_seconds", 21_600)
    lock_status = _acquire_cache_lock(
        lock_path,
        timeout_seconds,
        cache_path=cache_path,
        cache_read=cache_read,
    )
    if lock_status == "cache_ready":
        print_mapping(
            "labeled_cache",
            {"action": "read_after_wait", "path": str(cache_path)},
        )
        return _apply_candidate_filter_from_config(
            _load_labeled_cache(cache_path),
            config,
        )
    if lock_status == "timeout":
        if cache_read and cache_path.exists():
            print_mapping(
                "labeled_cache",
                {"action": "read_after_wait", "path": str(cache_path)},
            )
            return _apply_candidate_filter_from_config(
                _load_labeled_cache(cache_path),
                config,
            )
        raise SystemExit(
            f"timed out waiting for labeled cache lock: {lock_path}; "
            "cache file was not created"
        )

    try:
        with _CacheLockHeartbeat(lock_path):
            if cache_read and cache_path.exists():
                print_mapping(
                    "labeled_cache",
                    {"action": "read_after_lock", "path": str(cache_path)},
                )
                base_labeled = _load_labeled_cache(cache_path)
            else:
                _clear_cache_ready(lock_path)
                base_labeled = _build_clickhouse_labeled_frame(args, config)
                _write_labeled_cache(base_labeled, cache_path)
                _mark_cache_ready(cache_path, lock_path)
                print_mapping(
                    "labeled_cache",
                    {
                        "action": "write",
                        "path": str(cache_path),
                        **dataset_summary(base_labeled),
                    },
                )
    finally:
        _release_cache_lock(lock_path)

    return _apply_candidate_filter_from_config(base_labeled, config)


def _load_labeled_pvc_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    path = _labeled_pvc_path(args, config)
    filters, filter_summary = _labeled_pvc_date_filters(args, config)
    print_mapping(
        "labeled_pvc",
        {"action": "read", "path": str(path), **filter_summary},
    )
    return _filter_labeled_frame(read_frame(path, filters=filters), config)


def _test_year_from_args(args: argparse.Namespace, config: dict, key: str) -> int | None:
    if key == "start":
        if args.test_start_year is not None:
            return args.test_start_year
        test_date = args.test_start_date or config_value(config, "window", "test_start_date", None)
        explicit = config_optional_int(config, "window", "test_start_year", None)
    else:
        if args.test_end_year is not None:
            return args.test_end_year
        test_date = args.test_end_date or config_value(config, "window", "test_end_date", None)
        explicit = config_optional_int(config, "window", "test_end_year", None)
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
                else config_int(config, "window", "train_months", 12)
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
                else config_optional_int(config, "window", "train_start_year", None)
            ),
            first_test_year=_test_year_from_args(args, config, "start"),
            last_test_year=_test_year_from_args(args, config, "end"),
            min_train_years=config_int(config, "window", "min_train_years", 1),
        )
    return [
        chronological_date_split(
            labeled,
            test_start_date=args.test_start_date
            or config_value(config, "window", "test_start_date", None),
            test_end_date=args.test_end_date
            or config_value(config, "window", "test_end_date", None),
            train_fraction=config_float(config, "window", "train_fraction", 0.8),
        )
    ]


def _resolved_window_mode(args: argparse.Namespace, config: dict) -> str:
    window_mode = _arg_value(args, "split_mode") or config_str(
        config,
        "window",
        "mode",
        "chronological",
    )
    if _arg_value(args, "rolling_monthly", False):
        return "rolling_monthly"
    if _arg_value(args, "rolling_annual", False):
        return "rolling_annual"
    return window_mode


def _evaluation_settings(config: dict, args: argparse.Namespace) -> dict[str, object]:
    bucket_mode = config_str(config, "evaluation", "bucket_mode", "daily")
    selection_mode = config_str(config, "evaluation", "selection_mode", "symbol_day")
    ic_mode = config_str(config, "evaluation", "ic_mode", bucket_mode)
    bucket_group_cols = group_cols_for_mode(bucket_mode)
    selection_group_cols = group_cols_for_mode(selection_mode)
    ic_group_cols = group_cols_for_mode(ic_mode)
    settings = {
        "score_bucket_mode": bucket_mode,
        "score_bucket_group_cols": format_group_cols(bucket_group_cols),
        "selection_mode": selection_mode,
        "selection_group_cols": format_group_cols(selection_group_cols),
        "ic_mode": ic_mode,
        "ic_group_cols": format_group_cols(ic_group_cols),
        "top_n": (
            args.top_n
            if args.top_n is not None
            else config_int(config, "evaluation", "top_n", 20)
        ),
        "score_bins": config_int(config, "evaluation", "score_bins", 5),
        "_bucket_group_cols": bucket_group_cols,
        "_selection_group_cols": selection_group_cols,
        "_ic_group_cols": ic_group_cols,
    }
    settings.update(
        stock_pool_evaluation_settings(stock_pool_config_from_mapping(config))
    )
    return settings


def _fit_prediction_model(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
):
    model_name = config_str(config, "model", "name", "ridge").strip().lower()
    feature_limit = _feature_limit(args, config)
    target_col = config_str(config, "model", "target_col", "label")
    feature_filters = _feature_filters_from_config(config)
    if model_name == "ridge":
        return fit_ridge_frame(
            train,
            alpha=alpha,
            feature_limit=feature_limit,
            target_col=target_col,
            feature_filters=feature_filters,
        )
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return fit_gbm_frame(
            train,
            feature_limit=feature_limit,
            target_col=target_col,
            feature_filters=feature_filters,
            max_iter=config_int(config, "model", "max_iter", 100),
            learning_rate=config_float(config, "model", "learning_rate", 0.05),
            max_leaf_nodes=config_int(config, "model", "max_leaf_nodes", 31),
            l2_regularization=config_float(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            random_state=config_int(config, "model", "random_state", 7),
        )
    if model_name in {"lightgbm", "lgbm"}:
        return fit_lightgbm_frame(
            train,
            feature_limit=feature_limit,
            target_col=target_col,
            sample_weight_col=config_str(config, "model", "sample_weight_col", ""),
            feature_filters=feature_filters,
            n_estimators=config_int(config, "model", "n_estimators", 300),
            learning_rate=config_float(config, "model", "learning_rate", 0.03),
            num_leaves=config_int(config, "model", "num_leaves", 63),
            max_depth=config_int(config, "model", "max_depth", -1),
            min_child_samples=config_int(config, "model", "min_child_samples", 200),
            subsample=config_float(config, "model", "subsample", 1.0),
            colsample_bytree=config_float(config, "model", "colsample_bytree", 1.0),
            reg_alpha=config_float(config, "model", "reg_alpha", 0.0),
            reg_lambda=config_float(config, "model", "reg_lambda", 0.0),
            random_state=config_int(config, "model", "random_state", 7),
            n_jobs=config_int(config, "model", "n_jobs", -1),
            device_type=config_str(config, "model", "device_type", "cpu"),
            max_bin=config_optional_int(config, "model", "max_bin", None),
            gpu_use_dp=config_bool(config, "model", "gpu_use_dp", False),
        )
    raise SystemExit(
        f"unsupported model.name={model_name!r}; expected ridge, gbm, or lightgbm"
    )


def _model_json(config: dict, alpha: float) -> dict[str, object]:
    model_name = config_str(config, "model", "name", "ridge").strip().lower()
    target_col = config_str(config, "model", "target_col", "label")
    if model_name == "ridge":
        return {"name": "ridge", "alpha": alpha, "target_col": target_col}
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return {
            "name": "gbm",
            "target_col": target_col,
            "max_iter": config_int(config, "model", "max_iter", 100),
            "learning_rate": config_float(config, "model", "learning_rate", 0.05),
            "max_leaf_nodes": config_int(config, "model", "max_leaf_nodes", 31),
            "l2_regularization": config_float(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            "random_state": config_int(config, "model", "random_state", 7),
        }
    if model_name in {"lightgbm", "lgbm"}:
        return {
            "name": "lightgbm",
            "target_col": target_col,
            "device_type": config_str(config, "model", "device_type", "cpu"),
            "n_estimators": config_int(config, "model", "n_estimators", 300),
            "learning_rate": config_float(config, "model", "learning_rate", 0.03),
            "num_leaves": config_int(config, "model", "num_leaves", 63),
            "max_depth": config_int(config, "model", "max_depth", -1),
            "min_child_samples": config_int(config, "model", "min_child_samples", 200),
            "subsample": config_float(config, "model", "subsample", 1.0),
            "colsample_bytree": config_float(config, "model", "colsample_bytree", 1.0),
            "reg_alpha": config_float(config, "model", "reg_alpha", 0.0),
            "reg_lambda": config_float(config, "model", "reg_lambda", 0.0),
            "random_state": config_int(config, "model", "random_state", 7),
            "n_jobs": config_int(config, "model", "n_jobs", -1),
            "max_bin": config_optional_int(config, "model", "max_bin", None),
            "gpu_use_dp": config_bool(config, "model", "gpu_use_dp", False),
            "sample_weight_col": config_str(config, "model", "sample_weight_col", ""),
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
    stock_pool_settings: StockPoolConfig | None = None,
    stock_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    stock_pool_settings = stock_pool_settings or stock_pool_config_from_mapping(config)
    train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
    test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()
    train = filter_configured_stock_pool_train(train, stock_pool_settings, stock_pool)
    model, train_stats = _fit_prediction_model(
        train,
        args=args,
        config=config,
        alpha=alpha,
    )
    predictions = predict_frame(model, test)
    if "valid_label" in predictions.columns:
        predictions = predictions.loc[predictions["valid_label"]].copy()
    predictions, selection_predictions, stock_pool_summary = configured_stock_pool_selection_frame(
        predictions,
        stock_pool_settings,
        stock_pool,
    )

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
        selection_predictions,
        top_n=int(evaluation_settings["top_n"]),
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary = summarize_trades(
        top_trades,
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary.update(stock_pool_summary)

    prediction_year = _test_year(split.test_start_date)
    prediction_period = (
        str(pd.Timestamp(split.test_start_date).to_period("M"))
        if split.test_start_date[:7] == split.test_end_date[:7]
        else str(prediction_year)
    )
    write_frame(predictions, output_dir / f"predictions_{prediction_period}.parquet")
    buckets.to_csv(output_dir / f"score_buckets_{prediction_period}.csv", index=False)
    if stock_pool is not None and stock_pool_settings.filter_selection:
        pool_buckets = score_bucket_returns(
            selection_predictions,
            bins=int(evaluation_settings["score_bins"]),
            group_cols=evaluation_settings["_bucket_group_cols"],
        )
        pool_buckets.to_csv(
            output_dir / f"score_buckets_{prediction_period}_stock_pool.csv",
            index=False,
        )
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
        target_col=model.target_col,
        evaluation_settings=evaluation_settings,
    )
    print_mapping(f"train_stats[{prediction_period}]", train_stats)
    print_mapping(f"prediction_metrics[{prediction_period}]", metrics)
    print_mapping(
        f"top_score_summary[{prediction_period},top_n={evaluation_settings['top_n']}]",
        top_summary,
    )
    if stock_pool_summary:
        print_mapping(f"stock_pool_summary[{prediction_period}]", stock_pool_summary)
    return predictions, metrics_row, train_stats


def train_from_args(args: argparse.Namespace) -> None:
    config = load_run_config(args.config)
    config = apply_stock_pool_cli_overrides(config, args)
    stock_pool_settings = stock_pool_config_from_mapping(config)
    tick_path = (
        args.input
        or config_str(config, "data", "tick_path", "")
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
        or config_str(config, "output", "local_dir", f"output/local/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_source == "clickhouse":
        labeled = _load_clickhouse_labeled_frame(args, config)
    elif data_source == "labeled_pvc":
        labeled = _load_labeled_pvc_frame(args, config)
    else:
        labeled = _load_training_frame(tick_path, args, config)
    labeled = _apply_guard_features_from_config(labeled, config)
    labeled = _apply_sample_weight_from_config(labeled, config)
    stock_pool = load_configured_stock_pool(stock_pool_settings)
    if stock_pool is not None:
        print_mapping("stock_pool", stock_pool_runtime_summary(stock_pool_settings, stock_pool))
        labeled = add_configured_stock_pool_feature(
            labeled,
            stock_pool_settings,
            stock_pool,
        )
    print_mapping("dataset", dataset_summary(labeled))

    alpha = (
        args.alpha
        if args.alpha is not None
        else config_float(config, "model", "alpha", 1.0)
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
            stock_pool_settings=stock_pool_settings,
            stock_pool=stock_pool,
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
    if stock_pool is not None and stock_pool_settings.filter_selection:
        _, combined_pool_predictions, _ = configured_stock_pool_selection_frame(
            combined_predictions,
            stock_pool_settings,
            stock_pool,
        )
        combined_pool_buckets = score_bucket_returns(
            combined_pool_predictions,
            bins=int(evaluation_settings["score_bins"]),
            group_cols=evaluation_settings["_bucket_group_cols"],
        )
        combined_pool_buckets.to_csv(
            output_dir / "score_buckets_stock_pool.csv",
            index=False,
        )

    metrics_by_window = pd.DataFrame(metric_rows)
    metrics_by_year = metrics_by_year_from_windows(metrics_by_window)
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
