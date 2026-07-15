from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from opening_strength_fit.cache_lock import (
    CacheLockHeartbeat as _CacheLockHeartbeat,
)
from opening_strength_fit.cache_lock import (
    acquire_cache_lock as _acquire_cache_lock,
)
from opening_strength_fit.cache_lock import (
    clear_cache_ready as _clear_cache_ready,
)
from opening_strength_fit.cache_lock import (
    mark_cache_ready as _mark_cache_ready,
)
from opening_strength_fit.cache_lock import (
    release_cache_lock as _release_cache_lock,
)
from opening_strength_fit.cache_manifest import (
    cache_manifest_path,
    publish_cache_manifest,
    validate_cache_manifest,
)
from opening_strength_fit.clickhouse_daily_reference import (
    DEFAULT_DAILY_MARKET_REFERENCE_TABLE,
    attach_daily_market_reference,
    query_lagged_daily_market_reference,
)
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
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_float_mapping,
    config_int,
    config_list,
    config_optional_int,
    config_str,
    config_value,
    run_id,
)
from opening_strength_fit.dataset import load_ticks
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.features import mechanismized_feature_value_reference_columns
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic
from opening_strength_fit.model import PREDICTION_CONTEXT_COLUMNS
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.training_labeled import (
    apply_candidate_filter_from_config,
    build_labeled_frame_from_config,
    filter_labeled_frame,
    looks_labeled,
)
from opening_strength_fit.training_windows import (
    period_end_date,
    resolve_window_mode,
    rolling_monthly_date_bounds,
    test_year_from_args,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX, load_symbol_list


def _input_kind(args: argparse.Namespace, config: dict) -> str:
    return args.input_kind or config_str(config, "data", "input_kind", "auto")


def load_training_frame(path: str, args: argparse.Namespace, config: dict) -> pd.DataFrame:
    frame = load_ticks(path)
    kind = _input_kind(args, config)
    if kind == "labeled" or (kind == "auto" and looks_labeled(frame)):
        return filter_labeled_frame(frame, config)
    if kind not in {"auto", "raw_ticks"}:
        raise SystemExit(f"unknown data.input_kind={kind!r}; expected auto, raw_ticks, or labeled")
    return build_labeled_frame_from_config(frame, config)


def resolve_data_source(args: argparse.Namespace, config: dict, tick_path: str) -> str:
    if args.input:
        return "path"
    if getattr(args, "labeled_input", None):
        return "labeled_pvc"
    source = args.data_source or config_str(config, "data", "source", "auto")
    source = source.strip().lower()
    if source == "auto":
        labeled_path = os.environ.get("OPENING_STRENGTH_LABELED_PATH", "") or config_value(
            config, "data", "labeled_path", ""
        )
        if labeled_path:
            return "labeled_pvc"
        return "path" if tick_path else "clickhouse"
    if source in {"path", "clickhouse", "labeled_pvc"}:
        return source
    raise SystemExit(
        f"unknown data.source={source!r}; expected auto, path, clickhouse, or labeled_pvc"
    )


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

    window_mode = resolve_window_mode(args, config)
    if window_mode == "rolling_monthly":
        train_months = (
            args.train_months
            if args.train_months is not None
            else config_int(config, "window", "train_months", 12)
        )
        first_test_month = args.test_start_month or config_value(
            config, "window", "test_start_month", None
        )
        last_test_month = args.test_end_month or config_value(
            config, "window", "test_end_month", None
        )
        if not first_test_month or not last_test_month:
            raise SystemExit(
                "ClickHouse rolling_monthly source needs [window].test_start_month "
                "and [window].test_end_month, or CLI overrides."
            )
        first_period = pd.Period(first_test_month, freq="M")
        last_period = pd.Period(last_test_month, freq="M")
        start_period = first_period - int(train_months)
        return str(start_period.to_timestamp().date()), period_end_date(last_period)

    if window_mode == "rolling_annual":
        train_start_year = (
            args.train_start_year
            if args.train_start_year is not None
            else config_optional_int(config, "window", "train_start_year", None)
        )
        first_test_year = test_year_from_args(args, config, "start")
        last_test_year = test_year_from_args(args, config, "end")
        if train_start_year is None or first_test_year is None or last_test_year is None:
            raise SystemExit(
                "ClickHouse rolling_annual source needs train_start_year and test start/end years."
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
    if test_start_date and str(pd.Timestamp(start_date).date()) >= str(
        pd.Timestamp(test_start_date).date()
    ):
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


def resolve_cache_path(config: dict) -> Path | None:
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


def _labeled_pvc_date_filters(
    args: argparse.Namespace,
    config: dict,
) -> tuple[list[tuple[str, str, object]] | None, dict[str, str]]:
    if resolve_window_mode(args, config) != "rolling_monthly":
        return None, {}
    bounds = rolling_monthly_date_bounds(args, config)
    if not bounds:
        return None, {}
    start_date, end_date = bounds
    return [
        ("date", ">=", start_date),
        ("date", "<=", end_date),
    ], {"date_start": start_date, "date_end": end_date}


def _mapping_keys(mapping: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(key) for key in mapping.keys())


def _existing_columns(available: set[str], columns: list[str] | tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in available]


def _matching_existing_columns(
    available: set[str],
    *,
    prefixes: tuple[str, ...] = (),
    patterns: tuple[str, ...] = (),
) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    out = []
    for column in sorted(available):
        if prefixes and column.startswith(prefixes):
            out.append(column)
            continue
        if compiled and any(pattern.search(column) for pattern in compiled):
            out.append(column)
    return out


def _postopen_decision_source_columns(config: dict) -> tuple[str, ...]:
    if not config_bool(config, "features", "include_postopen_decision", False):
        return ()
    return (
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_10",
        "bid_depth_10",
        "depth_imbalance_1",
        "depth_imbalance_10",
        "spread_bps",
        "mid_price",
        "ask_price_1",
        "bid_price_1",
        "volume",
        "turnover",
        "volume_diff_1t",
        *(f"ask_gap_{level}_bps" for level in range(2, 11)),
        *(f"bid_gap_{level}_bps" for level in range(2, 11)),
    )


def _postopen_v2_source_columns(config: dict, available: set[str]) -> list[str]:
    if not config_bool(config, "features", "include_postopen_v2", False):
        return []
    source = [
        "ask_price_1",
        "bid_price_1",
        "ask_volume_1",
        "bid_volume_1",
        "mid_price",
        "spread_bps",
        "volume",
        "turnover",
    ]
    source.extend(
        _matching_existing_columns(
            available,
            prefixes=(
                "ask_price_",
                "bid_price_",
                "ask_volume_",
                "bid_volume_",
                "ask_count_",
                "bid_count_",
                "ask_gap_",
                "bid_gap_",
                "ask_depth_",
                "bid_depth_",
                "depth_imbalance_",
                "volume_diff_",
                "turnover_diff_",
                "trade_vwap_",
                "return_",
            ),
        )
    )
    return source


def _cross_sectional_relative_source_columns(config: dict, available: set[str]) -> list[str]:
    if not config_bool(config, "features", "include_cross_sectional_relative", False):
        return []
    source = []
    source.extend(config_list(config, "features", "cross_sectional_relative_group_cols", []))
    source.extend(config_list(config, "features", "cross_sectional_relative_columns", []))
    source.extend(
        _matching_existing_columns(
            available,
            prefixes=tuple(
                config_list(config, "features", "cross_sectional_relative_prefixes", [])
            ),
            patterns=tuple(config_list(config, "features", "cross_sectional_relative_regexes", [])),
        )
    )
    return source


def _historical_surprise_source_columns(config: dict, available: set[str]) -> list[str]:
    if not config_bool(config, "features", "include_historical_same_minute_surprise", False):
        return []
    source = []
    source.extend(config_list(config, "features", "historical_surprise_columns", []))
    source.extend(
        _matching_existing_columns(
            available,
            prefixes=tuple(config_list(config, "features", "historical_surprise_prefixes", [])),
            patterns=tuple(config_list(config, "features", "historical_surprise_regexes", [])),
        )
    )
    return source


def _price_scale_source_columns(config: dict) -> list[str]:
    if not config_bool(config, "features", "include_price_scale_features", False):
        return []
    source = [
        config_str(config, "features", "price_scale_price_col", "ask_price_1"),
        "ask_price_1",
        "bid_price_1",
        "spread_abs",
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_3",
        "bid_depth_3",
        "ask_depth_10",
        "bid_depth_10",
        "postopen_v2_ask_depth_3",
        "postopen_v2_bid_depth_3",
        "postopen_v2_ask_depth_10",
        "postopen_v2_bid_depth_10",
        "volume_diff_1t",
        "volume_diff_3t",
    ]
    source.extend(f"ask_price_{level}" for level in range(2, 11))
    source.extend(f"bid_price_{level}" for level in range(2, 11))
    source.extend(config_list(config, "features", "price_scale_interaction_columns", []))
    return source


def _target_transform_source_columns(config: dict) -> list[str]:
    if not config_bool(config, "target_transform", "enabled", False):
        return []
    return [
        config_str(config, "target_transform", "source_col", "target_label"),
        *config_list(
            config,
            "target_transform",
            "group_cols",
            ["date", "decision_target_timestamp"],
        ),
    ]


def _guard_condition_columns(config: dict, section: str) -> tuple[str, ...]:
    return (
        *_mapping_keys(config_float_mapping(config, section, "min")),
        *_mapping_keys(config_float_mapping(config, section, "max")),
        *_mapping_keys(config_float_mapping(config, section, "rank_min")),
        *_mapping_keys(config_float_mapping(config, section, "rank_max")),
        *tuple(config_list(config, section, "rank_columns", [])),
        *tuple(config_list(config, section, "rank_group_cols", [])),
    )


def _labeled_pvc_read_columns(path: Path, config: dict) -> list[str] | None:
    feature_filters = feature_filters_from_config(config)
    has_include_filter = bool(
        feature_filters["include_columns"]
        or feature_filters["include_prefixes"]
        or feature_filters["include_patterns"]
    )
    explicit = tuple(config_list(config, "data", "read_columns", []))
    if not has_include_filter and not explicit:
        return None

    available = frame_columns(path)
    target_col = config_str(config, "model", "target_col", "label")
    sample_weight_col = config_str(config, "model", "sample_weight_col", "")
    sample_weight_output_col = config_str(
        config,
        "sample_weight",
        "output_col",
        "sample_weight",
    )
    required = [
        "date",
        "symbol",
        "timestamp",
        "decision_time",
        "decision_target_timestamp",
        "decision_lag_seconds",
        "status",
        "label",
        "valid_label",
        "gross_label",
        target_col,
        sample_weight_col,
        sample_weight_output_col,
        *_target_transform_source_columns(config),
        *PREDICTION_CONTEXT_COLUMNS,
        *_guard_condition_columns(config, "candidate_filter"),
        *_guard_condition_columns(config, "sample_weight"),
        *_guard_condition_columns(config, "guard_features"),
    ]

    selected = []
    selected.extend(explicit)
    selected.extend(required)
    selected.extend(feature_filters["include_columns"])
    selected.extend(_postopen_decision_source_columns(config))
    selected.extend(_postopen_v2_source_columns(config, available))
    selected.extend(_price_scale_source_columns(config))
    selected.extend(_cross_sectional_relative_source_columns(config, available))
    selected.extend(_historical_surprise_source_columns(config, available))
    value_transform = config_str(config, "features", "feature_value_transform", "")
    value_transform = value_transform.strip().lower().replace("-", "_")
    if value_transform.startswith(("mechanismized_v3", "mechanism_aware_v3")) or config_bool(
        config,
        "features",
        "include_historical_daily_activity_references",
        False,
    ):
        selected.extend(
            [
                config_str(config, "features", "historical_daily_activity_volume_col", "volume"),
                config_str(
                    config,
                    "features",
                    "historical_daily_activity_turnover_col",
                    "turnover",
                ),
            ]
        )
    if value_transform.startswith(("mechanismized", "mechanism_aware")):
        selected.extend(mechanismized_feature_value_reference_columns())
    selected.extend(
        _matching_existing_columns(
            available,
            prefixes=feature_filters["include_prefixes"],
            patterns=feature_filters["include_patterns"],
        )
    )
    return list(dict.fromkeys(_existing_columns(available, selected)))


def _labeled_pvc_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if files:
        return files
    files = sorted(path.rglob("*.csv")) + sorted(path.rglob("*.csv.gz"))
    if files:
        return files
    raise SystemExit(f"no parquet/csv files found under directory: {path}")


def _read_labeled_pvc_file(
    path: Path,
    *,
    columns: list[str] | None,
    filters: list[tuple[str, str, object]] | None,
) -> pd.DataFrame:
    file_columns = columns
    if columns is not None:
        available = frame_columns(path)
        file_columns = [column for column in columns if column in available]
    return read_frame(path, columns=file_columns, filters=filters)


def _downcast_labeled_pvc_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not config_bool(config, "data", "downcast_float32", False):
        return frame
    float_columns = frame.select_dtypes(include=["float64"]).columns
    if len(float_columns) == 0:
        return frame
    out = frame.copy()
    out[float_columns] = out[float_columns].astype("float32")
    return out


def _read_labeled_pvc_frame(
    path: Path,
    *,
    columns: list[str] | None,
    filters: list[tuple[str, str, object]] | None,
    config: dict,
) -> pd.DataFrame:
    files = _labeled_pvc_files(path)
    if len(files) == 1:
        return _downcast_labeled_pvc_frame(
            filter_labeled_frame(
                _read_labeled_pvc_file(
                    files[0],
                    columns=columns,
                    filters=filters,
                ),
                config,
            ),
            config,
        )

    parts = []
    for file in files:
        part = _read_labeled_pvc_file(
            file,
            columns=columns,
            filters=filters,
        )
        if part.empty:
            continue
        parts.append(part)
        print_mapping(
            "labeled_pvc_part",
            {
                "file": file.name,
                "rows": len(part),
                "columns": len(part.columns),
            },
        )
    if not parts:
        return pd.DataFrame()
    return _downcast_labeled_pvc_frame(
        filter_labeled_frame(pd.concat(parts, ignore_index=True), config),
        config,
    )


def _load_labeled_cache(path: Path, config: dict) -> pd.DataFrame:
    validate_cache_manifest(
        path,
        config,
        required=config_bool(config, "cache", "require_manifest", False),
    )
    labeled = read_frame(path)
    return ensure_timestamp_columns(standardize_columns(labeled))


def _write_labeled_cache(labeled: pd.DataFrame, path: Path) -> None:
    write_frame_atomic(labeled, path)


def _cache_ready_paths(cache_path: Path, config: dict, *, cache_write: bool) -> tuple[Path, ...]:
    if cache_write or config_bool(config, "cache", "require_manifest", False):
        return (cache_manifest_path(cache_path),)
    return ()


def _cache_artifacts_exist(cache_path: Path, ready_paths: tuple[Path, ...]) -> bool:
    return cache_path.exists() and all(path.exists() for path in ready_paths)


def _manifest_run_name(args: argparse.Namespace, config: dict, cache_path: Path) -> str:
    config_path = getattr(args, "config", None)
    if config_path:
        return run_id(config, config_path)
    return str(config.get("run", {}).get("id", cache_path.stem))


def _publish_labeled_cache(
    labeled: pd.DataFrame,
    path: Path,
    args: argparse.Namespace,
    config: dict,
) -> None:
    _write_labeled_cache(labeled, path)
    publish_cache_manifest(
        labeled,
        cache_path=path,
        config=config,
        run_name=_manifest_run_name(args, config, path),
        config_path=getattr(args, "config", "") or "",
    )


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
    daily_reference_enabled = config_bool(
        config,
        "daily_market_reference",
        "enabled",
        False,
    )
    daily_reference_table = config_str(
        config,
        "daily_market_reference",
        "table",
        DEFAULT_DAILY_MARKET_REFERENCE_TABLE,
    )
    daily_reference_lag_sessions = config_int(
        config,
        "daily_market_reference",
        "lag_sessions",
        1,
    )
    market_cap_unit_multiplier = config_float(
        config,
        "daily_market_reference",
        "market_cap_unit_multiplier",
        10_000.0,
    )
    share_unit_multiplier = config_float(
        config,
        "daily_market_reference",
        "share_unit_multiplier",
        10_000.0,
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
            "daily_market_reference": daily_reference_enabled,
            "daily_market_reference_table": (
                daily_reference_table if daily_reference_enabled else ""
            ),
            "daily_market_reference_lag_sessions": (
                daily_reference_lag_sessions if daily_reference_enabled else 0
            ),
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
        if daily_reference_enabled:
            reference = query_lagged_daily_market_reference(
                client,
                trading_day=trading_day,
                symbols=ticks["symbol"].dropna().astype(str).unique().tolist(),
                table=daily_reference_table,
                lag_sessions=daily_reference_lag_sessions,
                market_cap_unit_multiplier=market_cap_unit_multiplier,
                share_unit_multiplier=share_unit_multiplier,
            )
            reference_dates = (
                pd.to_datetime(
                    reference["market_cap_reference_date"], errors="coerce"
                )
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            print_mapping(
                f"daily_market_reference[{trading_day}]",
                {
                    "reference_date": reference_dates[0] if reference_dates else "",
                    "symbols_requested": int(ticks["symbol"].nunique()),
                    "symbols_returned": int(reference["symbol"].nunique()),
                    "symbols_with_total_market_cap": int(
                        reference["total_market_cap"].notna().sum()
                    ),
                },
            )
            ticks = attach_daily_market_reference(ticks, reference)
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


def load_clickhouse_labeled_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    cache_enabled = config_bool(config, "cache", "enabled", False)
    cache_path = resolve_cache_path(config) if cache_enabled else None
    cache_read = config_bool(config, "cache", "read", True)
    cache_write = config_bool(config, "cache", "write", True)

    ready_paths = (
        _cache_ready_paths(cache_path, config, cache_write=cache_write) if cache_path else ()
    )
    if cache_path and cache_read and _cache_artifacts_exist(cache_path, ready_paths):
        print_mapping("labeled_cache", {"action": "read", "path": str(cache_path)})
        return apply_candidate_filter_from_config(_load_labeled_cache(cache_path, config), config)
    if cache_path and cache_read and cache_path.exists() and ready_paths and not cache_write:
        print_mapping("labeled_cache", {"action": "read", "path": str(cache_path)})
        return apply_candidate_filter_from_config(_load_labeled_cache(cache_path, config), config)

    if not cache_path or not cache_write:
        return apply_candidate_filter_from_config(
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
        ready_paths=ready_paths,
    )
    if lock_status == "cache_ready":
        print_mapping(
            "labeled_cache",
            {"action": "read_after_wait", "path": str(cache_path)},
        )
        return apply_candidate_filter_from_config(
            _load_labeled_cache(cache_path, config),
            config,
        )
    if lock_status == "timeout":
        if cache_read and _cache_artifacts_exist(cache_path, ready_paths):
            print_mapping(
                "labeled_cache",
                {"action": "read_after_wait", "path": str(cache_path)},
            )
            return apply_candidate_filter_from_config(
                _load_labeled_cache(cache_path, config),
                config,
            )
        raise SystemExit(
            f"timed out waiting for labeled cache lock: {lock_path}; cache file was not created"
        )

    try:
        with _CacheLockHeartbeat(lock_path):
            if cache_read and _cache_artifacts_exist(cache_path, ready_paths):
                print_mapping(
                    "labeled_cache",
                    {"action": "read_after_lock", "path": str(cache_path)},
                )
                base_labeled = _load_labeled_cache(cache_path, config)
            else:
                _clear_cache_ready(lock_path)
                base_labeled = _build_clickhouse_labeled_frame(args, config)
                _publish_labeled_cache(base_labeled, cache_path, args, config)
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

    return apply_candidate_filter_from_config(base_labeled, config)


def load_labeled_pvc_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    path = _labeled_pvc_path(args, config)
    filters, filter_summary = _labeled_pvc_date_filters(args, config)
    columns = _labeled_pvc_read_columns(path, config)
    print_mapping(
        "labeled_pvc",
        {
            "action": "read",
            "path": str(path),
            "projected_columns": len(columns) if columns is not None else 0,
            **filter_summary,
        },
    )
    return _read_labeled_pvc_frame(
        path,
        columns=columns,
        filters=filters,
        config=config,
    )
