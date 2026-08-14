from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit import (
    cache_lock,
    training_pvc_columns,
    training_pvc_reuse,
    training_sources,
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
    config_int,
    config_list,
    config_str,
    config_value,
    run_id,
)
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.io import frame_columns, frame_files, read_frame, write_frame_atomic
from opening_strength_fit.model import feature_columns
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.schema import (
    DECISION_KEY_COLUMNS,
    ensure_timestamp_columns,
    normalize_decision_keys,
    standardize_columns,
)
from opening_strength_fit.training_labeled import (
    apply_candidate_filter_from_config,
    build_labeled_frame_from_config,
    filter_labeled_frame,
)
from opening_strength_fit.training_windows import (
    resolve_window_mode,
    rolling_monthly_date_bounds,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX, load_symbol_list

DATASET_JOIN_KEYS = DECISION_KEY_COLUMNS

_CacheLockHeartbeat = cache_lock.CacheLockHeartbeat
_acquire_cache_lock = cache_lock.acquire_cache_lock
_clear_cache_ready = cache_lock.clear_cache_ready
_mark_cache_ready = cache_lock.mark_cache_ready
_release_cache_lock = cache_lock.release_cache_lock
_labeled_pvc_read_columns = training_pvc_columns.labeled_pvc_read_columns
_attach_reused_labeled_features = training_pvc_reuse.attach_reused_labeled_features
_clickhouse_date_bounds = training_sources.clickhouse_date_bounds
_clickhouse_setting = training_sources.clickhouse_setting
_input_kind = training_sources.input_kind
load_training_frame = training_sources.load_training_frame
resolve_cache_path = training_sources.resolve_cache_path
resolve_data_source = training_sources.resolve_data_source


def _labeled_pvc_path(args: argparse.Namespace, config: dict) -> Path:
    raw = (
        args.labeled_input
        or os.environ.get("OPENING_STRENGTH_LABELED_PATH", "")
        or config_value(config, "data", "labeled_path", "")
        or config_value(config, "data", "feature_path", "")
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
            "No labeled PVC path supplied. Set [data].labeled_path or "
            "[data].feature_path + [data].label_path, "
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


def _labeled_pvc_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return frame_files(path)


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
    out = frame.copy(deep=False)
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
        labeled = _read_labeled_pvc_file(
            files[0],
            columns=columns,
            filters=filters,
        )
    else:
        parts = []
        for file in files:
            part = _read_labeled_pvc_file(file, columns=columns, filters=filters)
            if part.empty:
                continue
            parts.append(part)
            print_mapping(
                "labeled_pvc_part",
                {"file": file.name, "rows": len(part), "columns": len(part.columns)},
            )
        if not parts:
            return pd.DataFrame()
        labeled = pd.concat(parts, ignore_index=True)
    return _downcast_labeled_pvc_frame(filter_labeled_frame(labeled, config), config)


def _read_dataset_pvc_parts(
    path: Path,
    *,
    columns: list[str] | None,
    filters: list[tuple[str, str, object]] | None,
    kind: str,
) -> pd.DataFrame:
    parts = []
    for file in _labeled_pvc_files(path):
        part = _read_labeled_pvc_file(file, columns=columns, filters=filters)
        if part.empty:
            continue
        parts.append(part)
        print_mapping(
            f"{kind}_pvc_part",
            {"file": str(file), "rows": len(part), "columns": len(part.columns)},
        )
    if not parts:
        raise SystemExit(f"no filtered {kind} rows found under: {path}")
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def _normalize_dataset_join_keys(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    return normalize_decision_keys(
        standardize_columns(frame),
        drop_missing=False,
        require_unique=True,
        context=f"{kind} dataset",
    )


def _validate_model_ready_split_keys(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    chunk_rows: int = 1_000_000,
) -> None:
    """Validate every key without copying either wide model-ready dataset."""
    for kind, frame in (("feature", features), ("label", labels)):
        missing = [column for column in DATASET_JOIN_KEYS if column not in frame.columns]
        if missing:
            raise SystemExit(f"{kind} dataset is missing join keys {missing}")
    if len(features) != len(labels):
        raise SystemExit(
            "model-ready feature/label row count mismatch: "
            f"features={len(features)} labels={len(labels)}"
        )
    if not len(features):
        return

    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, len(features), chunk_rows):
        stop = min(start + chunk_rows, len(features))
        for column in DATASET_JOIN_KEYS:
            feature_keys = features[column].iloc[start:stop]
            label_keys = labels[column].iloc[start:stop]
            if column == "decision_target_timestamp":
                feature_values = pd.to_datetime(feature_keys, errors="coerce").to_numpy()
                label_values = pd.to_datetime(label_keys, errors="coerce").to_numpy()
            else:
                feature_values = feature_keys.astype(str).to_numpy()
                label_values = label_keys.astype(str).to_numpy()
            if not np.array_equal(feature_values, label_values):
                raise SystemExit(
                    "model-ready feature/label full key mismatch in "
                    f"{column!r} at rows {start}:{stop}; rebuild the published "
                    "datasets or disable data.trusted_model_ready_split for a key join"
                )


def _read_split_feature_label_pvc_frame(
    feature_path: Path,
    label_path: Path,
    *,
    filters: list[tuple[str, str, object]] | None,
    config: dict,
) -> pd.DataFrame:
    trusted_model_ready = config_bool(
        config,
        "data",
        "trusted_model_ready_split",
        False,
    )
    projected_feature_columns = _labeled_pvc_read_columns(feature_path, config)
    features = _read_dataset_pvc_parts(
        feature_path,
        columns=projected_feature_columns,
        filters=filters,
        kind="feature",
    )
    if not trusted_model_ready:
        features = _normalize_dataset_join_keys(features, kind="feature")
    configured_label_columns = config_list(
        config,
        "data",
        "label_columns",
        ["label_short", "label_next_close", "target_label"],
    )
    label_columns = list(dict.fromkeys([*DATASET_JOIN_KEYS, *configured_label_columns]))
    labels = _read_dataset_pvc_parts(
        label_path,
        columns=label_columns,
        filters=filters,
        kind="label",
    )
    if trusted_model_ready:
        _validate_model_ready_split_keys(features, labels)
    else:
        labels = _normalize_dataset_join_keys(labels, kind="label")
    missing_labels = [column for column in configured_label_columns if column not in labels.columns]
    if missing_labels:
        raise SystemExit(f"label dataset is missing columns {missing_labels}")
    overlapping_labels = sorted(set(configured_label_columns).intersection(features.columns))
    if overlapping_labels:
        raise SystemExit(
            f"feature dataset unexpectedly contains label columns {overlapping_labels}"
        )

    keys_already_aligned = trusted_model_ready or (
        len(features) == len(labels)
        and all(features[column].equals(labels[column]) for column in DATASET_JOIN_KEYS)
    )
    if keys_already_aligned:
        # Published feature/label shards preserve the same row order. Keep the 350-column
        # feature blocks shallow and attach only the three narrow label arrays; a full
        # merge would transiently duplicate the wide frame for no benefit.
        merged = features if trusted_model_ready else features.copy(deep=False)
        for column in configured_label_columns:
            merged[column] = labels[column].to_numpy(copy=False)
        feature_only = 0
        label_only = 0
        alignment = "trusted_full_key_order" if trusted_model_ready else "same_order"
    else:
        merged = features.merge(
            labels.loc[:, label_columns],
            on=list(DATASET_JOIN_KEYS),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        coverage = merged["_merge"].value_counts().to_dict()
        feature_only = int(coverage.get("left_only", 0))
        label_only = int(coverage.get("right_only", 0))
        if feature_only or label_only:
            raise SystemExit(
                "feature/label key coverage mismatch: "
                f"feature_only={feature_only} label_only={label_only}"
            )
        merged = merged.drop(columns="_merge")
        alignment = "key_join"

    short_column = config_str(config, "data", "short_label_column", "label_short")
    target_column = config_str(config, "model", "target_col", "target_label")
    if short_column not in merged.columns or target_column not in merged.columns:
        raise SystemExit(
            f"joined dataset requires short={short_column!r} and target={target_column!r}"
        )
    short = pd.to_numeric(merged[short_column], errors="coerce")
    target = pd.to_numeric(merged[target_column], errors="coerce")
    if not trusted_model_ready:
        if "timestamp" not in merged.columns:
            merged["timestamp"] = merged["decision_target_timestamp"]
        if "decision_time" not in merged.columns:
            merged["decision_time"] = merged["decision_target_timestamp"].dt.strftime("%H:%M:%S")
        if "decision_lag_seconds" not in merged.columns:
            merged["decision_lag_seconds"] = 0.0
    merged["label"] = short
    merged["gross_label"] = short
    merged["valid_label"] = short.notna() & target.notna()
    print_mapping(
        "feature_label_join",
        {
            "feature_path": str(feature_path),
            "label_path": str(label_path),
            "rows": len(merged),
            "valid_target_rows": int(merged["valid_label"].sum()),
            "feature_only": feature_only,
            "label_only": label_only,
            "alignment": alignment,
        },
    )
    result = (
        _downcast_labeled_pvc_frame(merged, config)
        if trusted_model_ready
        else _downcast_labeled_pvc_frame(filter_labeled_frame(merged, config), config)
    )
    expected_features = config_int(config, "data", "expected_feature_count", 0)
    if expected_features:
        selected_features = feature_columns(
            result,
            None,
            **feature_filters_from_config(config),
        )
        if len(selected_features) != expected_features:
            raise SystemExit(
                "joined training feature count mismatch: "
                f"expected={expected_features} actual={len(selected_features)}"
            )
    return result


def _load_labeled_cache(path: Path, config: dict) -> pd.DataFrame:
    validate_cache_manifest(
        path,
        config,
        required=config_bool(config, "cache", "require_manifest", False),
    )
    return ensure_timestamp_columns(standardize_columns(read_frame(path)))


def _load_filtered_labeled_cache(
    path: Path,
    config: dict,
    *,
    action: str,
) -> pd.DataFrame:
    print_mapping("labeled_cache", {"action": action, "path": str(path)})
    return apply_candidate_filter_from_config(_load_labeled_cache(path, config), config)


def _cache_artifacts_exist(cache_path: Path, ready_paths: tuple[Path, ...]) -> bool:
    return cache_path.exists() and all(path.exists() for path in ready_paths)


def _build_clickhouse_labeled_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    def source_setting(name: str, default, *, env_name: str | None = None):
        return _clickhouse_setting(
            args,
            config,
            f"clickhouse_{name}",
            name,
            env_name or f"CLICKHOUSE_{name.upper()}",
            default,
        )

    def offset_setting(name: str, default: int) -> int:
        cli_value = getattr(args, name)
        return int(
            cli_value
            if cli_value is not None
            else config_value(config, "clickhouse", name, default)
        )

    host = str(source_setting("host", DEFAULT_CLICKHOUSE_TICK_HOST))
    port = int(source_setting("port", DEFAULT_CLICKHOUSE_TICK_PORT))
    user = source_setting("user", None)
    password = source_setting("password", None)
    table = str(
        source_setting("table", DEFAULT_CLICKHOUSE_TICK_TABLE, env_name="CLICKHOUSE_TICK_TABLE")
    )
    start_offset_us = offset_setting("start_offset_us", DEFAULT_TICK_START_OFFSET_US)
    end_offset_us = offset_setting("end_offset_us", DEFAULT_TICK_END_OFFSET_US)
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
    reference_section = "daily_market_reference"
    daily_reference_enabled = config_bool(config, reference_section, "enabled", False)
    daily_reference_table = config_str(
        config, reference_section, "table", DEFAULT_DAILY_MARKET_REFERENCE_TABLE
    )
    daily_reference_lag_sessions = config_int(config, reference_section, "lag_sessions", 1)
    market_cap_unit_multiplier = config_float(
        config, reference_section, "market_cap_unit_multiplier", 10_000.0
    )
    share_unit_multiplier = config_float(
        config, reference_section, "share_unit_multiplier", 10_000.0
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
                pd.to_datetime(reference["market_cap_reference_date"], errors="coerce")
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
    return _attach_reused_labeled_features(
        pd.concat(labeled_parts, ignore_index=True),
        config,
    )


def load_clickhouse_labeled_frame(
    args: argparse.Namespace,
    config: dict,
) -> pd.DataFrame:
    cache_enabled = config_bool(config, "cache", "enabled", False)
    cache_path = resolve_cache_path(config) if cache_enabled else None
    cache_read = config_bool(config, "cache", "read", True)
    cache_write = config_bool(config, "cache", "write", True)

    ready_paths = (
        (cache_manifest_path(cache_path),)
        if cache_path and (cache_write or config_bool(config, "cache", "require_manifest", False))
        else ()
    )
    cache_is_readable = cache_path and (
        _cache_artifacts_exist(cache_path, ready_paths)
        or (cache_path.exists() and bool(ready_paths) and not cache_write)
    )
    if cache_read and cache_is_readable:
        return _load_filtered_labeled_cache(cache_path, config, action="read")

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
        return _load_filtered_labeled_cache(cache_path, config, action="read_after_wait")
    if lock_status == "timeout":
        if cache_read and _cache_artifacts_exist(cache_path, ready_paths):
            return _load_filtered_labeled_cache(cache_path, config, action="read_after_wait")
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
                config_path = getattr(args, "config", "") or ""
                write_frame_atomic(base_labeled, cache_path)
                publish_cache_manifest(
                    base_labeled,
                    cache_path=cache_path,
                    config=config,
                    run_name=(
                        run_id(config, config_path)
                        if config_path
                        else str(config.get("run", {}).get("id", cache_path.stem))
                    ),
                    config_path=config_path,
                )
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
    feature_path_raw = str(
        getattr(args, "feature_input", None) or config_str(config, "data", "feature_path", "")
    ).strip()
    label_path_raw = str(
        getattr(args, "label_input", None) or config_str(config, "data", "label_path", "")
    ).strip()
    if not getattr(args, "labeled_input", None) and (feature_path_raw or label_path_raw):
        if not feature_path_raw or not label_path_raw:
            raise SystemExit("[data] requires both feature_path and label_path")
        filters, filter_summary = _labeled_pvc_date_filters(args, config)
        print_mapping(
            "split_training_dataset",
            {
                "feature_path": feature_path_raw,
                "label_path": label_path_raw,
                **filter_summary,
            },
        )
        return _read_split_feature_label_pvc_frame(
            Path(feature_path_raw),
            Path(label_path_raw),
            filters=filters,
            config=config,
        )
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
