from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
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
    config_int,
    config_list,
    config_str,
    config_value,
    run_id,
)
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic
from opening_strength_fit.model import feature_columns
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.schema import (
    DECISION_KEY_COLUMNS,
    ensure_timestamp_columns,
    standardize_columns,
)
from opening_strength_fit.training_labeled import (
    apply_candidate_filter_from_config,
    build_labeled_frame_from_config,
    filter_labeled_frame,
)
from opening_strength_fit.training_pvc_columns import (
    labeled_pvc_read_columns as _labeled_pvc_read_columns,
)
from opening_strength_fit.training_pvc_reuse import (
    attach_reused_labeled_features as _attach_reused_labeled_features,
)
from opening_strength_fit.training_sources import (
    clickhouse_date_bounds as _clickhouse_date_bounds,
)
from opening_strength_fit.training_sources import (
    clickhouse_setting as _clickhouse_setting,
)
from opening_strength_fit.training_sources import (
    input_kind as _input_kind,
)
from opening_strength_fit.training_sources import (
    load_training_frame,
    resolve_cache_path,
    resolve_data_source,
)
from opening_strength_fit.training_windows import (
    resolve_window_mode,
    rolling_monthly_date_bounds,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX, load_symbol_list

DATASET_JOIN_KEYS = DECISION_KEY_COLUMNS

__all__ = [
    "_attach_reused_labeled_features",
    "_clickhouse_date_bounds",
    "_clickhouse_setting",
    "_input_kind",
    "_labeled_pvc_files",
    "_labeled_pvc_path",
    "_labeled_pvc_read_columns",
    "load_clickhouse_labeled_frame",
    "load_labeled_pvc_frame",
    "load_training_frame",
    "resolve_cache_path",
    "resolve_data_source",
]


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
    files = _unique_labeled_pvc_files(list(path.rglob("*.parquet")))
    if files:
        return files
    files = _unique_labeled_pvc_files(list(path.rglob("*.csv")) + list(path.rglob("*.csv.gz")))
    if files:
        return files
    raise SystemExit(f"no parquet/csv files found under directory: {path}")


def _unique_labeled_pvc_files(files: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for file in sorted(files, key=lambda item: (item.is_symlink(), str(item))):
        target = file.resolve()
        if target in seen:
            continue
        seen.add(target)
        unique.append(file)
    return unique


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
    out = standardize_columns(frame)
    missing = [column for column in DATASET_JOIN_KEYS if column not in out.columns]
    if missing:
        raise SystemExit(f"{kind} dataset is missing join keys {missing}")
    out = out.copy(deep=False)
    parsed_date = pd.to_datetime(out["date"], errors="coerce")
    parsed_timestamp = pd.to_datetime(out["decision_target_timestamp"], errors="coerce")
    out["date"] = parsed_date.dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype("string")
    out["decision_target_timestamp"] = parsed_timestamp
    null_keys = out.loc[:, list(DATASET_JOIN_KEYS)].isna().any(axis=1)
    if null_keys.any():
        raise SystemExit(f"{kind} dataset has {int(null_keys.sum())} null-key rows")
    duplicate = out.duplicated(list(DATASET_JOIN_KEYS), keep=False)
    if duplicate.any():
        raise SystemExit(f"{kind} dataset has {int(duplicate.sum())} duplicate-key rows")
    return out


def _validate_model_ready_split_keys(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    sample_rows: int = 4096,
) -> None:
    """Cheaply validate the row-order contract of published model-ready datasets."""
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

    count = min(max(2, int(sample_rows)), len(features))
    positions = np.unique(np.linspace(0, len(features) - 1, num=count, dtype=np.int64))
    for column in DATASET_JOIN_KEYS:
        feature_keys = features[column].iloc[positions].reset_index(drop=True)
        label_keys = labels[column].iloc[positions].reset_index(drop=True)
        if column == "decision_target_timestamp":
            feature_values = pd.to_datetime(feature_keys, errors="coerce").to_numpy()
            label_values = pd.to_datetime(label_keys, errors="coerce").to_numpy()
        else:
            feature_values = feature_keys.astype(str).to_numpy()
            label_values = label_keys.astype(str).to_numpy()
        if not np.array_equal(feature_values, label_values):
            raise SystemExit(
                "model-ready feature/label sampled key mismatch in "
                f"{column!r}; rebuild the published datasets or disable "
                "data.trusted_model_ready_split for a full key join"
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
        alignment = "trusted_sampled_order" if trusted_model_ready else "same_order"
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
