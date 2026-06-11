from __future__ import annotations

import re

import numpy as np
import pandas as pd

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
    config_str,
    config_value,
)
from opening_strength_fit.dataset import build_labeled_feature_frame
from opening_strength_fit.features import (
    add_cross_sectional_relative_features,
    add_historical_same_minute_surprise_features,
    add_path_shape_confirmation_features,
    add_postopen_decision_features,
    add_postopen_v2_decision_features,
    add_tree_sequence_features,
)
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, parse_clock_times
from opening_strength_fit.schema import (
    ensure_timestamp_columns,
    normalize_clock_time,
    standardize_columns,
)
from opening_strength_fit.universe import (
    DEFAULT_A_SHARE_SYMBOL_REGEX,
    filter_symbol_universe,
    load_symbol_list,
)


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


def _cross_sectional_relative_columns_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> list[str]:
    explicit = set(config_list(config, "features", "cross_sectional_relative_columns", []))
    prefixes = tuple(config_list(config, "features", "cross_sectional_relative_prefixes", []))
    patterns = [
        re.compile(pattern)
        for pattern in config_list(config, "features", "cross_sectional_relative_regexes", [])
    ]
    columns: list[str] = []
    for column in labeled.columns:
        name = str(column)
        if name in explicit:
            columns.append(name)
            continue
        if prefixes and name.startswith(prefixes):
            columns.append(name)
            continue
        if patterns and any(pattern.search(name) for pattern in patterns):
            columns.append(name)
    return list(dict.fromkeys(columns))


def _apply_cross_sectional_relative_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    relative_columns = _cross_sectional_relative_columns_from_config(labeled, config)
    if not relative_columns:
        raise SystemExit(
            "features.include_cross_sectional_relative=true but no "
            "cross_sectional_relative_columns/prefixes/regexes matched"
        )
    return add_cross_sectional_relative_features(
        labeled,
        columns=tuple(relative_columns),
        group_cols=tuple(
            config_list(
                config,
                "features",
                "cross_sectional_relative_group_cols",
                ["date", "decision_target_timestamp"],
            )
        ),
        modes=tuple(
            config_list(
                config,
                "features",
                "cross_sectional_relative_modes",
                ["zscore", "rank_centered"],
            )
        ),
        prefix=config_str(config, "features", "cross_sectional_relative_prefix", "xs_rel_"),
        rank_method=config_str(
            config,
            "features",
            "cross_sectional_relative_rank_method",
            "average",
        ),
    )


def _apply_feature_transforms_from_config(
    labeled: pd.DataFrame,
    config: dict,
    *,
    include_cross_sectional_relative: bool = True,
    drop_features: bool = True,
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
    if config_bool(config, "features", "include_path_shape_confirmation", False):
        labeled = add_path_shape_confirmation_features(
            labeled,
            windows=config_int_tuple(
                config,
                "features",
                "path_shape_windows",
                (2, 3, 5),
            ),
            prefix=config_str(
                config,
                "features",
                "path_shape_prefix",
                "path_shape_",
            ),
        )
    if config_bool(config, "features", "include_tree_sequence", False):
        labeled = add_tree_sequence_features(
            labeled,
            columns=tuple(config_list(config, "features", "tree_sequence_columns", [])),
            include_prefixes=tuple(
                config_list(config, "features", "tree_sequence_prefixes", [])
            ),
            include_patterns=tuple(
                config_list(config, "features", "tree_sequence_regexes", [])
            ),
            windows=config_int_tuple(
                config,
                "features",
                "tree_sequence_windows",
                (2, 3, 5),
            ),
            landmarks=tuple(
                config_list(
                    config,
                    "features",
                    "tree_sequence_landmarks",
                    ["09:31:00", "09:33:00", "09:35:00", "09:40:00"],
                )
            ),
            prefix=config_str(config, "features", "tree_sequence_prefix", "tree_seq_"),
        )
    if include_cross_sectional_relative and config_bool(
        config,
        "features",
        "include_cross_sectional_relative",
        False,
    ):
        labeled = _apply_cross_sectional_relative_from_config(labeled, config)
    if drop_features:
        labeled = _drop_features_from_config(labeled, config)
    return labeled


def _historical_surprise_columns_from_config(labeled: pd.DataFrame, config: dict) -> list[str]:
    explicit = set(config_list(config, "features", "historical_surprise_columns", []))
    prefixes = tuple(config_list(config, "features", "historical_surprise_prefixes", []))
    patterns = [
        re.compile(pattern)
        for pattern in config_list(config, "features", "historical_surprise_regexes", [])
    ]
    columns: list[str] = []
    for column in labeled.columns:
        name = str(column)
        if name in explicit:
            columns.append(name)
            continue
        if prefixes and name.startswith(prefixes):
            columns.append(name)
            continue
        if patterns and any(pattern.search(name) for pattern in patterns):
            columns.append(name)
    return list(dict.fromkeys(columns))


def _apply_post_sample_feature_transforms_from_config(
    labeled: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if config_bool(config, "features", "include_historical_same_minute_surprise", False):
        columns = _historical_surprise_columns_from_config(labeled, config)
        if not columns:
            raise SystemExit(
                "features.include_historical_same_minute_surprise=true but no "
                "historical_surprise_columns/prefixes/regexes matched"
            )
        labeled = add_historical_same_minute_surprise_features(
            labeled,
            columns=tuple(columns),
            windows=config_int_tuple(
                config,
                "features",
                "historical_surprise_windows",
                (20, 60),
            ),
            min_periods=config_int(
                config,
                "features",
                "historical_surprise_min_periods",
                10,
            ),
            modes=tuple(
                config_list(
                    config,
                    "features",
                    "historical_surprise_modes",
                    ["zscore", "ratio"],
                )
            ),
            prefix=config_str(
                config,
                "features",
                "historical_surprise_prefix",
                "hist_surprise_",
            ),
        )
    return labeled


def apply_target_transform_from_config(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not config_bool(config, "target_transform", "enabled", False):
        return frame
    mode = config_str(config, "target_transform", "mode", "rank_centered").strip().lower()
    if mode not in {"rank_centered", "rank_pct"}:
        raise SystemExit("target_transform.mode must be rank_centered or rank_pct")
    source_col = config_str(config, "target_transform", "source_col", "target_label")
    output_col = config_str(
        config,
        "target_transform",
        "output_col",
        f"{source_col}_{mode}",
    )
    if source_col not in frame.columns:
        raise SystemExit(f"target_transform source column not found: {source_col}")
    group_cols = [
        column
        for column in config_list(
            config,
            "target_transform",
            "group_cols",
            ["date", "decision_target_timestamp"],
        )
        if column in frame.columns
    ]
    if not group_cols:
        raise SystemExit("target_transform needs at least one available group column")

    out = frame.copy()
    values = pd.to_numeric(out[source_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.notna()
    if "valid_label" in out.columns:
        valid &= out["valid_label"].fillna(False).astype(bool)
    grouped = values.where(valid).groupby([out[column] for column in group_cols], sort=False)
    rank_pct = grouped.rank(method=config_str(config, "target_transform", "rank_method", "average"), pct=True)
    if mode == "rank_pct":
        target = rank_pct
    else:
        count = grouped.transform("count")
        rank_mean = (count + 1.0) / (2.0 * count)
        target = rank_pct - rank_mean
    out[output_col] = target.where(valid)
    if "valid_label" in out.columns:
        out["valid_label"] = out["valid_label"].fillna(False).astype(bool) & out[output_col].notna()
    return out


def apply_candidate_filter_from_config(
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


def apply_sample_weight_from_config(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if not config_bool(config, "sample_weight", "enabled", False):
        return frame
    output_col = config_str(config, "sample_weight", "output_col", "sample_weight")
    mode = config_str(config, "sample_weight", "mode", "candidate_mask").strip().lower()
    if mode in {"date_linear", "linear_date", "recent_date"}:
        if "date" not in frame.columns:
            raise SystemExit("sample_weight.mode='date_linear' requires a date column")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        configured_start = config_str(config, "sample_weight", "start_date", "")
        configured_end = config_str(config, "sample_weight", "end_date", "")
        start = pd.Timestamp(configured_start) if configured_start else dates.min()
        end = pd.Timestamp(configured_end) if configured_end else dates.max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            raise SystemExit("sample_weight.mode='date_linear' needs start_date < end_date")
        min_weight = config_float(config, "sample_weight", "min_weight", 0.75)
        max_weight = config_float(config, "sample_weight", "max_weight", 1.25)
        span = (end - start).days
        pct = ((dates - start).dt.days / span).clip(lower=0.0, upper=1.0)
        out = frame.copy()
        out[output_col] = (min_weight + pct * (max_weight - min_weight)).fillna(1.0)
        return out

    if mode not in {"candidate_mask", "guard", "rank_mask"}:
        raise SystemExit(
            "sample_weight.mode must be candidate_mask/rank_mask/guard or date_linear"
        )
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


def apply_guard_features_from_config(
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
        out[f"{prefix}{column}_rank_pct"] = values.groupby([out[col] for col in group_cols]).rank(
            method=rank_method, pct=True
        )

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
        series.astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False).fillna("")
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
        clock = pd.to_datetime(labeled[time_col], errors="coerce").dt.strftime("%H:%M:%S")
    mask = clock.isin(set(decision_times))

    max_lag = config_value(config, "sample", "decision_max_lag_seconds", None)
    if max_lag not in (None, "") and "decision_lag_seconds" in labeled.columns:
        lag = pd.to_numeric(labeled["decision_lag_seconds"], errors="coerce")
        mask &= lag.ge(0.0) & lag.le(float(max_lag))

    return labeled.loc[mask].copy()


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
    labeled = _apply_post_sample_feature_transforms_from_config(labeled, config)
    labeled = apply_target_transform_from_config(labeled, config)
    if apply_candidate_filter:
        return apply_candidate_filter_from_config(labeled, config)
    return labeled


def looks_labeled(frame: pd.DataFrame) -> bool:
    return {"date", "symbol", "timestamp", "label"}.issubset(frame.columns)


def filter_labeled_frame(labeled: pd.DataFrame, config: dict) -> pd.DataFrame:
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
    labeled = _apply_feature_transforms_from_config(
        labeled,
        config,
        include_cross_sectional_relative=False,
        drop_features=False,
    )
    labeled = _filter_labeled_sample_from_config(labeled, config)
    labeled = _apply_post_sample_feature_transforms_from_config(labeled, config)
    if config_bool(config, "features", "include_cross_sectional_relative", False):
        labeled = _apply_cross_sectional_relative_from_config(labeled, config)
    labeled = _drop_features_from_config(labeled, config)
    labeled = apply_target_transform_from_config(labeled, config)
    return apply_candidate_filter_from_config(labeled, config)
