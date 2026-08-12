from __future__ import annotations

import re
from pathlib import Path

from opening_strength_fit.config import (
    config_bool,
    config_float_mapping,
    config_list,
    config_str,
)
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.features import mechanismized_feature_value_reference_columns
from opening_strength_fit.features_postopen import POSTOPEN_DYNAMIC_COLUMNS
from opening_strength_fit.io import frame_columns
from opening_strength_fit.model import PREDICTION_CONTEXT_COLUMNS


def _matching_existing_columns(
    available: set[str],
    *,
    prefixes: tuple[str, ...] = (),
    patterns: tuple[str, ...] = (),
) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        column
        for column in sorted(available)
        if (prefixes and column.startswith(prefixes))
        or (compiled and any(pattern.search(column) for pattern in compiled))
    ]


def _postopen_decision_source_columns(config: dict) -> tuple[str, ...]:
    if not config_bool(config, "features", "include_postopen_decision", False):
        return ()
    return (
        *POSTOPEN_DYNAMIC_COLUMNS,
        "volume_diff_1t",
        *(f"ask_gap_{level}_bps" for level in range(2, 11)),
        *(f"bid_gap_{level}_bps" for level in range(2, 11)),
    )


def _postopen_v2_source_columns(config: dict, available: set[str]) -> list[str]:
    if not config_bool(config, "features", "include_postopen_v2", False):
        return []
    return [
        "ask_price_1",
        "bid_price_1",
        "ask_volume_1",
        "bid_volume_1",
        "mid_price",
        "spread_bps",
        "volume",
        "turnover",
        *_matching_existing_columns(
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
        ),
    ]


def _optional_feature_sources(
    config: dict,
    available: set[str],
    enabled_key: str,
    prefix: str,
) -> list[str]:
    if not config_bool(config, "features", enabled_key, False):
        return []
    return [
        *config_list(config, "features", f"{prefix}_group_cols", []),
        *config_list(config, "features", f"{prefix}_columns", []),
        *_matching_existing_columns(
            available,
            prefixes=tuple(config_list(config, "features", f"{prefix}_prefixes", [])),
            patterns=tuple(config_list(config, "features", f"{prefix}_regexes", [])),
        ),
    ]


def _multi_denominator_source_columns(config: dict) -> list[str]:
    if not config_bool(config, "features", "include_multi_denominator_features", False):
        return []
    source = [
        "volume",
        "turnover",
        "float_shares",
        "float_market_cap",
        *config_list(config, "features", "multi_denominator_turnover_columns", []),
        *config_list(config, "features", "multi_denominator_volume_columns", []),
        *config_list(config, "features", "multi_denominator_depth_columns", []),
        *config_list(
            config,
            "features",
            "multi_denominator_cross_sectional_median_columns",
            [],
        ),
        *config_list(
            config,
            "features",
            "multi_denominator_cross_sectional_group_cols",
            ["date", "decision_target_timestamp"],
        ),
    ]
    return list(dict.fromkeys(source))


def _price_scale_source_columns(config: dict) -> list[str]:
    if not config_bool(config, "features", "include_price_scale_features", False):
        return []
    return [
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
        *(f"ask_price_{level}" for level in range(2, 11)),
        *(f"bid_price_{level}" for level in range(2, 11)),
        *config_list(config, "features", "price_scale_interaction_columns", []),
    ]


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
        *map(str, config_float_mapping(config, section, "min")),
        *map(str, config_float_mapping(config, section, "max")),
        *map(str, config_float_mapping(config, section, "rank_min")),
        *map(str, config_float_mapping(config, section, "rank_max")),
        *config_list(config, section, "rank_columns", []),
        *config_list(config, section, "rank_group_cols", []),
    )


def labeled_pvc_read_columns(path: Path, config: dict) -> list[str] | None:
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
    for enabled_key, prefix in (
        ("include_cross_sectional_relative", "cross_sectional_relative"),
        ("include_historical_same_minute_surprise", "historical_surprise"),
    ):
        selected.extend(_optional_feature_sources(config, available, enabled_key, prefix))
    selected.extend(_multi_denominator_source_columns(config))
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
    return list(dict.fromkeys(column for column in selected if column in available))
