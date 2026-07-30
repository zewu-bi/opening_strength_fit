from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from opening_strength_fit.schema import PRICE_LEVELS

NON_FEATURE_COLUMNS = {
    "date",
    "year",
    "month",
    "minute_bucket",
    "time",
    "symbol",
    "timestamp",
    "decision_time",
    "decision_target_timestamp",
    "decision_source_timestamp",
    "decision_state_age_seconds",
    "decision_lag_seconds",
    "entry_timestamp",
    "entry_source_timestamp",
    "entry_state_age_seconds",
    "entry_delay_ticks",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "cross_section_ready_timestamp",
    "entry_after_cross_section_ready",
    "sell_start_target_timestamp",
    "sell_start_source_timestamp",
    "sell_start_state_age_seconds",
    "sell_end_target_timestamp",
    "sell_end_source_timestamp",
    "sell_end_state_age_seconds",
    "market_cap_reference_date",
    "market_cap_reference_lag_sessions",
    "entry_lag_seconds",
    "entry_status",
    "label",
    "label_raw",
    "label_xs_mean",
    "label_xs_std",
    "label_xs_count",
    "label_xs_rank_pct",
    "target_label",
    "gross_label",
    "valid_label",
    "buy_price",
    "sell_vwap",
    "sell_volume",
    "sell_turnover",
    "alpha_return_next_close",
    "candidate_alpha_score",
    "candidate_alpha_rank",
    "alpha_conditioning_prediction",
    "prediction",
    "sample_weight",
    "risk_sample_weight",
}
LEAKY_PREFIXES = (
    "label_xs_",
    "target_",
    "timestamp_sell_",
    "volume_sell_",
    "turnover_sell_",
    "timestamp_entry",
    "entry_ask_price_",
    "entry_ask_volume_",
)

ENTRY_ASK_CONTEXT_COLUMNS = tuple(
    column
    for level in PRICE_LEVELS
    for column in (f"entry_ask_price_{level}", f"entry_ask_volume_{level}")
)

PREDICTION_CONTEXT_COLUMNS = (
    "status",
    "decision_source_timestamp",
    "decision_state_age_seconds",
    "entry_status",
    "entry_timestamp",
    "entry_source_timestamp",
    "entry_state_age_seconds",
    "entry_delay_ticks",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "cross_section_ready_timestamp",
    "entry_after_cross_section_ready",
    "sell_start_target_timestamp",
    "sell_start_source_timestamp",
    "sell_start_state_age_seconds",
    "sell_end_target_timestamp",
    "sell_end_source_timestamp",
    "sell_end_state_age_seconds",
    "market_cap_reference_date",
    "market_cap_reference_lag_sessions",
    "gross_label",
    "buy_price",
    "volume",
    "turnover",
    "ask_price_1",
    "bid_price_1",
    "ask_volume_1",
    "bid_volume_1",
    "mid_price",
    "spread_bps",
    "ask1_to_limit_up_bps",
    "ask_depth_10",
    "bid_depth_10",
    "depth_imbalance_1",
    "depth_imbalance_10",
    "volume_diff_1t",
    "volume_diff_3t",
    "volume_diff_10t",
    "volume_diff_30t",
    "turnover_diff_1t",
    "turnover_diff_3t",
    "turnover_diff_10t",
    "turnover_diff_30t",
    "preopen_volume",
    "preopen_turnover",
    "return_10t",
    "return_30t",
    "preopen_return_vs_prev_close",
    *ENTRY_ASK_CONTEXT_COLUMNS,
)


@dataclass
class RidgePredictionModel:
    features: list[str]
    alpha: float
    pipeline: Pipeline
    model_name: str = "ridge"
    target_col: str = "label"
    source_features: list[str] | None = None
    feature_value_transform: str = "none"
    feature_value_transform_output: str = "replace"
    feature_value_transform_prefix: str = "mech_v3_"
    feature_value_transform_group_cols: tuple[str, ...] = (
        "date",
        "decision_target_timestamp",
    )
    feature_value_transform_rank_method: str = "average"
    feature_value_transform_tick_size: float = 0.01


@dataclass
class EnsemblePredictionModel:
    features: list[str]
    alpha: float
    models: list[RidgePredictionModel]
    weights: list[float]
    combine_mode: str = "rank"
    rank_group_cols: tuple[str, ...] = ("date", "decision_target_timestamp")
    model_name: str = "ensemble"
    target_col: str = "label"


@dataclass
class ClockSegmentPredictionModel:
    features: list[str]
    segment_models: list[tuple[str, tuple[str, ...], RidgePredictionModel]]
    fallback_model: RidgePredictionModel | None = None
    model_name: str = "clock_segment"
    target_col: str = "label"


@dataclass
class TorchMLPPredictionModel:
    features: list[str]
    module: Any
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    device: str
    batch_size: int
    feature_standardization: str = "global_zscore"
    standardization_group_col: str = "symbol"
    standardization_group_keys: np.ndarray | None = None
    standardization_group_mean: np.ndarray | None = None
    standardization_group_scale: np.ndarray | None = None
    feature_value_transform: str = "none"
    feature_value_transform_group_cols: tuple[str, ...] = ("date", "decision_target_timestamp")
    feature_value_transform_rank_method: str = "average"
    feature_value_transform_tick_size: float = 0.01
    diagnostics: dict[str, Any] | None = None
    model_name: str = "torch_mlp"
    target_col: str = "label"
