from __future__ import annotations

import tomllib
from pathlib import Path

from opening_strength_fit.feature_config import feature_filters_from_config

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
MODEL_CONFIG = (
    RUNS / "nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
)
MECH_V2_CONFIG = (
    RUNS
    / "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse_v1.toml"
)
MECH_V3_CONFIG = (
    RUNS
    / "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v3_histavg_activity_gelu_mse_v1.toml"
)
BASE_ROOT = "opening_2019_2025_delay2_base_labeled_v4_auction_fresh_mcap_lag1"
MIXED_ROOT = "opening_2019_2025_delay2_mixed_w030_labeled_v3_auction_fresh_mcap_lag1"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_annual_cache_configs_are_isolated_and_causal() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay2_{year}_auction_fresh_cache_v1.toml")

        assert BASE_ROOT in config["cache"]["path"]
        assert "opening_13y_201301_202512_delay2_base_labeled_v2" not in config["cache"]["path"]
        assert config["features"]["preopen_price_mode"] == "indicative_quote_v2"
        assert config["labels"]["entry_max_gap_seconds"] == 5
        assert config["labels"]["max_future_gap_seconds"] == 5
        assert config["labels"]["require_entry_after_cross_section_ready"] is True
        assert config["daily_market_reference"] == {
            "enabled": True,
            "table": "stock.daily_bar_jy",
            "lag_sessions": 1,
            "market_cap_unit_multiplier": 10_000.0,
            "share_unit_multiplier": 10_000.0,
        }


def test_target_configs_only_write_new_mixed_cache() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay2_{year}_auction_fresh_mixed_w030_target_v1.toml")

        assert BASE_ROOT in config["target_cache"]["input_path"]
        assert MIXED_ROOT in config["target_cache"]["output_path"]
        assert config["target_cache"]["long_label_weight"] == 0.30


def test_pruned_model_uses_minimal_price_basis_and_matched_queue_horizon() -> None:
    config = _load(MODEL_CONFIG)
    filters = feature_filters_from_config(config)
    dropped = set(filters["drop_columns"])
    included = set(filters["include_columns"])

    assert {"ask_price_1", "bid_price_1", "spread_abs"} <= dropped
    assert {"preopen_price_min", "preopen_price_max"} <= dropped
    assert {
        "postopen_v2_ask1_queue_replenish_vs_trade_1m",
        "postopen_v2_bid1_queue_replenish_vs_trade_1m",
    } <= dropped
    assert {"mid_price", "spread_bps"} <= included
    assert {
        "auction_price_range_bps",
        "auction_last_position_in_range",
        "postopen_v2_queue_ask1_replenish_vs_trade_1m",
        "postopen_v2_queue_bid1_replenish_vs_trade_1m",
    } <= included
    assert all("postopen_v2_ask1_queue" not in prefix for prefix in filters["include_prefixes"])
    assert all("postopen_v2_bid1_queue" not in prefix for prefix in filters["include_prefixes"])
    assert MIXED_ROOT in config["data"]["labeled_path"]
    assert len(config["k8s"]["wait_for_paths"]) == 7


def test_mech_v3_reuses_cached_cap_with_v2_memory_resources() -> None:
    v2 = _load(MECH_V2_CONFIG)
    v3 = _load(MECH_V3_CONFIG)

    assert MIXED_ROOT in v3["data"]["labeled_path"]
    assert v3["features"]["feature_value_transform"] == "mechanismized_v3_dimensionless_328"
    assert v3["features"]["include_historical_daily_activity_references"] is False
    assert v3["model"] == v2["model"]
    assert v3["window"] == v2["window"]
    assert v3["k8s"]["resources"]["memory_request"] == "384Gi"
    assert v3["k8s"]["resources"]["memory_limit"] == "768Gi"
    assert len(v3["k8s"]["wait_for_paths"]) == 7
    assert all(MIXED_ROOT in path for path in v3["k8s"]["wait_for_paths"])
