from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
BASE_ROOT = "opening_2019_2025_label_v2_tick2_unique_base_mcap_lag1"
MIXED_ROOT = "opening_2019_2025_label_v2_tick2_unique_mixed_w030_mcap_lag1"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_conservative_cache_counts_unique_ticks_without_strict_validity_gates() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay2_{year}_conservative_cap_cache_v1.toml")

        assert BASE_ROOT in config["cache"]["path"]
        assert config["data"]["tick_timestamp_deduplication"] == ("latest_local_timestamp")
        assert config["features"]["preopen_price_mode"] == "indicative_quote_v2"
        assert "entry_max_gap_seconds" not in config["labels"]
        assert "max_future_gap_seconds" not in config["labels"]
        assert "require_entry_after_cross_section_ready" not in config["labels"]
        assert config["daily_market_reference"] == {
            "enabled": True,
            "table": "stock.daily_bar_jy",
            "lag_sessions": 1,
            "market_cap_unit_multiplier": 10_000.0,
            "share_unit_multiplier": 10_000.0,
        }


def test_conservative_target_configs_are_isolated() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay2_{year}_conservative_cap_mixed_w030_target_v1.toml")

        assert BASE_ROOT in config["target_cache"]["input_path"]
        assert MIXED_ROOT in config["target_cache"]["output_path"]
        assert config["target_cache"]["long_label_weight"] == 0.30
