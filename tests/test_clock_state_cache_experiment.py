from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
BASE_ROOT = "opening_2019_2025_label_v4_clock6_state_unique_base_mcap_lag1"
MIXED_ROOT = "opening_2019_2025_label_v4_clock6_state_unique_mixed_w030_mcap_lag1"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_clock_state_cache_uses_fixed_wall_clock_boundaries() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay6_clock_state_{year}_cap_cache_v1.toml")

        assert BASE_ROOT in config["cache"]["path"]
        assert config["cache"]["schema_version"] == ("label_v4_clock6_state_unique_base_mcap_lag1")
        assert config["cache"]["path"].endswith(
            f"opening_{year}_label_v4_clock6_state_unique_base_mcap_lag1.parquet"
        )
        assert config["data"]["source"] == "clickhouse"
        assert config["data"]["tick_timestamp_deduplication"] == ("latest_local_timestamp")
        assert config["labels"]["entry_alignment"] == "clock_state"
        assert config["labels"]["entry_clock_delay_seconds"] == 6
        assert config["labels"]["future_alignment"] == "clock_state"
        assert config["labels"]["require_entry_after_cross_section_ready"] is True
        assert "entry_max_gap_seconds" not in config["labels"]
        assert "max_future_gap_seconds" not in config["labels"]


def test_clock_state_target_and_model_only_read_new_lineage() -> None:
    for year in range(2019, 2026):
        config = _load(RUNS / f"build_delay6_clock_state_{year}_cap_mixed_w030_target_v1.toml")

        assert BASE_ROOT in config["target_cache"]["input_path"]
        assert MIXED_ROOT in config["target_cache"]["output_path"]
        assert config["target_cache"]["input_path"].endswith(
            f"opening_{year}_label_v4_clock6_state_unique_base_mcap_lag1.parquet"
        )
        assert config["target_cache"]["output_path"].endswith(
            f"opening_{year}_label_v4_clock6_state_unique_mixed_w030_mcap_lag1.parquet"
        )
        assert (
            "opening_2013_2025_next_close_labels_v1" in config["target_cache"]["long_label_input"]
        )
        assert config["target_cache"]["long_label_weight"] == 0.30

    model = _load(
        RUNS
        / "nn_delay6_clock_state_36m_2022_2025_auction_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
    )
    assert MIXED_ROOT in model["data"]["labeled_path"]
    assert model["k8s"]["wait_for_paths"] == [
        f"/mnt/output/opening_strength_fit/cache/{MIXED_ROOT}/"
        f"opening_{year}_label_v4_clock6_state_unique_mixed_w030_mcap_lag1.parquet"
        for year in range(2019, 2026)
    ]
    assert model["labels"]["entry_alignment"] == "clock_state"
    assert model["labels"]["entry_clock_delay_seconds"] == 6
    assert model["labels"]["future_alignment"] == "clock_state"
    assert "entry_max_gap_seconds" not in model["labels"]
    assert "max_future_gap_seconds" not in model["labels"]
