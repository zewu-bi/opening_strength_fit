from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
BASE_ROOT = "opening_2019_2025_label_v4_clock6_state_unique_base_mcap_lag1"
MIXED_ROOT = "opening_2019_2025_label_v4_clock6_state_unique_mixed_w030_mcap_lag1"
CONTROL_MODEL = (
    RUNS
    / "nn_delay6_clock_state_36m_2022_2025_auction_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
)
MULTI_DEN_MODEL = (
    RUNS
    / "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
)


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
        assert config["sample"].get("decision_alignment", "next_tick") == "next_tick"
        assert config["labels"]["entry_alignment"] == "clock_state"
        assert config["labels"]["entry_clock_delay_seconds"] == 6
        assert config["labels"]["future_alignment"] == "clock_state"
        assert config["labels"]["require_entry_after_cross_section_ready"] is True
        assert "entry_max_gap_seconds" not in config["labels"]
        assert "max_future_gap_seconds" not in config["labels"]


def test_corrected_decision_clock_state_cache_has_distinct_lineage() -> None:
    corrected = _load(RUNS / "build_delay6_decision_clock_state_0930_0940_cache_v1.toml")
    assert corrected["cache"]["schema_version"] == (
        "label_v6_decision_clock_state_clock6_unique_base_mcap_lag1"
    )
    assert corrected["sample"]["decision_alignment"] == "clock_state"
    assert "decision_max_lag_seconds" not in corrected["sample"]
    assert corrected["labels"]["entry_alignment"] == "clock_state"
    assert corrected["labels"]["entry_clock_delay_seconds"] == 6
    assert corrected["labels"]["future_alignment"] == "clock_state"

    for window in ("1001_1010", "1101_1110", "1401_1410"):
        historical = _load(RUNS / f"build_delay6_clock_state_{window}_from_start_cache_v1.toml")
        assert historical["sample"].get("decision_alignment", "next_tick") == "next_tick"
        assert historical["sample"]["decision_max_lag_seconds"] == 5
        assert "label_v5_clock6_state_window_start" in historical["cache"]["schema_version"]


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

    model = _load(CONTROL_MODEL)
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


def test_clock_state_multi_denominator_challenger_only_adds_bounded_features() -> None:
    control = _load(CONTROL_MODEL)
    challenger = _load(MULTI_DEN_MODEL)

    for section in (
        "data",
        "universe",
        "sample",
        "labels",
        "filters",
        "window",
        "model",
        "evaluation",
    ):
        assert challenger[section] == control[section]

    features = challenger["features"]
    assert features["include_multi_denominator_features"] is True
    assert len(features["multi_denominator_turnover_columns"]) == 5
    assert len(features["multi_denominator_volume_columns"]) == 5
    assert len(features["multi_denominator_depth_columns"]) == 2
    assert features["multi_denominator_cross_sectional_median_columns"] == [
        "postopen_turnover_diff_1m"
    ]
    added_feature_count = 5 * 2 + 5 * 2 + 2 * 2 + 1
    assert added_feature_count == features["multi_denominator_min_features"] == 25
    assert added_feature_count <= features["multi_denominator_max_features"] == 40
    assert 325 + added_feature_count == 350 < 400
    assert "multi_den_ratio_" in features["include_feature_prefixes"]

    assert challenger["k8s"]["shard_parallelism"] == 4
    assert challenger["k8s"]["resources"]["memory_request"] == "608Gi"
    assert challenger["k8s"]["resources"]["memory_limit"] == "960Gi"
