from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
EVIDENCE = (
    ROOT
    / "experiments"
    / "evidence"
    / "backtests"
    / (
        "nn_delay6_clock_state_36m_2022_2025_w1001_1010_auction_pruned_"
        "multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1"
    )
)


def test_window_decay_cumulative_uses_pool_l_relative_lower_panel() -> None:
    trace = json.loads((EVIDENCE / "trace_optimization.json").read_text(encoding="utf-8"))
    assert trace["overlay_excess_horizon"] == "short"
    with (EVIDENCE / "01_signal_acceptance.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        overlay_columns = tuple(csv.DictReader(handle).fieldnames or ())
    assert "short_internal_excess_bps" in overlay_columns
    assert "next_internal_excess_bps" not in overlay_columns

    assert trace["cumulative_relative_mode"] == "pool_l"
    assert trace["plotted_series"]["cumulative_relative"] == [
        "baseline_pool_l",
        "w1001_1010",
    ]
    assert trace["plotted_series"]["cumulative_top"] == [
        "market",
        "background",
        "w1001_1010_pool_l",
        "baseline_pool_l",
        "w1001_1010",
    ]
    assert trace["plotted_series"]["cumulative_market_alpha"] == []
    assert trace["plotted_series"]["cumulative_pool_l_excess"] == [
        "baseline_pool_l",
        "w1001_1010",
    ]

    svg = (EVIDENCE / "02_top100_cumulative.svg").read_text(encoding="utf-8")
    assert "相对 pool_L 累和超额" in svg
    assert "对比全A股市场平均alpha" not in svg
    assert svg.count("<polyline") == 7

    with (EVIDENCE / "02_top100_cumulative.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    assert "next_cumulative_alpha_bps" in columns
    assert "next_cumulative_alpha_vs_visible_pool_l_bps" not in columns
    assert "next_cumulative_alpha_vs_market_bps" not in columns

    endpoints = {row["pool"]: row for row in rows}
    assert round(float(endpoints["background"]["next_cumulative_net_return_bps"]), 3) == 5063.158
    assert (
        round(
            float(endpoints["w1001_1010_pool_l"]["next_cumulative_net_return_bps"]),
            3,
        )
        == 3026.461
    )
    assert round(float(endpoints["w1001_1010"]["next_cumulative_net_return_bps"]), 3) == 2708.538
    assert round(float(endpoints["w1001_1010"]["next_cumulative_alpha_bps"]), 3) == -317.923
