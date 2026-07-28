from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opening_strength_fit.temporal_analysis import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    analyze_day_sequence,
    assemble_day_sequence,
    cross_section_rank_values,
    summarize_temporal_metrics,
    write_sequence_npz,
)


def test_assemble_day_sequence_aligns_symbols_clocks_labels_and_pool() -> None:
    rows = []
    for symbol, base in [("000001.SZ", 0.01), ("000002.SZ", 0.02)]:
        for offset, clock in enumerate(["09:30:00", "09:31:00"]):
            rows.append(
                {
                    "symbol": symbol,
                    "decision_target_timestamp": f"2025-01-02 {clock}",
                    "alpha_return_1m": base + offset / 1000,
                    "alpha_return_10m": base * 2 + offset / 1000,
                    "alpha_return_60m": base * 3 + offset / 1000,
                }
            )
    features = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {
            "symbol": ["000002.SZ", "000001.SZ"],
            TARGET_COLUMN: [0.03, -0.01],
        }
    )
    arrays = assemble_day_sequence(
        features,
        labels,
        clocks=["09:30:00", "09:31:00"],
        pool_symbols={"000002.SZ"},
    )

    assert arrays["values"].shape == (2, 3, 2)
    assert arrays["symbols"].tolist() == ["000001.SZ", "000002.SZ"]
    np.testing.assert_allclose(arrays["target"], [-0.01, 0.03])
    assert arrays["pool_member"].tolist() == [False, True]
    assert arrays["valid"].all()


def test_analyze_day_sequence_detects_monotonic_head() -> None:
    symbols = np.array([f"{index:06d}.SZ" for index in range(10)])
    target = np.arange(10, dtype=np.float32) / 100.0
    values = np.empty((10, len(FEATURE_COLUMNS), 2), dtype=np.float32)
    for channel in range(len(FEATURE_COLUMNS)):
        values[:, channel, 0] = target
        values[:, channel, 1] = -target
    arrays = {
        "symbols": symbols,
        "clock_seconds": np.array([34200, 34260], dtype=np.int32),
        "values": values,
        "valid": np.isfinite(values),
        "target": target,
        "pool_member": np.ones(10, dtype=bool),
    }

    metrics = analyze_day_sequence(
        arrays,
        date="2025-01-02",
        top_n=2,
        tail_fraction=0.2,
    )
    one_minute = metrics.loc[
        metrics["universe"].eq("all_a") & metrics["horizon"].eq("1m")
    ].sort_values("clock_seconds")
    np.testing.assert_allclose(one_minute["rank_ic"], [1.0, -1.0])
    assert one_minute.iloc[0]["top_n_excess"] > 0
    assert one_minute.iloc[0]["head_tail_spread"] > 0
    assert one_minute.iloc[1]["top_n_excess"] < 0


def test_summarize_temporal_metrics_outputs_overall_and_year() -> None:
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2025-01-02"],
            "universe": ["all_a", "all_a"],
            "horizon": ["1m", "1m"],
            "clock_seconds": [34200, 34200],
            "clock": ["09:30", "09:30"],
            "n": [100, 120],
            "rank_ic": [0.1, 0.2],
            "top_n_excess": [0.01, 0.02],
            "head_tail_spread": [0.03, 0.04],
        }
    )
    overall, annual = summarize_temporal_metrics(daily)
    assert len(overall) == 1
    assert len(annual) == 2
    assert overall.iloc[0]["days"] == 2
    assert overall.iloc[0]["mean_rank_ic"] == pytest.approx(0.15)


def test_sequence_npz_uses_float16_and_implicit_mask(tmp_path) -> None:
    arrays = {
        "values": np.array([[[0.01, np.nan]]], dtype=np.float32),
        "valid": np.array([[[True, False]]]),
        "target": np.array([0.02], dtype=np.float32),
    }
    path = tmp_path / "sequence.npz"
    write_sequence_npz(path, arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert loaded["values"].dtype == np.float16
        assert loaded["rank_values"].dtype == np.float16
        assert "valid" not in loaded.files


def test_cross_section_rank_values_are_centered_per_clock() -> None:
    values = np.array(
        [
            [[1.0, 3.0]],
            [[2.0, np.nan]],
            [[3.0, 1.0]],
        ],
        dtype=np.float32,
    )
    ranked = cross_section_rank_values(values, np.isfinite(values))
    np.testing.assert_allclose(ranked[:, 0, 0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ranked[[0, 2], 0, 1], [1.0, -1.0])
