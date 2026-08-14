from __future__ import annotations

import pandas as pd

from opening_strength_fit.labels import build_trade_labels
from opening_strength_fit.sampling import select_decision_points


def _state_ticks() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2025-01-02 09:30:57",
            "2025-01-02 09:31:03",
            "2025-01-02 09:31:06",
            "2025-01-02 09:31:12",
            "2025-01-02 09:32:03",
        ]
    )
    return pd.DataFrame(
        {
            "date": ["2025-01-02"] * len(timestamps),
            "symbol": ["000001.SZ"] * len(timestamps),
            "timestamp": timestamps,
            "ask_price_1": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100, 110, 120, 130, 140],
            "turnover": [1_000.0, 1_110.0, 1_230.0, 1_360.0, 1_500.0],
            "status": ["T0"] * len(timestamps),
        }
    )


def test_clock_state_decisions_carry_forward_without_looking_ahead() -> None:
    sampled = select_decision_points(
        _state_ticks(),
        decision_times=("09:31:00", "09:32:00"),
        alignment="clock_state",
    )

    assert sampled["decision_time"].tolist() == ["09:31:00", "09:32:00"]
    assert sampled["ask_price_1"].tolist() == [10.0, 13.0]
    assert (
        sampled["decision_source_timestamp"].tolist()
        == pd.to_datetime(["2025-01-02 09:30:57", "2025-01-02 09:31:12"]).tolist()
    )
    assert sampled["decision_state_age_seconds"].tolist() == [3.0, 48.0]
    assert sampled["decision_lag_seconds"].tolist() == [0.0, 0.0]


def test_clock_state_source_cutoff_keeps_logical_decision_clock() -> None:
    sampled = select_decision_points(
        _state_ticks(),
        decision_times=("09:31:00", "09:32:00"),
        alignment="clock_state",
        source_cutoff_seconds=2,
    )

    assert (
        sampled["decision_target_timestamp"].tolist()
        == pd.to_datetime(["2025-01-02 09:31:00", "2025-01-02 09:32:00"]).tolist()
    )
    assert (
        sampled["decision_source_cutoff_timestamp"].tolist()
        == pd.to_datetime(["2025-01-02 09:30:58", "2025-01-02 09:31:58"]).tolist()
    )
    assert (
        sampled["decision_source_timestamp"].tolist()
        == pd.to_datetime(["2025-01-02 09:30:57", "2025-01-02 09:31:12"]).tolist()
    )


def test_next_tick_decisions_keep_historical_forward_sampling() -> None:
    sampled = select_decision_points(
        _state_ticks(),
        decision_times=("09:31:00", "09:32:00"),
        alignment="next_tick",
        max_lag_seconds=5,
    )

    assert sampled["decision_time"].tolist() == ["09:31:00", "09:32:00"]
    assert sampled["ask_price_1"].tolist() == [11.0, 14.0]
    assert sampled["decision_lag_seconds"].tolist() == [3.0, 3.0]
    assert "decision_state_age_seconds" not in sampled


def test_clock_state_entry_delay_is_anchored_to_logical_decision_clock() -> None:
    state = _state_ticks()
    sampled = select_decision_points(
        state,
        decision_times=("09:31:00",),
        alignment="clock_state",
    )
    labeled = build_trade_labels(
        sampled,
        state_ticks=state,
        entry_target_timestamp_col="decision_target_timestamp",
        entry_alignment="clock_state",
        entry_clock_delay_seconds=6,
        future_alignment="clock_state",
        hold_seconds=0,
        sell_window_seconds=6,
        sample_start_time="09:31:00",
        sample_end_time="09:31:00",
    )

    row = labeled.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2025-01-02 09:30:57")
    assert row["decision_target_timestamp"] == pd.Timestamp("2025-01-02 09:31:00")
    assert row["entry_timestamp"] == pd.Timestamp("2025-01-02 09:31:06")
    assert row["entry_source_timestamp"] == pd.Timestamp("2025-01-02 09:31:06")
    assert row["buy_price"] == 12.0
