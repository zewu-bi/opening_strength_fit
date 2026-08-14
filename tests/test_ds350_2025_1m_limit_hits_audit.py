from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).parents[1] / "experiments/scripts/audit_ds350_2025_1m_limit_hits.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("audit_ds350_2025_1m_limit_hits", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
add_limit_checks = SCRIPT_MODULE.add_limit_checks
align_tick_states = SCRIPT_MODULE.align_tick_states
select_unique_limit_hits = SCRIPT_MODULE.select_unique_limit_hits


def test_select_unique_limit_hits_keeps_strongest_clock_per_symbol_day() -> None:
    selected = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3,
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "decision_target_timestamp": pd.to_datetime(
                ["2025-01-02 09:31:00", "2025-01-02 09:32:00", "2025-01-02 09:31:00"]
            ),
            "prediction": [0.4, 0.8, 0.9],
            "candidate_rows": [3000, 3000, 3000],
            "score_rank": [90, 20, 10],
            "score_percentile": [0.97, 0.993, 0.997],
        }
    )
    daily = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "pvc_daily_close": [11.0, 10.5],
            "pvc_prev_close": [10.0, 10.0],
            "pvc_daily_trade_status": ["T0", "T0"],
            "pvc_st_status": [0, 0],
            "pvc_updown_limit_status": [1, 0],
            "pvc_tick_close": [11.0, 10.5],
            "pvc_close_source_offset_us": [54_000_000_000, 54_000_000_000],
        }
    )

    hits, summary = select_unique_limit_hits(selected, daily)

    assert len(hits) == 1
    assert hits.iloc[0]["prediction"] == 0.8
    assert hits.iloc[0]["hit_clock_count"] == 2
    assert hits.iloc[0]["buy_target_timestamp"] == pd.Timestamp("2025-01-02 09:32:06")
    assert summary["limit_hit_rows_before_daily_dedup"] == 2
    assert summary["multi_clock_limit_hit_symbol_days"] == 1


def test_align_tick_states_uses_last_known_state_at_target() -> None:
    keys = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001.SZ"],
            "decision_target_timestamp": pd.to_datetime(["2025-01-02 09:31:00"]),
        }
    )
    ticks = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "ExchTimeOffsetUs": [34_259_000_000, 34_261_000_000],
            "Status": ["T0", "T0"],
            "AskPrice1": [10.1, 10.2],
            "AskVolume1": [1000, 2000],
            "BidPrice1": [10.0, 10.1],
            "BidVolume1": [900, 1900],
            "LastPrice": [10.05, 10.15],
            "HighPrice": [10.1, 10.2],
        }
    )

    aligned = align_tick_states(
        keys,
        ticks,
        target_column="decision_target_timestamp",
        prefix="pvc_decision",
    )

    assert aligned.iloc[0]["pvc_decision_source_offset_us"] == 34_259_000_000
    assert aligned.iloc[0]["pvc_decision_state_age_seconds"] == 1.0
    assert aligned.iloc[0]["pvc_decision_AskPrice1"] == 10.1


def test_add_limit_checks_supports_main_and_growth_boards() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "300001.SZ"],
            "pvc_prev_close": [10.0, 10.0],
            "pvc_daily_close": [11.0, 12.0],
            "pvc_st_status": [0, 0],
            "pvc_updown_limit_status": [1, 1],
        }
    )

    checked = add_limit_checks(frame, prefix="pvc")

    assert checked["pvc_rule_upper_limit"].tolist() == [11.0, 12.0]
    assert checked["pvc_close_equals_rule_limit"].all()
