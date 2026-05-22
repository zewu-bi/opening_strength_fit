from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

from opening_strength_fit.labels import build_trade_labels

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_opening_intraday_backtest import apply_static_constraints  # noqa: E402


def _ticks(offsets: list[int]) -> pd.DataFrame:
    base = pd.Timestamp("2022-01-04 09:30:00")
    rows = []
    for idx, seconds in enumerate(offsets):
        rows.append(
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "timestamp": base + pd.Timedelta(seconds=seconds),
                "ask_price_1": 10.0 + idx * 0.01,
                "ask_volume_1": 1000.0,
                "bid_price_1": 9.99,
                "bid_volume_1": 1000.0,
                "volume": 1000.0 + idx * 100.0,
                "turnover": (1000.0 + idx * 100.0) * (10.0 + idx * 0.01),
                "status": "TRADE",
            }
        )
    return pd.DataFrame(rows)


class DelayFreshnessTest(unittest.TestCase):
    def test_delay_seconds_do_not_count_as_entry_staleness(self) -> None:
        labeled = build_trade_labels(
            _ticks([0, 3, 6, 9, 12]),
            entry_tick_delay=2,
            entry_max_gap_seconds=5,
            hold_seconds=0,
            sell_window_seconds=3,
        )

        first = labeled.iloc[0]
        self.assertEqual(first["entry_delay_ticks"], 2.0)
        self.assertEqual(first["entry_delay_seconds"], 6.0)
        self.assertEqual(first["entry_max_tick_gap_seconds"], 3.0)
        self.assertFalse(pd.isna(first["entry_timestamp"]))

    def test_entry_max_gap_filters_sparse_tick_path(self) -> None:
        labeled = build_trade_labels(
            _ticks([0, 3, 13, 16, 19]),
            entry_tick_delay=2,
            entry_max_gap_seconds=5,
            hold_seconds=0,
            sell_window_seconds=3,
        )

        first = labeled.iloc[0]
        self.assertTrue(pd.isna(first["entry_timestamp"]))
        self.assertTrue(pd.isna(first["entry_delay_seconds"]))
        self.assertTrue(pd.isna(first["entry_max_tick_gap_seconds"]))

    def test_replay_entry_freshness_uses_tick_gap_not_total_delay(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2022-01-04",
                    "symbol": "000001.SZ",
                    "entry_delay_seconds": 6.0,
                    "entry_max_tick_gap_seconds": 3.0,
                }
            ]
        )

        constrained = apply_static_constraints(
            frame,
            run_label="test",
            top_n=20,
            fee_bps=0.0,
            slippage_bps=0.0,
            tradable_statuses=set(),
            require_entry_status=False,
            max_decision_lag_seconds=None,
            max_entry_tick_gap_seconds=5.0,
            max_spread_bps=None,
            min_limit_up_room_bps=None,
            min_ask_volume_1=None,
            min_bid_volume_1=None,
            capacity_notional_col="",
            capacity_volume_col="",
            capacity_price_col="ask_price_1",
            min_capacity_notional=0.0,
            max_participation_rate=0.0,
            capital_per_cycle=0.0,
            ask_depth_levels=0,
            ask_depth_participation_rate=1.0,
            ask_depth_fill_mode="filter",
            allow_decision_depth_fallback=False,
            max_symbol_weight=0.0,
            missing_policy="error",
        )

        self.assertEqual(len(constrained), 1)


if __name__ == "__main__":
    unittest.main()
