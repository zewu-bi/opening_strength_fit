import unittest

import pandas as pd

from opening_strength_fit.labels import build_trade_labels


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

if __name__ == "__main__":
    unittest.main()
