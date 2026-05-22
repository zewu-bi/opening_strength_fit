from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from materialize_labeled_caches import _labeled_from_feature_frame  # noqa: E402


def _feature_rows(offsets: list[int]) -> pd.DataFrame:
    base = pd.Timestamp("2022-01-04 09:30:00")
    rows = []
    cumulative_volume = 1000.0
    cumulative_turnover = 10_000.0
    for idx, seconds in enumerate(offsets):
        price = 10.0 + idx * 0.01
        cumulative_volume += 100.0
        cumulative_turnover += 100.0 * price
        rows.append(
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "timestamp": base + pd.Timedelta(seconds=seconds),
                "ask_price_1": price,
                "ask_volume_1": 1000.0,
                "bid_price_1": price - 0.01,
                "bid_volume_1": 1000.0,
                "volume": cumulative_volume,
                "turnover": cumulative_turnover,
                "status": "TRADE",
            }
        )
    return pd.DataFrame(rows)


def _config(entry_max_gap_seconds=None) -> dict:
    return {
        "sample": {
            "mode": "decision_points",
            "decision_times": ["09:30:00"],
            "decision_max_lag_seconds": 0,
            "start_time": "09:30:00",
            "end_time": "09:40:00",
        },
        "labels": {
            "buy_price_col": "ask_price_1",
            "volume_col": "volume",
            "turnover_col": "turnover",
            "hold_seconds": 60,
            "sell_window_seconds": 60,
            "volume_unit_multiplier": 1.0,
            "fee_bps": 0.0,
            "entry_max_gap_seconds": entry_max_gap_seconds,
        },
        "filters": {"tradable_statuses": ["TRADE"]},
    }


class MaterializeLabeledCachesTest(unittest.TestCase):
    def test_delay_freshness_uses_tick_gap_not_total_delay(self) -> None:
        labeled = _labeled_from_feature_frame(
            _feature_rows([0, 3, 63, 123]),
            _config(entry_max_gap_seconds=5),
            delay=1,
        )

        row = labeled.iloc[0]
        self.assertEqual(row["entry_delay_ticks"], 1.0)
        self.assertEqual(row["entry_delay_seconds"], 3.0)
        self.assertEqual(row["entry_max_tick_gap_seconds"], 3.0)
        self.assertFalse(pd.isna(row["entry_timestamp"]))
        self.assertTrue(bool(row["valid_label"]))
        self.assertEqual(row["entry_ask_price_1"], 10.01)
        self.assertEqual(row["entry_status"], "TRADE")

    def test_entry_max_gap_filters_sparse_entry_path(self) -> None:
        labeled = _labeled_from_feature_frame(
            _feature_rows([0, 6, 66, 126]),
            _config(entry_max_gap_seconds=5),
            delay=1,
        )

        row = labeled.iloc[0]
        self.assertTrue(pd.isna(row["entry_timestamp"]))
        self.assertTrue(pd.isna(row["entry_delay_seconds"]))
        self.assertTrue(pd.isna(row["entry_max_tick_gap_seconds"]))
        self.assertFalse(bool(row["valid_label"]))


if __name__ == "__main__":
    unittest.main()
