import unittest

import pandas as pd

from opening_strength_fit.labels import build_trade_labels
from opening_strength_fit.sampling import require_entry_after_cross_section_ready


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

    def test_clock_state_entry_uses_fixed_six_second_target(self) -> None:
        labeled = build_trade_labels(
            _ticks([0, 6, 12, 18]),
            entry_tick_delay=2,
            entry_alignment="clock_state",
            entry_clock_delay_seconds=6,
            future_alignment="clock_state",
            hold_seconds=0,
            sell_window_seconds=6,
        )

        first = labeled.iloc[0]
        base = pd.Timestamp("2022-01-04 09:30:00")
        self.assertEqual(first["entry_timestamp"], base + pd.Timedelta(seconds=6))
        self.assertEqual(
            first["entry_source_timestamp"],
            base + pd.Timedelta(seconds=6),
        )
        self.assertEqual(first["entry_delay_seconds"], 6.0)
        self.assertEqual(first["entry_state_age_seconds"], 0.0)
        self.assertEqual(first["buy_price"], 10.01)

    def test_clock_state_entry_carries_forward_without_looking_ahead(self) -> None:
        labeled = build_trade_labels(
            _ticks([0, 9, 12, 18]),
            entry_tick_delay=2,
            entry_alignment="clock_state",
            entry_clock_delay_seconds=6,
            future_alignment="clock_state",
            hold_seconds=0,
            sell_window_seconds=6,
        )

        first = labeled.iloc[0]
        base = pd.Timestamp("2022-01-04 09:30:00")
        self.assertEqual(first["entry_timestamp"], base + pd.Timedelta(seconds=6))
        self.assertEqual(first["entry_source_timestamp"], base)
        self.assertEqual(first["entry_state_age_seconds"], 6.0)
        self.assertEqual(first["buy_price"], 10.0)

    def test_clock_state_future_boundaries_use_last_known_cumulative_state(self) -> None:
        labeled = build_trade_labels(
            _ticks([0, 6, 65, 72, 125, 132]),
            entry_tick_delay=2,
            entry_alignment="clock_state",
            entry_clock_delay_seconds=6,
            future_alignment="clock_state",
            hold_seconds=60,
            sell_window_seconds=60,
        )

        first = labeled.iloc[0]
        base = pd.Timestamp("2022-01-04 09:30:00")
        self.assertEqual(
            first["sell_start_target_timestamp"],
            base + pd.Timedelta(seconds=66),
        )
        self.assertEqual(
            first["sell_start_source_timestamp"],
            base + pd.Timedelta(seconds=65),
        )
        self.assertEqual(first["sell_start_state_age_seconds"], 1.0)
        self.assertEqual(
            first["sell_end_target_timestamp"],
            base + pd.Timedelta(seconds=126),
        )
        self.assertEqual(
            first["sell_end_source_timestamp"],
            base + pd.Timedelta(seconds=125),
        )
        self.assertEqual(first["sell_end_state_age_seconds"], 1.0)

    def test_clock_state_rejects_observed_gap_validity_gates(self) -> None:
        with self.assertRaisesRegex(SystemExit, "entry_max_gap_seconds"):
            build_trade_labels(
                _ticks([0, 6, 12]),
                entry_alignment="clock_state",
                entry_clock_delay_seconds=6,
                entry_max_gap_seconds=5,
            )
        with self.assertRaisesRegex(SystemExit, "max_future_gap_seconds"):
            build_trade_labels(
                _ticks([0, 6, 12]),
                entry_alignment="clock_state",
                entry_clock_delay_seconds=6,
                future_alignment="clock_state",
                max_future_gap_seconds=5,
            )

    def test_cross_section_readiness_invalidates_early_entry_only(self) -> None:
        target = pd.Timestamp("2022-01-04 09:31:00")
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "symbol": ["600520.SH", "000001.SZ"],
                "decision_target_timestamp": [target, target],
                "timestamp": [
                    target + pd.Timedelta(seconds=3),
                    target + pd.Timedelta(seconds=5),
                ],
                "entry_timestamp": [
                    target + pd.Timedelta(seconds=4),
                    target + pd.Timedelta(seconds=6),
                ],
                "valid_label": [True, True],
            }
        )

        out = require_entry_after_cross_section_ready(frame)

        self.assertEqual(
            out["cross_section_ready_timestamp"].tolist(),
            [target + pd.Timedelta(seconds=5)] * 2,
        )
        self.assertEqual(out["entry_after_cross_section_ready"].tolist(), [False, True])
        self.assertEqual(out["valid_label"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
