from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.daily_return_labels import (
    CLOSE_TO_NEXT_CLOSE_LABEL_COL,
    NEXT_SESSION_OPEN_CLOSE_LABEL_COL,
    build_close_to_next_close_labels,
    build_next_session_open_close_labels,
)


class DailyReturnLabelsTest(unittest.TestCase):
    def test_builds_next_trading_session_open_close_label(self) -> None:
        bars = pd.DataFrame(
            {
                "date": [
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ],
                "symbol": [
                    "000001.SZ",
                    "900001.SH",
                    "000001.SZ",
                    "900001.SH",
                    "000001.SZ",
                    "900001.SH",
                ],
                "open_price": [10.0, 5.0, 10.0, 5.0, 11.0, 5.0],
                "close_price": [10.0, 5.0, 11.0, 5.0, 9.9, 5.0],
            }
        )

        labels = build_next_session_open_close_labels(
            bars,
            feature_start_date="2025-01-03",
            feature_end_date="2025-01-06",
        )

        self.assertEqual(labels["date"].tolist(), ["2025-01-03", "2025-01-06"])
        self.assertEqual(labels["target_date"].tolist(), ["2025-01-06", "2025-01-07"])
        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ", "000001.SZ"])
        self.assertAlmostEqual(labels.loc[0, NEXT_SESSION_OPEN_CLOSE_LABEL_COL], 0.1)
        self.assertAlmostEqual(labels.loc[1, NEXT_SESSION_OPEN_CLOSE_LABEL_COL], -0.1)

    def test_drops_invalid_prices_and_applies_fee(self) -> None:
        bars = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03", "2025-01-03"],
                "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "open_price": [10.0, 10.0, 0.0],
                "close_price": [10.0, 10.1, 2.0],
            }
        )

        labels = build_next_session_open_close_labels(
            bars,
            feature_start_date="2025-01-02",
            feature_end_date="2025-01-02",
            fee_bps=8.0,
        )

        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])
        self.assertAlmostEqual(labels.loc[0, NEXT_SESSION_OPEN_CLOSE_LABEL_COL], 0.0092)

    def test_rejects_duplicate_daily_keys(self) -> None:
        bars = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-02"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "open_price": [10.0, 10.0],
                "close_price": [10.1, 10.1],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            build_next_session_open_close_labels(
                bars,
                feature_start_date="2025-01-02",
                feature_end_date="2025-01-02",
            )

    def test_close_to_next_close_uses_adjusted_target_preclose(self) -> None:
        bars = pd.DataFrame(
            {
                "date": ["2025-06-11", "2025-06-12"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "close_price": [11.85, 11.68],
                "preclose_price": [11.90, 11.49],
            }
        )

        labels = build_close_to_next_close_labels(
            bars,
            feature_start_date="2025-06-11",
            feature_end_date="2025-06-11",
        )

        self.assertEqual(labels["date"].tolist(), ["2025-06-11"])
        self.assertEqual(labels["target_date"].tolist(), ["2025-06-12"])
        self.assertAlmostEqual(
            labels.loc[0, CLOSE_TO_NEXT_CLOSE_LABEL_COL],
            11.68 / 11.49 - 1.0,
        )
        self.assertGreater(labels.loc[0, CLOSE_TO_NEXT_CLOSE_LABEL_COL], 0)

    def test_close_to_next_close_requires_symbol_on_both_sessions(self) -> None:
        bars = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03", "2025-01-03"],
                "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "close_price": [10.0, 10.2, 8.0],
                "preclose_price": [9.9, 10.0, 7.9],
            }
        )

        labels = build_close_to_next_close_labels(
            bars,
            feature_start_date="2025-01-02",
            feature_end_date="2025-01-02",
        )

        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])

    def test_close_to_next_close_requires_valid_entry_close(self) -> None:
        bars = pd.DataFrame(
            {
                "date": [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                ],
                "symbol": [
                    "000001.SZ",
                    "000002.SZ",
                    "000001.SZ",
                    "000002.SZ",
                ],
                "close_price": [10.0, 0.0, 10.2, 8.0],
                "preclose_price": [9.9, 7.8, 10.0, 7.9],
            }
        )

        labels = build_close_to_next_close_labels(
            bars,
            feature_start_date="2025-01-02",
            feature_end_date="2025-01-02",
        )

        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
