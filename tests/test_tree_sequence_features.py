from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.features import add_tree_sequence_features


def _sequence_rows() -> pd.DataFrame:
    rows = []
    for minute, mid_price in enumerate((10.0, 10.1, 10.3)):
        rows.append(
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "timestamp": pd.Timestamp("2022-01-04 09:31:00")
                + pd.Timedelta(minutes=minute),
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00")
                + pd.Timedelta(minutes=minute),
                "mid_price": mid_price,
                "spread_bps": 20.0 - minute,
                "bid_depth_10": 1000.0 + minute * 30.0,
                "ask_depth_10": 900.0 - minute * 15.0,
                "volume_diff_1t": 10.0 + minute * 10.0,
            }
        )
    rows.append(
        {
            "date": "2022-01-04",
            "symbol": "000002.SZ",
            "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
            "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
            "mid_price": 20.0,
            "spread_bps": 12.0,
            "bid_depth_10": 500.0,
            "ask_depth_10": 450.0,
            "volume_diff_1t": 5.0,
        }
    )
    return pd.DataFrame(rows)


class TreeSequenceFeatureTest(unittest.TestCase):
    def test_sequence_features_use_past_rows_and_landmark_masking(self) -> None:
        out = add_tree_sequence_features(
            _sequence_rows(),
            columns=("mid_price", "spread_bps", "bid_depth_10", "ask_depth_10", "volume_diff_1t"),
            windows=(2, 3),
            landmarks=("09:31:00", "09:33:00"),
        )

        first_symbol = out[out["symbol"] == "000001.SZ"].reset_index(drop=True)
        self.assertAlmostEqual(first_symbol.loc[0, "tree_seq_mid_price_at_093100"], 10.0)
        self.assertAlmostEqual(first_symbol.loc[1, "tree_seq_mid_price_vs_093100"], 0.1)
        self.assertTrue(pd.isna(first_symbol.loc[1, "tree_seq_mid_price_at_093300"]))
        self.assertAlmostEqual(first_symbol.loc[2, "tree_seq_mid_price_at_093300"], 10.3)
        self.assertAlmostEqual(first_symbol.loc[2, "tree_seq_mid_price_vs_093300"], 0.0)

        self.assertAlmostEqual(first_symbol.loc[1, "tree_seq_mid_price_roll2_change"], 0.1)
        self.assertAlmostEqual(first_symbol.loc[2, "tree_seq_mid_price_roll3_slope"], 0.15)
        self.assertAlmostEqual(
            first_symbol.loc[1, "tree_seq_bid_depth_replenish_to_trade_roll2"],
            1.0,
        )

        second_symbol = out[out["symbol"] == "000002.SZ"].reset_index(drop=True)
        self.assertTrue(pd.isna(second_symbol.loc[0, "tree_seq_mid_price_roll2_change"]))
        self.assertAlmostEqual(second_symbol.loc[0, "tree_seq_mid_price_at_093100"], 20.0)


if __name__ == "__main__":
    unittest.main()
