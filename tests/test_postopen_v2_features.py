from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.features import add_postopen_v2_decision_features


def _decision_rows() -> pd.DataFrame:
    rows = []
    for minute, volume in enumerate((1000.0, 1060.0, 1140.0)):
        row: dict[str, object] = {
            "date": "2022-01-04",
            "symbol": "000001.SZ",
            "timestamp": pd.Timestamp("2022-01-04 09:30:00") + pd.Timedelta(minutes=minute),
            "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00")
            + pd.Timedelta(minutes=minute),
            "mid_price": 10.0 + minute * 0.02,
            "ask_price_1": 10.01 + minute * 0.02,
            "bid_price_1": 9.99 + minute * 0.02,
            "spread_bps": 20.0 - minute,
            "volume": volume,
            "turnover": volume * (10.0 + minute * 0.02),
            "volume_diff_1t": 60.0 + minute * 10.0,
            "turnover_diff_1t": (60.0 + minute * 10.0) * (10.0 + minute * 0.02),
            "trade_vwap_1t": 10.0 + minute * 0.02,
        }
        for level in range(1, 11):
            row[f"ask_price_{level}"] = 10.01 + level * 0.01 + minute * 0.02
            row[f"bid_price_{level}"] = 9.99 - level * 0.01 + minute * 0.02
            row[f"ask_volume_{level}"] = 100.0 * level + minute * 10.0
            row[f"bid_volume_{level}"] = 120.0 * level + minute * 20.0
            if level > 1:
                row[f"ask_gap_{level}_bps"] = float(level)
                row[f"bid_gap_{level}_bps"] = float(level) * 0.5
        row["ask_depth_10"] = sum(float(row[f"ask_volume_{level}"]) for level in range(1, 11))
        row["bid_depth_10"] = sum(float(row[f"bid_volume_{level}"]) for level in range(1, 11))
        row["depth_imbalance_10"] = (row["bid_depth_10"] - row["ask_depth_10"]) / (
            row["bid_depth_10"] + row["ask_depth_10"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


class PostopenV2FeatureTest(unittest.TestCase):
    def test_v2_features_use_current_and_past_decision_rows(self) -> None:
        frame = _decision_rows()
        out = add_postopen_v2_decision_features(
            frame,
            windows=(1,),
            depth_levels=(3, 10),
        )

        self.assertIn("postopen_v2_ask_depth_3", out.columns)
        self.assertIn("postopen_v2_ask_volume_1_diff_1m", out.columns)
        self.assertTrue(pd.isna(out.loc[0, "postopen_v2_ask_volume_1_diff_1m"]))
        self.assertAlmostEqual(out.loc[1, "postopen_v2_ask_volume_1_diff_1m"], 10.0)
        self.assertAlmostEqual(out.loc[0, "postopen_v2_ask_volume_1_from_open_diff"], 0.0)
        self.assertAlmostEqual(out.loc[2, "postopen_v2_ask_volume_1_from_open_diff"], 20.0)

        ask_depth_3 = sum(frame.loc[0, f"ask_volume_{level}"] for level in range(1, 4))
        ask_depth_10 = sum(frame.loc[0, f"ask_volume_{level}"] for level in range(1, 11))
        self.assertAlmostEqual(out.loc[0, "postopen_v2_ask_depth_3"], ask_depth_3)
        self.assertAlmostEqual(
            out.loc[0, "postopen_v2_ask_depth_concentration_3_10"],
            ask_depth_3 / ask_depth_10,
        )

    def test_trade_impact_features_are_emitted_when_tick_diffs_exist(self) -> None:
        out = add_postopen_v2_decision_features(
            _decision_rows(),
            windows=(1,),
            depth_levels=(3, 10),
        )

        self.assertIn("postopen_v2_trade_volume_to_ask1_1t", out.columns)
        self.assertIn("postopen_v2_trade_vwap_vs_mid_1t_bps", out.columns)
        self.assertAlmostEqual(out.loc[0, "postopen_v2_trade_vwap_vs_mid_1t_bps"], 0.0)
        self.assertIn("postopen_v2_queue_ask1_replenish_vs_trade_1m", out.columns)
        self.assertIn("postopen_v2_queue_bid_depth10_replenish_vs_trade_1m", out.columns)

    def test_minute_lag_does_not_bridge_a_missing_decision_point(self) -> None:
        frame = _decision_rows().iloc[[0, 2]].reset_index(drop=True)

        out = add_postopen_v2_decision_features(
            frame,
            windows=(1, 2),
            depth_levels=(3, 10),
        )

        self.assertTrue(pd.isna(out.loc[1, "postopen_v2_ask_volume_1_diff_1m"]))
        self.assertAlmostEqual(out.loc[1, "postopen_v2_ask_volume_1_diff_2m"], 20.0)


if __name__ == "__main__":
    unittest.main()
