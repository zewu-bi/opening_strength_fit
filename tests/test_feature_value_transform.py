from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.features import (
    transform_cross_sectional_feature_values,
    transform_mechanismized_feature_values,
    transform_mechanismized_v2_feature_values,
)
from opening_strength_fit.model_torch import _torch_feature_value_frame


class FeatureValueTransformTest(unittest.TestCase):
    def test_cross_sectional_rank_centered_replaces_values_in_place(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04", "2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:32:00"),
                    pd.Timestamp("2022-01-04 09:32:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "spread_bps": [30.0, 10.0, 100.0, 140.0],
                "ask_depth_10": [100.0, 300.0, 500.0, 400.0],
                "label": [0.1, 0.2, 0.3, 0.4],
            }
        )

        out = transform_cross_sectional_feature_values(
            frame,
            columns=("spread_bps", "ask_depth_10"),
            mode="rank_centered",
        )

        self.assertEqual(out.columns.tolist(), frame.columns.tolist())
        self.assertEqual(out["spread_bps"].tolist(), [0.5, 0.0, 0.0, 0.5])
        self.assertEqual(out["ask_depth_10"].tolist(), [0.0, 0.5, 0.5, 0.0])
        self.assertEqual(frame["spread_bps"].tolist(), [30.0, 10.0, 100.0, 140.0])

    def test_torch_feature_value_frame_keeps_feature_names_and_leaves_input_raw(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ"],
                "spread_bps": [30.0, 10.0],
                "target_label": [0.1, 0.2],
                "valid_label": [True, True],
            }
        )

        out = _torch_feature_value_frame(
            frame,
            ["spread_bps"],
            feature_value_transform="cross_sectional_rank_centered",
            group_cols=("date", "decision_target_timestamp"),
            rank_method="average",
            extra_columns=("target_label", "valid_label", "symbol"),
        )

        self.assertIn("spread_bps", out.columns)
        self.assertNotIn("xs_rel_spread_bps_rank_centered", out.columns)
        self.assertEqual(out["spread_bps"].tolist(), [0.5, 0.0])
        self.assertEqual(frame["spread_bps"].tolist(), [30.0, 10.0])

    def test_mechanismized_transform_ranks_share_volume_by_notional(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ"],
                "ask_price_1": [10.00, 100.00],
                "bid_price_1": [9.99, 99.98],
                "mid_price": [9.995, 99.99],
                "spread_abs": [0.01, 0.02],
                "volume_diff_1t": [1000.0, 200.0],
                "trade_vwap_1t": [10.02, 100.01],
                "postopen_v2_ask_depth_concentration_3_10": [0.80, 0.40],
            }
        )

        out = transform_mechanismized_feature_values(
            frame,
            columns=(
                "mid_price",
                "spread_abs",
                "volume_diff_1t",
                "trade_vwap_1t",
                "postopen_v2_ask_depth_concentration_3_10",
            ),
            group_cols=("date", "decision_target_timestamp"),
            rank_method="average",
            tick_size=0.01,
        )

        self.assertEqual(out.columns.tolist(), frame.columns.tolist())
        self.assertEqual(out["volume_diff_1t"].tolist(), [0.0, 0.5])
        self.assertEqual(out["mid_price"].tolist(), [0.5, 0.0])
        self.assertEqual(out["spread_abs"].tolist(), [0.0, 0.5])
        self.assertEqual(out["trade_vwap_1t"].tolist(), [0.5, 0.0])
        self.assertEqual(
            out["postopen_v2_ask_depth_concentration_3_10"].tolist(),
            [0.5, 0.0],
        )
        self.assertEqual(frame["volume_diff_1t"].tolist(), [1000.0, 200.0])

    def test_torch_mechanismized_transform_can_use_reference_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ"],
                "ask_price_1": [10.0, 100.0],
                "volume_diff_1t": [1000.0, 200.0],
                "target_label": [0.1, 0.2],
                "valid_label": [True, True],
            }
        )

        out = _torch_feature_value_frame(
            frame,
            ["volume_diff_1t"],
            feature_value_transform="mechanismized_dimensionless_328",
            group_cols=("date", "decision_target_timestamp"),
            rank_method="average",
            extra_columns=("target_label", "valid_label", "symbol"),
        )

        self.assertIn("volume_diff_1t", out.columns)
        self.assertNotIn("xs_rel_volume_diff_1t_rank_centered", out.columns)
        self.assertEqual(out["volume_diff_1t"].tolist(), [0.0, 0.5])
        self.assertEqual(frame["volume_diff_1t"].tolist(), [1000.0, 200.0])

    def test_mechanismized_v2_keeps_volume_depth_and_notional_semantics(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ"],
                "ask_price_1": [10.00, 100.00],
                "bid_price_1": [9.99, 99.98],
                "mid_price": [9.995, 99.99],
                "volume_diff_1t": [1000.0, 200.0],
                "turnover_diff_1t": [10_000.0, 20_000.0],
                "ask_depth_10": [10_000.0, 100_000.0],
                "ask_volume_1": [1_000.0, 2_000.0],
                "hist_surprise_volume_diff_1t_20d_ratio": [2.0, 2.0],
            }
        )

        out = transform_mechanismized_v2_feature_values(
            frame,
            columns=(
                "volume_diff_1t",
                "turnover_diff_1t",
                "ask_depth_10",
                "ask_volume_1",
            ),
            group_cols=("date", "decision_target_timestamp"),
            cross_sectional_mode="none",
        )

        self.assertEqual(out.columns.tolist(), frame.columns.tolist())
        self.assertAlmostEqual(out["volume_diff_1t"].iloc[0], out["volume_diff_1t"].iloc[1])
        self.assertGreater(out["turnover_diff_1t"].iloc[1], out["turnover_diff_1t"].iloc[0])
        self.assertGreater(out["ask_depth_10"].iloc[1], out["ask_depth_10"].iloc[0])
        self.assertAlmostEqual(out["ask_volume_1"].iloc[0], 0.10, places=6)
        self.assertAlmostEqual(out["ask_volume_1"].iloc[1], 0.02, places=6)

    def test_torch_mechanismized_v2_transform_uses_robust_zscore(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ"],
                "ask_price_1": [10.0, 100.0],
                "volume_diff_1t": [1000.0, 200.0],
                "hist_surprise_volume_diff_1t_20d_ratio": [2.0, 4.0],
                "target_label": [0.1, 0.2],
                "valid_label": [True, True],
            }
        )

        out = _torch_feature_value_frame(
            frame,
            ["volume_diff_1t"],
            feature_value_transform="mechanismized_v2_dimensionless_328",
            group_cols=("date", "decision_target_timestamp"),
            rank_method="average",
            extra_columns=("target_label", "valid_label", "symbol"),
        )

        self.assertIn("volume_diff_1t", out.columns)
        self.assertLess(out["volume_diff_1t"].iloc[0], 0.0)
        self.assertGreater(out["volume_diff_1t"].iloc[1], 0.0)
        self.assertEqual(frame["volume_diff_1t"].tolist(), [1000.0, 200.0])


if __name__ == "__main__":
    unittest.main()
