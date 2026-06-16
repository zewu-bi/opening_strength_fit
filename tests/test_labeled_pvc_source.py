from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opening_strength_fit.training_data import load_labeled_pvc_frame, resolve_data_source


class LabeledPvcSourceTest(unittest.TestCase):
    def test_labeled_pvc_source_reads_data_labeled_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                        "label": 0.01,
                        "valid_label": True,
                    }
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)
        self.assertIn("label", labeled.columns)
        self.assertEqual(str(labeled.loc[0, "symbol"]), "000001.SZ")

    def test_labeled_pvc_source_filters_configured_decision_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                        "decision_time": "09:30:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "valid_label": True,
                    },
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:01"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": 1.0,
                        "label": 0.02,
                        "valid_label": True,
                    },
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled.iloc[0]["decision_time"], "09:31:00")

    def test_labeled_pvc_postopen_features_are_built_before_decision_filter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            rows = []
            for minute, ask_volume in enumerate((100.0, 125.0)):
                rows.append(
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00")
                        + pd.Timedelta(minutes=minute),
                        "decision_time": f"09:{30 + minute:02d}:00",
                        "decision_target_timestamp": pd.Timestamp(
                            f"2022-01-04 09:{30 + minute:02d}:00"
                        ),
                        "decision_lag_seconds": 0.0,
                        "ask_volume_1": ask_volume,
                        "bid_volume_1": 200.0,
                        "ask_price_1": 10.01,
                        "bid_price_1": 9.99,
                        "ask_depth_10": ask_volume,
                        "bid_depth_10": 200.0,
                        "depth_imbalance_10": 0.1,
                        "spread_bps": 20.0,
                        "mid_price": 10.0,
                        "volume": 1000.0 + minute,
                        "turnover": 10000.0 + minute,
                        "label": 0.01,
                        "valid_label": True,
                    }
                )
            pd.DataFrame(rows).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {
                    "include_postopen_v2": True,
                    "postopen_v2_windows": [1],
                    "postopen_v2_depth_levels": [10],
                },
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)
        self.assertAlmostEqual(
            labeled.iloc[0]["postopen_v2_ask_volume_1_diff_1m"],
            25.0,
        )
        self.assertIn("volume", labeled.columns)

    def test_labeled_pvc_relative_features_are_built_after_decision_filter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            rows = []
            for symbol, spread, lag in (
                ("000001.SZ", 10.0, 0.0),
                ("000002.SZ", 20.0, 0.0),
                ("000003.SZ", 1000.0, 10.0),
            ):
                rows.append(
                    {
                        "date": "2022-01-04",
                        "symbol": symbol,
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": lag,
                        "spread_bps": spread,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                    }
                )
            pd.DataFrame(rows).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {
                    "include_cross_sectional_relative": True,
                    "cross_sectional_relative_columns": ["spread_bps"],
                    "cross_sectional_relative_modes": ["rank_centered"],
                },
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(labeled["symbol"].tolist(), ["000001.SZ", "000002.SZ"])
        self.assertEqual(labeled["xs_rel_spread_bps_rank_centered"].tolist(), [0.0, 0.5])

    def test_labeled_pvc_projects_columns_when_feature_includes_are_configured(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "keep_feature": 1.0,
                        "unused_heavy_feature": 999.0,
                    }
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {"include_feature_columns": ["keep_feature"]},
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertIn("keep_feature", labeled.columns)
        self.assertIn("target_label", labeled.columns)
        self.assertNotIn("unused_heavy_feature", labeled.columns)

    def test_labeled_pvc_projection_keeps_postopen_transform_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            rows = []
            for minute, ask_volume in enumerate((100.0, 125.0)):
                rows.append(
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00")
                        + pd.Timedelta(minutes=minute),
                        "decision_time": f"09:{30 + minute:02d}:00",
                        "decision_target_timestamp": pd.Timestamp(
                            f"2022-01-04 09:{30 + minute:02d}:00"
                        ),
                        "decision_lag_seconds": 0.0,
                        "ask_volume_1": ask_volume,
                        "bid_volume_1": 200.0,
                        "ask_price_1": 10.01,
                        "bid_price_1": 9.99,
                        "ask_depth_10": ask_volume,
                        "bid_depth_10": 200.0,
                        "depth_imbalance_10": 0.1,
                        "spread_bps": 20.0,
                        "mid_price": 10.0,
                        "volume": 1000.0 + minute,
                        "turnover": 10000.0 + minute,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "unused_heavy_feature": 999.0,
                    }
                )
            pd.DataFrame(rows).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {
                    "include_postopen_v2": True,
                    "postopen_v2_windows": [1],
                    "postopen_v2_depth_levels": [10],
                    "include_feature_prefixes": ["postopen_v2_"],
                },
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertAlmostEqual(
            labeled.iloc[0]["postopen_v2_ask_volume_1_diff_1m"],
            25.0,
        )
        self.assertNotIn("unused_heavy_feature", labeled.columns)

    def test_labeled_pvc_projection_keeps_cross_sectional_relative_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "raw_depth_state": 10.0,
                        "unused_heavy_feature": 999.0,
                    },
                    {
                        "date": "2022-01-04",
                        "symbol": "000002.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.02,
                        "target_label": 0.3,
                        "valid_label": True,
                        "raw_depth_state": 30.0,
                        "unused_heavy_feature": 999.0,
                    },
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {
                    "include_cross_sectional_relative": True,
                    "cross_sectional_relative_columns": ["raw_depth_state"],
                    "cross_sectional_relative_modes": ["demean"],
                    "include_feature_prefixes": ["xs_rel_"],
                },
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertIn("xs_rel_raw_depth_state_demean", labeled.columns)
        self.assertEqual(labeled["xs_rel_raw_depth_state_demean"].tolist(), [-10.0, 10.0])
        self.assertNotIn("unused_heavy_feature", labeled.columns)

    def test_labeled_pvc_projection_keeps_price_scale_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "ask_price_1": 2.00,
                        "bid_price_1": 1.99,
                        "ask_price_2": 2.01,
                        "bid_price_2": 1.98,
                        "spread_bps": 50.0,
                        "unused_heavy_feature": 999.0,
                    }
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
                "sample": {
                    "mode": "decision_points",
                    "decision_times": ["09:31:00"],
                    "decision_max_lag_seconds": 5,
                },
                "features": {
                    "include_price_scale_features": True,
                    "include_feature_prefixes": ["price_scale_"],
                },
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertIn("price_scale_tick_bps", labeled.columns)
        self.assertIn("price_scale_ask_gap_2_ticks", labeled.columns)
        self.assertAlmostEqual(labeled.iloc[0]["price_scale_tick_bps"], 50.0)
        self.assertNotIn("unused_heavy_feature", labeled.columns)

    def test_labeled_pvc_directory_projects_each_file_before_concat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2020-09-01",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2020-09-01 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2020-09-01 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "keep_feature": 1.0,
                        "unused_heavy_feature": 999.0,
                    }
                ]
            ).to_parquet(root / "part_2020.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2021-09-30",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2021-09-30 09:31:00"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp("2021-09-30 09:31:00"),
                        "decision_lag_seconds": 0.0,
                        "label": 0.03,
                        "target_label": 0.4,
                        "valid_label": True,
                        "keep_feature": 2.0,
                        "unused_heavy_feature": 999.0,
                    }
                ]
            ).to_parquet(root / "part_2021.parquet", index=False)

            args = argparse.Namespace(
                labeled_input=None,
                split_mode=None,
                rolling_monthly=True,
                rolling_annual=False,
                train_months=12,
                test_start_month="2021-09",
                test_end_month="2021-09",
            )
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(root)},
                "universe": {"enabled": False},
                "features": {"include_feature_columns": ["keep_feature"]},
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(labeled["date"].tolist(), ["2020-09-01", "2021-09-30"])
        self.assertEqual(labeled["keep_feature"].tolist(), [1.0, 2.0])
        self.assertNotIn("unused_heavy_feature", labeled.columns)

    def test_labeled_pvc_can_downcast_float64_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "label": 0.01,
                        "target_label": 0.2,
                        "valid_label": True,
                        "keep_feature": 1.0,
                    }
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "labeled_path": str(path),
                    "downcast_float32": True,
                },
                "universe": {"enabled": False},
                "features": {"include_feature_columns": ["keep_feature"]},
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(str(labeled["keep_feature"].dtype), "float32")
        self.assertEqual(str(labeled["target_label"].dtype), "float32")

    def test_labeled_pvc_rolling_monthly_reads_needed_date_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2020-08-31",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2020-08-31 09:31:00"),
                        "label": 0.01,
                        "valid_label": True,
                    },
                    {
                        "date": "2020-09-01",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2020-09-01 09:31:00"),
                        "label": 0.02,
                        "valid_label": True,
                    },
                    {
                        "date": "2021-09-30",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2021-09-30 09:31:00"),
                        "label": 0.03,
                        "valid_label": True,
                    },
                    {
                        "date": "2021-10-01",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2021-10-01 09:31:00"),
                        "label": 0.04,
                        "valid_label": True,
                    },
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(
                labeled_input=None,
                split_mode=None,
                rolling_monthly=True,
                rolling_annual=False,
                train_months=12,
                test_start_month="2021-09",
                test_end_month="2021-09",
            )
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(labeled["date"].tolist(), ["2020-09-01", "2021-09-30"])

    def test_labeled_pvc_is_explicit_data_source(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "labeled_pvc"}}

        self.assertEqual(resolve_data_source(args, config, ""), "labeled_pvc")

    def test_auto_source_prefers_data_labeled_path(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "auto", "labeled_path": "/mnt/cache/labeled.parquet"}}

        self.assertEqual(resolve_data_source(args, config, ""), "labeled_pvc")


if __name__ == "__main__":
    unittest.main()
