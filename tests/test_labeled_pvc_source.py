from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import opening_strength_fit.training_data as training_data
from opening_strength_fit.training_data import (
    _labeled_pvc_read_columns,
    load_labeled_pvc_frame,
    resolve_data_source,
)


class LabeledPvcSourceTest(unittest.TestCase):
    def test_model_ready_split_uses_sampled_alignment_without_generic_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            label_root = root / "labels"
            feature_root.mkdir()
            label_root.mkdir()
            keys = {
                "date": ["2022-01-04", "2022-01-05"],
                "symbol": ["000001.SZ", "000002.SZ"],
                "decision_target_timestamp": pd.to_datetime(
                    ["2022-01-04 09:31:00", "2022-01-05 09:31:00"]
                ),
            }
            pd.DataFrame({**keys, "feature_a": pd.Series([1.0, 2.0], dtype="float32")}).to_parquet(
                feature_root / "features.parquet", index=False
            )
            pd.DataFrame(
                {
                    **keys,
                    "label_short": [0.01, 0.02],
                    "label_next_close": [0.03, 0.04],
                    "target_label": [0.5, 1.0],
                }
            ).to_parquet(label_root / "labels.parquet", index=False)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "feature_path": str(feature_root),
                    "label_path": str(label_root),
                    "trusted_model_ready_split": True,
                    "downcast_float32": True,
                },
                "universe": {"enabled": True},
                "model": {"target_col": "target_label"},
            }

            with (
                mock.patch.object(
                    training_data,
                    "_normalize_dataset_join_keys",
                    side_effect=AssertionError("model-ready keys must not be normalized"),
                ),
                mock.patch.object(
                    training_data,
                    "filter_labeled_frame",
                    side_effect=AssertionError("model-ready input must not be cleaned again"),
                ),
            ):
                labeled = load_labeled_pvc_frame(argparse.Namespace(labeled_input=None), config)

        self.assertEqual(len(labeled), 2)
        self.assertNotIn("timestamp", labeled.columns)
        self.assertNotIn("decision_time", labeled.columns)
        self.assertEqual(str(labeled["feature_a"].dtype), "float32")

    def test_model_ready_split_rejects_sampled_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            label_root = root / "labels"
            feature_root.mkdir()
            label_root.mkdir()
            timestamp = pd.to_datetime(["2022-01-04 09:31:00"])
            pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000001.SZ"],
                    "decision_target_timestamp": timestamp,
                    "feature_a": [1.0],
                }
            ).to_parquet(feature_root / "features.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000002.SZ"],
                    "decision_target_timestamp": timestamp,
                    "label_short": [0.01],
                    "label_next_close": [0.03],
                    "target_label": [0.5],
                }
            ).to_parquet(label_root / "labels.parquet", index=False)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "feature_path": str(feature_root),
                    "label_path": str(label_root),
                    "trusted_model_ready_split": True,
                },
                "model": {"target_col": "target_label"},
            }

            with self.assertRaisesRegex(SystemExit, "sampled key mismatch"):
                load_labeled_pvc_frame(argparse.Namespace(labeled_input=None), config)

    def test_labeled_pvc_joins_separate_feature_and_label_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features" / "year=2022"
            label_root = root / "labels" / "year=2022"
            feature_root.mkdir(parents=True)
            label_root.mkdir(parents=True)
            keys = {
                "date": ["2022-01-04", "2022-01-04"],
                "symbol": ["000001.SZ", "000002.SZ"],
                "decision_target_timestamp": pd.to_datetime(
                    ["2022-01-04 09:31:00", "2022-01-04 09:31:00"]
                ),
            }
            pd.DataFrame({**keys, "feature_a": [1.0, 2.0]}).to_parquet(
                feature_root / "features.parquet", index=False
            )
            pd.DataFrame(
                {
                    **keys,
                    "label_short": [0.01, 0.02],
                    "label_next_close": [0.03, 0.04],
                    "target_label": [0.5, 1.0],
                }
            ).to_parquet(label_root / "labels.parquet", index=False)
            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "feature_path": str(root / "features"),
                    "label_path": str(root / "labels"),
                    "downcast_float32": True,
                },
                "universe": {"enabled": False},
                "model": {"target_col": "target_label"},
            }

            with mock.patch.object(
                pd.DataFrame,
                "merge",
                side_effect=AssertionError("same-order input should not use a wide merge"),
            ):
                labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 2)
        self.assertEqual(labeled["feature_a"].tolist(), [1.0, 2.0])
        np.testing.assert_allclose(labeled["label"], [0.01, 0.02])
        np.testing.assert_allclose(labeled["gross_label"], [0.01, 0.02])
        self.assertTrue(labeled["valid_label"].all())

    def test_labeled_pvc_key_joins_when_label_order_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            label_root = root / "labels"
            feature_root.mkdir()
            label_root.mkdir()
            timestamps = pd.to_datetime(["2022-01-04 09:31:00"] * 2)
            pd.DataFrame(
                {
                    "date": ["2022-01-04"] * 2,
                    "symbol": ["000001.SZ", "000002.SZ"],
                    "decision_target_timestamp": timestamps,
                    "feature_a": [1.0, 2.0],
                }
            ).to_parquet(feature_root / "features.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2022-01-04"] * 2,
                    "symbol": ["000002.SZ", "000001.SZ"],
                    "decision_target_timestamp": timestamps,
                    "label_short": [0.02, 0.01],
                    "label_next_close": [0.04, 0.03],
                    "target_label": [1.0, 0.5],
                }
            ).to_parquet(label_root / "labels.parquet", index=False)
            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "feature_path": str(feature_root),
                    "label_path": str(label_root),
                },
                "universe": {"enabled": False},
                "model": {"target_col": "target_label"},
            }

            labeled = load_labeled_pvc_frame(args, config)

        assert labeled["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
        np.testing.assert_allclose(labeled["label_short"], [0.01, 0.02])

    def test_labeled_pvc_rejects_split_dataset_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            label_root = root / "labels"
            feature_root.mkdir()
            label_root.mkdir()
            pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000001.SZ"],
                    "decision_target_timestamp": pd.to_datetime(["2022-01-04 09:31:00"]),
                    "feature_a": [1.0],
                }
            ).to_parquet(feature_root / "features.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000002.SZ"],
                    "decision_target_timestamp": pd.to_datetime(["2022-01-04 09:31:00"]),
                    "label_short": [0.01],
                    "label_next_close": [0.03],
                    "target_label": [0.5],
                }
            ).to_parquet(label_root / "labels.parquet", index=False)
            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {
                    "source": "labeled_pvc",
                    "feature_path": str(feature_root),
                    "label_path": str(label_root),
                },
                "universe": {"enabled": False},
                "model": {"target_col": "target_label"},
            }

            with self.assertRaisesRegex(SystemExit, "key coverage mismatch"):
                load_labeled_pvc_frame(args, config)

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

    def test_labeled_pvc_directory_deduplicates_compatibility_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            path = root / "opening_2025_label_v4.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2025-01-02",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2025-01-02 09:30:00"),
                        "label": 0.01,
                        "valid_label": True,
                    }
                ]
            ).to_parquet(path, index=False)
            (root / "opening_2025_old_name.parquet").symlink_to(path.name)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(root)},
                "universe": {"enabled": False},
            }

            labeled = load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)

    def test_labeled_pvc_mechanismized_transform_reads_reference_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                        "ask_price_1": 10.01,
                        "volume": 1000.0,
                        "turnover": 10_000.0,
                        "market_cap": 100_000_000.0,
                        "total_shares": 10_000_000.0,
                        "label": 0.01,
                        "valid_label": True,
                    }
                ]
            ).to_parquet(path, index=False)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "features": {
                    "include_feature_columns": ["ask_price_1"],
                    "feature_value_transform": "mechanismized_v3_dimensionless_328",
                },
            }

            columns = _labeled_pvc_read_columns(path, config)

        self.assertIsNotNone(columns)
        assert columns is not None
        self.assertIn("ask_price_1", columns)
        self.assertIn("volume", columns)
        self.assertIn("turnover", columns)
        self.assertIn("market_cap", columns)
        self.assertIn("total_shares", columns)

    def test_labeled_pvc_multi_denominator_features_read_simple_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                        "volume": 1_000.0,
                        "turnover": 10_000.0,
                        "float_market_cap": 100_000_000.0,
                        "float_shares": 10_000_000.0,
                        "ask_depth_10": 100.0,
                        "label": 0.01,
                        "valid_label": True,
                    }
                ]
            ).to_parquet(path, index=False)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "features": {
                    "include_feature_prefixes": ["multi_den_ratio_"],
                    "include_multi_denominator_features": True,
                    "multi_denominator_depth_columns": ["ask_depth_10"],
                },
            }

            columns = _labeled_pvc_read_columns(path, config)

        self.assertIsNotNone(columns)
        assert columns is not None
        self.assertIn("volume", columns)
        self.assertIn("turnover", columns)
        self.assertIn("float_market_cap", columns)
        self.assertIn("float_shares", columns)
        self.assertIn("ask_depth_10", columns)

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

    def test_labeled_pvc_downcast_is_invariant_to_file_partitioning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_path = root / "single.parquet"
            split_path = root / "split"
            split_path.mkdir()
            frame = pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp(f"2022-01-04 09:{minute}:00"),
                        "decision_time": f"09:{minute}:00",
                        "decision_target_timestamp": pd.Timestamp(f"2022-01-04 09:{minute}:00"),
                        "decision_lag_seconds": 0.0,
                        "ask_volume_1": ask_volume,
                        "label": 0.01,
                        "valid_label": True,
                    }
                    for minute, ask_volume in ((30, 100_000_000.0), (31, 100_000_001.0))
                ]
            )
            frame.to_parquet(single_path, index=False)
            frame.iloc[[0]].to_parquet(split_path / "part_1.parquet", index=False)
            frame.iloc[[1]].to_parquet(split_path / "part_2.parquet", index=False)

            def load(path: Path) -> pd.DataFrame:
                return load_labeled_pvc_frame(
                    argparse.Namespace(labeled_input=None),
                    {
                        "data": {
                            "source": "labeled_pvc",
                            "labeled_path": str(path),
                            "downcast_float32": True,
                        },
                        "universe": {"enabled": False},
                        "sample": {
                            "mode": "decision_points",
                            "decision_times": ["09:31:00"],
                            "decision_max_lag_seconds": 5,
                        },
                        "features": {
                            "include_postopen_v2": True,
                            "postopen_v2_windows": [1],
                        },
                    },
                )

            single = load(single_path)
            split = load(split_path)

        column = "postopen_v2_ask_volume_1_diff_1m"
        self.assertEqual(str(single[column].dtype), "float32")
        self.assertEqual(str(split[column].dtype), "float32")
        self.assertEqual(single[column].tolist(), [1.0])
        self.assertEqual(split[column].tolist(), [1.0])

    def test_labeled_pvc_downcast_does_not_copy_unchanged_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "value": pd.Series([1.0, 2.0], dtype="float64"),
                "count": pd.Series([1, 2], dtype="int64"),
            }
        )

        result = training_data._downcast_labeled_pvc_frame(
            frame,
            {"data": {"downcast_float32": True}},
        )

        self.assertEqual(str(frame["value"].dtype), "float64")
        self.assertEqual(str(result["value"].dtype), "float32")
        self.assertTrue(np.shares_memory(frame["count"].to_numpy(), result["count"].to_numpy()))

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

    def test_labeled_pvc_directory_builds_relative_features_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            for part, symbol, spread in (
                (1, "000001.SZ", 10.0),
                (2, "000002.SZ", 30.0),
            ):
                pd.DataFrame(
                    [
                        {
                            "date": "2022-01-04",
                            "symbol": symbol,
                            "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                            "decision_time": "09:31:00",
                            "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                            "decision_lag_seconds": 0.0,
                            "spread_bps": spread,
                            "label": 0.01,
                            "valid_label": True,
                        }
                    ]
                ).to_parquet(root / f"part_{part}.parquet", index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(root)},
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
            }

            labeled = load_labeled_pvc_frame(args, config)

        relative = labeled.set_index("symbol")["xs_rel_spread_bps_rank_centered"]
        self.assertEqual(relative.to_dict(), {"000001.SZ": 0.0, "000002.SZ": 0.5})

    def test_labeled_pvc_directory_keeps_historical_rolling_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir()
            for file_name, rows in (
                ("part_1.parquet", (("2022-01-03", 10.0), ("2022-01-04", 12.0))),
                ("part_2.parquet", (("2022-01-05", 14.0),)),
            ):
                pd.DataFrame(
                    [
                        {
                            "date": date,
                            "symbol": "000001.SZ",
                            "timestamp": pd.Timestamp(f"{date} 09:31:00"),
                            "decision_target_timestamp": pd.Timestamp(f"{date} 09:31:00"),
                            "volume_diff_1t": value,
                            "label": 0.01,
                            "valid_label": True,
                        }
                        for date, value in rows
                    ]
                ).to_parquet(root / file_name, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(root)},
                "universe": {"enabled": False},
                "features": {
                    "include_historical_same_minute_surprise": True,
                    "historical_surprise_columns": ["volume_diff_1t"],
                    "historical_surprise_windows": [2],
                    "historical_surprise_min_periods": 2,
                    "historical_surprise_modes": ["ratio"],
                },
            }

            labeled = load_labeled_pvc_frame(args, config)

        surprise = labeled.set_index("date")["hist_surprise_volume_diff_1t_2d_ratio"]
        self.assertTrue(surprise.loc["2022-01-03":"2022-01-04"].isna().all())
        self.assertAlmostEqual(surprise.loc["2022-01-05"], 14.0 / 11.0)

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
