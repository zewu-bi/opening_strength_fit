from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from opening_strength_fit.training import _load_labeled_pvc_frame
from opening_strength_fit.training import _resolved_data_source


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

            labeled = _load_labeled_pvc_frame(args, config)

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
                        "decision_target_timestamp": pd.Timestamp(
                            "2022-01-04 09:30:00"
                        ),
                        "decision_lag_seconds": 0.0,
                        "label": 0.01,
                        "valid_label": True,
                    },
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:31:01"),
                        "decision_time": "09:31:00",
                        "decision_target_timestamp": pd.Timestamp(
                            "2022-01-04 09:31:00"
                        ),
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

            labeled = _load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled.iloc[0]["decision_time"], "09:31:00")

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

            labeled = _load_labeled_pvc_frame(args, config)

        self.assertEqual(labeled["date"].tolist(), ["2020-09-01", "2021-09-30"])

    def test_labeled_pvc_is_explicit_data_source(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "labeled_pvc"}}

        self.assertEqual(_resolved_data_source(args, config, ""), "labeled_pvc")

    def test_auto_source_prefers_data_labeled_path(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "auto", "labeled_path": "/mnt/cache/labeled.parquet"}}

        self.assertEqual(_resolved_data_source(args, config, ""), "labeled_pvc")


if __name__ == "__main__":
    unittest.main()
