from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from opening_strength_fit.rolling import DateSplit
from opening_strength_fit.reports import metrics_by_year_from_windows
from opening_strength_fit.stock_pool import (
    apply_stock_pool_cli_overrides,
    load_stock_pool,
    parse_stock_pool_location,
    stock_pool_membership_mask,
)
from opening_strength_fit.training import _evaluation_settings, _fit_predict_split


class StockPoolTest(unittest.TestCase):
    def test_parse_ceph_pool_location(self) -> None:
        location = parse_stock_pool_location("lml.bzw@ssd/data/pool_S.parquet")

        self.assertEqual(location.bucket, "lml.bzw")
        self.assertEqual(location.key, "data/pool_S.parquet")
        self.assertEqual(
            location.endpoint_url,
            "http://ceph-s3-ssd.prod.highfortfunds.com",
        )
        self.assertTrue(location.is_remote)

    def test_cli_pool_shortcut_defaults_to_selection_mask(self) -> None:
        args = argparse.Namespace(
            pool="S",
            pool_path=None,
            pool_date_lag_sessions=1,
            pool_filter_train=False,
            pool_add_feature=False,
        )

        config = apply_stock_pool_cli_overrides({}, args)

        self.assertEqual(
            config["stock_pool"]["path"],
            "lml.bzw@ssd/data/pool_S.parquet",
        )
        self.assertEqual(config["stock_pool"]["name"], "pool_S")
        self.assertTrue(config["stock_pool"]["enabled"])
        self.assertFalse(config["stock_pool"]["filter_train"])
        self.assertTrue(config["stock_pool"]["filter_selection"])
        self.assertEqual(config["stock_pool"]["date_lag_sessions"], 1)

    def test_stock_pool_membership_supports_session_lag(self) -> None:
        pool = pd.DataFrame(
            {
                "000001.SZ": [True, False],
                "000002.SZ": [False, True],
            },
            index=pd.Index(["2022-01-03", "2022-01-04"], name="date"),
        )
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04", "2022-01-04"],
                "symbol": ["000001.SZ", "000002.SZ", "999999.SZ"],
            }
        )

        same_day = stock_pool_membership_mask(frame, pool)
        lagged = stock_pool_membership_mask(frame, pool, date_lag_sessions=1)

        self.assertEqual(same_day.tolist(), [False, True, False])
        self.assertEqual(lagged.tolist(), [True, False, False])

    def test_load_local_stock_pool_normalizes_date_and_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.parquet"
            pd.DataFrame(
                {
                    "date": ["2022-01-03", "2022-01-04"],
                    "000001.sz": [1, 0],
                    " 000002.SH ": [0, 1],
                }
            ).to_parquet(path, index=False)

            pool = load_stock_pool(path)

        self.assertEqual(pool.index.tolist(), ["2022-01-03", "2022-01-04"])
        self.assertEqual(pool.columns.tolist(), ["000001.SZ", "000002.SH"])
        self.assertEqual(pool.dtypes.astype(str).unique().tolist(), ["bool"])

    def test_training_top_selection_can_be_limited_to_stock_pool(self) -> None:
        labeled = pd.DataFrame(
            {
                "date": [
                    "2022-01-03",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-04",
                    "2022-01-05",
                    "2022-01-05",
                ],
                "symbol": [
                    "000001.SZ",
                    "000002.SZ",
                    "000001.SZ",
                    "000002.SZ",
                    "000001.SZ",
                    "000002.SZ",
                ],
                "timestamp": pd.to_datetime(
                    [
                        "2022-01-03 09:31:00",
                        "2022-01-03 09:31:00",
                        "2022-01-04 09:31:00",
                        "2022-01-04 09:31:00",
                        "2022-01-05 09:31:00",
                        "2022-01-05 09:31:00",
                    ]
                ),
                "decision_target_timestamp": pd.to_datetime(
                    [
                        "2022-01-03 09:31:00",
                        "2022-01-03 09:31:00",
                        "2022-01-04 09:31:00",
                        "2022-01-04 09:31:00",
                        "2022-01-05 09:31:00",
                        "2022-01-05 09:31:00",
                    ]
                ),
                "feature": [0.0, 1.0, 0.0, 1.0, 0.2, 2.0],
                "label": [0.0, 1.0, 0.0, 1.0, 0.2, 2.0],
                "valid_label": [True] * 6,
            }
        )
        pool = pd.DataFrame(
            {
                "000001.SZ": [True],
                "000002.SZ": [False],
            },
            index=pd.Index(["2022-01-05"], name="date"),
        )
        config = {
            "model": {"name": "ridge", "alpha": 1.0},
            "evaluation": {
                "score_bins": 2,
                "bucket_mode": "cross_section",
                "selection_mode": "cross_section",
                "ic_mode": "cross_section",
                "top_n": 1,
            },
            "stock_pool": {
                "enabled": True,
                "name": "pool_test",
                "filter_selection": True,
                "membership_col": "pool_member",
            },
        }
        args = argparse.Namespace(feature_limit=None, top_n=None)

        with tempfile.TemporaryDirectory() as directory:
            predictions, metrics_row, _ = _fit_predict_split(
                labeled=labeled,
                split=DateSplit(
                    train_dates=["2022-01-03", "2022-01-04"],
                    test_dates=["2022-01-05"],
                ),
                run_name="stock_pool_test",
                output_dir=Path(directory),
                args=args,
                config=config,
                alpha=1.0,
                evaluation_settings=_evaluation_settings(config, args),
                stock_pool=pool,
            )

        self.assertIn("pool_member", predictions.columns)
        self.assertEqual(predictions["pool_member"].tolist(), [1, 0])
        self.assertEqual(metrics_row["top_score_trades"], 1)
        self.assertEqual(metrics_row["top_score_stock_pool_candidate_rows"], 1)
        self.assertTrue(metrics_row["stock_pool_filter_selection"])

    def test_yearly_metrics_aggregate_stock_pool_candidate_columns(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "run_id": "pool_run",
                    "test_year": 2022,
                    "test_month": "2022-01",
                    "train_start_date": "2021-01-01",
                    "train_end_date": "2021-12-31",
                    "test_start_date": "2022-01-04",
                    "test_end_date": "2022-01-31",
                    "test_rows": 10,
                    "top_score_stock_pool_candidate_rows": 4,
                    "top_score_stock_pool_candidate_dates": 2,
                    "top_score_stock_pool_candidate_symbols": 3,
                    "top_score_stock_pool_candidate_row_fraction": 0.4,
                },
                {
                    "run_id": "pool_run",
                    "test_year": 2022,
                    "test_month": "2022-02",
                    "train_start_date": "2021-01-01",
                    "train_end_date": "2022-01-31",
                    "test_start_date": "2022-02-01",
                    "test_end_date": "2022-02-28",
                    "test_rows": 30,
                    "top_score_stock_pool_candidate_rows": 18,
                    "top_score_stock_pool_candidate_dates": 3,
                    "top_score_stock_pool_candidate_symbols": 5,
                    "top_score_stock_pool_candidate_row_fraction": 0.6,
                },
            ]
        )

        yearly = metrics_by_year_from_windows(metrics)

        self.assertEqual(int(yearly.loc[0, "top_score_stock_pool_candidate_rows"]), 22)
        self.assertEqual(int(yearly.loc[0, "top_score_stock_pool_candidate_dates"]), 5)
        self.assertEqual(int(yearly.loc[0, "top_score_stock_pool_candidate_symbols"]), 5)
        self.assertAlmostEqual(
            float(yearly.loc[0, "top_score_stock_pool_candidate_row_fraction"]),
            0.55,
        )


if __name__ == "__main__":
    unittest.main()
