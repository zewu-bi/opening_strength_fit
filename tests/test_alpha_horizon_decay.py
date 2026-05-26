from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_alpha_horizon_decay import (  # noqa: E402
    HorizonSpec,
    build_summary_tables,
    compute_sampled_intraday_labels,
    compute_tick_horizon_labels,
    horizon_specs,
    load_sample_context,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "entry_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "buy_price": 10.0,
                "prediction": 0.9,
                "branch": "Universe",
            },
            {
                "date": "2022-01-04",
                "symbol": "000002.SZ",
                "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "entry_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "buy_price": 10.0,
                "prediction": 0.1,
                "branch": "Universe",
            },
        ]
    )


def _tick_frame() -> pd.DataFrame:
    rows = []
    for date in ("2022-01-04", "2022-01-05"):
        for symbol, price_offset in (("000001.SZ", 0.0), ("000002.SZ", -0.2)):
            for seconds, volume, turnover, price in (
                (0, 0.0, 0.0, 10.00),
                (30, 100.0, 1_010.0, 10.10),
                (60, 200.0, 2_020.0, 10.10),
                (90, 300.0, 3_050.0, 10.30),
                (120, 400.0, 4_080.0, 10.30),
                (300, 500.0, 5_120.0, 10.40),
                (360, 600.0, 6_170.0, 10.50),
                (19_800, 700.0, 7_255.0, 10.85),
            ):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(f"{date} 09:30:00")
                        + pd.Timedelta(seconds=seconds),
                        "volume": volume,
                        "turnover": turnover + price_offset * volume,
                        "last_price": price + price_offset,
                    }
                )
    return pd.DataFrame(rows)


class AlphaHorizonDecayTest(unittest.TestCase):
    def test_horizon_specs_accept_min_alias(self) -> None:
        specs = horizon_specs(["30s", "5min", "next open"])
        self.assertEqual([spec.name for spec in specs], ["30s", "5m", "next_open"])
        self.assertEqual([spec.seconds for spec in specs], [30, 300, None])

    def test_compute_intraday_horizon_labels_from_ticks(self) -> None:
        horizons = [
            HorizonSpec(name="60s", label="60s", seconds=60),
            HorizonSpec(name="5m", label="5m", seconds=300),
            HorizonSpec(name="next_open", label="next open", seconds=None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            tick_path = Path(directory) / "ticks.parquet"
            _tick_frame().to_parquet(tick_path, index=False)
            labels = compute_tick_horizon_labels(
                _prediction_frame(),
                tick_path,
                horizons,
                volume_col="volume",
                turnover_col="turnover",
                volume_unit_multiplier=1.0,
                sell_window_seconds=60,
                fee_bps=0.0,
                price_col="last_price",
                open_time="09:30:00",
                close_time="15:00:00",
                max_future_gap_seconds=None,
                max_price_gap_seconds=None,
            )

        self.assertAlmostEqual(labels.loc[0, "alpha_return_60s"], 0.03)
        self.assertAlmostEqual(labels.loc[0, "alpha_return_5m"], 0.05)
        self.assertAlmostEqual(labels.loc[0, "alpha_return_next_open"], 0.0)

    def test_compute_sampled_intraday_decay_from_decision_rows(self) -> None:
        frame = _prediction_frame()
        context = pd.DataFrame(
            [
                {
                    "date": "2022-01-04",
                    "symbol": "000001.SZ",
                    "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                    "mid_price": 10.20,
                },
                {
                    "date": "2022-01-04",
                    "symbol": "000002.SZ",
                    "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                    "mid_price": 9.90,
                },
            ]
        )
        context = load_sample_context(context, "", exit_price_col="mid_price")
        labels = compute_sampled_intraday_labels(
            frame,
            context,
            [HorizonSpec(name="1m", label="1m", seconds=60)],
            exit_price_col="mid_price",
            fee_bps=0.0,
            target_end_seconds=None,
        )

        self.assertAlmostEqual(labels.loc[0, "alpha_return_1m"], 0.02)
        self.assertAlmostEqual(labels.loc[1, "alpha_return_1m"], -0.01)

    def test_summary_uses_top_score_alpha_return(self) -> None:
        frame = _prediction_frame()
        frame["alpha_return_60s"] = [0.02, -0.01]
        summary, buckets, missing = build_summary_tables(
            frame,
            horizons=[HorizonSpec(name="60s", label="60s", seconds=60)],
            top_n=1,
            score_bins=2,
            group_cols=("date", "decision_target_timestamp"),
            allow_missing=False,
        )

        self.assertEqual(missing, [])
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.loc[0, "mean_alpha_return_bps"], 200.0)
        self.assertFalse(buckets.empty)


if __name__ == "__main__":
    unittest.main()
