from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

from opening_strength_fit.labels import build_trade_labels

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_lgbm_delay_replays import DEFAULT_SCENARIOS, SCENARIOS  # noqa: E402
from run_lgbm_delay_replays import validate_context_delay  # noqa: E402
from run_lgbm_delay_replays import validate_prediction_interface  # noqa: E402
from run_opening_intraday_backtest import apply_static_constraints  # noqa: E402


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


def _cpu_lgbm_prediction_row(*, delay: int = 1) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2022-01-04",
        "symbol": "000001.SZ",
        "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
        "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
        "decision_lag_seconds": 0.0,
        "entry_delay_ticks": float(delay),
        "entry_delay_seconds": float(delay * 3),
        "entry_max_tick_gap_seconds": 3.0,
        "status": "TRADE",
        "entry_status": "TRADE",
        "label": 0.01,
        "valid_label": True,
        "prediction": 0.02,
        "spread_bps": 10.0,
        "ask_price_1": 10.01,
        "bid_price_1": 9.99,
        "ask_volume_1": 10_000.0,
        "bid_volume_1": 10_000.0,
        "turnover_diff_30t": 1_000_000.0,
    }
    for level in range(1, 11):
        row[f"entry_ask_price_{level}"] = 10.01 + level * 0.01
        row[f"entry_ask_volume_{level}"] = 10_000.0
    return row


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

    def test_replay_entry_freshness_uses_tick_gap_not_total_delay(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2022-01-04",
                    "symbol": "000001.SZ",
                    "entry_delay_seconds": 6.0,
                    "entry_max_tick_gap_seconds": 3.0,
                }
            ]
        )

        constrained = apply_static_constraints(
            frame,
            run_label="test",
            top_n=20,
            fee_bps=0.0,
            slippage_bps=0.0,
            tradable_statuses=set(),
            require_entry_status=False,
            max_decision_lag_seconds=None,
            max_entry_tick_gap_seconds=5.0,
            max_spread_bps=None,
            min_limit_up_room_bps=None,
            min_ask_volume_1=None,
            min_bid_volume_1=None,
            capacity_notional_col="",
            capacity_volume_col="",
            capacity_price_col="ask_price_1",
            min_capacity_notional=0.0,
            max_participation_rate=0.0,
            capital_per_cycle=0.0,
            ask_depth_levels=0,
            ask_depth_participation_rate=1.0,
            ask_depth_fill_mode="filter",
            allow_decision_depth_fallback=False,
            max_symbol_weight=0.0,
            missing_policy="error",
        )

        self.assertEqual(len(constrained), 1)

    def test_lgbm_replay_rejects_labeled_context_for_wrong_delay(self) -> None:
        context = pd.DataFrame(
            {
                "date": ["2022-01-04"],
                "symbol": ["000001.SZ"],
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:30:00")],
                "entry_delay_ticks": [1.0],
            }
        )

        with self.assertRaises(SystemExit) as caught:
            validate_context_delay(context, delay="delay2")

        self.assertIn("context delay mismatch", str(caught.exception))

    def test_lgbm_replay_requires_context_delay_metadata(self) -> None:
        context = pd.DataFrame(
            {
                "date": ["2022-01-04"],
                "symbol": ["000001.SZ"],
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:30:00")],
            }
        )

        with self.assertRaises(SystemExit) as caught:
            validate_context_delay(context, delay="delay1")

        self.assertIn("entry_delay_ticks", str(caught.exception))

    def test_lgbm_replay_accepts_cpu_prediction_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_all.parquet"
            pd.DataFrame([_cpu_lgbm_prediction_row(delay=1)]).to_parquet(
                path,
                index=False,
            )

            result = validate_prediction_interface(
                path,
                delay="delay1",
                scenarios=[SCENARIOS[name] for name in DEFAULT_SCENARIOS],
            )

        self.assertEqual(result["delay_source"], "prediction")

    def test_lgbm_replay_rejects_missing_default_depth_interface(self) -> None:
        row = _cpu_lgbm_prediction_row(delay=1)
        row.pop("entry_ask_volume_10")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_all.parquet"
            pd.DataFrame([row]).to_parquet(path, index=False)

            with self.assertRaises(SystemExit) as caught:
                validate_prediction_interface(
                    path,
                    delay="delay1",
                    scenarios=[SCENARIOS[name] for name in DEFAULT_SCENARIOS],
                )

        self.assertIn("entry_ask_volume_10", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
