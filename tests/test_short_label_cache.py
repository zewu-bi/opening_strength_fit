from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opening_strength_fit.commands.short_label_cache import (
    compute_short_vwap_labels,
    read_short_label_base,
    validate_source_cache,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001.SZ"],
            "decision_target_timestamp": [pd.Timestamp("2025-01-02 09:31:00")],
            "entry_timestamp": [pd.Timestamp("2025-01-02 09:31:06")],
            "entry_source_timestamp": [pd.Timestamp("2025-01-02 09:31:05")],
            "entry_state_age_seconds": [1.0],
            "buy_price": [10.0],
            "status": ["T0"],
            "entry_status": ["T0"],
        }
    )


class ShortLabelCacheTest(unittest.TestCase):
    def test_hold_180_then_60_second_vwap_uses_backward_clock_state(self) -> None:
        ticks = pd.DataFrame(
            {
                "TradingDay": ["2025-01-02"] * 4,
                "Symbol": ["000001.SZ"] * 4,
                "ExchTimeOffsetUs": [
                    34_445_000_000,  # 09:34:05: sell start source
                    34_447_000_000,  # 09:34:07: must not leak into start
                    34_505_000_000,  # 09:35:05: sell end source
                    34_507_000_000,  # 09:35:07: must not leak into end
                ],
                "LocalTimeStamp": [1, 2, 3, 4],
                "TradeNum": [10, 11, 20, 21],
                "Volume": [100.0, 120.0, 200.0, 240.0],
                "Turnover": [1_000.0, 1_210.0, 2_020.0, 2_440.0],
            }
        )

        labels = compute_short_vwap_labels(
            _base_frame(),
            ticks,
            hold_seconds=180,
            sell_window_seconds=60,
            volume_unit_multiplier=1.0,
            fee_bps=0.0,
            tradable_statuses=("T0", "20", "TRADE"),
        )

        self.assertEqual(labels.loc[0, "sell_start_target_timestamp"].second, 6)
        self.assertEqual(
            labels.loc[0, "sell_start_source_timestamp"],
            pd.Timestamp("2025-01-02 09:34:05"),
        )
        self.assertEqual(
            labels.loc[0, "sell_end_source_timestamp"],
            pd.Timestamp("2025-01-02 09:35:05"),
        )
        self.assertAlmostEqual(labels.loc[0, "sell_volume"], 100.0)
        self.assertAlmostEqual(labels.loc[0, "sell_turnover"], 1_020.0)
        self.assertAlmostEqual(labels.loc[0, "sell_vwap"], 10.2)
        self.assertAlmostEqual(labels.loc[0, "label"], 0.02)
        self.assertTrue(bool(labels.loc[0, "valid_label"]))

    def test_duplicate_exchange_timestamp_keeps_latest_local_state(self) -> None:
        ticks = pd.DataFrame(
            {
                "TradingDay": ["2025-01-02"] * 3,
                "Symbol": ["000001.SZ"] * 3,
                "ExchTimeOffsetUs": [34_446_000_000, 34_446_000_000, 34_506_000_000],
                "LocalTimeStamp": [1, 2, 3],
                "TradeNum": [10, 11, 20],
                "Volume": [90.0, 100.0, 200.0],
                "Turnover": [900.0, 1_000.0, 2_020.0],
            }
        )

        labels = compute_short_vwap_labels(
            _base_frame(),
            ticks,
            hold_seconds=180,
            sell_window_seconds=60,
            volume_unit_multiplier=1.0,
            fee_bps=0.0,
            tradable_statuses=("T0",),
        )

        self.assertAlmostEqual(labels.loc[0, "sell_start_volume"], 100.0)
        self.assertAlmostEqual(labels.loc[0, "sell_vwap"], 10.2)

    def test_read_base_rejects_wrong_entry_delay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.parquet"
            base = _base_frame()
            base["entry_timestamp"] = pd.Timestamp("2025-01-02 09:31:07")
            base.to_parquet(path, index=False)

            with self.assertRaisesRegex(SystemExit, "entry delay mismatch"):
                read_short_label_base(
                    path,
                    decision_times=("09:31:00",),
                    expected_entry_delay_seconds=6,
                    tradable_statuses=("T0",),
                )

    def test_source_manifest_schema_and_size_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.parquet"
            _base_frame().to_parquet(path, index=False)
            manifest_path = path.with_name(f"{path.name}.manifest.json")
            manifest_path.write_text(
                (
                    '{"cache_schema_version":"expected-v6",'
                    f'"cache_file":{{"bytes":{path.stat().st_size}}}}}'
                ),
                encoding="utf-8",
            )

            manifest = validate_source_cache(
                path,
                require_manifest=True,
                expected_schema_version="expected-v6",
            )
            self.assertEqual(manifest["cache_schema_version"], "expected-v6")
            with self.assertRaisesRegex(SystemExit, "schema mismatch"):
                validate_source_cache(
                    path,
                    require_manifest=True,
                    expected_schema_version="wrong-v6",
                )


if __name__ == "__main__":
    unittest.main()
