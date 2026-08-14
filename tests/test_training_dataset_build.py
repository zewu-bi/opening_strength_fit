from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.short_label_cache import compute_short_vwap_labels
from opening_strength_fit.commands.training_dataset_build import (
    _apply_post_sample_feature_transforms_from_config,
    _clock_partition,
    _drop_features_from_config,
    _feature_base_shard_root,
    _sort_feature_source,
)
from opening_strength_fit.config import load_toml
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.model_features import feature_columns
from opening_strength_fit.model_preprocessing import lightgbm_feature_value_frame
from opening_strength_fit.raw_source import (
    DAILY_REFERENCE_COLUMNS,
    TICK_COLUMNS,
)
from opening_strength_fit.training_dataset_features import (
    build_raw_feature_day,
    decode_clickhouse_text,
    normalize_clickhouse_date,
)
from opening_strength_fit.training_dataset_labels import (
    KEY_COLUMNS,
)
from opening_strength_fit.training_dataset_labels import (
    build_label_base as _build_label_base,
)
from opening_strength_fit.training_dataset_labels import (
    compute_clock_vwap_label_set as compute_short_label_set,
)
from opening_strength_fit.training_dataset_labels import (
    filter_decision_clocks as _filter_decision_clocks,
)
from opening_strength_fit.training_dataset_labels import (
    mixed_target_label as _mixed_label,
)

CANONICAL_CONFIG = Path(
    "experiments/runs/nn_v6_w0931_0940_short3m_corrected_nextclose_36m_grouped_gated_v2_mse.toml"
)


def _offset(clock: str) -> int:
    timestamp = pd.Timestamp(f"2000-01-01 {clock}")
    return int((timestamp - timestamp.normalize()) / pd.Timedelta(microseconds=1))


def _raw_tick_row(date: str, symbol: str, clock: str, sequence: int) -> dict[str, object]:
    row: dict[str, object] = {column: float(sequence + 1) for column in TICK_COLUMNS}
    row.update(
        {
            "TradingDay": date,
            "Symbol": symbol,
            "ExchTimeOffsetUs": _offset(clock),
            "HighPrice": 10.2,
            "LowPrice": 9.8,
            "LastPrice": 10.0 + sequence * 0.001,
            "TradeNum": sequence + 1,
            "Volume": 1000.0 + sequence * 100.0,
            "Turnover": 10_000.0 + sequence * 1005.0,
            "Status": "T0",
            "AvgAskPrice": 10.1,
            "AvgBidPrice": 9.9,
            "TotalAskVolume": 10_000.0,
            "TotalBidVolume": 11_000.0,
            "TotalAskCount": 100.0,
            "TotalBidCount": 110.0,
            "IOPV": 0.0,
        }
    )
    for level in range(1, 11):
        row[f"AskPrice{level}"] = 10.0 + level * 0.01
        row[f"BidPrice{level}"] = 10.0 - level * 0.01
        row[f"AskVolume{level}"] = 1000.0 + level
        row[f"BidVolume{level}"] = 1100.0 + level
        row[f"AskCount{level}"] = 10.0 + level
        row[f"BidCount{level}"] = 11.0 + level
    return row


def _daily_row(date: str, symbol: str, scale: float) -> dict[str, object]:
    row: dict[str, object] = {column: 0 for column in DAILY_REFERENCE_COLUMNS}
    row.update(
        {
            "TradingDay": date,
            "Symbol": symbol,
            "OpenPrice": 10.0,
            "ClosePrice": 10.1,
            "PreClosePrice": 9.9,
            "TradeStatus": 1,
            "STStatus": 0,
            "UpdownLimitStatus": 0,
            "TotalMarketValue": 100_000.0 * scale,
            "TotalFloatMarketValue": 80_000.0 * scale,
            "TotalShareToday": 10_000.0 * scale,
            "FloatAShare": 8_000.0 * scale,
            "FreeShareToday": 7_000.0 * scale,
        }
    )
    return row


class TrainingDatasetBuildTest(unittest.TestCase):
    def test_feature_base_clock_layout_and_reduce_order_are_stable(self) -> None:
        root = Path("/tmp/feature-base")
        self.assertEqual(_clock_partition("09:31:00"), "093100")
        self.assertEqual(
            _feature_base_shard_root(root, 2025, "09:31:00"),
            root / "year=2025" / "clock=093100",
        )
        frame = pd.DataFrame(
            {
                "date": ["2025-01-03", "2025-01-02", "2025-01-02"],
                "symbol": ["000002.SZ", "000002.SZ", "000001.SZ"],
                "decision_target_timestamp": pd.to_datetime(
                    [
                        "2025-01-03 09:31:00",
                        "2025-01-02 09:32:00",
                        "2025-01-02 09:31:00",
                    ]
                ),
                "value": [3.0, 2.0, 1.0],
            }
        )

        sorted_frame = _sort_feature_source(frame)

        self.assertEqual(sorted_frame["value"].tolist(), [1.0, 2.0, 3.0])

    def test_clickhouse_parquet_date_and_binary_text_are_decoded(self) -> None:
        dates = normalize_clickhouse_date(pd.Series([20090], dtype="uint16"))
        symbols = decode_clickhouse_text(pd.Series([b"000001.SZ"]))

        self.assertEqual(dates.tolist(), ["2025-01-02"])
        self.assertEqual(symbols.tolist(), ["000001.SZ"])

    def test_raw_feature_day_produces_canonical_350_model_ready_features(self) -> None:
        date = "2025-01-02"
        clocks = ["09:24:59", "09:25:00", *[f"09:{minute:02d}:00" for minute in range(30, 41)]]
        rows = [
            _raw_tick_row(date, symbol, clock, index)
            for symbol in ("000001.SZ", "000002.SZ")
            for index, clock in enumerate(clocks)
        ]
        daily = [
            _daily_row(day, symbol, scale)
            for day, scale in (("2024-12-31", 0.9), (date, 1.0))
            for symbol in ("000001.SZ", "000002.SZ")
        ]
        with tempfile.TemporaryDirectory() as directory:
            year_root = Path(directory) / "year=2025"
            tick_path = year_root / "ticks" / f"date={date}.parquet"
            tick_path.parent.mkdir(parents=True)
            pd.DataFrame(rows).to_parquet(tick_path, index=False)
            pd.DataFrame(daily).to_parquet(year_root / "daily_reference.parquet", index=False)

            feature_config = load_toml(CANONICAL_CONFIG)
            dataset_config = {
                "dataset": {
                    "context_decision_times": [f"09:{minute:02d}:00" for minute in range(30, 41)],
                    "feature_tick_start_offset_us": _offset("09:15:00"),
                    "market_cap_unit_multiplier": 10_000.0,
                    "share_unit_multiplier": 10_000.0,
                }
            }
            day = build_raw_feature_day(
                tick_path,
                date,
                feature_config,
                dataset_config,
                {},
            )
            transformed = _apply_post_sample_feature_transforms_from_config(day, feature_config)
            transformed = _drop_features_from_config(transformed, feature_config)
            transformed = _filter_decision_clocks(
                transformed,
                tuple(f"09:{minute:02d}:00" for minute in range(31, 41)),
            )
            selected = feature_columns(
                transformed,
                None,
                **feature_filters_from_config(feature_config),
            )

            self.assertEqual(len(selected), 350)
            ready, ready_features = lightgbm_feature_value_frame(
                transformed,
                selected,
                feature_value_transform="mechanismized_v3_dimensionless_328",
                extra_columns=KEY_COLUMNS,
            )
            self.assertEqual(ready_features, selected)
            model_ready = ready[[*KEY_COLUMNS, *selected]]
            self.assertEqual(set(model_ready.columns), {*KEY_COLUMNS, *selected})

    def test_clock_state_base_uses_last_state_at_decision_and_entry(self) -> None:
        date = "2025-01-02"
        raw = pd.DataFrame(
            {
                "Symbol": ["000001.SZ"] * 3,
                "ExchTimeOffsetUs": [
                    _offset("09:30:59"),
                    _offset("09:31:05"),
                    _offset("09:31:07"),
                ],
                "AskPrice1": [10.0, 10.1, 10.2],
                "Status": ["T0", "T0", "T0"],
                "Volume": [100.0, 110.0, 120.0],
                "Turnover": [1000.0, 1110.0, 1220.0],
            }
        )

        base = _build_label_base(
            raw,
            trading_day=date,
            decision_times=("09:31:00",),
            feature_tick_start_offset_us=_offset("09:30:00"),
            entry_delay_seconds=6,
        )

        self.assertEqual(base.loc[0, "timestamp"], pd.Timestamp(f"{date} 09:30:59"))
        self.assertEqual(base.loc[0, "entry_timestamp"], pd.Timestamp(f"{date} 09:31:06"))
        self.assertAlmostEqual(base.loc[0, "buy_price"], 10.1)
        self.assertTrue(bool(base.loc[0, "entry_after_cross_section_ready"]))

    def test_multi_horizon_short_labels_match_existing_clock_state_builder(self) -> None:
        date = "2025-01-02"
        base = pd.DataFrame(
            {
                "date": [date],
                "symbol": ["000001.SZ"],
                "decision_target_timestamp": [pd.Timestamp(f"{date} 09:31:00")],
                "entry_timestamp": [pd.Timestamp(f"{date} 09:31:06")],
                "buy_price": [10.0],
                "status": ["T0"],
                "entry_status": ["T0"],
                "entry_after_cross_section_ready": [True],
            }
        )
        clocks = ["09:32:06", "09:33:06", "09:34:06", "09:35:06", "09:36:06", "09:37:06"]
        raw = pd.DataFrame(
            {
                "TradingDay": [date] * len(clocks),
                "Symbol": ["000001.SZ"] * len(clocks),
                "ExchTimeOffsetUs": [_offset(clock) for clock in clocks],
                "LocalTimeStamp": np.arange(len(clocks)),
                "TradeNum": np.arange(len(clocks)),
                "Volume": [100.0, 200.0, 300.0, 420.0, 550.0, 700.0],
                "Turnover": [1000.0, 2020.0, 3050.0, 4280.0, 5610.0, 7150.0],
            }
        )
        projected = raw[["Symbol", "ExchTimeOffsetUs", "Volume", "Turnover"]]
        actual = compute_short_label_set(
            base,
            projected,
            horizons=(60, 180, 300),
            sell_window_seconds=60,
            volume_unit_multiplier=1.0,
            fee_bps=0.0,
            tradable_statuses=("T0",),
        )

        for horizon in (60, 180, 300):
            expected = compute_short_vwap_labels(
                base,
                raw,
                hold_seconds=horizon,
                sell_window_seconds=60,
                volume_unit_multiplier=1.0,
                fee_bps=0.0,
                tradable_statuses=("T0",),
            )
            minutes = horizon // 60
            self.assertAlmostEqual(
                actual.loc[0, f"label_short_{minutes}m"], expected.loc[0, "label"]
            )
            self.assertEqual(
                bool(actual.loc[0, f"valid_short_{minutes}m"]),
                bool(expected.loc[0, "valid_label"]),
            )

    def test_mixed_label_uses_one_minute_and_next_close_cross_sectional_zscores(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-02"] * 3,
                "decision_target_timestamp": [pd.Timestamp("2025-01-02 09:31:00")] * 3,
                "label_short_1m": [1.0, 2.0, 3.0],
                "label_next_close": [30.0, 10.0, 20.0],
                "valid_short_1m": [True] * 3,
                "valid_next_close": [True] * 3,
            }
        )

        mixed, valid = _mixed_label(frame, weight=0.30, min_group_size=2)

        np.testing.assert_allclose(
            mixed.to_numpy(),
            [-0.857321, -0.367423, 1.224745],
            atol=1e-6,
        )
        self.assertTrue(valid.all())


if __name__ == "__main__":
    unittest.main()
