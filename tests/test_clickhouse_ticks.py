from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    deduplicate_tick_timestamps,
    normalize_clickhouse_ticks,
)
from opening_strength_fit.labels import build_trade_labels


class ClickHouseTicksTest(unittest.TestCase):
    def test_normalize_clickhouse_ticks_serializes_numpy_scalars_in_objects(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "TradingDay": ["2024-10-28"],
                "Symbol": ["000001.SZ"],
                "ExchTimeOffsetUs": [33_300_000_000],
                "LocalTimeStamp": [
                    {
                        "AnXin-SH": np.uint64(1_730_078_100_360_189),
                        "Nested": [np.int64(3), np.float64(4.5)],
                    }
                ],
            }
        )

        out = normalize_clickhouse_ticks(frame)
        payload = json.loads(out.loc[0, "local_timestamp"])

        self.assertEqual(payload["AnXin-SH"], 1_730_078_100_360_189)
        self.assertEqual(payload["Nested"], [3, 4.5])
        self.assertEqual(out.loc[0, "date"], "2024-10-28")
        self.assertEqual(out.loc[0, "time"], "09:15:00")

    def test_deduplicate_tick_timestamps_keeps_latest_received_snapshot(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2019-04-19"] * 3,
                "symbol": ["000001.SZ"] * 3,
                "timestamp": pd.to_datetime(
                    [
                        "2019-04-19 09:30:00",
                        "2019-04-19 09:30:00",
                        "2019-04-19 09:30:03",
                    ]
                ),
                "local_timestamp": [
                    json.dumps({"feed": 100}),
                    json.dumps({"feed": 200}),
                    json.dumps({"feed": 300}),
                ],
                "trade_num": [10, 11, 12],
                "volume": [100, 110, 120],
                "turnover": [1_000, 1_100, 1_200],
                "ask_price_1": [10.0, 10.1, 10.2],
            }
        )

        out = deduplicate_tick_timestamps(frame)

        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[0, "ask_price_1"], 10.1)
        self.assertEqual(
            out.attrs["tick_timestamp_deduplication"]["rows_removed"],
            1,
        )

    def test_delay_two_counts_distinct_exchange_timestamps(self) -> None:
        seconds = [0, 0, 3, 6, 66, 69]
        frame = pd.DataFrame(
            {
                "date": ["2019-04-19"] * len(seconds),
                "symbol": ["000001.SZ"] * len(seconds),
                "timestamp": [
                    pd.Timestamp("2019-04-19 09:30:00") + pd.Timedelta(seconds=value)
                    for value in seconds
                ],
                "local_timestamp": [100, 200, 300, 400, 500, 600],
                "ask_price_1": [10.0] * len(seconds),
                "volume": [0, 0, 1, 2, 12, 13],
                "turnover": [0, 0, 10, 20, 120, 130],
            }
        )

        unique_ticks = deduplicate_tick_timestamps(frame)
        labeled = build_trade_labels(
            unique_ticks,
            entry_tick_delay=2,
            hold_seconds=60,
            sell_window_seconds=3,
            sample_start_time="09:30:00",
            sample_end_time="09:30:01",
        )

        self.assertEqual(len(labeled), 1)
        self.assertEqual(
            labeled.loc[0, "entry_timestamp"],
            pd.Timestamp("2019-04-19 09:30:06"),
        )
        self.assertEqual(labeled.loc[0, "entry_delay_seconds"], 6.0)

    def test_legacy_delay_two_still_counts_physical_rows_without_dedup(self) -> None:
        seconds = [0, 0, 3, 6, 66, 69]
        frame = pd.DataFrame(
            {
                "date": ["2019-04-19"] * len(seconds),
                "symbol": ["000001.SZ"] * len(seconds),
                "timestamp": [
                    pd.Timestamp("2019-04-19 09:30:00") + pd.Timedelta(seconds=value)
                    for value in seconds
                ],
                "ask_price_1": [10.0] * len(seconds),
                "volume": [0, 0, 1, 2, 12, 13],
                "turnover": [0, 0, 10, 20, 120, 130],
            }
        )

        labeled = build_trade_labels(
            frame,
            entry_tick_delay=2,
            hold_seconds=60,
            sell_window_seconds=3,
            sample_start_time="09:30:00",
            sample_end_time="09:30:01",
        )

        self.assertEqual(
            labeled.loc[0, "entry_timestamp"],
            pd.Timestamp("2019-04-19 09:30:03"),
        )
        self.assertEqual(labeled.loc[0, "entry_delay_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
