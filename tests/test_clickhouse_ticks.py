from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import normalize_clickhouse_ticks


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


if __name__ == "__main__":
    unittest.main()
