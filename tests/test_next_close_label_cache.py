from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_next_close_labels import (  # noqa: E402
    _read_base_frame,
    fetch_next_close_labels,
)


class NextCloseLabelCacheTest(unittest.TestCase):
    def test_read_base_frame_filters_decision_times_and_renames_buy_price(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.parquet"
            pd.DataFrame(
                {
                    "date": ["2022-01-04"] * 4,
                    "symbol": ["000001.SZ"] * 4,
                    "decision_target_timestamp": pd.to_datetime(
                        [
                            "2022-01-04 09:30:00",
                            "2022-01-04 09:31:00",
                            "2022-01-04 09:40:00",
                            "2022-01-04 09:41:00",
                        ]
                    ),
                    "ask_price_1": [9.9, 10.0, 10.2, 10.3],
                }
            ).to_parquet(path, index=False)

            frame = _read_base_frame(
                path,
                buy_price_col="ask_price_1",
                decision_times=("09:31:00", "09:40:00"),
            )

        self.assertEqual(frame["decision_target_timestamp"].dt.strftime("%H:%M:%S").tolist(), ["09:31:00", "09:40:00"])
        self.assertEqual(frame["buy_price"].tolist(), [10.0, 10.2])

    def test_fetch_next_close_labels_normalizes_non_finite_returns(self) -> None:
        base = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04"],
                "symbol": ["000001.SZ", "000002.SZ"],
                "decision_target_timestamp": pd.to_datetime(
                    ["2022-01-04 09:31:00", "2022-01-04 09:31:00"]
                ),
                "buy_price": [10.0, 0.0],
            }
        )
        returned = base[["date", "symbol", "decision_target_timestamp"]].copy()
        returned["alpha_return_next_close"] = [0.01, float("inf")]

        with patch(
            "build_next_close_labels.compute_clickhouse_close_labels",
            return_value=returned,
        ):
            labels = fetch_next_close_labels(
                base,
                host="localhost",
                port=8123,
                username="user",
                password="pass",
                table="stock.tick",
            )

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])
        self.assertAlmostEqual(labels.loc[0, "alpha_return_next_close"], 0.01)


if __name__ == "__main__":
    unittest.main()
