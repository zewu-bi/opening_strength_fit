from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from opening_strength_fit.commands.next_close_label_cache import (  # noqa: E402
    _read_base_frame,
    fetch_next_close_labels,
    main,
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

        self.assertEqual(
            frame["decision_target_timestamp"].dt.strftime("%H:%M:%S").tolist(),
            ["09:31:00", "09:40:00"],
        )
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
            "opening_strength_fit.commands.next_close_label_cache.compute_clickhouse_close_labels",
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

    def test_main_reads_input_and_output_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "run.toml"
            input_path = root / "input.parquet"
            output_path = root / "labels.parquet"
            output_dir = root / "out"
            pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000001.SZ"],
                    "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")],
                    "buy_price": [10.0],
                }
            ).to_parquet(input_path, index=False)
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'id = "build_next_close_test"',
                        "",
                        "[next_close_labels]",
                        f'input_path = "{input_path}"',
                        f'output_path = "{output_path}"',
                        'decision_times = ["09:31:00"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            returned = pd.DataFrame(
                {
                    "date": ["2022-01-04"],
                    "symbol": ["000001.SZ"],
                    "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")],
                    "alpha_return_next_close": [0.02],
                }
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "osf-build-next-close-labels",
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                patch(
                    "opening_strength_fit.commands.next_close_label_cache.compute_clickhouse_close_labels",
                    return_value=returned,
                ),
            ):
                main()

            labels = pd.read_parquet(output_path)

        self.assertEqual(labels["alpha_return_next_close"].tolist(), [0.02])


if __name__ == "__main__":
    unittest.main()
