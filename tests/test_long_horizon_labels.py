from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.long_horizon_label_split import split_label_year
from opening_strength_fit.commands.long_horizon_labels import same_day_close_label
from opening_strength_fit.commands.long_label_raw_source import TICK_COLUMNS
from opening_strength_fit.config import load_toml
from opening_strength_fit.io import read_frame, write_frame_atomic
from opening_strength_fit.raw_source import tick_source_sql
from opening_strength_fit.training_dataset_labels import compute_clock_vwap_label_set


def _offset(clock: str) -> int:
    value = pd.Timestamp(f"2000-01-01 {clock}")
    return int((value - value.normalize()) / pd.Timedelta(microseconds=1))


def test_projected_long_label_raw_query_keeps_only_cumulative_trade_state() -> None:
    sql = tick_source_sql(
        "stock.tick",
        ((_offset("10:19:00"), _offset("11:12:00")),),
        output_columns=TICK_COLUMNS,
    )

    assert TICK_COLUMNS == (
        "TradingDay",
        "Symbol",
        "ExchTimeOffsetUs",
        "Volume",
        "Turnover",
    )
    assert "AskVolume10" not in sql
    assert "limit 1 by Symbol, ExchTimeOffsetUs" in sql
    assert "arrayMax(mapValues(LocalTimeStamp)) desc" in sql


def test_timed_vwap_and_same_day_close_labels_use_clock6_buy_price() -> None:
    decision = pd.Timestamp("2025-01-02 09:31:00")
    base = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001.SZ"],
            "decision_target_timestamp": [decision],
            "entry_timestamp": [decision + pd.Timedelta(seconds=6)],
            "buy_price": [10.0],
            "status": ["T0"],
            "entry_status": ["T0"],
            "entry_after_cross_section_ready": [True],
        }
    )
    state = pd.DataFrame(
        {
            "Symbol": ["000001.SZ"] * 4,
            "ExchTimeOffsetUs": [
                _offset("09:41:06"),
                _offset("09:42:06"),
                _offset("10:31:06"),
                _offset("10:32:06"),
            ],
            "Volume": [100.0, 110.0, 200.0, 210.0],
            "Turnover": [1000.0, 1101.0, 2020.0, 2122.0],
        }
    )
    timed = compute_clock_vwap_label_set(
        base,
        state,
        horizons=(600, 3600),
        sell_window_seconds=60,
        volume_unit_multiplier=1.0,
        fee_bps=0.0,
        tradable_statuses=("T0", "20", "TRADE"),
    )

    np.testing.assert_allclose(timed.loc[0, "label_short_10m"], 0.01, rtol=0, atol=1e-12)
    np.testing.assert_allclose(timed.loc[0, "label_short_60m"], 0.02, rtol=0, atol=1e-12)

    close = pd.DataFrame(
        {"TradingDay": ["2025-01-02"], "Symbol": ["000001.SZ"], "ClosePrice": [10.3]}
    )
    same_close = same_day_close_label(
        base,
        close,
        tradable_statuses=("T0", "20", "TRADE"),
        fee_bps=0.0,
    )
    np.testing.assert_allclose(same_close.loc[0, "label_same_day_close"], 0.03, rtol=0, atol=1e-12)


def test_long_label_split_reuses_next_close_for_three_mixed_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "combined"
    source_year = source_root / "year=2025"
    keys = {
        "date": ["2025-01-02", "2025-01-02"],
        "symbol": ["000001.SZ", "000002.SZ"],
        "decision_target_timestamp": [
            pd.Timestamp("2025-01-02 09:31:00"),
            pd.Timestamp("2025-01-02 09:31:00"),
        ],
    }
    source = pd.DataFrame(
        {
            **keys,
            "label_hold_10m": [-1.0, 1.0],
            "label_hold_1h": [1.0, -1.0],
            "label_same_day_close": [-2.0, 2.0],
            "label_next_close": [-3.0, 3.0],
        }
    )
    write_frame_atomic(source, source_year / "labels.parquet")
    (source_year / "_SUCCESS").touch()
    outputs = {name: tmp_path / name for name in ("10m", "1h", "close")}
    config_path = tmp_path / "split.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                'id = "test_long_split"',
                "[dataset]",
                f'label_output_root = "{source_root}"',
                "mixed_next_close_weight = 0.30",
                "mixed_min_group_size = 2",
                *[
                    "\n".join(
                        [
                            "[[dataset.mixed_labels]]",
                            f'name = "{name}"',
                            f'source_column = "{column}"',
                            f'output_root = "{outputs[name]}"',
                        ]
                    )
                    for name, column in (
                        ("10m", "label_hold_10m"),
                        ("1h", "label_hold_1h"),
                        ("close", "label_same_day_close"),
                    )
                ],
            ]
        ),
        encoding="utf-8",
    )

    manifests = split_label_year(load_toml(config_path), config_path, year=2025, overwrite=False)

    assert [item["horizon_name"] for item in manifests] == ["10m", "1h", "close"]
    ten = read_frame(outputs["10m"] / "year=2025" / "labels.parquet")
    assert ten["label_next_close"].tolist() == [-3.0, 3.0]
    np.testing.assert_allclose(ten["target_label"], [-1.3, 1.3])


def test_long_label_configs_cover_only_the_first_two_windows() -> None:
    expected = {
        "opening_0931_0940_labels_10m_1h_close.toml": "09:31:00",
        "opening_1001_1010_labels_10m_1h_close.toml": "10:01:00",
    }
    for name, first_clock in expected.items():
        config = load_toml(Path("experiments/runs") / name)
        assert config["dataset"]["horizons_seconds"] == [600, 3600]
        assert config["dataset"]["decision_times"][0] == first_clock
        assert config["dataset"]["next_close_label_root"].endswith("labels_h1m_v2")
