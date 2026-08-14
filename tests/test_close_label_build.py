from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.close_label_build import build_close_label_year
from opening_strength_fit.commands.horizon_label_split import OUTPUT_COLUMNS
from opening_strength_fit.io import read_frame, write_frame_atomic


def _offset(clock: str) -> int:
    value = pd.Timestamp(f"2000-01-01 {clock}")
    return int((value - value.normalize()) / pd.Timedelta(microseconds=1))


def test_build_close_label_year_writes_model_ready_mixed_labels(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_year = raw_root / "year=2025"
    next_root = tmp_path / "h1m"
    next_year = next_root / "year=2025"
    output_root = tmp_path / "hclose"
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    asks = [10.0, 20.0, 30.0]
    ticks = pd.DataFrame(
        {
            "Symbol": np.repeat(symbols, 2),
            "ExchTimeOffsetUs": [_offset("11:01:00"), _offset("11:01:06")] * 3,
            "Volume": [100.0, 101.0] * 3,
            "Turnover": [1000.0, 1010.0] * 3,
            "AskPrice1": np.repeat(asks, 2),
            "Status": ["T0"] * 6,
        }
    )
    write_frame_atomic(ticks, raw_year / "ticks/date=2025-01-02.parquet")
    write_frame_atomic(
        pd.DataFrame(
            {
                "TradingDay": ["2025-01-02"] * 3,
                "Symbol": symbols,
                "ClosePrice": [11.0, 18.0, 33.0],
            }
        ),
        raw_year / "close_reference.parquet",
    )
    raw_year.joinpath("_SUCCESS").touch()

    keys = {
        "date": ["2025-01-02"] * 3,
        "symbol": symbols,
        "decision_target_timestamp": pd.to_datetime(["2025-01-02 11:01:00"] * 3),
    }
    write_frame_atomic(
        pd.DataFrame(
            {
                **keys,
                "label_short": [0.01, 0.02, 0.03],
                "label_next_close": [0.2, 0.0, -0.2],
                "target_label": [0.0, 0.0, 0.0],
            }
        ),
        next_year / "labels.parquet",
    )
    next_year.joinpath("_SUCCESS").touch()

    config = {
        "run": {"id": "test_close_labels"},
        "dataset": {
            "years": [2025],
            "raw_source_root": str(raw_root),
            "next_close_label_root": str(next_root),
            "label_output_root": str(output_root),
            "decision_times": ["11:01:00"],
            "feature_tick_start_offset_us": _offset("10:50:00"),
            "entry_delay_seconds": 6,
            "tradable_statuses": ["T0", "20", "TRADE"],
            "mixed_next_close_weight": 0.3,
            "mixed_min_group_size": 2,
        },
    }
    build_close_label_year(
        config,
        tmp_path / "close.toml",
        year=2025,
        start_date="2025-01-01",
        end_date="2025-12-31",
        overwrite=False,
    )

    output = read_frame(output_root / "year=2025/labels.parquet")
    assert list(output.columns) == list(OUTPUT_COLUMNS)
    np.testing.assert_allclose(output["label_short"], [0.1, -0.1, 0.1])
    assert output["target_label"].notna().all()
    assert output["target_label"].dtype == np.dtype("float32")
    assert (output_root / "year=2025/_SUCCESS").exists()
