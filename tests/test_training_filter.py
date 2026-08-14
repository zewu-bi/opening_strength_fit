from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.training_modeling import filter_configured_training_rows


def test_training_filter_keeps_only_ordinary_daily_limit_rows(tmp_path: Path) -> None:
    root = tmp_path / "raw_source" / "year=2022"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "TradingDay": ["2022-01-03", "2022-01-03", "2022-01-03"],
            "Symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "UpdownLimitStatus": [0, 1, -1],
        }
    ).to_parquet(root / "daily_reference.parquet", index=False)
    train = pd.DataFrame(
        {
            "date": ["2022-01-03"] * 4,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "label": [1.0, 2.0, 3.0, 4.0],
        }
    )

    filtered, stats = filter_configured_training_rows(
        train,
        {
            "training_filter": {
                "enabled": True,
                "mode": "ordinary_daily_limit",
                "daily_reference_root": str(tmp_path / "raw_source"),
            }
        },
    )

    assert filtered["symbol"].tolist() == ["000001.SZ"]
    assert stats["training_filter_input_rows"] == 4
    assert stats["training_filter_output_rows"] == 1
    assert stats["training_filter_limit_up_rows"] == 1
    assert stats["training_filter_limit_down_rows"] == 1
    assert stats["training_filter_unknown_rows"] == 1
