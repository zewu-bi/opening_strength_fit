from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.horizon_label_split import split_label_year
from opening_strength_fit.io import read_frame, write_frame_atomic
from opening_strength_fit.label_splitting import OUTPUT_COLUMNS


def test_split_label_year_writes_three_training_datasets_without_valid_flags(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "combined"
    source_year = source_root / "year=2025"
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": pd.to_datetime(["2025-01-02 10:01:00"] * 3),
            "label_short_1m": [1.0, 2.0, 3.0],
            "label_short_3m": [3.0, 2.0, 1.0],
            "label_short_5m": [1.0, 1.0, 2.0],
            "label_next_close": [2.0, 4.0, np.nan],
        }
    )
    write_frame_atomic(frame, source_year / "labels.parquet")
    (source_year / "_SUCCESS").touch()
    config = {
        "run": {"id": "split_test"},
        "dataset": {
            "label_output_root": str(source_root),
            "horizon_label_output_template": str(tmp_path / "labels_h{horizon_minutes}m_v2"),
            "mixed_next_close_weight": 0.30,
            "mixed_min_group_size": 2,
        },
    }

    manifests = split_label_year(
        config,
        tmp_path / "config.toml",
        year=2025,
        overwrite=False,
    )

    assert [manifest["horizon_minutes"] for manifest in manifests] == [1, 3, 5]
    one = read_frame(tmp_path / "labels_h1m_v2/year=2025/labels.parquet")
    three = read_frame(tmp_path / "labels_h3m_v2/year=2025/labels.parquet")
    five = read_frame(tmp_path / "labels_h5m_v2/year=2025/labels.parquet")
    assert list(one.columns) == list(OUTPUT_COLUMNS)
    assert not any(column.startswith("valid_") for column in one.columns)
    assert one["target_label"].dtype == np.dtype("float32")
    np.testing.assert_allclose(one["target_label"].iloc[:2], [-1.3, 1.3])
    np.testing.assert_allclose(three["target_label"].iloc[:2], [0.7, -0.7])
    assert np.isnan(one["target_label"].iloc[2])
    assert five["target_label"].isna().all()
    manifest = json.loads((tmp_path / "labels_h1m_v2/year=2025/manifest.json").read_text())
    assert manifest["contains_validity_flags"] is False
    assert manifest["non_null_rows"]["target_label"] == 2
    assert (tmp_path / "labels_h1m_v2/year=2025/_SUCCESS").exists()
