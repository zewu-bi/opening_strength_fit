from __future__ import annotations

import pandas as pd

from opening_strength_fit.io import write_frame_atomic


def test_write_frame_atomic_writes_readable_parquet(tmp_path):
    path = tmp_path / "predictions.parquet"
    frame = pd.DataFrame({"score": [0.1, 0.2], "symbol": ["000001.SZ", "600000.SH"]})

    write_frame_atomic(frame, path)

    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert not list(tmp_path.glob(".*.tmp*"))


def test_write_frame_atomic_replaces_existing_csv(tmp_path):
    path = tmp_path / "metrics_by_year.csv"
    old = pd.DataFrame({"year": [2022], "value": [1.0]})
    new = pd.DataFrame({"year": [2023], "value": [2.0]})

    write_frame_atomic(old, path)
    write_frame_atomic(new, path)

    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(loaded, new)
    assert not list(tmp_path.glob(".*.tmp*"))
