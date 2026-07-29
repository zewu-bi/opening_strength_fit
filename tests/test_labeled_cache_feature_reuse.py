from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from opening_strength_fit.commands.labeled_cache_build import _apply_cache_year_shard
from opening_strength_fit.training_data import _attach_reused_labeled_features


def _reuse_config(path: Path) -> dict:
    return {
        "features": {
            "reuse_labeled_path": str(path),
            "reuse_feature_prefixes": ["preopen_", "auction_"],
            "reuse_key_columns": ["date", "symbol"],
            "reuse_join": "inner",
            "reuse_require_constant": True,
        }
    }


def test_attach_reused_labeled_features_projects_and_deduplicates(tmp_path: Path) -> None:
    source_path = tmp_path / "opening.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "preopen_volume": [100.0, 100.0, 200.0],
            "auction_price_range_bps": [5.0, 5.0, 7.0],
            "label": [0.1, 0.2, 0.3],
        }
    ).to_parquet(source_path, index=False)
    destination = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ", "000003.SZ"],
            "decision_time": ["10:01:00", "10:02:00", "10:01:00"],
        }
    )

    out = _attach_reused_labeled_features(destination, _reuse_config(source_path))

    assert len(out) == 2
    assert out["symbol"].eq("000001.SZ").all()
    assert out["preopen_volume"].eq(100.0).all()
    assert out["auction_price_range_bps"].eq(5.0).all()
    assert "label" not in out.columns


def test_attach_reused_labeled_features_rejects_nonconstant_source(tmp_path: Path) -> None:
    source_path = tmp_path / "opening.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "preopen_volume": [100.0, 101.0],
        }
    ).to_parquet(source_path, index=False)
    destination = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["000001.SZ"],
        }
    )

    with pytest.raises(SystemExit, match="not constant"):
        _attach_reused_labeled_features(destination, _reuse_config(source_path))


def test_apply_cache_year_shard_formats_paths_and_dates() -> None:
    config = {
        "run": {"id": "build_window_cache"},
        "data": {},
        "cache": {},
        "features": {},
        "cache_shards": {
            "years": [2019, 2020],
            "cache_path_template": "/cache/window_{year}.parquet",
            "reuse_labeled_path_template": "/cache/opening_{year}.parquet",
        },
    }

    year = _apply_cache_year_shard(config, index=1)

    assert year == 2020
    assert config["run"]["id"] == "build_window_cache_2020"
    assert config["data"] == {
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
    }
    assert config["cache"]["path"] == "/cache/window_2020.parquet"
    assert config["features"]["reuse_labeled_path"] == "/cache/opening_2020.parquet"
