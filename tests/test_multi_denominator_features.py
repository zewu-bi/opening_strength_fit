from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opening_strength_fit.features_multi_denominator import (
    add_multi_denominator_ratio_features,
)
from opening_strength_fit.torch_model.architectures import _feature_group_name_v2

TURNOVER_COLUMNS = (
    "preopen_turnover",
    "turnover_diff_1t",
    "postopen_turnover_diff_1m",
    "postopen_turnover_diff_3m",
    "postopen_turnover_diff_5m",
)
VOLUME_COLUMNS = (
    "preopen_volume",
    "volume_diff_1t",
    "postopen_volume_diff_1m",
    "postopen_volume_diff_3m",
    "postopen_volume_diff_5m",
)
DEPTH_COLUMNS = ("ask_depth_10", "bid_depth_10")
XS_COLUMNS = ("postopen_turnover_diff_1m",)


def _frame() -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(("000001.SZ", "000002.SZ", "000003.SZ"), start=1):
        scale = float(symbol_index)
        row = {
            "date": "2025-01-02",
            "symbol": symbol,
            "decision_target_timestamp": pd.Timestamp("2025-01-02 09:35:00"),
            "float_market_cap": 1_000_000.0 * scale,
            "float_shares": 100_000.0 * scale,
            "hist_avg_daily_turnover_60d": 100_000.0 * scale,
            "hist_avg_daily_volume_60d": 10_000.0 * scale,
            "turnover": 50_000.0 * scale,
            "volume": 5_000.0 * scale,
            "ask_depth_10": 500.0 * scale,
            "bid_depth_10": 600.0 * scale,
        }
        for index, column in enumerate(TURNOVER_COLUMNS, start=1):
            row[column] = float(100 * index * symbol_index)
        for index, column in enumerate(VOLUME_COLUMNS, start=1):
            row[column] = float(10 * index * symbol_index)
        rows.append(row)
    return pd.DataFrame(rows)


def _add(frame: pd.DataFrame, **overrides) -> pd.DataFrame:
    kwargs = {
        "turnover_columns": TURNOVER_COLUMNS,
        "volume_columns": VOLUME_COLUMNS,
        "depth_columns": DEPTH_COLUMNS,
        "cross_sectional_median_columns": XS_COLUMNS,
        "min_features": 25,
        "max_features": 40,
    }
    kwargs.update(overrides)
    return add_multi_denominator_ratio_features(frame, **kwargs)


def test_multi_denominator_family_is_bounded_distinct_and_float32() -> None:
    frame = _frame()
    out = _add(frame)
    added = [column for column in out.columns if column.startswith("multi_den_ratio_")]

    assert len(added) == 25
    assert len(added) == len(set(added))
    assert all(out[column].dtype == np.dtype("float32") for column in added)
    assert np.isfinite(out[added].to_numpy()[~np.isnan(out[added].to_numpy())]).all()
    assert out.loc[
        0,
        "multi_den_ratio_postopen_turnover_diff_1m_to_float_market_cap",
    ] == pytest.approx(300.0 / 1_000_000.0)
    assert out.loc[
        0,
        "multi_den_ratio_postopen_volume_diff_1m_to_hist_avg_daily_volume_60d",
    ] == pytest.approx(30.0 / 10_000.0)
    assert out.loc[
        0,
        "multi_den_ratio_postopen_turnover_diff_1m_to_xs_median",
    ] == pytest.approx(0.5)


def test_multi_denominator_family_fails_if_a_required_reference_is_missing() -> None:
    with pytest.raises(SystemExit, match="expected at least 25"):
        _add(_frame().drop(columns="float_shares"))


def test_multi_denominator_features_land_in_semantic_model_groups() -> None:
    assert (
        _feature_group_name_v2("multi_den_ratio_preopen_turnover_to_float_market_cap")
        == "preopen_auction"
    )
    assert (
        _feature_group_name_v2("multi_den_ratio_ask_depth_10_to_float_shares") == "book_depth_level"
    )
    assert (
        _feature_group_name_v2("multi_den_ratio_postopen_turnover_diff_1m_to_xs_median")
        == "trade_activity"
    )
