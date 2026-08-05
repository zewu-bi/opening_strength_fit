from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.torch_model.preprocessing import (
    _fit_symbol_train_standardization,
    _standardized_float_matrix,
)


def test_global_standardization_can_select_rows_without_feature_frame_copy() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": pd.Series([1.0, np.inf, 3.0, 5.0], dtype="float32"),
            "feature_b": pd.Series([2.0, 4.0, np.nan, 8.0], dtype="float32"),
            "target": [1.0, np.nan, 2.0, 3.0],
        }
    )
    selected = frame["target"].notna()

    values, mean, scale = _standardized_float_matrix(
        frame,
        ["feature_a", "feature_b"],
        row_mask=selected,
        column_block_size=1,
        stats_row_block_size=2,
    )

    np.testing.assert_allclose(mean, [3.0, 5.0])
    np.testing.assert_allclose(scale, [np.sqrt(8.0 / 3.0), 3.0], rtol=1e-6)
    np.testing.assert_allclose(values[0], [-np.sqrt(1.5), -1.0], rtol=1e-6)
    np.testing.assert_allclose(values[1], [0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(values[2], [np.sqrt(1.5), 1.0], rtol=1e-6)


def test_symbol_train_zscore_uses_each_symbol_own_training_scale() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B", "C"],
            "feature_price": [10.0, 14.0, 100.0, 120.0, 5.0],
            "feature_depth": [1.0, 3.0, 10.0, 14.0, 7.0],
        }
    )
    features = ["feature_price", "feature_depth"]
    mean, scale, keys, group_mean, group_scale = _fit_symbol_train_standardization(
        frame,
        features,
        group_col="symbol",
    )

    values, _, _ = _standardized_float_matrix(
        frame,
        features,
        mean=mean,
        scale=scale,
        group_col="symbol",
        group_keys=keys,
        group_mean=group_mean,
        group_scale=group_scale,
    )

    np.testing.assert_allclose(values[:2, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(values[2:4, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(values[:2, 1], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(values[2:4, 1], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(values[4], [0.0, 0.0], atol=1e-6)


def test_symbol_train_zscore_falls_back_to_global_for_unseen_symbol() -> None:
    train = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "feature_price": [10.0, 14.0, 100.0, 120.0],
        }
    )
    features = ["feature_price"]
    mean, scale, keys, group_mean, group_scale = _fit_symbol_train_standardization(
        train,
        features,
        group_col="symbol",
    )
    test = pd.DataFrame({"symbol": ["A", "D"], "feature_price": [16.0, mean[0] + scale[0]]})

    values, _, _ = _standardized_float_matrix(
        test,
        features,
        mean=mean,
        scale=scale,
        group_col="symbol",
        group_keys=keys,
        group_mean=group_mean,
        group_scale=group_scale,
    )

    np.testing.assert_allclose(values[0, 0], 2.0, atol=1e-6)
    np.testing.assert_allclose(values[1, 0], 1.0, atol=1e-6)
