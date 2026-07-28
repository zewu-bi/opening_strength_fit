from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from opening_strength_fit.model import fit_lightgbm_frame, predict_frame
from opening_strength_fit.model_preprocessing import lightgbm_feature_value_frame


class _FakeLGBMRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **params) -> None:
        self.params = params
        self.fit_x = None
        self.predict_x = None

    def fit(self, x, y, sample_weight=None):
        self.fit_x = x.copy()
        self.fit_y = np.asarray(y)
        self.fit_sample_weight = sample_weight
        self.is_fitted_ = True
        return self

    def predict(self, x):
        self.predict_x = x.copy()
        return np.zeros(len(x), dtype="float64")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_target_timestamp": [
                pd.Timestamp("2025-01-02 09:31:00"),
                pd.Timestamp("2025-01-02 09:31:00"),
            ],
            "mid_price": [10.0, 20.0],
            "turnover_diff_1t": [1_000_000.0, 4_000_000.0],
            "spread_bps": [10.0, 20.0],
            "multi_den_ratio_turnover_diff_1t_to_float_market_cap": [0.01, 0.02],
            "total_market_cap": [100_000_000.0, 200_000_000.0],
            "total_shares": [10_000_000.0, 10_000_000.0],
            "target_label": [0.1, 0.2],
            "valid_label": [True, True],
        }
    )


def test_lightgbm_append_mode_adds_only_materially_changed_v3_features() -> None:
    source_features = [
        "mid_price",
        "turnover_diff_1t",
        "spread_bps",
        "multi_den_ratio_turnover_diff_1t_to_float_market_cap",
    ]

    out, model_features = lightgbm_feature_value_frame(
        _frame(),
        source_features,
        feature_value_transform="mechanismized_v3_dimensionless_328",
        feature_value_transform_output="raw_plus_transformed",
        feature_value_transform_prefix="mech_v3_",
        extra_columns=("target_label", "valid_label"),
    )

    assert model_features == [
        *source_features,
        "mech_v3_mid_price",
        "mech_v3_turnover_diff_1t",
    ]
    assert out["turnover_diff_1t"].tolist() == [1_000_000.0, 4_000_000.0]
    assert out["mech_v3_turnover_diff_1t"].tolist() == pytest.approx([0.01, 0.02])
    assert "mech_v3_spread_bps" not in out
    assert "mech_v3_multi_den_ratio_turnover_diff_1t_to_float_market_cap" not in out


def test_lightgbm_uses_native_nan_real_bagging_and_prediction_transform(
    monkeypatch,
) -> None:
    fake_module = types.ModuleType("lightgbm")
    fake_module.LGBMRegressor = _FakeLGBMRegressor
    monkeypatch.setitem(sys.modules, "lightgbm", fake_module)
    frame = _frame()
    frame.loc[1, "spread_bps"] = np.nan

    model, stats = fit_lightgbm_frame(
        frame,
        target_col="target_label",
        feature_filters={
            "include_columns": (
                "mid_price",
                "turnover_diff_1t",
                "spread_bps",
            ),
            "include_prefixes": (),
            "include_patterns": (),
            "drop_columns": (),
            "drop_prefixes": (),
            "drop_patterns": (),
        },
        subsample=0.85,
        subsample_freq=1,
        feature_value_transform="mechanismized_v3_dimensionless_328",
        feature_value_transform_output="raw_plus_transformed",
    )

    assert list(model.pipeline.named_steps) == ["lightgbm"]
    estimator = model.pipeline.named_steps["lightgbm"]
    assert estimator.params["subsample"] == 0.85
    assert estimator.params["subsample_freq"] == 1
    assert np.isnan(estimator.fit_x["spread_bps"].iloc[1])
    assert stats["features"] == 5

    predictions = predict_frame(model, frame)

    assert predictions["prediction"].tolist() == [0.0, 0.0]
    assert estimator.predict_x.columns.tolist() == model.features
    assert estimator.predict_x["mech_v3_turnover_diff_1t"].tolist() == pytest.approx([0.01, 0.02])
