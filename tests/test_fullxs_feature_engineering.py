import numpy as np
import pandas as pd

from opening_strength_fit.features import (
    add_historical_same_minute_surprise_features,
    add_path_shape_confirmation_features,
)
from opening_strength_fit.model import (
    ClockSegmentPredictionModel,
    RidgePredictionModel,
    predict_frame,
)
from opening_strength_fit.training_labeled import apply_target_transform_from_config


class ConstantPipeline:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, dtype="float64")


def test_historical_same_minute_surprise_uses_prior_dates_only():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "decision_target_timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:31:00",
                    "2024-01-02 09:31:00",
                    "2024-01-03 09:31:00",
                ]
            ),
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:31:00",
                    "2024-01-02 09:31:00",
                    "2024-01-03 09:31:00",
                ]
            ),
            "volume_diff_1t": [10.0, 12.0, 14.0],
        }
    )

    out = add_historical_same_minute_surprise_features(
        frame,
        columns=("volume_diff_1t",),
        windows=(2,),
        min_periods=2,
        modes=("zscore", "ratio"),
    )

    assert out["hist_surprise_volume_diff_1t_2d_zscore"].iloc[:2].isna().all()
    assert np.isclose(
        out["hist_surprise_volume_diff_1t_2d_zscore"].iloc[2],
        (14.0 - 11.0) / np.std([10.0, 12.0], ddof=1),
    )
    assert np.isclose(out["hist_surprise_volume_diff_1t_2d_ratio"].iloc[2], 14.0 / 11.0)


def test_path_shape_confirmation_keeps_series_semantics():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "symbol": ["000001.SZ"] * 3,
            "decision_target_timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:32:00",
                    "2024-01-02 09:33:00",
                ]
            ),
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:31:00",
                    "2024-01-02 09:32:00",
                    "2024-01-02 09:33:00",
                ]
            ),
            "mid_price": [10.0, 10.1, 10.05],
            "spread_bps": [20.0, 18.0, 19.0],
            "depth_imbalance_10": [0.10, 0.15, 0.12],
            "bid_depth_10": [1000.0, 1030.0, 1020.0],
            "ask_depth_10": [900.0, 890.0, 910.0],
        }
    )

    out = add_path_shape_confirmation_features(frame, windows=(2, 3))

    assert "path_shape_return_positive_fraction" in out.columns
    assert np.isclose(out.loc[1, "path_shape_return_positive_fraction"], 0.5)
    assert np.isfinite(out.loc[1, "path_shape_spread_compress_after_upmove"])
    assert np.isclose(out.loc[1, "path_shape_imbalance_slope_roll2"], 0.05)
    assert np.isclose(out.loc[2, "path_shape_imbalance_slope_roll2"], -0.03)
    assert np.isclose(out.loc[2, "path_shape_imbalance_slope_roll3"], 0.01)


def test_rank_centered_target_transform_groups_by_date_and_clock():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "decision_target_timestamp": pd.to_datetime(["2024-01-02 09:31:00"] * 3),
            "target_label": [2.0, 4.0, 6.0],
            "valid_label": [True, True, True],
        }
    )
    config = {
        "target_transform": {
            "enabled": True,
            "mode": "rank_centered",
            "source_col": "target_label",
            "output_col": "target_label_rank_centered",
            "group_cols": ["date", "decision_target_timestamp"],
        }
    }

    out = apply_target_transform_from_config(frame, config)

    assert np.allclose(out["target_label_rank_centered"], [-1.0 / 3.0, 0.0, 1.0 / 3.0])
    assert out["valid_label"].all()


def test_clock_segment_prediction_routes_by_decision_time():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_time": ["09:31:00", "09:36:00"],
            "x": [1.0, 2.0],
            "label": [0.0, 0.0],
        }
    )
    early = RidgePredictionModel(features=["x"], alpha=np.nan, pipeline=ConstantPipeline(1.0))
    late = RidgePredictionModel(features=["x"], alpha=np.nan, pipeline=ConstantPipeline(2.0))
    model = ClockSegmentPredictionModel(
        features=["x"],
        segment_models=[
            ("early", ("09:31:00",), early),
            ("late", ("09:36:00",), late),
        ],
    )

    out = predict_frame(model, frame)

    assert out["prediction"].tolist() == [1.0, 2.0]
