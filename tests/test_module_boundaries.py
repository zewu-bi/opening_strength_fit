from __future__ import annotations

from pathlib import Path

import opening_strength_fit.features as features
import opening_strength_fit.model as model

ROOT = Path(__file__).resolve().parents[1]


def _line_count(path: str) -> int:
    return len((ROOT / path).read_text(encoding="utf-8").splitlines())


def test_compatibility_modules_stay_thin() -> None:
    assert _line_count("src/opening_strength_fit/features.py") <= 120
    assert _line_count("src/opening_strength_fit/model.py") <= 120


def test_feature_public_api_is_exported_from_compatibility_module() -> None:
    assert features.add_postopen_v2_decision_features.__module__.endswith("features_postopen")
    assert features.add_historical_same_minute_surprise_features.__module__.endswith(
        "features_history"
    )
    assert features.add_price_scale_features.__module__.endswith("features_relative")
    assert features.build_feature_frame.__module__.endswith("features_base")


def test_model_public_api_is_exported_from_compatibility_module() -> None:
    assert model.feature_columns.__module__.endswith("model_features")
    assert model.fit_lightgbm_frame.__module__.endswith("model_sklearn")
    assert model.fit_torch_mlp_frame.__module__.endswith("model_torch")
    assert model.predict_frame.__module__.endswith("model_prediction")
    assert model.evaluate_prediction_frame.__module__.endswith("model_metrics")
