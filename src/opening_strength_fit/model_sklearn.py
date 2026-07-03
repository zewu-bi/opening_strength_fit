from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opening_strength_fit.model_features import _clean_xy, feature_columns
from opening_strength_fit.model_types import RidgePredictionModel


def fit_ridge_frame(
    train: pd.DataFrame,
    *,
    alpha: float = 1.0,
    feature_limit: int | None = None,
    target_col: str = "label",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(x, y)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    return (
        RidgePredictionModel(
            features=features,
            alpha=alpha,
            pipeline=pipeline,
            model_name="ridge",
            target_col=target_col,
        ),
        stats,
    )


def fit_gbm_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    max_iter: int = 100,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.0,
    random_state: int = 7,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if sample_weight_col:
        features = [column for column in features if column != sample_weight_col]
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "gbm",
                HistGradientBoostingRegressor(
                    max_iter=int(max_iter),
                    learning_rate=float(learning_rate),
                    max_leaf_nodes=int(max_leaf_nodes),
                    l2_regularization=float(l2_regularization),
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    pipeline.fit(x, y)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    return (
        RidgePredictionModel(
            features=features,
            alpha=float("nan"),
            pipeline=pipeline,
            model_name="gbm",
            target_col=target_col,
        ),
        stats,
    )


def fit_lightgbm_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    max_depth: int = -1,
    min_child_samples: int = 200,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    random_state: int = 7,
    n_jobs: int = -1,
    device_type: str = "cpu",
    max_bin: int | None = None,
    gpu_use_dp: bool = False,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise SystemExit(
            "model.name='lightgbm' requires the lightgbm package. "
            "Install project dependencies or rebuild the training image."
        ) from exc

    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    sample_weight = None
    if sample_weight_col:
        if sample_weight_col not in train.columns:
            raise SystemExit(f"missing sample weight column: {sample_weight_col}")
        sample_weight = (
            pd.to_numeric(train.loc[x.index, sample_weight_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
    lightgbm_params = {
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "min_child_samples": int(min_child_samples),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    device_type = str(device_type or "cpu").strip().lower()
    if device_type not in {"", "auto"}:
        lightgbm_params["device_type"] = device_type
    if max_bin is not None and int(max_bin) > 0:
        lightgbm_params["max_bin"] = int(max_bin)
    if device_type in {"gpu", "cuda"}:
        lightgbm_params["gpu_use_dp"] = bool(gpu_use_dp)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("lightgbm", LGBMRegressor(**lightgbm_params)),
        ]
    )
    fit_params = (
        {"lightgbm__sample_weight": sample_weight.to_numpy(dtype="float64")}
        if sample_weight is not None
        else {}
    )
    pipeline.fit(x, y, **fit_params)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    if sample_weight is not None:
        stats["sample_weight_mean"] = float(sample_weight.mean())
        stats["sample_weight_zero_rate"] = float((sample_weight <= 0.0).mean())
    return (
        RidgePredictionModel(
            features=features,
            alpha=float("nan"),
            pipeline=pipeline,
            model_name=f"lightgbm_{device_type or 'cpu'}",
            target_col=target_col,
        ),
        stats,
    )
