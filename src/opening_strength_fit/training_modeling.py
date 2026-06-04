from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_optional_int,
    config_str,
)
from opening_strength_fit.evaluation import (
    score_bucket_returns,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.feature_config import (
    feature_filters_from_config,
    feature_limit,
)
from opening_strength_fit.io import write_frame
from opening_strength_fit.model import (
    evaluate_prediction_frame,
    fit_gbm_frame,
    fit_lightgbm_frame,
    fit_ridge_frame,
    predict_frame,
)
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.stock_pool import (
    StockPoolConfig,
    configured_stock_pool_selection_frame,
    filter_configured_stock_pool_train,
    stock_pool_config_from_mapping,
)


def test_period_year(value: str) -> int:
    return int(pd.Timestamp(value).year)


def prediction_r2(predictions: pd.DataFrame, *, target_col: str = "label") -> float:
    target_col = target_col if target_col in predictions.columns else "label"
    frame = predictions.loc[predictions[target_col].notna() & predictions["prediction"].notna()]
    if len(frame) < 2:
        return float("nan")
    y = frame[target_col].astype("float64").to_numpy()
    y_hat = frame["prediction"].astype("float64").to_numpy()
    total = float(np.square(y - y.mean()).sum())
    if total == 0.0:
        return float("nan")
    residual = float(np.square(y - y_hat).sum())
    return 1.0 - residual / total


def metrics_row(
    *,
    run_name: str,
    split,
    train_stats: dict[str, int],
    predictions: pd.DataFrame,
    metrics: dict[str, object],
    top_summary: dict[str, object],
    model_name: str,
    alpha: float,
    feature_count: int,
    target_col: str,
    evaluation_settings: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_name,
        "test_year": test_period_year(split.test_start_date),
        "test_month": str(pd.Timestamp(split.test_start_date).to_period("M")),
        "train_start_date": split.train_start_date,
        "train_end_date": split.train_end_date,
        "test_start_date": split.test_start_date,
        "test_end_date": split.test_end_date,
        "train_rows": int(train_stats["rows"]),
        "train_dates": int(train_stats["dates"]),
        "train_symbols": int(train_stats["symbols"]),
        "test_rows": int(len(predictions)),
        "test_dates": int(metrics.get("dates", 0)),
        "test_symbols": int(metrics.get("symbols", 0)),
        "features": int(feature_count),
        "model_name": model_name,
        "alpha": float(alpha),
        "model_target_col": target_col,
        "model_test_r2": prediction_r2(predictions, target_col=target_col),
        "ic_mode": str(evaluation_settings["ic_mode"]),
        "selection_mode": str(evaluation_settings["selection_mode"]),
        "top_n": int(evaluation_settings["top_n"]),
        "stock_pool_enabled": bool(evaluation_settings.get("stock_pool_enabled", False)),
        "stock_pool_name": str(evaluation_settings.get("stock_pool_name", "")),
        "stock_pool_path": str(evaluation_settings.get("stock_pool_path", "")),
        "stock_pool_date_lag_sessions": int(
            evaluation_settings.get("stock_pool_date_lag_sessions", 0)
        ),
        "stock_pool_filter_train": bool(evaluation_settings.get("stock_pool_filter_train", False)),
        "stock_pool_filter_selection": bool(
            evaluation_settings.get("stock_pool_filter_selection", False)
        ),
        "stock_pool_add_feature": bool(evaluation_settings.get("stock_pool_add_feature", False)),
    }
    row.update(metrics)
    for key, value in top_summary.items():
        row[f"top_score_{key}"] = value
    return row


def fit_prediction_model(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
):
    model_name = config_str(config, "model", "name", "ridge").strip().lower()
    configured_feature_limit = feature_limit(args, config)
    target_col = config_str(config, "model", "target_col", "label")
    configured_feature_filters = feature_filters_from_config(config)
    if model_name == "ridge":
        return fit_ridge_frame(
            train,
            alpha=alpha,
            feature_limit=configured_feature_limit,
            target_col=target_col,
            feature_filters=configured_feature_filters,
        )
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return fit_gbm_frame(
            train,
            feature_limit=configured_feature_limit,
            target_col=target_col,
            feature_filters=configured_feature_filters,
            max_iter=config_int(config, "model", "max_iter", 100),
            learning_rate=config_float(config, "model", "learning_rate", 0.05),
            max_leaf_nodes=config_int(config, "model", "max_leaf_nodes", 31),
            l2_regularization=config_float(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            random_state=config_int(config, "model", "random_state", 7),
        )
    if model_name in {"lightgbm", "lgbm"}:
        return fit_lightgbm_frame(
            train,
            feature_limit=configured_feature_limit,
            target_col=target_col,
            sample_weight_col=config_str(config, "model", "sample_weight_col", ""),
            feature_filters=configured_feature_filters,
            n_estimators=config_int(config, "model", "n_estimators", 300),
            learning_rate=config_float(config, "model", "learning_rate", 0.03),
            num_leaves=config_int(config, "model", "num_leaves", 63),
            max_depth=config_int(config, "model", "max_depth", -1),
            min_child_samples=config_int(config, "model", "min_child_samples", 200),
            subsample=config_float(config, "model", "subsample", 1.0),
            colsample_bytree=config_float(config, "model", "colsample_bytree", 1.0),
            reg_alpha=config_float(config, "model", "reg_alpha", 0.0),
            reg_lambda=config_float(config, "model", "reg_lambda", 0.0),
            random_state=config_int(config, "model", "random_state", 7),
            n_jobs=config_int(config, "model", "n_jobs", -1),
            device_type=config_str(config, "model", "device_type", "cpu"),
            max_bin=config_optional_int(config, "model", "max_bin", None),
            gpu_use_dp=config_bool(config, "model", "gpu_use_dp", False),
        )
    raise SystemExit(f"unsupported model.name={model_name!r}; expected ridge, gbm, or lightgbm")


def model_config_payload(config: dict, alpha: float) -> dict[str, object]:
    model_name = config_str(config, "model", "name", "ridge").strip().lower()
    target_col = config_str(config, "model", "target_col", "label")
    if model_name == "ridge":
        return {"name": "ridge", "alpha": alpha, "target_col": target_col}
    if model_name in {"gbm", "hist_gbm", "hist_gradient_boosting"}:
        return {
            "name": "gbm",
            "target_col": target_col,
            "max_iter": config_int(config, "model", "max_iter", 100),
            "learning_rate": config_float(config, "model", "learning_rate", 0.05),
            "max_leaf_nodes": config_int(config, "model", "max_leaf_nodes", 31),
            "l2_regularization": config_float(
                config,
                "model",
                "l2_regularization",
                0.0,
            ),
            "random_state": config_int(config, "model", "random_state", 7),
        }
    if model_name in {"lightgbm", "lgbm"}:
        return {
            "name": "lightgbm",
            "target_col": target_col,
            "device_type": config_str(config, "model", "device_type", "cpu"),
            "n_estimators": config_int(config, "model", "n_estimators", 300),
            "learning_rate": config_float(config, "model", "learning_rate", 0.03),
            "num_leaves": config_int(config, "model", "num_leaves", 63),
            "max_depth": config_int(config, "model", "max_depth", -1),
            "min_child_samples": config_int(config, "model", "min_child_samples", 200),
            "subsample": config_float(config, "model", "subsample", 1.0),
            "colsample_bytree": config_float(config, "model", "colsample_bytree", 1.0),
            "reg_alpha": config_float(config, "model", "reg_alpha", 0.0),
            "reg_lambda": config_float(config, "model", "reg_lambda", 0.0),
            "random_state": config_int(config, "model", "random_state", 7),
            "n_jobs": config_int(config, "model", "n_jobs", -1),
            "max_bin": config_optional_int(config, "model", "max_bin", None),
            "gpu_use_dp": config_bool(config, "model", "gpu_use_dp", False),
            "sample_weight_col": config_str(config, "model", "sample_weight_col", ""),
        }
    return {"name": model_name}


def fit_predict_split(
    *,
    labeled: pd.DataFrame,
    split,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
    evaluation_settings: dict[str, object],
    stock_pool_settings: StockPoolConfig | None = None,
    stock_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    stock_pool_settings = stock_pool_settings or stock_pool_config_from_mapping(config)
    train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
    test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()
    train = filter_configured_stock_pool_train(train, stock_pool_settings, stock_pool)
    model, train_stats = fit_prediction_model(
        train,
        args=args,
        config=config,
        alpha=alpha,
    )
    predictions = predict_frame(model, test)
    if "valid_label" in predictions.columns:
        predictions = predictions.loc[predictions["valid_label"]].copy()
    predictions, selection_predictions, stock_pool_summary = configured_stock_pool_selection_frame(
        predictions,
        stock_pool_settings,
        stock_pool,
    )

    metrics = evaluate_prediction_frame(
        predictions,
        group_cols=evaluation_settings["_ic_group_cols"],
    )
    buckets = score_bucket_returns(
        predictions,
        bins=int(evaluation_settings["score_bins"]),
        group_cols=evaluation_settings["_bucket_group_cols"],
    )
    top_trades = top_score_trades(
        selection_predictions,
        top_n=int(evaluation_settings["top_n"]),
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary = summarize_trades(
        top_trades,
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary.update(stock_pool_summary)

    prediction_year = test_period_year(split.test_start_date)
    prediction_period = (
        str(pd.Timestamp(split.test_start_date).to_period("M"))
        if split.test_start_date[:7] == split.test_end_date[:7]
        else str(prediction_year)
    )
    write_frame(predictions, output_dir / f"predictions_{prediction_period}.parquet")
    buckets.to_csv(output_dir / f"score_buckets_{prediction_period}.csv", index=False)
    if stock_pool is not None and stock_pool_settings.filter_selection:
        pool_buckets = score_bucket_returns(
            selection_predictions,
            bins=int(evaluation_settings["score_bins"]),
            group_cols=evaluation_settings["_bucket_group_cols"],
        )
        pool_buckets.to_csv(
            output_dir / f"score_buckets_{prediction_period}_stock_pool.csv",
            index=False,
        )
    row = metrics_row(
        run_name=run_name,
        split=split,
        train_stats=train_stats,
        predictions=predictions,
        metrics=metrics,
        top_summary=top_summary,
        model_name=model.model_name,
        alpha=alpha,
        feature_count=len(model.features),
        target_col=model.target_col,
        evaluation_settings=evaluation_settings,
    )
    print_mapping(f"train_stats[{prediction_period}]", train_stats)
    print_mapping(f"prediction_metrics[{prediction_period}]", metrics)
    print_mapping(
        f"top_score_summary[{prediction_period},top_n={evaluation_settings['top_n']}]",
        top_summary,
    )
    if stock_pool_summary:
        print_mapping(f"stock_pool_summary[{prediction_period}]", stock_pool_summary)
    return predictions, row, train_stats
