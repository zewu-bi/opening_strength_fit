from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_optional_int,
    config_str,
    config_value,
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
    ClockSegmentPredictionModel,
    EnsemblePredictionModel,
    evaluate_prediction_frame,
    fit_gbm_frame,
    fit_lightgbm_frame,
    fit_ridge_frame,
    fit_torch_mlp_frame,
    predict_frame,
)
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import normalize_clock_time
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


def _model_group_embedding_dims(config: dict) -> dict[str, int]:
    value = config_value(config, "model", "group_embedding_dims", {})
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise SystemExit("[model].group_embedding_dims must be a table of group = dim")
    return {str(group): int(dim) for group, dim in value.items() if dim not in (None, "")}


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
    if model_name == "ensemble":
        return fit_ensemble_prediction_model(
            train,
            args=args,
            config=config,
            alpha=alpha,
        )
    if model_name in {"clock_segment_lightgbm", "clock_segment_lgbm", "segmented_lightgbm"}:
        return fit_clock_segment_prediction_model(
            train,
            args=args,
            config=config,
            alpha=alpha,
        )
    return fit_single_prediction_model(
        train,
        args=args,
        config=config,
        alpha=alpha,
    )


def fit_single_prediction_model(
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
    if model_name in {"torch_mlp", "mlp", "nn"}:
        hidden_layers = tuple(
            int(value)
            for value in config_list(config, "model", "hidden_layers", ["512", "256", "128"])
        )
        return fit_torch_mlp_frame(
            train,
            feature_limit=configured_feature_limit,
            target_col=target_col,
            sample_weight_col=config_str(config, "model", "sample_weight_col", ""),
            feature_filters=configured_feature_filters,
            hidden_layers=hidden_layers,
            architecture=config_str(config, "model", "architecture", "mlp"),
            group_embedding_dim=config_int(config, "model", "group_embedding_dim", 48),
            group_embedding_dims=_model_group_embedding_dims(config),
            fusion_dim=config_int(config, "model", "fusion_dim", 256),
            block_hidden_dim=config_int(config, "model", "block_hidden_dim", 512),
            num_blocks=config_int(config, "model", "num_blocks", 2),
            transformer_heads=config_int(config, "model", "transformer_heads", 4),
            dropout=config_float(config, "model", "dropout", 0.1),
            activation=config_str(config, "model", "activation", "relu"),
            batch_size=config_int(config, "model", "batch_size", 32768),
            predict_batch_size=config_int(config, "model", "predict_batch_size", 65536),
            learning_rate=config_float(config, "model", "learning_rate", 3e-4),
            weight_decay=config_float(config, "model", "weight_decay", 1e-4),
            max_epochs=config_int(config, "model", "max_epochs", 8),
            validation_fraction=config_float(config, "model", "validation_fraction", 0.02),
            validation_max_rows=config_int(config, "model", "validation_max_rows", 250_000),
            early_stopping_patience=config_int(config, "model", "early_stopping_patience", 2),
            loss=config_str(config, "model", "loss", "mse"),
            device=config_str(config, "model", "device", "auto"),
            random_state=config_int(config, "model", "random_state", 7),
            num_workers=config_int(config, "model", "num_workers", 0),
            gate_diagnostics_max_rows=config_int(
                config,
                "model",
                "gate_diagnostics_max_rows",
                200_000,
            ),
        )
    raise SystemExit(
        f"unsupported model.name={model_name!r}; "
        "expected ridge, gbm, lightgbm, torch_mlp, ensemble, or clock_segment_lightgbm"
    )


def _clock_series(frame: pd.DataFrame) -> pd.Series:
    if "decision_time" in frame.columns:
        raw = frame["decision_time"].astype(str)
        extracted = raw.str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False).fillna("")
        return extracted.map(lambda value: normalize_clock_time(value) if value else "")
    time_col = "decision_target_timestamp" if "decision_target_timestamp" in frame else "timestamp"
    return pd.to_datetime(frame[time_col], errors="coerce").dt.strftime("%H:%M:%S").fillna("")


def _segment_model_config(config: dict, segment: dict, base_model_name: str) -> dict:
    merged = {
        section: dict(values) if isinstance(values, dict) else values
        for section, values in config.items()
    }
    model_section = dict(merged.get("model", {}))
    for key in (
        "segments",
        "base_model_name",
        "fallback",
        "fallback_model_name",
        "fallback_min_rows",
    ):
        model_section.pop(key, None)
    model_section["name"] = str(segment.get("model_name", base_model_name))
    for key, value in segment.items():
        if key in {"segment_name", "decision_times", "model_name", "fallback"}:
            continue
        model_section[key] = value
    merged["model"] = model_section
    return merged


def fit_clock_segment_prediction_model(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
):
    model_section = config.get("model", {})
    segments = model_section.get("segments", []) if isinstance(model_section, dict) else []
    if not isinstance(segments, list) or not segments:
        raise SystemExit(
            "model.name='clock_segment_lightgbm' requires at least one [[model.segments]] table"
        )

    base_model_name = config_str(config, "model", "base_model_name", "lightgbm")
    clock = _clock_series(train)
    fitted_segments = []
    segment_stats = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise SystemExit("each [[model.segments]] entry must be a table")
        decision_times = tuple(
            normalize_clock_time(str(value))
            for value in segment.get("decision_times", [])
            if str(value).strip()
        )
        if not decision_times:
            raise SystemExit("each [[model.segments]] entry requires decision_times")
        mask = clock.isin(set(decision_times))
        if not bool(mask.any()):
            raise SystemExit(
                f"clock segment {segment.get('segment_name', index)!r} has no training rows"
            )

        segment_config = _segment_model_config(config, segment, base_model_name)
        model, stats = fit_single_prediction_model(
            train.loc[mask].copy(),
            args=args,
            config=segment_config,
            alpha=alpha,
        )
        segment_name = str(segment.get("segment_name", f"segment_{index}"))
        fitted_segments.append((segment_name, decision_times, model))
        segment_stats.append(
            {
                "index": index,
                "segment_name": segment_name,
                "decision_times": list(decision_times),
                "model_name": model.model_name,
                **stats,
            }
        )

    features = list(
        dict.fromkeys(feature for _, _, model in fitted_segments for feature in model.features)
    )
    target_col = config_str(config, "model", "target_col", "label")
    stats = {
        "rows": int(sum(item["rows"] for item in segment_stats)),
        "dates": int(train["date"].nunique()) if "date" in train.columns else 0,
        "symbols": int(train["symbol"].nunique()) if "symbol" in train.columns else 0,
        "features": len(features),
        "segments": segment_stats,
    }
    return (
        ClockSegmentPredictionModel(
            features=features,
            segment_models=fitted_segments,
            fallback_model=None,
            model_name="clock_segment_lightgbm",
            target_col=target_col,
        ),
        stats,
    )


def _member_model_config(config: dict, member: dict) -> dict:
    merged = {
        section: dict(values) if isinstance(values, dict) else values
        for section, values in config.items()
    }
    model_section = dict(merged.get("model", {}))
    for key in ("members", "weights", "combine_mode", "rank_group_cols"):
        model_section.pop(key, None)
    for key, value in member.items():
        if key == "weight":
            continue
        model_section[key] = value
    merged["model"] = model_section
    return merged


def fit_ensemble_prediction_model(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    alpha: float,
):
    model_section = config.get("model", {})
    members = model_section.get("members", []) if isinstance(model_section, dict) else []
    if not isinstance(members, list) or not members:
        raise SystemExit("model.name='ensemble' requires at least one [[model.members]] table")

    fitted_models = []
    weights = []
    member_stats = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise SystemExit("each [[model.members]] entry must be a table")
        member_config = _member_model_config(config, member)
        member_alpha = float(member.get("alpha", alpha))
        model, stats = fit_single_prediction_model(
            train,
            args=args,
            config=member_config,
            alpha=member_alpha,
        )
        fitted_models.append(model)
        weights.append(float(member.get("weight", 1.0)))
        member_stats.append(
            {
                "index": index,
                "model_name": model.model_name,
                "weight": float(member.get("weight", 1.0)),
                "features": int(len(model.features)),
                **stats,
            }
        )

    features = list(dict.fromkeys(feature for model in fitted_models for feature in model.features))
    base_stats = member_stats[0]
    stats = {
        "rows": int(base_stats["rows"]),
        "dates": int(base_stats["dates"]),
        "symbols": int(base_stats["symbols"]),
        "features": len(features),
        "members": member_stats,
    }
    member_names = "+".join(model.model_name for model in fitted_models)
    target_col = config_str(config, "model", "target_col", "label")
    combine_mode = config_str(config, "model", "combine_mode", "rank")
    return (
        EnsemblePredictionModel(
            features=features,
            alpha=float("nan"),
            models=fitted_models,
            weights=weights,
            combine_mode=combine_mode,
            rank_group_cols=tuple(
                config_list(
                    config,
                    "model",
                    "rank_group_cols",
                    ["date", "decision_target_timestamp"],
                )
            ),
            model_name=f"ensemble_{combine_mode}_{member_names}",
            target_col=target_col,
        ),
        stats,
    )


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
    if model_name in {"torch_mlp", "mlp", "nn"}:
        return {
            "name": "torch_mlp",
            "target_col": target_col,
            "hidden_layers": config_list(
                config,
                "model",
                "hidden_layers",
                ["512", "256", "128"],
            ),
            "architecture": config_str(config, "model", "architecture", "mlp"),
            "group_embedding_dim": config_int(config, "model", "group_embedding_dim", 48),
            "group_embedding_dims": _model_group_embedding_dims(config),
            "fusion_dim": config_int(config, "model", "fusion_dim", 256),
            "block_hidden_dim": config_int(config, "model", "block_hidden_dim", 512),
            "num_blocks": config_int(config, "model", "num_blocks", 2),
            "transformer_heads": config_int(config, "model", "transformer_heads", 4),
            "dropout": config_float(config, "model", "dropout", 0.1),
            "activation": config_str(config, "model", "activation", "relu"),
            "batch_size": config_int(config, "model", "batch_size", 32768),
            "predict_batch_size": config_int(config, "model", "predict_batch_size", 65536),
            "learning_rate": config_float(config, "model", "learning_rate", 3e-4),
            "weight_decay": config_float(config, "model", "weight_decay", 1e-4),
            "max_epochs": config_int(config, "model", "max_epochs", 8),
            "validation_fraction": config_float(config, "model", "validation_fraction", 0.02),
            "validation_max_rows": config_int(config, "model", "validation_max_rows", 250_000),
            "early_stopping_patience": config_int(
                config,
                "model",
                "early_stopping_patience",
                2,
            ),
            "loss": config_str(config, "model", "loss", "mse"),
            "device": config_str(config, "model", "device", "auto"),
            "random_state": config_int(config, "model", "random_state", 7),
            "num_workers": config_int(config, "model", "num_workers", 0),
            "sample_weight_col": config_str(config, "model", "sample_weight_col", ""),
            "gate_diagnostics_max_rows": config_int(
                config,
                "model",
                "gate_diagnostics_max_rows",
                200_000,
            ),
        }
    if model_name == "ensemble":
        model_section = config.get("model", {})
        members = model_section.get("members", []) if isinstance(model_section, dict) else []
        return {
            "name": "ensemble",
            "target_col": target_col,
            "combine_mode": config_str(config, "model", "combine_mode", "rank"),
            "rank_group_cols": config_list(
                config,
                "model",
                "rank_group_cols",
                ["date", "decision_target_timestamp"],
            ),
            "members": members,
        }
    if model_name in {"clock_segment_lightgbm", "clock_segment_lgbm", "segmented_lightgbm"}:
        model_section = config.get("model", {})
        segments = model_section.get("segments", []) if isinstance(model_section, dict) else []
        return {
            "name": "clock_segment_lightgbm",
            "target_col": target_col,
            "base_model_name": config_str(config, "model", "base_model_name", "lightgbm"),
            "segments": segments,
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
