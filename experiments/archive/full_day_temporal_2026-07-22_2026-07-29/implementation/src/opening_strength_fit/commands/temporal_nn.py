from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.config import (
    config_float,
    config_int,
    config_str,
    config_value,
    load_toml,
    run_id,
)
from opening_strength_fit.io import write_frame_atomic, write_json
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.temporal_dataset import (
    DEFAULT_LATEST_CLOCKS,
    list_sequence_paths,
    rolling_date_bounds,
)
from opening_strength_fit.temporal_modeling import (
    evaluate_temporal_model,
    fit_temporal_model,
)


def _latest_clocks(config: dict) -> dict[str, str]:
    configured = config_value(config, "temporal_features", "latest_clocks", {})
    if not isinstance(configured, dict):
        raise SystemExit("[temporal_features].latest_clocks must be a TOML table")
    latest = dict(DEFAULT_LATEST_CLOCKS)
    latest.update({str(key): str(value) for key, value in configured.items()})
    return latest


def _raw_scales(config: dict) -> dict[str, float]:
    configured = config_value(config, "temporal_features", "raw_scales", {})
    if not isinstance(configured, dict):
        raise SystemExit("[temporal_features].raw_scales must be a TOML table")
    return {str(key): float(value) for key, value in configured.items()}


def _metrics_by_period(
    predictions: pd.DataFrame,
    *,
    evaluation_universe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = str(evaluation_universe).strip().lower()
    eligible = predictions["evaluation_eligible"].eq(1)
    if normalized == "all_a":
        work = predictions.loc[eligible].copy()
    elif normalized == "pool_l":
        work = predictions.loc[eligible & predictions["stock_pool_member"].eq(1)].copy()
    else:
        raise ValueError(f"unsupported evaluation_universe={evaluation_universe!r}")
    work["year"] = work["date"].astype(str).str[:4].astype(int)
    work["month"] = work["date"].astype(str).str[:7]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        daily = []
        top = []
        for _, day in group.groupby("date", observed=True, sort=False):
            daily.append(day["score"].rank().corr(day["target"].rank()))
            count = min(100, len(day))
            selected = day.nlargest(count, "score")
            top.append(float(selected["target"].mean() - day["target"].mean()))
        return pd.Series(
            {
                "rows": len(group),
                "days": group["date"].nunique(),
                "daily_rank_ic": float(np.nanmean(daily)),
                "daily_rank_ic_std": float(np.nanstd(daily, ddof=1)),
                "top100_excess": float(np.nanmean(top)),
            }
        )

    yearly = (
        work.groupby("year", observed=True, sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    monthly = (
        work.groupby("month", observed=True, sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    return yearly, monthly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a sequence-to-one neural network on full-day return paths."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-monthly", action="store_true")
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--test-stride-months", type=int, default=None)
    parser.add_argument("--test-start-month", default="")
    parser.add_argument("--test-end-month", default="")
    args = parser.parse_args()

    import torch

    config = load_toml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_root = Path(config_str(config, "temporal_data", "sequence_root", ""))
    if not (sequence_root / "_SUCCESS").exists():
        raise SystemExit(f"temporal sequence cache is incomplete: {sequence_root}")
    input_mask_root_raw = config_str(
        config,
        "temporal_data",
        "input_mask_sequence_root",
        "",
    ).strip()
    input_mask_sequence_root = Path(input_mask_root_raw) if input_mask_root_raw else None
    if (
        input_mask_sequence_root is not None
        and not (input_mask_sequence_root / "_SUCCESS").exists()
    ):
        raise SystemExit(
            f"temporal input mask sequence cache is incomplete: {input_mask_sequence_root}"
        )
    train_months = args.train_months or config_int(config, "window", "train_months", 36)
    validation_months = config_int(config, "window", "validation_months", 3)
    test_start_month = args.test_start_month or config_str(config, "window", "test_start_month", "")
    test_end_month = args.test_end_month or config_str(config, "window", "test_end_month", "")
    if not test_start_month or not test_end_month:
        raise SystemExit("temporal NN requires test_start_month and test_end_month")
    bounds = rolling_date_bounds(
        test_start_month=test_start_month,
        test_end_month=test_end_month,
        train_months=train_months,
        validation_months=validation_months,
    )
    train_paths = list_sequence_paths(
        sequence_root,
        start_date=bounds["train_start_date"],
        end_date=bounds["train_end_date"],
    )
    validation_paths = list_sequence_paths(
        sequence_root,
        start_date=bounds["validation_start_date"],
        end_date=bounds["validation_end_date"],
    )
    test_paths = list_sequence_paths(
        sequence_root,
        start_date=bounds["test_start_date"],
        end_date=bounds["test_end_date"],
    )
    device = config_str(
        config,
        "training",
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("temporal NN requested CUDA but no GPU is available")
    latest_clocks = _latest_clocks(config)
    evaluation_universe = config_str(
        config,
        "training",
        "evaluation_universe",
        "pool_l",
    )
    common = {
        "device": device,
        "batch_size": config_int(config, "training", "batch_size", 1024),
        "value_mode": config_str(
            config,
            "temporal_features",
            "value_mode",
            "cross_section_rank",
        ),
        "latest_clocks": latest_clocks,
        "raw_scale": config_float(config, "temporal_features", "raw_scale", 0.02),
        "raw_scales": _raw_scales(config),
        "evaluation_universe": evaluation_universe,
        "top_n": config_int(config, "evaluation", "top_n", 100),
        "target_mode": config_str(config, "training", "target_mode", "rank"),
        "input_mask_sequence_root": input_mask_sequence_root,
    }
    model, history, fit_trace = fit_temporal_model(
        train_paths,
        validation_paths,
        architecture=config_str(config, "model", "architecture", "tcn"),
        train_universe=config_str(config, "training", "train_universe", "pool_l"),
        hidden_width=config_int(config, "model", "hidden_width", 64),
        dropout=config_float(config, "model", "dropout", 0.10),
        epochs=config_int(config, "training", "epochs", 6),
        learning_rate=config_float(config, "training", "learning_rate", 0.001),
        weight_decay=config_float(config, "training", "weight_decay", 0.0001),
        loss_name=config_str(config, "training", "loss", "huber"),
        head_fraction=config_float(config, "training", "head_fraction", 0.10),
        huber_delta=config_float(config, "training", "huber_delta", 0.25),
        target_winsor_lower_quantile=config_float(
            config,
            "training",
            "target_winsor_lower_quantile",
            0.01,
        ),
        target_winsor_upper_quantile=config_float(
            config,
            "training",
            "target_winsor_upper_quantile",
            0.99,
        ),
        selection_metric=config_str(
            config,
            "training",
            "selection_metric",
            "top_n_excess",
        ),
        patience=config_int(config, "training", "patience", 2),
        seed=config_int(config, "training", "seed", 20260724),
        **common,
    )
    target_bounds_raw = fit_trace.get("target_winsor_bounds")
    target_winsor_bounds = (
        (float(target_bounds_raw[0]), float(target_bounds_raw[1]))
        if isinstance(target_bounds_raw, list) and len(target_bounds_raw) == 2
        else None
    )
    test_metrics, predictions = evaluate_temporal_model(
        model,
        test_paths,
        include_predictions=True,
        target_winsor_bounds=target_winsor_bounds,
        **common,
    )
    yearly, monthly = _metrics_by_period(
        predictions,
        evaluation_universe=evaluation_universe,
    )
    history.to_csv(output_dir / "training_history.csv", index=False)
    write_frame_atomic(predictions, output_dir / "predictions.parquet")
    yearly.to_csv(output_dir / "metrics_by_year.csv", index=False)
    monthly.to_csv(output_dir / "metrics_by_month.csv", index=False)
    torch.save(model.state_dict(), output_dir / "model.pt")
    trace = {
        "run_id": run_id(config, args.config),
        "architecture": config_str(config, "model", "architecture", "tcn"),
        "bounds": bounds,
        "train_days": len(train_paths),
        "validation_days": len(validation_paths),
        "test_days": len(test_paths),
        "latest_clocks": latest_clocks,
        "input_mask_sequence_root": str(input_mask_sequence_root)
        if input_mask_sequence_root is not None
        else None,
        "test_metrics": test_metrics,
        **fit_trace,
    }
    write_json(output_dir / "temporal_nn_trace.json", trace, atomic=True)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    print_mapping("temporal_nn", trace)
