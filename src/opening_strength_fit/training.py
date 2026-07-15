from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import (
    config_float,
    config_str,
    run_id,
)
from opening_strength_fit.evaluation import score_bucket_returns
from opening_strength_fit.io import write_frame_atomic, write_json
from opening_strength_fit.reports import (
    dataset_summary,
    metrics_by_year_from_windows,
    print_mapping,
)
from opening_strength_fit.stock_pool import (
    add_configured_stock_pool_feature,
    apply_stock_pool_cli_overrides,
    configured_stock_pool_selection_frame,
    load_configured_stock_pool,
    stock_pool_config_from_mapping,
    stock_pool_runtime_summary,
)
from opening_strength_fit.training_args import load_run_config
from opening_strength_fit.training_data import (
    load_clickhouse_labeled_frame,
    load_labeled_pvc_frame,
    load_training_frame,
    resolve_data_source,
)
from opening_strength_fit.training_labeled import (
    apply_guard_features_from_config,
    apply_sample_weight_from_config,
)
from opening_strength_fit.training_modeling import (
    fit_predict_split,
    model_config_payload,
)
from opening_strength_fit.training_windows import (
    build_evaluation_settings,
    date_splits,
    resolve_window_mode,
)


def _write_success_marker(output_dir: Path, *, run_name: str, windows: int) -> None:
    write_json(
        output_dir / "_SUCCESS",
        {
            "run_id": run_name,
            "windows": windows,
            "status": "completed",
            "format_version": 1,
        },
        atomic=True,
    )


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_from_args(args: argparse.Namespace) -> None:
    config = load_run_config(args.config)
    config = apply_stock_pool_cli_overrides(config, args)
    stock_pool_settings = stock_pool_config_from_mapping(config)
    tick_path = (
        args.input
        or config_str(config, "data", "tick_path", "")
        or os.environ.get("OPENING_STRENGTH_TICK_PATH", "")
    )
    data_source = resolve_data_source(args, config, tick_path)
    if data_source == "path" and not tick_path:
        raise SystemExit(
            "No tick data path supplied. Set [data].tick_path, --input, "
            'OPENING_STRENGTH_TICK_PATH, or use [data].source = "clickhouse".'
        )

    run_name = run_id(config, args.config) if args.config else "local_ridge_opening"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_source == "clickhouse":
        labeled = load_clickhouse_labeled_frame(args, config)
    elif data_source == "labeled_pvc":
        labeled = load_labeled_pvc_frame(args, config)
    else:
        labeled = load_training_frame(tick_path, args, config)
    labeled = apply_guard_features_from_config(labeled, config)
    labeled = apply_sample_weight_from_config(labeled, config)
    stock_pool = load_configured_stock_pool(stock_pool_settings)
    if stock_pool is not None:
        print_mapping("stock_pool", stock_pool_runtime_summary(stock_pool_settings, stock_pool))
        labeled = add_configured_stock_pool_feature(
            labeled,
            stock_pool_settings,
            stock_pool,
        )
    print_mapping("dataset", dataset_summary(labeled))

    alpha = args.alpha if args.alpha is not None else config_float(config, "model", "alpha", 1.0)
    evaluation_settings = build_evaluation_settings(config, args)
    splits = date_splits(labeled, args, config)
    print_mapping(
        "split_plan",
        {
            "windows": len(splits),
            "first_test": splits[0].test_start_date,
            "last_test": splits[-1].test_end_date,
            "mode": resolve_window_mode(args, config),
        },
    )

    prediction_frames = []
    metric_rows = []
    train_stats_by_window = {}
    for split in splits:
        predictions, metrics_row, train_stats = fit_predict_split(
            labeled=labeled,
            split=split,
            run_name=run_name,
            output_dir=output_dir,
            args=args,
            config=config,
            alpha=alpha,
            evaluation_settings=evaluation_settings,
            stock_pool_settings=stock_pool_settings,
            stock_pool=stock_pool,
        )
        prediction_frames.append(predictions)
        metric_rows.append(metrics_row)
        train_stats_by_window[str(metrics_row["test_month"])] = train_stats

    combined_predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    if not combined_predictions.empty:
        sort_cols = [
            column
            for column in ["date", "symbol", "timestamp", "decision_time"]
            if column in combined_predictions.columns
        ]
        combined_predictions = combined_predictions.sort_values(sort_cols)
    write_frame_atomic(combined_predictions, output_dir / "predictions.parquet")

    combined_buckets = score_bucket_returns(
        combined_predictions,
        bins=int(evaluation_settings["score_bins"]),
        group_cols=evaluation_settings["_bucket_group_cols"],
    )
    write_frame_atomic(combined_buckets, output_dir / "score_buckets.csv")
    if stock_pool is not None and stock_pool_settings.filter_selection:
        _, combined_pool_predictions, _ = configured_stock_pool_selection_frame(
            combined_predictions,
            stock_pool_settings,
            stock_pool,
        )
        combined_pool_buckets = score_bucket_returns(
            combined_pool_predictions,
            bins=int(evaluation_settings["score_bins"]),
            group_cols=evaluation_settings["_bucket_group_cols"],
        )
        write_frame_atomic(combined_pool_buckets, output_dir / "score_buckets_stock_pool.csv")

    metrics_by_window = pd.DataFrame(metric_rows)
    metrics_by_year = metrics_by_year_from_windows(metrics_by_window)
    write_frame_atomic(metrics_by_year, output_dir / "metrics_by_year.csv")
    write_frame_atomic(metrics_by_year, output_dir / "metrics_by_year.parquet")
    if not metrics_by_window["test_month"].is_unique or len(metrics_by_window) != len(
        metrics_by_year
    ):
        write_frame_atomic(metrics_by_window, output_dir / "metrics_by_month.csv")
        write_frame_atomic(metrics_by_window, output_dir / "metrics_by_month.parquet")

    write_json(
        output_dir / "metrics.json",
        {
            "run_id": run_name,
            "reproducibility": {
                "config_path": args.config,
                "config_sha256": _file_sha256(args.config) if args.config else "",
                "source_revision": os.environ.get("OPENING_STRENGTH_SOURCE_REVISION", ""),
            },
            "windows": len(splits),
            "train_window": f"{splits[0].train_start_date} -> {splits[-1].train_end_date}",
            "test_window": f"{splits[0].test_start_date} -> {splits[-1].test_end_date}",
            "train_stats_by_window": train_stats_by_window,
            "model": model_config_payload(config, alpha),
            "evaluation": {
                key: value for key, value in evaluation_settings.items() if not key.startswith("_")
            },
            "metrics_by_window": metric_rows,
            "metrics_by_year": metrics_by_year.to_dict(orient="records"),
        },
        atomic=True,
    )
    _write_success_marker(output_dir, run_name=run_name, windows=len(splits))

    print_mapping(
        "evaluation_settings",
        {key: value for key, value in evaluation_settings.items() if not key.startswith("_")},
    )
    print(f"\nwrote outputs: {output_dir}")
