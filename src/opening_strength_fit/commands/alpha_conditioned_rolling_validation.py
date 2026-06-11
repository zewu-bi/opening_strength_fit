from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.alpha_conditioning import (  # noqa: E402
    KEY_COLUMNS,
    add_alpha_conditioned_risk_targets,
    add_group_rank,
    fit_lgbm_config_section,
    predict_model_score,
)
from opening_strength_fit.analysis import (
    json_safe as shared_json_safe,
)
from opening_strength_fit.analysis import write_json
from opening_strength_fit.commands.learned_risk_layer import (
    load_or_fetch_next_close_labels,  # noqa: E402
)
from opening_strength_fit.config import (  # noqa: E402
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import write_frame  # noqa: E402
from opening_strength_fit.reports import dataset_summary, print_mapping  # noqa: E402
from opening_strength_fit.score_variant_eval import (  # noqa: E402
    score_variants,
    summarize_group_metrics,
)
from opening_strength_fit.stock_pool import (  # noqa: E402
    apply_stock_pool_cli_overrides,
    load_configured_stock_pool,
    stock_pool_config_from_mapping,
    stock_pool_membership_mask,
    stock_pool_runtime_summary,
)
from opening_strength_fit.training_args import build_training_parser  # noqa: E402
from opening_strength_fit.training_data import load_labeled_pvc_frame  # noqa: E402
from opening_strength_fit.training_windows import date_splits  # noqa: E402

DEFAULT_VARIANTS = (
    {"variant": "alpha_rank", "risk_model": "", "penalty": 0.0, "candidate_alpha_rank_min": 0.0},
    {
        "variant": "gap_penalty_030_p80",
        "risk_model": "gap",
        "penalty": 0.30,
        "candidate_alpha_rank_min": 0.80,
    },
    {
        "variant": "gap_penalty_035_p80",
        "risk_model": "gap",
        "penalty": 0.35,
        "candidate_alpha_rank_min": 0.80,
    },
    {
        "variant": "gap_penalty_030_p90",
        "risk_model": "gap",
        "penalty": 0.30,
        "candidate_alpha_rank_min": 0.90,
    },
    {
        "variant": "binary_penalty_035_p80",
        "risk_model": "binary",
        "penalty": 0.35,
        "candidate_alpha_rank_min": 0.80,
    },
)


def parse_args() -> argparse.Namespace:
    parser = build_training_parser("Rolling validation for alpha-conditioned Top100 risk scores.")
    parser.add_argument("--next-close-label-input", default="")
    parser.add_argument("--close-offset-us", type=int, default=54_000_000_000)
    parser.add_argument("--close-lookback-seconds", type=int, default=1_800)
    parser.add_argument("--calendar-days-after", type=int, default=10)
    return parser.parse_args()


def json_safe(value):
    return shared_json_safe(value)


def variant_specs(config: dict) -> list[dict[str, object]]:
    variants = config.get("score", {}).get("variants", [])
    if not variants:
        return [dict(item) for item in DEFAULT_VARIANTS]
    specs = []
    for item in variants:
        specs.append(
            {
                "variant": str(item.get("variant", "")).strip(),
                "risk_model": str(item.get("risk_model", "") or "").strip().lower(),
                "penalty": float(item.get("penalty", 0.0) or 0.0),
                "candidate_alpha_rank_min": float(item.get("candidate_alpha_rank_min", 0.0) or 0.0),
            }
        )
    return [spec for spec in specs if spec["variant"]]


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    config = apply_stock_pool_cli_overrides(config, args)
    stock_pool_settings = stock_pool_config_from_mapping(config)
    if stock_pool_settings.enabled and stock_pool_settings.filter_train:
        raise SystemExit(
            "alpha-conditioned rolling validation keeps training on the full universe; "
            "do not set stock_pool.filter_train or --pool-filter-train."
        )
    if stock_pool_settings.enabled and stock_pool_settings.add_feature:
        raise SystemExit(
            "alpha-conditioned rolling validation does not add stock_pool_member as a "
            "model feature; use stock_pool.filter_selection for selection-only masks."
        )
    stock_pool = load_configured_stock_pool(stock_pool_settings)
    if stock_pool is not None:
        print_mapping("stock_pool", stock_pool_runtime_summary(stock_pool_settings, stock_pool))

    run_name = run_id(config, args.config) if args.config else "rolling_alpha_conditioned_top100"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = load_labeled_pvc_frame(args, config)
    labels = load_or_fetch_next_close_labels(
        labeled,
        args=args,
        config=config,
        output_dir=output_dir,
    )
    labeled = labeled.merge(labels, on=list(KEY_COLUMNS), how="inner")
    labeled["decision_target_timestamp"] = pd.to_datetime(labeled["decision_target_timestamp"])
    before_label_filter = len(labeled)
    for column in ("label", "alpha_return_next_close"):
        labeled[column] = pd.to_numeric(labeled[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    labeled = labeled.dropna(subset=["label", "alpha_return_next_close"]).copy()
    print_mapping(
        "rolling_label_filter",
        {
            "rows_before": before_label_filter,
            "rows_after": len(labeled),
            "dropped_non_finite": before_label_filter - len(labeled),
        },
    )
    print_mapping("rolling_dataset", dataset_summary(labeled))

    splits = date_splits(labeled, args, config)
    print_mapping(
        "rolling_split_plan",
        {
            "windows": len(splits),
            "first_test": splits[0].test_start_date,
            "last_test": splits[-1].test_end_date,
        },
    )
    top_n = args.top_n or config_int(config, "score", "top_n", 100)
    variants = variant_specs(config)

    group_metric_frames = []
    prediction_paths = []
    train_stats = {}
    for split in splits:
        month = str(pd.Timestamp(split.test_start_date).to_period("M"))
        print(f"\nrolling window: {month}")
        train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
        test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()

        alpha_model, alpha_stats = fit_lgbm_config_section(
            train,
            args=args,
            config=config,
            section="alpha_conditioning",
            target_col="label",
            sample_weight_col=config_str(config, "alpha_conditioning", "sample_weight_col", ""),
            random_state_default=config_int(config, "model", "random_state", 7) + 1000,
        )
        train["candidate_alpha_score"] = predict_model_score(alpha_model, train)
        test["candidate_alpha_score"] = predict_model_score(alpha_model, test)
        del alpha_model
        gc.collect()
        train = add_group_rank(train, "candidate_alpha_score", "candidate_alpha_rank")
        test = add_group_rank(test, "candidate_alpha_score", "candidate_alpha_rank")

        risk_train = add_alpha_conditioned_risk_targets(train, config, copy_frame=False)
        gap_model, gap_stats = fit_lgbm_config_section(
            risk_train,
            args=args,
            config=config,
            section="gap_model",
            target_col="target_alpha_conditioned_gap_risk",
            sample_weight_col="risk_sample_weight",
            random_state_default=config_int(config, "model", "random_state", 43),
        )
        test["gap_risk_prediction"] = np.clip(predict_model_score(gap_model, test), 0.0, 1.0)
        del gap_model
        gc.collect()

        binary_model, binary_stats = fit_lgbm_config_section(
            risk_train,
            args=args,
            config=config,
            section="binary_model",
            target_col="target_alpha_conditioned_binary_risk",
            sample_weight_col="risk_sample_weight",
            random_state_default=config_int(config, "model", "random_state", 44),
        )
        test["binary_risk_prediction"] = np.clip(predict_model_score(binary_model, test), 0.0, 1.0)
        del binary_model
        gc.collect()
        test = add_group_rank(test, "gap_risk_prediction", "gap_risk_rank")
        test = add_group_rank(test, "binary_risk_prediction", "binary_risk_rank")
        selection_mask_col = ""
        if stock_pool is not None:
            pool_mask = stock_pool_membership_mask(
                test,
                stock_pool,
                date_lag_sessions=stock_pool_settings.date_lag_sessions,
            )
            if stock_pool_settings.annotate_predictions or stock_pool_settings.filter_selection:
                test[stock_pool_settings.membership_col] = pool_mask.astype("int8").to_numpy()
            if stock_pool_settings.filter_selection:
                selection_mask_col = stock_pool_settings.membership_col

        group_metrics = score_variants(
            test,
            month=month,
            variants=variants,
            top_n=int(top_n),
            selection_mask_col=selection_mask_col,
        )
        group_metric_frames.append(group_metrics)
        prediction_columns = [
            *KEY_COLUMNS,
            "label",
            "alpha_return_next_close",
            "candidate_alpha_score",
            "candidate_alpha_rank",
            "gap_risk_prediction",
            "gap_risk_rank",
            "binary_risk_prediction",
            "binary_risk_rank",
        ]
        if stock_pool_settings.membership_col in test.columns:
            prediction_columns.append(stock_pool_settings.membership_col)
        shard_output_dir = output_dir / f"month_{month}" if len(splits) > 1 else output_dir
        shard_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = shard_output_dir / "predictions.parquet"
        write_frame(test[prediction_columns].copy(), prediction_path)
        prediction_paths.append(str(prediction_path))
        shard_month_summary, shard_summary = summarize_group_metrics(group_metrics)
        group_metrics.to_csv(shard_output_dir / "rolling_group_metrics.csv", index=False)
        shard_month_summary.to_csv(shard_output_dir / "rolling_month_summary.csv", index=False)
        shard_summary.to_csv(shard_output_dir / "rolling_summary.csv", index=False)
        train_stats[month] = {
            "train_start_date": split.train_start_date,
            "train_end_date": split.train_end_date,
            "test_start_date": split.test_start_date,
            "test_end_date": split.test_end_date,
            "alpha": alpha_stats,
            "gap": gap_stats,
            "binary": binary_stats,
        }
        print_mapping(
            f"rolling_window_summary[{month}]",
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "groups": int(
                    group_metrics[["date", "decision_target_timestamp"]].drop_duplicates().shape[0]
                ),
                "stock_pool_filter_selection": bool(selection_mask_col),
                "stock_pool_candidate_rows": (
                    float(group_metrics["stock_pool_candidate_rows"].mean())
                    if selection_mask_col
                    else None
                ),
            },
        )
        del (
            train,
            test,
            risk_train,
            group_metrics,
            shard_month_summary,
            shard_summary,
        )
        gc.collect()

    group_metrics = pd.concat(group_metric_frames, ignore_index=True)
    month_summary, summary = summarize_group_metrics(group_metrics)
    group_metrics.to_csv(output_dir / "rolling_group_metrics.csv", index=False)
    month_summary.to_csv(output_dir / "rolling_month_summary.csv", index=False)
    summary.to_csv(output_dir / "rolling_summary.csv", index=False)

    trace = {
        "run_id": run_name,
        "top_n": int(top_n),
        "variants": variants,
        "risk_layer": config.get("risk_layer", {}),
        "stock_pool": {
            "enabled": stock_pool_settings.enabled,
            "name": stock_pool_settings.name,
            "path": stock_pool_settings.path,
            "date_lag_sessions": stock_pool_settings.date_lag_sessions,
            "filter_selection": stock_pool_settings.filter_selection,
            "membership_col": stock_pool_settings.membership_col,
        },
        "windows": len(splits),
        "rows": int(len(labeled)),
        "outputs": {
            "predictions_by_window": prediction_paths,
            "group_metrics": str(output_dir / "rolling_group_metrics.csv"),
            "month_summary": str(output_dir / "rolling_month_summary.csv"),
            "summary": str(output_dir / "rolling_summary.csv"),
        },
        "train_stats_by_window": train_stats,
    }
    write_json(output_dir / "rolling_trace.json", trace)
    print("\nrolling_summary")
    print(
        summary[
            [
                "variant",
                "months",
                "short_top_excess_bps",
                "next_top_excess_bps",
                "next_positive_minute_count",
                "next_positive_month_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
