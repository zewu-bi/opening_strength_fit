from __future__ import annotations

import os

import numpy as np
import pandas as pd

from opening_strength_fit.commands.arguments import add_arguments
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_str,
    prepare_output_dir,
    run_id,
)
from opening_strength_fit.evaluation import (
    group_cols_for_mode,
    resolve_group_cols,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.feature_groups import (
    FeatureGroup,
)
from opening_strength_fit.feature_groups import (
    feature_group_name as _feature_group_name,
)
from opening_strength_fit.feature_groups import (
    feature_groups as _feature_groups,
)
from opening_strength_fit.feature_groups import (
    matching_features as _matching_features,
)
from opening_strength_fit.io import write_frame, write_json
from opening_strength_fit.model import evaluate_prediction_frame, predict_frame
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.stock_pool import (
    configured_stock_pool_selection_frame,
    load_configured_stock_pool,
    stock_pool_config_from_mapping,
    stock_pool_runtime_summary,
)
from opening_strength_fit.training_args import build_training_parser, load_run_config
from opening_strength_fit.training_data import (
    load_clickhouse_labeled_frame,
    load_labeled_pvc_frame,
    load_training_frame,
    resolve_data_source,
)
from opening_strength_fit.training_modeling import (
    fit_prediction_model,
    model_config_payload,
)
from opening_strength_fit.training_windows import (
    build_evaluation_settings,
    date_splits,
    resolve_window_mode,
)


def _load_labeled(args, config: dict) -> pd.DataFrame:
    tick_path = (
        args.input
        or config_str(config, "data", "tick_path", "")
        or os.environ.get("OPENING_STRENGTH_TICK_PATH", "")
    )
    data_source = resolve_data_source(args, config, tick_path)
    if data_source == "clickhouse":
        return load_clickhouse_labeled_frame(args, config)
    if data_source == "labeled_pvc":
        return load_labeled_pvc_frame(args, config)
    if not tick_path:
        raise SystemExit(
            "No input path supplied. Set [data].tick_path, --input, "
            'or use [data].source = "labeled_pvc"/"clickhouse".'
        )
    return load_training_frame(tick_path, args, config)


def _summary_row(
    *,
    run_name: str,
    variant: str,
    split_label: str,
    predictions: pd.DataFrame,
    features: int,
    dropped_features: int,
    evaluation_settings: dict[str, object],
    train_rows: int | None,
    repeat: int | None = None,
    baseline: dict[str, object] | None = None,
    stock_pool_settings=None,
    stock_pool: pd.DataFrame | None = None,
) -> dict[str, object]:
    if stock_pool is not None and stock_pool_settings is not None:
        predictions, selection_predictions, stock_pool_summary = (
            configured_stock_pool_selection_frame(
                predictions,
                stock_pool_settings,
                stock_pool,
            )
        )
    else:
        selection_predictions = predictions
        stock_pool_summary = {}
    metrics = evaluate_prediction_frame(
        selection_predictions,
        group_cols=evaluation_settings["_ic_group_cols"],
    )
    top = top_score_trades(
        selection_predictions,
        top_n=int(evaluation_settings["top_n"]),
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    top_summary = summarize_trades(
        top,
        group_cols=evaluation_settings["_selection_group_cols"],
    )
    row: dict[str, object] = {
        "run_id": run_name,
        "variant": variant,
        "split": split_label,
        "repeat": repeat,
        "features": int(features),
        "dropped_features": int(dropped_features),
        "train_rows": train_rows,
        "rows": metrics["rows"],
        "dates": metrics["dates"],
        "symbols": metrics["symbols"],
        "ic_grouping": metrics["ic_grouping"],
        "ic_groups": metrics["ic_groups"],
        "group_ic_mean": metrics["group_ic_mean"],
        "group_rank_ic_mean": metrics["group_rank_ic_mean"],
        "group_rank_ic_ir": metrics["group_rank_ic_ir"],
        "daily_rank_ic_mean": metrics["daily_rank_ic_mean"],
        "top_score_trades": top_summary["trades"],
        "top_score_groups": top_summary["groups"],
        "top_score_mean_return": top_summary["mean_return"],
        "top_score_mean_return_bps": top_summary["mean_return"] * 10_000,
        "top_score_win_rate": top_summary["win_rate"],
    }
    row.update(stock_pool_summary)
    if baseline is not None:
        row["delta_group_rank_ic_mean"] = row["group_rank_ic_mean"] - baseline["group_rank_ic_mean"]
        row["delta_top_score_mean_return_bps"] = (
            row["top_score_mean_return_bps"] - baseline["top_score_mean_return_bps"]
        )
    return row


def _predict_valid(model, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = predict_frame(model, frame)
    if "valid_label" in predictions.columns:
        predictions = predictions.loc[predictions["valid_label"]].copy()
    return predictions


def _final_estimator(model):
    return model.pipeline.steps[-1][1]


def _estimator_feature_names(model) -> list[str]:
    imputer = model.pipeline.named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "get_feature_names_out"):
        return [str(name) for name in imputer.get_feature_names_out(model.features)]
    return list(model.features)


def _indexed_value(values, index: int) -> float:
    if values is None or index >= len(values):
        return np.nan
    return float(values[index])


def _importance_rows(
    *,
    model,
    split_label: str,
    groups: list[FeatureGroup],
) -> list[dict[str, object]]:
    estimator = _final_estimator(model)
    features = _estimator_feature_names(model)
    rows: list[dict[str, object]] = []
    split_importance = getattr(estimator, "feature_importances_", None)
    gain_importance = None
    booster = getattr(estimator, "booster_", None)
    if booster is not None:
        gain_importance = booster.feature_importance(importance_type="gain")
    coefficients = getattr(estimator, "coef_", None)

    for index, feature in enumerate(features):
        row = {
            "split": split_label,
            "feature": feature,
            "group": _feature_group_name(feature, groups),
            "importance_split": _indexed_value(split_importance, index),
            "importance_gain": _indexed_value(gain_importance, index),
            "coefficient": _indexed_value(coefficients, index),
        }
        row["abs_coefficient"] = (
            abs(row["coefficient"]) if not pd.isna(row["coefficient"]) else np.nan
        )
        rows.append(row)
    return rows


def _shuffle_feature_columns(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    group_cols: tuple[str, ...],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = frame.copy()
    columns = [column for column in columns if column in out.columns]
    if not columns:
        return out
    resolved_group_cols = resolve_group_cols(out, group_cols)
    if not resolved_group_cols:
        for column in columns:
            values = out[column].to_numpy(copy=True)
            rng.shuffle(values)
            out[column] = values
        return out

    for _, index in out.groupby(list(resolved_group_cols), sort=False).groups.items():
        if len(index) < 2:
            continue
        for column in columns:
            values = out.loc[index, column].to_numpy(copy=True)
            rng.shuffle(values)
            out.loc[index, column] = values
    return out


def main() -> None:
    parser = build_training_parser(
        "Audit opening-strength feature dependence with grouped importance, "
        "permutation, and drop-retrain ablations."
    )
    add_arguments(parser, "skip-permutation skip-ablation write-predictions", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    run_name = run_id(config, args.config) if args.config else "local_feature_audit"
    output_dir = prepare_output_dir(config, args.output_dir, run_name)

    labeled = _load_labeled(args, config)
    print_mapping("dataset", dataset_summary(labeled))
    stock_pool_settings = stock_pool_config_from_mapping(config)
    stock_pool = load_configured_stock_pool(stock_pool_settings)
    if stock_pool is not None:
        print_mapping("stock_pool", stock_pool_runtime_summary(stock_pool_settings, stock_pool))

    alpha = args.alpha if args.alpha is not None else config_float(config, "model", "alpha", 1.0)
    evaluation_settings = build_evaluation_settings(config, args)
    splits = date_splits(labeled, args, config)
    groups = _feature_groups(config)
    permutation_repeats = config_int(config, "feature_audit", "permutation_repeats", 1)
    permutation_mode = config_str(config, "feature_audit", "permutation_mode", "cross_section")
    run_ablation = not args.skip_ablation and config_bool(
        config, "feature_audit", "run_ablation", True
    )
    run_permutation = not args.skip_permutation and config_bool(
        config, "feature_audit", "run_permutation", True
    )
    random_state = config_int(config, "feature_audit", "random_state", 7)
    print_mapping(
        "feature_audit_settings",
        {
            "groups": [group.name for group in groups],
            "permutation_repeats": permutation_repeats,
            "permutation_mode": permutation_mode,
            "ablation": run_ablation,
            "permutation": run_permutation,
            "write_predictions": args.write_predictions,
            "window_mode": resolve_window_mode(args, config),
        },
    )

    metric_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for split in splits:
        split_label = (
            str(pd.Timestamp(split.test_start_date).to_period("M"))
            if split.test_start_date[:7] == split.test_end_date[:7]
            else f"{split.test_start_date}:{split.test_end_date}"
        )
        train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
        test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()
        model, train_stats = fit_prediction_model(
            train,
            args=args,
            config=config,
            alpha=alpha,
        )
        baseline_predictions = _predict_valid(model, test)
        baseline_row = _summary_row(
            run_name=run_name,
            variant="baseline",
            split_label=split_label,
            predictions=baseline_predictions,
            features=len(model.features),
            dropped_features=0,
            train_rows=train_stats["rows"],
            evaluation_settings=evaluation_settings,
            stock_pool_settings=stock_pool_settings,
            stock_pool=stock_pool,
        )
        metric_rows.append(baseline_row)
        importance_rows.extend(
            _importance_rows(
                model=model,
                split_label=split_label,
                groups=groups,
            )
        )
        if args.write_predictions:
            write_frame(
                baseline_predictions,
                output_dir / f"predictions_{split_label}_baseline.parquet",
            )

        group_columns = {group.name: _matching_features(model.features, group) for group in groups}

        if run_permutation:
            permutation_group_cols = group_cols_for_mode(permutation_mode)
            for repeat in range(permutation_repeats):
                rng = np.random.default_rng(random_state + repeat)
                for group_name, columns in group_columns.items():
                    if not columns:
                        continue
                    shuffled = _shuffle_feature_columns(
                        test,
                        columns=columns,
                        group_cols=permutation_group_cols,
                        rng=rng,
                    )
                    predictions = _predict_valid(model, shuffled)
                    row = _summary_row(
                        run_name=run_name,
                        variant=f"permute_{group_name}",
                        split_label=split_label,
                        repeat=repeat,
                        predictions=predictions,
                        features=len(model.features),
                        dropped_features=0,
                        train_rows=train_stats["rows"],
                        evaluation_settings=evaluation_settings,
                        baseline=baseline_row,
                        stock_pool_settings=stock_pool_settings,
                        stock_pool=stock_pool,
                    )
                    row["group"] = group_name
                    row["permuted_features"] = len(columns)
                    permutation_rows.append(row)

        if run_ablation:
            for group_name, columns in group_columns.items():
                if not columns:
                    continue
                ablated_train = train.drop(columns=columns, errors="ignore")
                ablated_test = test.drop(columns=columns, errors="ignore")
                ablated_model, ablated_train_stats = fit_prediction_model(
                    ablated_train,
                    args=args,
                    config=config,
                    alpha=alpha,
                )
                predictions = _predict_valid(ablated_model, ablated_test)
                row = _summary_row(
                    run_name=run_name,
                    variant=f"drop_{group_name}",
                    split_label=split_label,
                    predictions=predictions,
                    features=len(ablated_model.features),
                    dropped_features=len(columns),
                    train_rows=ablated_train_stats["rows"],
                    evaluation_settings=evaluation_settings,
                    baseline=baseline_row,
                    stock_pool_settings=stock_pool_settings,
                    stock_pool=stock_pool,
                )
                row["group"] = group_name
                metric_rows.append(row)
                if args.write_predictions:
                    write_frame(
                        predictions,
                        output_dir / f"predictions_{split_label}_drop_{group_name}.parquet",
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "feature_audit_metrics.csv", index=False)
    if permutation_rows:
        permutation = pd.DataFrame(permutation_rows)
        permutation.to_csv(output_dir / "feature_audit_permutation.csv", index=False)
    else:
        permutation = pd.DataFrame()
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    if not importance.empty:
        group_importance = (
            importance.groupby(["split", "group"], as_index=False)
            .agg(
                features=("feature", "size"),
                importance_split_sum=("importance_split", "sum"),
                importance_gain_sum=("importance_gain", "sum"),
                abs_coefficient_sum=("abs_coefficient", "sum"),
            )
            .sort_values(["split", "importance_gain_sum", "importance_split_sum"], ascending=False)
        )
        group_importance.to_csv(output_dir / "feature_group_importance.csv", index=False)

    trace = {
        "run_id": run_name,
        "windows": len(splits),
        "model": model_config_payload(config, alpha),
        "evaluation": {
            key: value for key, value in evaluation_settings.items() if not key.startswith("_")
        },
        "outputs": {
            "metrics": str(output_dir / "feature_audit_metrics.csv"),
            "permutation": str(output_dir / "feature_audit_permutation.csv"),
            "importance": str(output_dir / "feature_importance.csv"),
            "group_importance": str(output_dir / "feature_group_importance.csv"),
        },
    }
    write_json(output_dir / "feature_audit_trace.json", trace)

    print("\nfeature_audit_metrics:")
    print(metrics.to_string(index=False))
    if not permutation.empty:
        print("\nfeature_audit_permutation:")
        columns = [
            "variant",
            "split",
            "repeat",
            "permuted_features",
            "delta_group_rank_ic_mean",
            "delta_top_score_mean_return_bps",
        ]
        print(permutation[columns].to_string(index=False))
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
