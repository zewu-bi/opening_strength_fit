from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_str,
    run_id,
)
from opening_strength_fit.evaluation import (
    group_cols_for_mode,
    resolve_group_cols,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.io import write_frame
from opening_strength_fit.model import evaluate_prediction_frame, predict_frame
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.training import (
    _date_splits,
    _evaluation_settings,
    _fit_prediction_model,
    _load_clickhouse_labeled_frame,
    _load_labeled_pvc_frame,
    _load_training_frame,
    _model_json,
    _resolved_data_source,
    _resolved_window_mode,
    build_training_parser,
    load_run_config,
)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()

    def matches(self, feature: str) -> bool:
        if self.exclude_prefixes and feature.startswith(self.exclude_prefixes):
            return False
        if feature in self.columns:
            return True
        if self.prefixes and feature.startswith(self.prefixes):
            return True
        return bool(self.contains and any(token in feature for token in self.contains))


DEFAULT_GROUPS = (
    FeatureGroup("preopen", prefixes=("preopen_",)),
    FeatureGroup("postopen_v2", prefixes=("postopen_v2_",)),
    FeatureGroup(
        "postopen_v1",
        prefixes=("postopen_",),
        exclude_prefixes=("postopen_v2_",),
    ),
    FeatureGroup("raw_cumulative_trade", columns=("volume", "turnover")),
    FeatureGroup(
        "trade_flow",
        prefixes=("volume_diff_", "turnover_diff_", "trade_vwap_"),
    ),
    FeatureGroup(
        "orderbook_depth",
        prefixes=(
            "ask_volume_",
            "bid_volume_",
            "ask_depth_",
            "bid_depth_",
            "depth_imbalance_",
            "ask_gap_",
            "bid_gap_",
        ),
        columns=(
            "ask_price_1",
            "bid_price_1",
            "mid_price",
            "spread_abs",
            "spread_bps",
            "ask1_to_limit_up_bps",
        ),
    ),
    FeatureGroup(
        "momentum",
        prefixes=("return_",),
        columns=("return_vs_prev_close", "return_vs_open"),
    ),
)


def _tuple_config(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace(",", " ").split() if part)
    return tuple(str(item) for item in value if str(item))


def _feature_groups(config: dict) -> list[FeatureGroup]:
    groups = {group.name: group for group in DEFAULT_GROUPS}
    custom_groups = config.get("feature_audit", {}).get("groups", {})
    if isinstance(custom_groups, dict):
        for name, spec in custom_groups.items():
            if not isinstance(spec, dict):
                continue
            groups[str(name)] = FeatureGroup(
                name=str(name),
                columns=_tuple_config(spec.get("columns")),
                prefixes=_tuple_config(spec.get("prefixes")),
                contains=_tuple_config(spec.get("contains")),
                exclude_prefixes=_tuple_config(spec.get("exclude_prefixes")),
            )

    enabled = _tuple_config(config.get("feature_audit", {}).get("enabled_groups"))
    ordered = list(groups.values())
    if enabled:
        enabled_set = set(enabled)
        ordered = [group for group in ordered if group.name in enabled_set]
    return ordered


def _matching_features(features: list[str], group: FeatureGroup) -> list[str]:
    return [feature for feature in features if group.matches(feature)]


def _feature_group_name(feature: str, groups: list[FeatureGroup]) -> str:
    for group in groups:
        if group.matches(feature):
            return group.name
    return "other"


def _load_labeled(args, config: dict) -> pd.DataFrame:
    tick_path = (
        args.input
        or config_str(config, "data", "tick_path", "")
        or os.environ.get("OPENING_STRENGTH_TICK_PATH", "")
    )
    data_source = _resolved_data_source(args, config, tick_path)
    if data_source == "clickhouse":
        return _load_clickhouse_labeled_frame(args, config)
    if data_source == "labeled_pvc":
        return _load_labeled_pvc_frame(args, config)
    if not tick_path:
        raise SystemExit(
            "No input path supplied. Set [data].tick_path, --input, "
            "or use [data].source = \"labeled_pvc\"/\"clickhouse\"."
        )
    return _load_training_frame(tick_path, args, config)


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
) -> dict[str, object]:
    metrics = evaluate_prediction_frame(
        predictions,
        group_cols=evaluation_settings["_ic_group_cols"],
    )
    top = top_score_trades(
        predictions,
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
    if baseline is not None:
        row["delta_group_rank_ic_mean"] = (
            row["group_rank_ic_mean"] - baseline["group_rank_ic_mean"]
        )
        row["delta_top_score_mean_return_bps"] = (
            row["top_score_mean_return_bps"]
            - baseline["top_score_mean_return_bps"]
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
        try:
            return [str(name) for name in imputer.get_feature_names_out(model.features)]
        except Exception:
            pass
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
        try:
            gain_importance = booster.feature_importance(importance_type="gain")
        except Exception:
            gain_importance = None
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
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    run_name = run_id(config, args.config) if args.config else "local_feature_audit"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/local/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = _load_labeled(args, config)
    print_mapping("dataset", dataset_summary(labeled))

    alpha = (
        args.alpha
        if args.alpha is not None
        else config_float(config, "model", "alpha", 1.0)
    )
    evaluation_settings = _evaluation_settings(config, args)
    splits = _date_splits(labeled, args, config)
    groups = _feature_groups(config)
    print_mapping(
        "feature_audit_settings",
        {
            "groups": [group.name for group in groups],
            "permutation_repeats": config_int(
                config,
                "feature_audit",
                "permutation_repeats",
                1,
            ),
            "permutation_mode": config_str(
                config,
                "feature_audit",
                "permutation_mode",
                "cross_section",
            ),
            "ablation": not args.skip_ablation
            and config_bool(config, "feature_audit", "run_ablation", True),
            "permutation": not args.skip_permutation
            and config_bool(config, "feature_audit", "run_permutation", True),
            "write_predictions": args.write_predictions,
            "window_mode": _resolved_window_mode(args, config),
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
        model, train_stats = _fit_prediction_model(
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

        group_columns = {
            group.name: _matching_features(model.features, group)
            for group in groups
        }

        if not args.skip_permutation and config_bool(
            config,
            "feature_audit",
            "run_permutation",
            True,
        ):
            repeat_count = config_int(
                config,
                "feature_audit",
                "permutation_repeats",
                1,
            )
            mode = config_str(
                config,
                "feature_audit",
                "permutation_mode",
                "cross_section",
            )
            permutation_group_cols = group_cols_for_mode(mode)
            seed = config_int(config, "feature_audit", "random_state", 7)
            for repeat in range(repeat_count):
                rng = np.random.default_rng(seed + repeat)
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
                    )
                    row["group"] = group_name
                    row["permuted_features"] = len(columns)
                    permutation_rows.append(row)

        if not args.skip_ablation and config_bool(
            config,
            "feature_audit",
            "run_ablation",
            True,
        ):
            for group_name, columns in group_columns.items():
                if not columns:
                    continue
                ablated_train = train.drop(columns=columns, errors="ignore")
                ablated_test = test.drop(columns=columns, errors="ignore")
                ablated_model, ablated_train_stats = _fit_prediction_model(
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
        "model": _model_json(config, alpha),
        "evaluation": {
            key: value
            for key, value in evaluation_settings.items()
            if not key.startswith("_")
        },
        "outputs": {
            "metrics": str(output_dir / "feature_audit_metrics.csv"),
            "permutation": str(output_dir / "feature_audit_permutation.csv"),
            "importance": str(output_dir / "feature_importance.csv"),
            "group_importance": str(output_dir / "feature_group_importance.csv"),
        },
    }
    (output_dir / "feature_audit_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
