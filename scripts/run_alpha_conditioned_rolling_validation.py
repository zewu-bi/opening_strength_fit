from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPT_DIR = Path("/app/opening_strength_fit/scripts")
for path in (SCRIPT_DIR, REPO_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _bootstrap  # noqa: F401,E402
from opening_strength_fit.alpha_conditioning import (  # noqa: E402
    KEY_COLUMNS,
    add_alpha_conditioned_risk_targets,
    add_group_rank,
    fit_lgbm_config_section,
    predict_model_score,
)
from opening_strength_fit.config import (  # noqa: E402
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import write_frame  # noqa: E402
from opening_strength_fit.reports import dataset_summary, print_mapping  # noqa: E402
from opening_strength_fit.training import (  # noqa: E402
    _date_splits,
    _load_labeled_pvc_frame,
    build_training_parser,
)
from run_learned_risk_layer import load_or_fetch_next_close_labels  # noqa: E402


DEFAULT_VARIANTS = (
    {"variant": "alpha_rank", "risk_model": "", "penalty": 0.0, "candidate_alpha_rank_min": 0.0},
    {"variant": "gap_penalty_030_p80", "risk_model": "gap", "penalty": 0.30, "candidate_alpha_rank_min": 0.80},
    {"variant": "gap_penalty_035_p80", "risk_model": "gap", "penalty": 0.35, "candidate_alpha_rank_min": 0.80},
    {"variant": "gap_penalty_030_p90", "risk_model": "gap", "penalty": 0.30, "candidate_alpha_rank_min": 0.90},
    {"variant": "binary_penalty_035_p80", "risk_model": "binary", "penalty": 0.35, "candidate_alpha_rank_min": 0.80},
)


def parse_args() -> argparse.Namespace:
    parser = build_training_parser("Rolling validation for alpha-conditioned Top100 risk scores.")
    parser.add_argument("--next-close-label-input", default="")
    parser.add_argument("--close-offset-us", type=int, default=54_000_000_000)
    parser.add_argument("--close-lookback-seconds", type=int, default=1_800)
    parser.add_argument("--calendar-days-after", type=int, default=10)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


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


def score_variants(
    test: pd.DataFrame,
    *,
    month: str,
    variants: list[dict[str, object]],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for spec in variants:
        variant = str(spec["variant"])
        risk_model = str(spec.get("risk_model", "") or "").lower()
        penalty = float(spec.get("penalty", 0.0) or 0.0)
        candidate_min = float(spec.get("candidate_alpha_rank_min", 0.0) or 0.0)
        if risk_model == "gap":
            risk_col = "gap_risk_rank"
        elif risk_model == "binary":
            risk_col = "binary_risk_rank"
        else:
            risk_col = ""

        work = test.loc[test["candidate_alpha_rank"].ge(candidate_min)].copy()
        risk_values = work[risk_col] if risk_col else 0.0
        work["final_score"] = work["candidate_alpha_rank"] - penalty * risk_values
        for (date, timestamp), group in work.groupby(["date", "decision_target_timestamp"], sort=True):
            full_group = test.loc[
                test["date"].eq(date) & test["decision_target_timestamp"].eq(timestamp)
            ]
            selected = group.sort_values("final_score", ascending=False).head(top_n)
            short_mean = float(selected["label"].mean()) if len(selected) else float("nan")
            next_mean = (
                float(selected["alpha_return_next_close"].mean())
                if len(selected)
                else float("nan")
            )
            rows.append(
                {
                    "test_month": month,
                    "variant": variant,
                    "risk_model": risk_model,
                    "penalty": penalty,
                    "candidate_alpha_rank_min": candidate_min,
                    "date": str(date),
                    "decision_target_timestamp": pd.Timestamp(timestamp),
                    "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                    "rows": int(len(full_group)),
                    "candidate_rows": int(len(group)),
                    "selected_rows": int(len(selected)),
                    "short_top_mean_bps": short_mean * 10_000.0,
                    "short_top_excess_bps": (
                        short_mean - float(full_group["label"].mean())
                    )
                    * 10_000.0,
                    "next_top_mean_bps": next_mean * 10_000.0,
                    "next_top_excess_bps": (
                        next_mean - float(full_group["alpha_return_next_close"].mean())
                    )
                    * 10_000.0,
                    "selected_gap_risk_rank": (
                        float(selected["gap_risk_rank"].mean())
                        if len(selected) and "gap_risk_rank" in selected
                        else float("nan")
                    ),
                    "selected_binary_risk_rank": (
                        float(selected["binary_risk_rank"].mean())
                        if len(selected) and "binary_risk_rank" in selected
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_group_metrics(group_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["variant", "risk_model", "penalty", "candidate_alpha_rank_min"]
    month_summary = (
        group_metrics.groupby(["test_month", *keys], as_index=False)
        .agg(
            groups=("date", "size"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_excess_positive_rate=("next_top_excess_bps", lambda s: float((s > 0).mean())),
            selected_gap_risk_rank=("selected_gap_risk_rank", "mean"),
            selected_binary_risk_rank=("selected_binary_risk_rank", "mean"),
        )
        .sort_values(["test_month", "next_top_excess_bps"], ascending=[True, False])
    )
    minute_positive = (
        group_metrics.groupby([*keys, "clock"])["next_top_excess_bps"]
        .mean()
        .reset_index()
        .groupby(keys)["next_top_excess_bps"]
        .apply(lambda s: int((s > 0).sum()))
        .reset_index(name="next_positive_minute_count")
    )
    monthly_positive = (
        month_summary.groupby(keys)["next_top_excess_bps"]
        .apply(lambda s: int((s > 0).sum()))
        .reset_index(name="next_positive_month_count")
    )
    summary = (
        group_metrics.groupby(keys, as_index=False)
        .agg(
            groups=("date", "size"),
            months=("test_month", "nunique"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_excess_positive_rate=("next_top_excess_bps", lambda s: float((s > 0).mean())),
            selected_gap_risk_rank=("selected_gap_risk_rank", "mean"),
            selected_binary_risk_rank=("selected_binary_risk_rank", "mean"),
        )
        .merge(minute_positive, on=keys, how="left")
        .merge(monthly_positive, on=keys, how="left")
        .sort_values(["next_top_excess_bps", "short_top_excess_bps"], ascending=[False, False])
    )
    return month_summary, summary


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "rolling_alpha_conditioned_top100"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/local/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = _load_labeled_pvc_frame(args, config)
    labels = load_or_fetch_next_close_labels(
        labeled,
        args=args,
        config=config,
        output_dir=output_dir,
    )
    labeled = labeled.merge(labels, on=list(KEY_COLUMNS), how="inner")
    labeled["decision_target_timestamp"] = pd.to_datetime(labeled["decision_target_timestamp"])
    print_mapping("rolling_dataset", dataset_summary(labeled))

    splits = _date_splits(labeled, args, config)
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
        train = add_group_rank(train, "candidate_alpha_score", "candidate_alpha_rank")
        test = add_group_rank(test, "candidate_alpha_score", "candidate_alpha_rank")

        risk_train = add_alpha_conditioned_risk_targets(train, config)
        gap_model, gap_stats = fit_lgbm_config_section(
            risk_train,
            args=args,
            config=config,
            section="gap_model",
            target_col="target_alpha_conditioned_gap_risk",
            sample_weight_col="risk_sample_weight",
            random_state_default=config_int(config, "model", "random_state", 43),
        )
        binary_model, binary_stats = fit_lgbm_config_section(
            risk_train,
            args=args,
            config=config,
            section="binary_model",
            target_col="target_alpha_conditioned_binary_risk",
            sample_weight_col="risk_sample_weight",
            random_state_default=config_int(config, "model", "random_state", 44),
        )

        test["gap_risk_prediction"] = np.clip(predict_model_score(gap_model, test), 0.0, 1.0)
        test["binary_risk_prediction"] = np.clip(predict_model_score(binary_model, test), 0.0, 1.0)
        test = add_group_rank(test, "gap_risk_prediction", "gap_risk_rank")
        test = add_group_rank(test, "binary_risk_prediction", "binary_risk_rank")

        group_metrics = score_variants(test, month=month, variants=variants, top_n=int(top_n))
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
                "groups": int(group_metrics[["date", "decision_target_timestamp"]].drop_duplicates().shape[0]),
            },
        )
        del (
            train,
            test,
            risk_train,
            alpha_model,
            gap_model,
            binary_model,
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
    (output_dir / "rolling_trace.json").write_text(
        json.dumps(json_safe(trace), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
