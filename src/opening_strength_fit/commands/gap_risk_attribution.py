from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import finite_mean
from opening_strength_fit.commands.arguments import add_arguments, command_context
from opening_strength_fit.config import (
    coerce_str_list,
    config_int,
    config_list,
    config_value,
    prepare_output_dir,
)
from opening_strength_fit.io import frame_columns, write_json
from opening_strength_fit.schema import DECISION_KEY_COLUMNS, normalize_decision_keys
from opening_strength_fit.score_variant_eval import (
    GAP_P80_VARIANTS,
    configured_score_variants,
)

KEY_COLUMNS = DECISION_KEY_COLUMNS
DEFAULT_FEATURES = tuple(
    "spread_bps turnover_diff_1t turnover_diff_3t turnover_diff_10t turnover_diff_30t "
    "volume_diff_10t volume_diff_30t return_10t return_30t ask_depth_10 bid_depth_10 "
    "ask_volume_1 bid_volume_1 depth_imbalance_1 depth_imbalance_10 preopen_turnover "
    "preopen_volume buy_price mid_price".split()
)
DEFAULT_CONTROLS = tuple(
    "turnover_diff_10t return_10t spread_bps ask_depth_10 depth_imbalance_10 "
    "preopen_turnover buy_price".split()
)
DEFAULT_VARIANTS = GAP_P80_VARIANTS
RISK_RANK_BY_MODEL = {
    "gap": "gap_risk_rank",
    "binary": "binary_risk_rank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute gap-risk Top100 changes: compare baseline Top100, "
            "penalized-out names, replacements, and retained names."
        )
    )
    add_arguments(parser, "config prediction-dir labeled-path output-dir", default="")
    parser.add_argument(
        "--months",
        default="",
        help="Comma/space separated months such as 2021-08,2021-09. Overrides config.",
    )
    parser.add_argument("--top-n", type=int, default=None)
    return parser.parse_args()


def month_sequence(config: dict, override: str = "") -> list[str]:
    explicit = coerce_str_list(override)
    if explicit:
        return explicit
    explicit = config_list(config, "attribution", "months", [])
    if explicit:
        return explicit
    start = config_value(config, "window", "test_start_month", "")
    end = config_value(config, "window", "test_end_month", "")
    if start and end:
        return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]
    raise SystemExit("gap attribution requires --months or [attribution].months")


def prediction_path(prediction_dir: Path, month: str) -> Path:
    candidates = (
        prediction_dir / f"predictions_{month}.parquet",
        prediction_dir / f"month_{month}" / "predictions.parquet",
        prediction_dir / f"month_{month}" / f"predictions_{month}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    for shard_dir in sorted(prediction_dir.glob("fold_*_*")):
        try:
            start_month, end_month = shard_dir.name.removeprefix("fold_").split("_", 1)
        except ValueError:
            continue
        if start_month <= month <= end_month:
            path = shard_dir / "predictions.parquet"
            if path.exists():
                return path
    raise SystemExit(f"no prediction shard found for {month} under {prediction_dir}")


def read_predictions(path: Path) -> pd.DataFrame:
    required = [
        *KEY_COLUMNS,
        "label",
        "alpha_return_next_close",
        "candidate_alpha_rank",
        "gap_risk_rank",
    ]
    available = frame_columns(path)
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{path}: prediction shard missing columns: {missing}")
    optional = [column for column in ["binary_risk_rank"] if column in available]
    columns = [*required, *optional]
    frame = pd.read_parquet(path, columns=columns)
    return normalize_frame(frame)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys(frame)
    for column in set(out.columns) - set(KEY_COLUMNS):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def read_feature_month(
    labeled_path: Path,
    *,
    month: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    available = frame_columns(labeled_path)
    features = [column for column in feature_columns if column in available]
    if not features:
        raise SystemExit(f"{labeled_path}: none of requested feature columns exist")
    columns = [column for column in [*KEY_COLUMNS, *features] if column in available]
    start = pd.Period(month, freq="M").start_time.strftime("%Y-%m-%d")
    end = pd.Period(month, freq="M").end_time.strftime("%Y-%m-%d")
    try:
        frame = pd.read_parquet(
            labeled_path,
            columns=columns,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )
    except Exception as exc:
        print(f"warning: filtered read failed for {labeled_path}: {exc}; reading columns")
        frame = pd.read_parquet(labeled_path, columns=columns)
        frame["date"] = frame["date"].astype(str)
        frame = frame.loc[frame["date"].between(start, end)].copy()
    return normalize_frame(frame)


def add_group_feature_scales(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rank = x.groupby(groupers).rank(method="average", pct=True)
        mean = x.groupby(groupers).transform("mean")
        std = x.groupby(groupers).transform("std").replace(0.0, np.nan)
        frame[f"{feature}__rank"] = rank
        frame[f"{feature}__z"] = ((x - mean) / std).replace([np.inf, -np.inf], np.nan)
    return frame


def add_group_label_excess(frame: pd.DataFrame) -> pd.DataFrame:
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    for source, target in [
        ("label", "short_excess_bps"),
        ("alpha_return_next_close", "next_excess_bps"),
    ]:
        x = pd.to_numeric(frame[source], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        mean = x.groupby(groupers).transform("mean")
        frame[target] = (x - mean) * 10_000.0
    return frame


def selected_ids(group: pd.DataFrame, variant: dict[str, object], top_n: int) -> set[int]:
    risk_model = str(variant.get("risk_model", "") or "").lower()
    penalty = float(variant.get("penalty", 0.0) or 0.0)
    candidate_min = float(variant.get("candidate_alpha_rank_min", 0.0) or 0.0)
    candidates = group.loc[group["candidate_alpha_rank"].ge(candidate_min)].copy()
    if candidates.empty:
        return set()
    risk_col = RISK_RANK_BY_MODEL.get(risk_model, "")
    risk_values = candidates[risk_col].fillna(0.0) if risk_col else 0.0
    candidates["final_score"] = candidates["candidate_alpha_rank"] - penalty * risk_values
    selected = candidates.sort_values("final_score", ascending=False).head(top_n)
    return set(selected["row_id"].astype(int))


def build_membership_and_group_metrics(
    frame: pd.DataFrame,
    *,
    month: str,
    variants: list[dict[str, object]],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    group_cols = ["date", "decision_target_timestamp"]
    baseline_spec = {
        "variant": "alpha_rank",
        "risk_model": "",
        "penalty": 0.0,
        "candidate_alpha_rank_min": 0.0,
    }
    for (date, timestamp), group in frame.groupby(group_cols, sort=True):
        baseline_ids = selected_ids(group, baseline_spec, top_n)
        full_short = finite_mean(group["label"])
        full_next = finite_mean(group["alpha_return_next_close"])

        for variant in variants:
            variant_name = str(variant["variant"])
            variant_ids = selected_ids(group, variant, top_n)
            cohorts = {
                "baseline_top100": baseline_ids,
                "variant_top100": variant_ids,
                "baseline_kept": baseline_ids & variant_ids,
                "penalized_out": baseline_ids - variant_ids,
                "replacement_in": variant_ids - baseline_ids,
            }
            for cohort, row_ids in cohorts.items():
                for row_id in row_ids:
                    membership_rows.append(
                        {
                            "test_month": month,
                            "variant": variant_name,
                            "date": str(date),
                            "decision_target_timestamp": pd.Timestamp(timestamp),
                            "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                            "row_id": int(row_id),
                            "cohort": cohort,
                        }
                    )
                selected = group.loc[group["row_id"].isin(row_ids)]
                short_mean = finite_mean(selected["label"]) if len(selected) else float("nan")
                next_mean = (
                    finite_mean(selected["alpha_return_next_close"])
                    if len(selected)
                    else float("nan")
                )
                metric_rows.append(
                    {
                        "test_month": month,
                        "variant": variant_name,
                        "date": str(date),
                        "decision_target_timestamp": pd.Timestamp(timestamp),
                        "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                        "cohort": cohort,
                        "rows": int(len(selected)),
                        "short_mean_bps": short_mean * 10_000.0,
                        "short_excess_bps": (short_mean - full_short) * 10_000.0,
                        "next_mean_bps": next_mean * 10_000.0,
                        "next_excess_bps": (next_mean - full_next) * 10_000.0,
                    }
                )
    return pd.DataFrame(membership_rows), pd.DataFrame(metric_rows)


def summarize_group_metrics(group_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def aggregate(keys: list[str]) -> pd.DataFrame:
        return (
            group_metrics.groupby(keys, dropna=False)
            .agg(
                groups=("rows", "size"),
                mean_rows=("rows", "mean"),
                short_mean_bps=("short_mean_bps", "mean"),
                short_excess_bps=("short_excess_bps", "mean"),
                next_mean_bps=("next_mean_bps", "mean"),
                next_excess_bps=("next_excess_bps", "mean"),
                next_positive_rate=("next_excess_bps", lambda values: float((values > 0).mean())),
            )
            .reset_index()
        )

    return (
        aggregate(["test_month", "variant", "cohort"]),
        aggregate(["variant", "cohort"]),
    )


def summarize_feature_exposure(
    membership: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = membership.merge(
        frame[
            [
                "row_id",
                *features,
                *(f"{feature}__rank" for feature in features),
                *(f"{feature}__z" for feature in features),
            ]
        ],
        on="row_id",
        how="left",
    )
    summaries = []
    keys = ["test_month", "variant", "cohort"]
    for feature in features:
        summary = selected.groupby(keys, as_index=False).agg(
            rows=(feature, "size"),
            mean_raw=(feature, finite_mean),
            median_raw=(feature, "median"),
            mean_rank=(f"{feature}__rank", finite_mean),
            mean_z=(f"{feature}__z", finite_mean),
        )
        summary.insert(len(keys), "feature", feature)
        summaries.append(summary)
    month_summary = pd.concat(summaries, ignore_index=True)
    overall = (
        month_summary.groupby(["variant", "cohort", "feature"], dropna=False)
        .agg(
            months=("test_month", "nunique"),
            rows=("rows", "sum"),
            mean_raw=("mean_raw", "mean"),
            median_raw=("median_raw", "mean"),
            mean_rank=("mean_rank", "mean"),
            mean_z=("mean_z", "mean"),
        )
        .reset_index()
    )
    columns = ["variant", "feature", "mean_rank", "mean_z"]
    kept = overall.loc[overall["cohort"].eq("baseline_kept"), columns]
    penalized = overall.loc[overall["cohort"].eq("penalized_out"), columns]
    delta = penalized.merge(kept, on=["variant", "feature"], suffixes=("", "_kept"))
    if delta.empty:
        delta = pd.DataFrame()
    else:
        for metric in ("rank", "z"):
            delta[f"penalized_minus_kept_{metric}"] = (
                delta[f"mean_{metric}"] - delta[f"mean_{metric}_kept"]
            )
        delta = delta[["variant", "feature", "penalized_minus_kept_rank", "penalized_minus_kept_z"]]
    return month_summary, overall, delta


def fit_residual(values: pd.DataFrame, control_columns: list[str]) -> pd.Series:
    y = pd.to_numeric(values["next_excess_bps"], errors="coerce").to_numpy(dtype=float)
    controls = values[[f"{column}__z" for column in control_columns]]
    controls = controls.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xmat = np.column_stack([np.ones(len(values)), controls])
    mask = np.isfinite(y) & np.isfinite(xmat).all(axis=1)
    residual = np.full(len(values), np.nan, dtype=float)
    if mask.sum() <= xmat.shape[1] + 5:
        return pd.Series(residual, index=values.index)
    beta, *_ = np.linalg.lstsq(xmat[mask], y[mask], rcond=None)
    residual[mask] = y[mask] - xmat[mask] @ beta
    return pd.Series(residual, index=values.index)


def residual_penalized_vs_kept(
    membership: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    controls: list[str],
) -> pd.DataFrame:
    wanted = membership.loc[membership["cohort"].isin(["baseline_kept", "penalized_out"])].copy()
    columns = [
        "row_id",
        "next_excess_bps",
        *(f"{control}__z" for control in controls),
    ]
    values = wanted.merge(frame[columns], on="row_id", how="left")
    rows = []
    for keys, part in values.groupby(["test_month", "variant"], dropna=False):
        month, variant = keys
        part = part.copy()
        part["next_residual_bps"] = fit_residual(part, controls)
        by_cohort = (
            part.groupby("cohort")
            .agg(
                rows=("row_id", "size"),
                next_excess_bps=("next_excess_bps", "mean"),
                next_residual_bps=("next_residual_bps", "mean"),
            )
            .reset_index()
            .set_index("cohort")
        )
        if {"baseline_kept", "penalized_out"} <= set(by_cohort.index):
            kept = by_cohort.loc["baseline_kept"]
            penalized = by_cohort.loc["penalized_out"]
            row = {
                "test_month": month,
                "variant": variant,
                "controls": " ".join(controls),
                "kept_rows": int(kept["rows"]),
                "penalized_rows": int(penalized["rows"]),
            }
            for metric in ("next_excess_bps", "next_residual_bps"):
                row[f"kept_{metric}"] = float(kept[metric])
                row[f"penalized_{metric}"] = float(penalized[metric])
                row[f"penalized_minus_kept_{metric}"] = float(penalized[metric] - kept[metric])
            rows.append(row)
    month_summary = pd.DataFrame(rows)
    if month_summary.empty:
        return month_summary
    aggregations = {"test_month": ("test_month", lambda _: "OVERALL")}
    for column in month_summary.columns:
        if column in {"test_month", "variant"}:
            continue
        operation = (
            "first" if column == "controls" else "sum" if column.endswith("_rows") else "mean"
        )
        aggregations[column] = (column, operation)
    overall = month_summary.groupby("variant", dropna=False).agg(**aggregations).reset_index()
    return pd.concat([month_summary, overall], ignore_index=True)


def main() -> None:
    args = parse_args()
    config, settings, run_name = command_context(
        args, "attribution", default_run_name="gap_risk_attribution"
    )
    prediction_dir = Path(settings.string("prediction_dir"))
    labeled_path = Path(settings.string("labeled_path"))
    if not prediction_dir:
        raise SystemExit(
            "gap attribution requires --prediction-dir or [attribution].prediction_dir"
        )
    if not labeled_path:
        raise SystemExit("gap attribution requires --labeled-path or [attribution].labeled_path")
    output_dir = prepare_output_dir(config, args.output_dir, run_name)

    months = month_sequence(config, args.months)
    top_n = args.top_n or config_int(config, "attribution", "top_n", 100)
    variants = configured_score_variants(
        config,
        "attribution",
        DEFAULT_VARIANTS,
        default_risk_model="gap",
    )
    features = config_list(config, "attribution", "feature_columns", DEFAULT_FEATURES)
    controls = config_list(config, "attribution", "control_columns", DEFAULT_CONTROLS)
    requested_features = sorted(set(features) | set(controls))
    available = frame_columns(labeled_path)
    active_features = [column for column in requested_features if column in available]
    active_controls = [column for column in controls if column in active_features]
    if not active_features:
        raise SystemExit("no configured feature/control columns exist in labeled_path")

    all_group_metrics = []
    all_membership = []
    all_frames = []
    trace = {
        "run_id": run_name,
        "prediction_dir": str(prediction_dir),
        "labeled_path": str(labeled_path),
        "months": months,
        "top_n": top_n,
        "variants": variants,
        "features": active_features,
        "controls": active_controls,
    }
    for month in months:
        print(f"\nloading month {month}")
        pred = read_predictions(prediction_path(prediction_dir, month))
        feature_frame = read_feature_month(
            labeled_path,
            month=month,
            feature_columns=active_features,
        )
        frame = pred.merge(feature_frame, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        frame = frame.dropna(
            subset=["label", "alpha_return_next_close", "candidate_alpha_rank"]
        ).copy()
        frame["row_id"] = np.arange(len(frame), dtype=np.int64) + sum(
            len(item) for item in all_frames
        )
        active_for_month = [column for column in active_features if column in frame.columns]
        frame = add_group_feature_scales(frame, active_for_month)
        frame = add_group_label_excess(frame)
        membership, group_metrics = build_membership_and_group_metrics(
            frame,
            month=month,
            variants=variants,
            top_n=top_n,
        )
        all_membership.append(membership)
        all_group_metrics.append(group_metrics)
        keep_columns = [
            *"row_id date symbol decision_target_timestamp label alpha_return_next_close candidate_alpha_rank gap_risk_rank short_excess_bps next_excess_bps".split(),
            *active_for_month,
            *(f"{feature}__rank" for feature in active_for_month),
            *(f"{feature}__z" for feature in active_for_month),
        ]
        all_frames.append(frame[keep_columns])
        print(
            json.dumps(
                {
                    "month": month,
                    "prediction_rows": len(pred),
                    "feature_rows": len(feature_frame),
                    "merged_rows": len(frame),
                    "membership_rows": len(membership),
                },
                ensure_ascii=False,
            )
        )

    membership = pd.concat(all_membership, ignore_index=True)
    group_metrics = pd.concat(all_group_metrics, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)
    month_outcome, overall_outcome = summarize_group_metrics(group_metrics)
    feature_month, feature_overall, feature_delta = summarize_feature_exposure(
        membership,
        frames,
        features=[column for column in active_features if column in frames.columns],
    )
    residual = residual_penalized_vs_kept(
        membership,
        frames,
        controls=[column for column in active_controls if column in frames.columns],
    )

    group_metrics.to_csv(output_dir / "gap_attribution_group_metrics.csv", index=False)
    month_outcome.to_csv(output_dir / "gap_attribution_outcomes_by_month.csv", index=False)
    overall_outcome.to_csv(output_dir / "gap_attribution_outcomes_overall.csv", index=False)
    feature_month.to_csv(output_dir / "gap_attribution_feature_exposure_by_month.csv", index=False)
    feature_overall.to_csv(output_dir / "gap_attribution_feature_exposure_overall.csv", index=False)
    feature_delta.to_csv(output_dir / "gap_attribution_penalized_feature_delta.csv", index=False)
    residual.to_csv(output_dir / "gap_attribution_residual_penalized_vs_kept.csv", index=False)
    write_json(output_dir / "gap_attribution_trace.json", trace)

    print("\ngap_attribution_outcomes_overall")
    print(overall_outcome.to_string(index=False))
    if not residual.empty:
        print("\ngap_attribution_residual_penalized_vs_kept")
        print(residual.loc[residual["test_month"].eq("OVERALL")].to_string(index=False))
    print(f"\noutput_dir={output_dir}")


if __name__ == "__main__":
    main()
