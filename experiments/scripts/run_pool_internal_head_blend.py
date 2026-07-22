from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL
from opening_strength_fit.config import load_toml
from opening_strength_fit.pool_internal_eval import evaluate_pool
from opening_strength_fit.schema import normalize_decision_keys
from opening_strength_fit.stock_pool import load_stock_pool, stock_pool_membership_mask

GROUP_COLS = ["date", "decision_target_timestamp"]
PREDICTION_COLS = [*KEY_COLUMNS, "prediction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pool-internal head-only score blend.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def prediction_path(root: Path, month: str) -> Path:
    shard = root / f"month_{month}"
    preferred = shard / "predictions.parquet"
    if preferred.exists():
        return preferred
    fallback = shard / f"predictions_{month[:4]}.parquet"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"missing prediction shard under {shard}")


def next_label_path(root: Path, year: str) -> Path:
    preferred = root / f"opening_{year}_next_close_labels_v1.parquet"
    if preferred.exists():
        return preferred
    matches = sorted(root.glob(f"*{year}*next_close*.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing next-close label shard for {year} under {root}")


def read_predictions(path: Path, score_name: str, *, include_label: bool) -> pd.DataFrame:
    columns = [*PREDICTION_COLS, "label"] if include_label else PREDICTION_COLS
    frame = pd.read_parquet(path, columns=columns)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"duplicate prediction keys: {path}")
    frame = normalize_decision_keys(frame, drop_missing=True)
    return frame.rename(columns={"prediction": score_name})


def read_next_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*KEY_COLUMNS, NEXT_CLOSE_LABEL_COL])
    frame = normalize_decision_keys(frame, drop_missing=True)
    frame = frame.dropna(subset=[NEXT_CLOSE_LABEL_COL])
    return frame.drop_duplicates(list(KEY_COLUMNS), keep="last")


def normalized_descending_rank(frame: pd.DataFrame, score_col: str) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(GROUP_COLS, sort=False, observed=True)
    rank = grouped[score_col].rank(ascending=False, method="first", na_option="bottom")
    size = grouped[score_col].transform("size").astype("float64")
    denominator = (size - 1.0).clip(lower=1.0)
    score = 1.0 - (rank.astype("float64") - 1.0) / denominator
    return rank, score


def build_variant_score(frame: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    mode = str(spec.get("mode", "")).strip().lower()
    if mode == "base":
        return frame["base_rank_score"].copy()
    if mode == "overlay":
        return frame["overlay_rank_score"].copy()
    if mode != "head_boost":
        raise ValueError(f"unsupported blend variant mode: {mode!r}")

    overlay_head_n = int(spec.get("overlay_head_n", 0))
    base_gate_n = int(spec.get("base_gate_n", 0))
    weight = float(spec.get("weight", 0.0))
    if overlay_head_n <= 0 or weight <= 0.0:
        raise ValueError(f"invalid head boost variant: {spec}")

    eligible = frame["overlay_available"] & frame["overlay_rank"].le(overlay_head_n)
    if base_gate_n > 0:
        eligible &= frame["base_rank"].le(base_gate_n)
    head_strength = ((overlay_head_n + 1.0 - frame["overlay_rank"]) / overlay_head_n).clip(
        lower=0.0,
        upper=1.0,
    )
    boost = head_strength.where(eligible, 0.0).fillna(0.0)
    return frame["base_rank_score"] + weight * boost


def summarize_group_metrics(
    group_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        "short_rank_ic",
        "next_rank_ic",
    ]
    monthly = (
        group_metrics.groupby(["variant", "top_n", "test_month"], observed=True)[metric_cols]
        .mean()
        .reset_index()
    )
    quarterly_frame = group_metrics.copy()
    quarterly_frame["quarter"] = (
        pd.to_datetime(quarterly_frame["date"]).dt.to_period("Q").astype(str)
    )
    quarterly = (
        quarterly_frame.groupby(["variant", "top_n", "quarter"], observed=True)[metric_cols]
        .mean()
        .reset_index()
    )
    summary = (
        group_metrics.groupby(["variant", "top_n"], observed=True)
        .agg(
            groups=("next_internal_excess_bps", "size"),
            months=("test_month", "nunique"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            next_group_p10_bps=("next_internal_excess_bps", lambda s: float(s.quantile(0.10))),
            next_group_median_bps=("next_internal_excess_bps", "median"),
            next_group_p90_bps=("next_internal_excess_bps", lambda s: float(s.quantile(0.90))),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
        )
        .reset_index()
    )
    positive = (
        monthly.groupby(["variant", "top_n"], observed=True)
        .agg(
            short_positive_months=("short_internal_excess_bps", lambda s: int((s > 0).sum())),
            next_positive_months=("next_internal_excess_bps", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    return summary.merge(positive, on=["variant", "top_n"], how="left"), monthly, quarterly


def tail_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, top_n), item in selected.groupby(["variant", "top_n"], observed=True):
        values = item["next_excess_bps"].dropna().to_numpy(dtype="float64")
        row: dict[str, object] = {
            "variant": variant,
            "top_n": int(top_n),
            "rows": len(values),
            "raw_mean_bps": float(np.mean(values)),
        }
        for quantile in (0.95, 0.99):
            threshold = float(np.quantile(values, quantile))
            winsor = np.minimum(values, threshold)
            suffix = int(quantile * 100)
            row[f"p{suffix}_threshold_bps"] = threshold
            row[f"p{suffix}_winsor_mean_bps"] = float(np.mean(winsor))
            row[f"p{suffix}_upper_tail_contribution_bps"] = float(np.mean(values - winsor))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_toml(args.config)
    blend = config["blend"]
    variants = list(blend.get("variants", []))
    if not variants:
        raise SystemExit("[blend] requires at least one [[blend.variants]] entry")

    output_dir = Path(args.output_dir or config["output"]["local_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_root = Path(str(blend["base_prediction_root"]))
    overlay_root = Path(str(blend["overlay_prediction_root"]))
    next_root = Path(str(blend["next_close_label_root"]))
    months = [str(value) for value in blend["months"]]
    top_n_list = [int(value) for value in blend.get("top_n_list", [100])]
    bucket_count = int(blend.get("bucket_count", 20))

    print(f"loading stock pool: {blend['pool_path']}", flush=True)
    pool = load_stock_pool(str(blend["pool_path"]))
    label_cache: dict[str, pd.DataFrame] = {}
    group_metric_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    overlap_parts: list[pd.DataFrame] = []
    bucket_month_parts: list[pd.DataFrame] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "base_prediction_root": str(base_root),
        "overlay_prediction_root": str(overlay_root),
        "next_close_label_root": str(next_root),
        "pool_path": str(blend["pool_path"]),
        "months": {},
        "top_n_list": top_n_list,
        "bucket_count": bucket_count,
        "variants": variants,
        "formula": "base_pool_rank_score + weight * linear_overlay_head_strength, optionally gated by base rank",
        "weighting": "decision_group_equal for acceptance and bucket summaries",
    }

    for month in months:
        print(f"processing shard {month}", flush=True)
        base_path = prediction_path(base_root, month)
        overlay_path = prediction_path(overlay_root, month)
        base = read_predictions(base_path, "base_prediction", include_label=True)
        overlay = read_predictions(overlay_path, "overlay_prediction", include_label=False)
        frame = base.merge(overlay, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        frame["overlay_available"] = frame["overlay_prediction"].notna()
        pool_mask = stock_pool_membership_mask(frame, pool)
        frame = frame.loc[pool_mask].copy()

        year = month[:4]
        if year not in label_cache:
            label_cache.clear()
            label_cache[year] = read_next_labels(next_label_path(next_root, year))
        frame = frame.merge(
            label_cache[year], on=list(KEY_COLUMNS), how="left", validate="many_to_one"
        )
        missing_next = int(frame[NEXT_CLOSE_LABEL_COL].isna().sum())
        if missing_next:
            raise ValueError(f"missing next-close labels for {month}: {missing_next}")

        frame["base_rank"], frame["base_rank_score"] = normalized_descending_rank(
            frame, "base_prediction"
        )
        frame["overlay_rank"], frame["overlay_rank_score"] = normalized_descending_rank(
            frame, "overlay_prediction"
        )
        frame["pool_next_mean"] = frame.groupby(GROUP_COLS, observed=True)[
            NEXT_CLOSE_LABEL_COL
        ].transform("mean")
        frame["pool_short_mean"] = frame.groupby(GROUP_COLS, observed=True)["label"].transform(
            "mean"
        )
        frame["next_excess_bps"] = (
            frame[NEXT_CLOSE_LABEL_COL] - frame["pool_next_mean"]
        ) * 10_000.0
        frame["short_excess_bps"] = (frame["label"] - frame["pool_short_mean"]) * 10_000.0

        month_trace = {
            "base_path": str(base_path),
            "overlay_path": str(overlay_path),
            "base_rows": len(base),
            "overlay_rows": len(overlay),
            "pool_rows": len(frame),
            "groups": int(frame.groupby(GROUP_COLS, observed=True).ngroups),
            "overlay_coverage": float(frame["overlay_available"].mean()),
            "missing_next_labels": missing_next,
        }
        trace["months"][month] = month_trace

        for spec in variants:
            variant = str(spec["name"])
            score_col = f"score__{variant}"
            frame[score_col] = build_variant_score(frame, spec)
            final_rank = frame.groupby(GROUP_COLS, sort=False, observed=True)[score_col].rank(
                ascending=False,
                method="first",
            )

            for top_n in top_n_list:
                metrics = evaluate_pool(
                    frame,
                    pool_name="pool_L",
                    score_col=score_col,
                    short_label_col="label",
                    next_label_col=NEXT_CLOSE_LABEL_COL,
                    top_n=top_n,
                )
                metrics.insert(0, "top_n", top_n)
                metrics.insert(0, "variant", variant)
                group_metric_parts.append(metrics)

                selected = frame.loc[final_rank.le(top_n)].copy()
                selected["variant"] = variant
                selected["top_n"] = top_n
                selected["final_rank"] = final_rank.loc[selected.index].astype("int32")
                selected["variant_score"] = selected[score_col]
                selected["base_top_n"] = selected["base_rank"].le(top_n)
                selected["overlay_top_n"] = selected["overlay_rank"].le(top_n)
                if top_n == min(top_n_list):
                    selected_parts.append(
                        selected[
                            [
                                *KEY_COLUMNS,
                                "variant",
                                "top_n",
                                "variant_score",
                                "final_rank",
                                "base_rank",
                                "overlay_rank",
                                "base_top_n",
                                "overlay_top_n",
                                "label",
                                NEXT_CLOSE_LABEL_COL,
                                "short_excess_bps",
                                "next_excess_bps",
                            ]
                        ]
                    )
                overlap = (
                    selected.groupby(GROUP_COLS, observed=True)
                    .agg(
                        selected_rows=("symbol", "size"),
                        overlap_with_base=("base_top_n", "sum"),
                        overlap_with_overlay=("overlay_top_n", "sum"),
                    )
                    .reset_index()
                )
                overlap["variant"] = variant
                overlap["top_n"] = top_n
                overlap["month"] = pd.to_datetime(overlap["date"]).dt.to_period("M").astype(str)
                overlap_parts.append(overlap)

            group_size = frame.groupby(GROUP_COLS, observed=True)[score_col].transform("size")
            bucket = np.floor((final_rank - 1.0) * bucket_count / group_size).astype("int16") + 1
            bucket = bucket.clip(lower=1, upper=bucket_count)
            bucket_frame = frame[GROUP_COLS + ["next_excess_bps", "short_excess_bps"]].copy()
            bucket_frame["bucket"] = bucket
            bucket_frame["month"] = (
                pd.to_datetime(bucket_frame["date"]).dt.to_period("M").astype(str)
            )
            group_bucket = (
                bucket_frame.groupby(GROUP_COLS + ["month", "bucket"], observed=True)
                .agg(
                    rows=("next_excess_bps", "size"),
                    next_excess_bps=("next_excess_bps", "mean"),
                    short_excess_bps=("short_excess_bps", "mean"),
                )
                .reset_index()
            )
            bucket_month = (
                group_bucket.groupby(["month", "bucket"], observed=True)
                .agg(
                    groups=("next_excess_bps", "size"),
                    rows=("rows", "sum"),
                    next_excess_bps=("next_excess_bps", "mean"),
                    short_excess_bps=("short_excess_bps", "mean"),
                )
                .reset_index()
            )
            bucket_month.insert(0, "variant", variant)
            bucket_month_parts.append(bucket_month)

    group_metrics = pd.concat(group_metric_parts, ignore_index=True)
    summary, monthly, quarterly = summarize_group_metrics(group_metrics)
    selected = pd.concat(selected_parts, ignore_index=True)
    overlap = pd.concat(overlap_parts, ignore_index=True)
    overlap_summary = (
        overlap.groupby(["variant", "top_n"], observed=True)
        .agg(
            groups=("selected_rows", "size"),
            selected_rows=("selected_rows", "mean"),
            overlap_with_base=("overlap_with_base", "mean"),
            overlap_with_overlay=("overlap_with_overlay", "mean"),
        )
        .reset_index()
    )
    overlap_summary["base_overlap_rate"] = (
        overlap_summary["overlap_with_base"] / overlap_summary["selected_rows"]
    )
    overlap_summary["overlay_overlap_rate"] = (
        overlap_summary["overlap_with_overlay"] / overlap_summary["selected_rows"]
    )
    overlap_monthly = (
        overlap.groupby(["variant", "top_n", "month"], observed=True)
        .agg(
            groups=("selected_rows", "size"),
            selected_rows=("selected_rows", "mean"),
            overlap_with_base=("overlap_with_base", "mean"),
            overlap_with_overlay=("overlap_with_overlay", "mean"),
        )
        .reset_index()
    )
    overlap_monthly["base_overlap_rate"] = (
        overlap_monthly["overlap_with_base"] / overlap_monthly["selected_rows"]
    )
    overlap_monthly["overlay_overlap_rate"] = (
        overlap_monthly["overlap_with_overlay"] / overlap_monthly["selected_rows"]
    )

    bucket_monthly = pd.concat(bucket_month_parts, ignore_index=True)
    bucket_summary = (
        bucket_monthly.groupby(["variant", "bucket"], observed=True)
        .agg(
            months=("month", "nunique"),
            groups=("groups", "sum"),
            rows=("rows", "sum"),
            next_excess_bps=("next_excess_bps", "mean"),
            short_excess_bps=("short_excess_bps", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(output_dir / "blend_summary.csv", index=False, float_format="%.6f")
    monthly.to_csv(output_dir / "blend_monthly.csv", index=False, float_format="%.6f")
    quarterly.to_csv(output_dir / "blend_quarterly.csv", index=False, float_format="%.6f")
    group_metrics.to_parquet(output_dir / "blend_group_metrics.parquet", index=False)
    selected.to_parquet(output_dir / "blend_selected_top100.parquet", index=False)
    tail_summary(selected).to_csv(
        output_dir / "blend_selected_tail_summary.csv", index=False, float_format="%.6f"
    )
    overlap_summary.to_csv(
        output_dir / "blend_overlap_summary.csv", index=False, float_format="%.6f"
    )
    overlap_monthly.to_csv(
        output_dir / "blend_overlap_monthly.csv", index=False, float_format="%.6f"
    )
    bucket_summary.to_csv(
        output_dir / "blend_pool_l_20bucket_summary.csv", index=False, float_format="%.6f"
    )
    bucket_monthly.to_csv(
        output_dir / "blend_pool_l_20bucket_monthly.csv", index=False, float_format="%.6f"
    )
    with (output_dir / "blend_trace.json").open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=True, indent=2)
    (output_dir / "_SUCCESS").touch()
    print(summary.sort_values(["top_n", "next_internal_excess_bps"], ascending=[True, False]))
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
