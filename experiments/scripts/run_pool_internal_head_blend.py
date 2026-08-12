from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL
from opening_strength_fit.config import load_toml
from opening_strength_fit.io import write_frame, write_json
from opening_strength_fit.pool_internal_eval import evaluate_pool
from opening_strength_fit.schema import normalize_decision_keys
from opening_strength_fit.stock_pool import load_stock_pool, stock_pool_membership_mask

GROUP_COLS = ["date", "decision_target_timestamp"]
METRICS = ["short_internal_excess_bps", "next_internal_excess_bps", "short_rank_ic", "next_rank_ic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pool-internal head-only score blend.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def prediction_path(root: Path, month: str) -> Path:
    shard = root / f"month_{month}"
    for name in ("predictions.parquet", f"predictions_{month[:4]}.parquet"):
        path = shard / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing prediction shard under {shard}")


def next_label_path(root: Path, year: str) -> Path:
    preferred = root / f"opening_{year}_next_close_labels_v1.parquet"
    matches = [preferred, *sorted(root.glob(f"*{year}*next_close*.parquet"))]
    if path := next((path for path in matches if path.exists()), None):
        return path
    raise FileNotFoundError(f"missing next-close label shard for {year} under {root}")


def read_predictions(path: Path, score_name: str, *, include_label: bool) -> pd.DataFrame:
    columns = [*KEY_COLUMNS, "prediction", *(["label"] if include_label else [])]
    frame = normalize_decision_keys(pd.read_parquet(path, columns=columns), drop_missing=True)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"duplicate prediction keys: {path}")
    return frame.rename(columns={"prediction": score_name})


def read_next_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*KEY_COLUMNS, NEXT_CLOSE_LABEL_COL])
    frame = normalize_decision_keys(frame, drop_missing=True).dropna(subset=[NEXT_CLOSE_LABEL_COL])
    return frame.drop_duplicates(list(KEY_COLUMNS), keep="last")


def normalized_descending_rank(frame: pd.DataFrame, score_col: str) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(GROUP_COLS, sort=False, observed=True)[score_col]
    rank = grouped.rank(ascending=False, method="first", na_option="bottom")
    denominator = (grouped.transform("size").astype("float64") - 1.0).clip(lower=1.0)
    return rank, 1.0 - (rank.astype("float64") - 1.0) / denominator


def build_variant_score(frame: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    mode = str(spec.get("mode", "")).strip().lower()
    if mode in {"base", "overlay"}:
        return frame[f"{mode}_rank_score"].copy()
    if mode != "head_boost":
        raise ValueError(f"unsupported blend variant mode: {mode!r}")
    head_n = int(spec.get("overlay_head_n", 0))
    gate_n = int(spec.get("base_gate_n", 0))
    weight = float(spec.get("weight", 0.0))
    if head_n <= 0 or weight <= 0:
        raise ValueError(f"invalid head boost variant: {spec}")
    eligible = frame["overlay_available"] & frame["overlay_rank"].le(head_n)
    if gate_n > 0:
        eligible &= frame["base_rank"].le(gate_n)
    strength = ((head_n + 1.0 - frame["overlay_rank"]) / head_n).clip(0.0, 1.0)
    return frame["base_rank_score"] + weight * strength.where(eligible, 0.0).fillna(0.0)


def summarize_group_metrics(
    group_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = (
        group_metrics.groupby(["variant", "top_n", "test_month"], observed=True)[METRICS]
        .mean()
        .reset_index()
    )
    quarter_frame = group_metrics.assign(
        quarter=pd.to_datetime(group_metrics["date"]).dt.to_period("Q").astype(str)
    )
    quarterly = (
        quarter_frame.groupby(["variant", "top_n", "quarter"], observed=True)[METRICS]
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
            short_positive_months=("short_internal_excess_bps", lambda s: int(s.gt(0).sum())),
            next_positive_months=("next_internal_excess_bps", lambda s: int(s.gt(0).sum())),
        )
        .reset_index()
    )
    return summary.merge(positive, on=["variant", "top_n"]), monthly, quarterly


def tail_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, top_n), item in selected.groupby(["variant", "top_n"], observed=True):
        values = item["next_excess_bps"].dropna().to_numpy(dtype="float64")
        row = {
            "variant": variant,
            "top_n": int(top_n),
            "rows": len(values),
            "raw_mean_bps": np.mean(values),
        }
        for quantile in (0.95, 0.99):
            suffix = int(quantile * 100)
            threshold = float(np.quantile(values, quantile))
            winsor = np.minimum(values, threshold)
            row.update(
                {
                    f"p{suffix}_threshold_bps": threshold,
                    f"p{suffix}_winsor_mean_bps": np.mean(winsor),
                    f"p{suffix}_upper_tail_contribution_bps": np.mean(values - winsor),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _overlap_summary(overlap: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    summary = (
        overlap.groupby(keys, observed=True)
        .agg(
            groups=("selected_rows", "size"),
            selected_rows=("selected_rows", "mean"),
            overlap_with_base=("overlap_with_base", "mean"),
            overlap_with_overlay=("overlap_with_overlay", "mean"),
        )
        .reset_index()
    )
    summary["base_overlap_rate"] = summary["overlap_with_base"] / summary["selected_rows"]
    summary["overlay_overlap_rate"] = summary["overlap_with_overlay"] / summary["selected_rows"]
    return summary


def _load_month(blend: dict, month: str, pool, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base_path = prediction_path(Path(str(blend["base_prediction_root"])), month)
    overlay_path = prediction_path(Path(str(blend["overlay_prediction_root"])), month)
    base = read_predictions(base_path, "base_prediction", include_label=True)
    overlay = read_predictions(overlay_path, "overlay_prediction", include_label=False)
    frame = base.merge(overlay, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    frame["overlay_available"] = frame["overlay_prediction"].notna()
    frame = frame.loc[stock_pool_membership_mask(frame, pool)].copy()
    frame = frame.merge(labels, on=list(KEY_COLUMNS), how="left", validate="many_to_one")
    missing = int(frame[NEXT_CLOSE_LABEL_COL].isna().sum())
    if missing:
        raise ValueError(f"missing next-close labels for {month}: {missing}")
    for name in ("base", "overlay"):
        frame[f"{name}_rank"], frame[f"{name}_rank_score"] = normalized_descending_rank(
            frame, f"{name}_prediction"
        )
    for prefix, column in (("next", NEXT_CLOSE_LABEL_COL), ("short", "label")):
        mean = frame.groupby(GROUP_COLS, observed=True)[column].transform("mean")
        frame[f"{prefix}_excess_bps"] = (frame[column] - mean) * 10_000.0
    return frame, {
        "base_path": str(base_path),
        "overlay_path": str(overlay_path),
        "base_rows": len(base),
        "overlay_rows": len(overlay),
        "pool_rows": len(frame),
        "groups": int(frame.groupby(GROUP_COLS, observed=True).ngroups),
        "overlay_coverage": float(frame["overlay_available"].mean()),
        "missing_next_labels": missing,
    }


def run(config_path: Path, output_override: str = "") -> None:
    config = load_toml(config_path)
    blend = config["blend"]
    variants = list(blend.get("variants", []))
    if not variants:
        raise SystemExit("[blend] requires at least one [[blend.variants]] entry")
    output_dir = Path(output_override or config["output"]["local_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(str(blend["pool_path"]))
    months = [str(value) for value in blend["months"]]
    top_n_list = [int(value) for value in blend.get("top_n_list", [100])]
    bucket_count = int(blend.get("bucket_count", 20))
    labels_by_year: dict[str, pd.DataFrame] = {}
    metric_parts, selected_parts, overlap_parts, bucket_parts = [], [], [], []
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "base_prediction_root": str(blend["base_prediction_root"]),
        "overlay_prediction_root": str(blend["overlay_prediction_root"]),
        "next_close_label_root": str(blend["next_close_label_root"]),
        "pool_path": str(blend["pool_path"]),
        "months": {},
        "top_n_list": top_n_list,
        "bucket_count": bucket_count,
        "variants": variants,
        "formula": "base_pool_rank_score + weight * linear_overlay_head_strength, optionally gated by base rank",
        "weighting": "decision_group_equal for acceptance and bucket summaries",
    }
    for month in months:
        year = month[:4]
        if year not in labels_by_year:
            labels_by_year.clear()
            labels_by_year[year] = read_next_labels(
                next_label_path(Path(str(blend["next_close_label_root"])), year)
            )
        frame, month_trace = _load_month(blend, month, pool, labels_by_year[year])
        trace["months"][month] = month_trace
        for spec in variants:
            variant = str(spec["name"])
            score_col = f"score__{variant}"
            frame[score_col] = build_variant_score(frame, spec)
            final_rank = frame.groupby(GROUP_COLS, observed=True)[score_col].rank(
                ascending=False, method="first"
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
                metric_parts.append(metrics)
                selected = frame.loc[final_rank.le(top_n)].copy()
                selected["variant"], selected["top_n"] = variant, top_n
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
                overlap["variant"], overlap["top_n"] = variant, top_n
                overlap["month"] = pd.to_datetime(overlap["date"]).dt.to_period("M").astype(str)
                overlap_parts.append(overlap)
            size = frame.groupby(GROUP_COLS, observed=True)[score_col].transform("size")
            bucket = (
                np.floor((final_rank - 1.0) * bucket_count / size).clip(0, bucket_count - 1) + 1
            )
            bucket_frame = frame[GROUP_COLS + ["next_excess_bps", "short_excess_bps"]].assign(
                bucket=bucket.astype("int16"), month=month
            )
            part = (
                bucket_frame.groupby(GROUP_COLS + ["month", "bucket"], observed=True)
                .agg(
                    rows=("next_excess_bps", "size"),
                    next_excess_bps=("next_excess_bps", "mean"),
                    short_excess_bps=("short_excess_bps", "mean"),
                )
                .reset_index()
                .groupby(["month", "bucket"], observed=True)
                .agg(
                    groups=("next_excess_bps", "size"),
                    rows=("rows", "sum"),
                    next_excess_bps=("next_excess_bps", "mean"),
                    short_excess_bps=("short_excess_bps", "mean"),
                )
                .reset_index()
            )
            part.insert(0, "variant", variant)
            bucket_parts.append(part)

    group_metrics = pd.concat(metric_parts, ignore_index=True)
    summary, monthly, quarterly = summarize_group_metrics(group_metrics)
    selected = pd.concat(selected_parts, ignore_index=True)
    overlap = pd.concat(overlap_parts, ignore_index=True)
    bucket_monthly = pd.concat(bucket_parts, ignore_index=True)
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
    outputs = {
        "blend_summary.csv": summary,
        "blend_monthly.csv": monthly,
        "blend_quarterly.csv": quarterly,
        "blend_group_metrics.parquet": group_metrics,
        "blend_selected_top100.parquet": selected,
        "blend_selected_tail_summary.csv": tail_summary(selected),
        "blend_overlap_summary.csv": _overlap_summary(overlap, ["variant", "top_n"]),
        "blend_overlap_monthly.csv": _overlap_summary(overlap, ["variant", "top_n", "month"]),
        "blend_pool_l_20bucket_summary.csv": bucket_summary,
        "blend_pool_l_20bucket_monthly.csv": bucket_monthly,
    }
    for name, frame in outputs.items():
        write_frame(frame, output_dir / name)
    write_json(output_dir / "blend_trace.json", trace, ensure_ascii=True)
    (output_dir / "_SUCCESS").touch()
    print(summary.sort_values(["top_n", "next_internal_excess_bps"], ascending=[True, False]))
    print(f"output_dir={output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run(Path(args.config), args.output_dir)


if __name__ == "__main__":
    main()
