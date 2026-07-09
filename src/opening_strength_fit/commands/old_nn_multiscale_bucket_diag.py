from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.stock_pool import load_stock_pool


DEFAULT_VARIANTS = {
    "nn_mlp_base": "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1",
    "nn_mlp_base_plus_mse": (
        "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_plus_mse_v1"
    ),
    "nn_deep_gelu_mse": (
        "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_deep_gelu_mse_v1"
    ),
}
DEFAULT_MONTHS = (
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
)
GROUP_COLS = ["date", "decision_target_timestamp"]
PREDICTION_COLS = ["date", "symbol", "decision_target_timestamp", "prediction"]
LABEL_COLS = ["date", "symbol", "decision_target_timestamp", "alpha_return_next_close"]


def parse_csv_ints(value: str, *, default: Iterable[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiscale Top1000 bucket diagnostics for the legacy three NN models."
        )
    )
    parser.add_argument(
        "--prediction-root",
        default="/mnt/output/opening_strength_fit/nn",
        help="PVC directory containing per-run prediction shard directories.",
    )
    parser.add_argument(
        "--next-label-root",
        default=(
            "/mnt/output/opening_strength_fit/cache/"
            "opening_13y_201301_202512_delay2_next_close_labels_v1"
        ),
        help="Directory containing opening_YYYY_next_close_labels_v1.parquet files.",
    )
    parser.add_argument(
        "--pool-path",
        default="lml.bzw@ssd/data/pool_L.parquet",
        help="Stock pool path understood by opening_strength_fit.stock_pool.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/output/opening_strength_fit/old_nn_multiscale_bucket_diag_v1",
    )
    parser.add_argument(
        "--months",
        default=",".join(DEFAULT_MONTHS),
        help="Comma-separated half-year month directories to process.",
    )
    parser.add_argument("--bucket-widths", default="50,100,200")
    parser.add_argument("--top-k", default="50,100,150,200,500,1000")
    parser.add_argument("--window-widths", default="50,100,200")
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(DEFAULT_VARIANTS),
        help="Variant to process. May be repeated. Defaults to all legacy variants.",
    )
    return parser.parse_args()


def normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def prediction_path(prediction_root: Path, run_id: str, month: str) -> Path:
    preferred = prediction_root / run_id / f"month_{month}" / "predictions.parquet"
    if preferred.exists():
        return preferred
    year = month.split("-", 1)[0]
    fallback = prediction_root / run_id / f"month_{month}" / f"predictions_{year}.parquet"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"missing prediction shard for {run_id} {month}: {preferred}")


def next_label_path(next_label_root: Path, year: str) -> Path:
    path = next_label_root / f"opening_{year}_next_close_labels_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing next-close label file: {path}")
    return path


def stock_pool_membership_mask(frame: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    dates = frame["date"].to_numpy(dtype=object)
    symbols = frame["symbol"].to_numpy(dtype=object)
    date_pos = pool.index.get_indexer(dates)
    symbol_pos = pool.columns.get_indexer(symbols)
    valid = (date_pos >= 0) & (symbol_pos >= 0)
    out = np.zeros(len(frame), dtype=bool)
    if valid.any():
        values = pool.to_numpy(dtype=bool, copy=False)
        out[valid] = values[date_pos[valid], symbol_pos[valid]]
    return out


def fixed_score_spearman(excess_bps: np.ndarray) -> float:
    n = len(excess_bps)
    if n < 3:
        return math.nan
    if not np.isfinite(excess_bps).all() or np.nanstd(excess_bps) == 0:
        return math.nan
    order = np.argsort(excess_bps, kind="mergesort")
    y_rank = np.empty(n, dtype="float64")
    y_rank[order] = np.arange(1, n + 1, dtype="float64")
    x_rank = np.arange(n, 0, -1, dtype="float64")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize_monthly_positive(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    monthly = (
        frame.groupby(group_cols + ["month"], observed=True)[value_col]
        .mean()
        .reset_index(name=f"{value_col}_month_mean")
    )
    summary = (
        monthly.groupby(group_cols, observed=True)
        .agg(
            months=("month", "nunique"),
            positive_months=(f"{value_col}_month_mean", lambda s: int((s > 0).sum())),
            month_min=(f"{value_col}_month_mean", "min"),
            month_max=(f"{value_col}_month_mean", "max"),
        )
        .reset_index()
    )
    return summary


def summarize_variant_average(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        col
        for col in frame.columns
        if col not in set(key_cols + ["variant", "rank_slice", "window"])
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not metric_cols:
        return pd.DataFrame()
    out = frame.groupby(key_cols, observed=True)[metric_cols].mean().reset_index()
    out.insert(0, "variant", "old3_mean")
    if "rank_slice" in frame.columns:
        rank_slice = frame.drop_duplicates(key_cols)[key_cols + ["rank_slice"]]
        out = out.merge(rank_slice, on=key_cols, how="left")
    if "window" in frame.columns:
        window = frame.drop_duplicates(key_cols)[key_cols + ["window"]]
        out = out.merge(window, on=key_cols, how="left")
    return out


def add_bucket_summaries(
    top_rows: pd.DataFrame,
    *,
    variant: str,
    bucket_widths: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    month_summaries: list[pd.DataFrame] = []
    for width in bucket_widths:
        frame = top_rows.loc[top_rows["score_rank"] <= (top_rows["top_n"] // width) * width].copy()
        frame["bucket"] = ((frame["score_rank"] - 1) // width + 1).astype(int)
        frame["rank_slice"] = (
            ((frame["bucket"] - 1) * width + 1).astype(str)
            + "-"
            + (frame["bucket"] * width).astype(str)
        )
        frame["positive_excess"] = frame["excess_bps"] > 0
        frame["positive_part_bps"] = frame["excess_bps"].clip(lower=0)
        frame["negative_part_bps"] = frame["excess_bps"].clip(upper=0)
        frame["big_win_100bps"] = frame["excess_bps"] >= 100.0
        frame["big_win_300bps"] = frame["excess_bps"] >= 300.0
        frame["big_win_500bps"] = frame["excess_bps"] >= 500.0

        group_cols = ["variant", "bucket_width", "bucket", "rank_slice"]
        frame["variant"] = variant
        frame["bucket_width"] = width
        summary = (
            frame.groupby(group_cols, observed=True)
            .agg(
                rows=("excess_bps", "size"),
                mean_excess_bps=("excess_bps", "mean"),
                median_excess_bps=("excess_bps", "median"),
                positive_excess_rate=("positive_excess", "mean"),
                positive_part_bps=("positive_part_bps", "mean"),
                negative_part_bps=("negative_part_bps", "mean"),
                p90_excess_bps=("excess_bps", lambda s: float(s.quantile(0.90))),
                p95_excess_bps=("excess_bps", lambda s: float(s.quantile(0.95))),
                p99_excess_bps=("excess_bps", lambda s: float(s.quantile(0.99))),
                realized_pool_top5_rate=("realized_pool_top5", "mean"),
                realized_pool_top10_rate=("realized_pool_top10", "mean"),
                big_win_100bps_rate=("big_win_100bps", "mean"),
                big_win_300bps_rate=("big_win_300bps", "mean"),
                big_win_500bps_rate=("big_win_500bps", "mean"),
            )
            .reset_index()
        )
        summary["groups"] = summary["rows"] / width
        monthly = (
            frame.groupby(group_cols + ["month"], observed=True)
            .agg(
                mean_excess_bps=("excess_bps", "mean"),
                median_excess_bps=("excess_bps", "median"),
                positive_excess_rate=("positive_excess", "mean"),
                realized_pool_top10_rate=("realized_pool_top10", "mean"),
                big_win_300bps_rate=("big_win_300bps", "mean"),
            )
            .reset_index()
        )
        monthly_positive = summarize_monthly_positive(
            frame,
            group_cols,
            "excess_bps",
        )
        summary = summary.merge(monthly_positive, on=group_cols, how="left")
        summaries.append(summary)
        month_summaries.append(monthly)
    return pd.concat(summaries, ignore_index=True), pd.concat(month_summaries, ignore_index=True)


def build_shape_summary(top_rows: pd.DataFrame, *, variant: str, top_k: list[int]) -> pd.DataFrame:
    rows = []
    top_rows = top_rows.copy()
    top_rows["positive_excess"] = top_rows["excess_bps"] > 0
    for k in top_k:
        frame = top_rows.loc[top_rows["score_rank"] <= k]
        rows.append(
            {
                "variant": variant,
                "top_k": k,
                "rows": len(frame),
                "groups": len(frame) / k,
                "mean_excess_bps": frame["excess_bps"].mean(),
                "median_excess_bps": frame["excess_bps"].median(),
                "positive_excess_rate": frame["positive_excess"].mean(),
                "realized_pool_top10_rate": frame["realized_pool_top10"].mean(),
                "realized_pool_top5_rate": frame["realized_pool_top5"].mean(),
            }
        )
    return pd.DataFrame(rows)


def add_ic_rows(
    top: pd.DataFrame,
    *,
    variant: str,
    top_k: list[int],
    bucket_widths: list[int],
    window_widths: list[int],
    window_stride: int,
    top_n: int,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    topk_rows: list[tuple] = []
    bucket_ic_rows: list[tuple] = []
    window_rows: list[tuple] = []
    for _, group in top.groupby(GROUP_COLS, sort=False, observed=True):
        y = group["excess_bps"].to_numpy(dtype="float64", copy=False)
        month = group["month"].iloc[0]
        n = len(y)
        for k in top_k:
            if n >= k:
                topk_rows.append((variant, month, k, fixed_score_spearman(y[:k])))
        for width in bucket_widths:
            max_bucket = min(top_n, n) // width
            for bucket in range(1, max_bucket + 1):
                start = (bucket - 1) * width
                end = bucket * width
                bucket_ic_rows.append(
                    (variant, month, width, bucket, start + 1, end, fixed_score_spearman(y[start:end]))
                )
        for width in window_widths:
            if n < width:
                continue
            max_start = min(top_n, n) - width
            for start in range(0, max_start + 1, window_stride):
                end = start + width
                window_rows.append(
                    (variant, month, width, start + 1, end, fixed_score_spearman(y[start:end]))
                )
    return topk_rows, bucket_ic_rows, window_rows


def summarize_ic(rows: list[tuple], columns: list[str], key_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(key_cols, observed=True)
        .agg(
            groups=("spearman_ic", "count"),
            mean_spearman_ic=("spearman_ic", "mean"),
            median_spearman_ic=("spearman_ic", "median"),
            p10_spearman_ic=("spearman_ic", lambda s: float(s.quantile(0.10))),
            p90_spearman_ic=("spearman_ic", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
    )
    month_positive = summarize_monthly_positive(frame, key_cols, "spearman_ic")
    return summary.merge(month_positive, on=key_cols, how="left")


def load_label_for_year(path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(path, columns=LABEL_COLS)
    labels["date"] = normalize_date(labels["date"])
    return labels


def process_shard(
    *,
    pred_path: Path,
    labels: pd.DataFrame,
    pool: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pred = pd.read_parquet(pred_path, columns=PREDICTION_COLS)
    pred["date"] = normalize_date(pred["date"])
    pred = pred.loc[stock_pool_membership_mask(pred, pool)].copy()
    pred_rows = len(pred)
    frame = pred.merge(labels, on=PREDICTION_COLS[:3], how="inner", validate="one_to_one")
    joined_rows = len(frame)
    frame["pool_mean"] = frame.groupby(GROUP_COLS, observed=True)[
        "alpha_return_next_close"
    ].transform("mean")
    frame["excess_bps"] = (frame["alpha_return_next_close"] - frame["pool_mean"]) * 10000.0
    frame["realized_pool_pct"] = frame.groupby(GROUP_COLS, observed=True)[
        "alpha_return_next_close"
    ].rank(ascending=False, method="average", pct=True)
    frame["realized_pool_top5"] = frame["realized_pool_pct"] <= 0.05
    frame["realized_pool_top10"] = frame["realized_pool_pct"] <= 0.10
    frame = frame.sort_values(
        GROUP_COLS + ["prediction"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    frame["score_rank"] = frame.groupby(GROUP_COLS, observed=True).cumcount() + 1
    top = frame.loc[frame["score_rank"] <= top_n].copy()
    top["month"] = top["date"].str.slice(0, 7)
    top["top_n"] = top_n
    return top[
        GROUP_COLS
        + [
            "month",
            "score_rank",
            "excess_bps",
            "realized_pool_top5",
            "realized_pool_top10",
            "top_n",
        ]
    ], {
        "prediction_pool_rows": pred_rows,
        "joined_rows": joined_rows,
        "top_rows": len(top),
    }


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root)
    next_label_root = Path(args.next_label_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    months = [part.strip() for part in args.months.split(",") if part.strip()]
    variants = args.variant or list(DEFAULT_VARIANTS)
    bucket_widths = parse_csv_ints(args.bucket_widths, default=(50, 100, 200))
    top_k = parse_csv_ints(args.top_k, default=(50, 100, 150, 200, 500, 1000))
    window_widths = parse_csv_ints(args.window_widths, default=(50, 100, 200))

    print("loading stock pool", args.pool_path, flush=True)
    pool = load_stock_pool(args.pool_path)
    print(f"pool shape={pool.shape}", flush=True)

    all_bucket_summary: list[pd.DataFrame] = []
    all_bucket_month: list[pd.DataFrame] = []
    all_shape_summary: list[pd.DataFrame] = []
    all_topk_ic_rows: list[tuple] = []
    all_bucket_ic_rows: list[tuple] = []
    all_window_ic_rows: list[tuple] = []
    label_cache: dict[str, pd.DataFrame] = {}
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root),
        "pool_path": args.pool_path,
        "variants": {},
        "months": months,
        "bucket_widths": bucket_widths,
        "top_k": top_k,
        "window_widths": window_widths,
        "window_stride": args.window_stride,
        "top_n": args.top_n,
    }

    for variant in variants:
        run_id = DEFAULT_VARIANTS[variant]
        print(f"processing {variant} ({run_id})", flush=True)
        top_parts: list[pd.DataFrame] = []
        variant_trace = {"run_id": run_id, "months": {}}
        for month in months:
            year = month.split("-", 1)[0]
            if year not in label_cache:
                print(f"  loading labels {year}", flush=True)
                label_cache.clear()
                label_cache[year] = load_label_for_year(next_label_path(next_label_root, year))
            pred_path = prediction_path(prediction_root, run_id, month)
            print(f"  shard {month}: {pred_path}", flush=True)
            top, shard_trace = process_shard(
                pred_path=pred_path,
                labels=label_cache[year],
                pool=pool,
                top_n=args.top_n,
            )
            shard_topk, shard_bucket_ic, shard_window = add_ic_rows(
                top,
                variant=variant,
                top_k=top_k,
                bucket_widths=bucket_widths,
                window_widths=window_widths,
                window_stride=args.window_stride,
                top_n=args.top_n,
            )
            all_topk_ic_rows.extend(shard_topk)
            all_bucket_ic_rows.extend(shard_bucket_ic)
            all_window_ic_rows.extend(shard_window)
            top_parts.append(top)
            variant_trace["months"][month] = shard_trace

        top_rows = pd.concat(top_parts, ignore_index=True)
        bucket_summary, bucket_month = add_bucket_summaries(
            top_rows,
            variant=variant,
            bucket_widths=bucket_widths,
        )
        shape_summary = build_shape_summary(top_rows, variant=variant, top_k=top_k)
        all_bucket_summary.append(bucket_summary)
        all_bucket_month.append(bucket_month)
        all_shape_summary.append(shape_summary)
        trace["variants"][variant] = variant_trace
        del top_parts, top_rows

    bucket_summary = pd.concat(all_bucket_summary, ignore_index=True)
    bucket_month = pd.concat(all_bucket_month, ignore_index=True)
    shape_summary = pd.concat(all_shape_summary, ignore_index=True)
    topk_ic = summarize_ic(
        all_topk_ic_rows,
        ["variant", "month", "top_k", "spearman_ic"],
        ["variant", "top_k"],
    )
    bucket_ic = summarize_ic(
        all_bucket_ic_rows,
        ["variant", "month", "bucket_width", "bucket", "start_rank", "end_rank", "spearman_ic"],
        ["variant", "bucket_width", "bucket", "start_rank", "end_rank"],
    )
    bucket_ic["rank_slice"] = bucket_ic["start_rank"].astype(str) + "-" + bucket_ic[
        "end_rank"
    ].astype(str)
    window_ic = summarize_ic(
        all_window_ic_rows,
        ["variant", "month", "window_width", "start_rank", "end_rank", "spearman_ic"],
        ["variant", "window_width", "start_rank", "end_rank"],
    )
    window_ic["window"] = window_ic["start_rank"].astype(str) + "-" + window_ic[
        "end_rank"
    ].astype(str)

    bucket_summary = pd.concat(
        [
            bucket_summary,
            summarize_variant_average(
                bucket_summary,
                ["bucket_width", "bucket"],
            ),
        ],
        ignore_index=True,
    )
    shape_summary = pd.concat(
        [shape_summary, summarize_variant_average(shape_summary, ["top_k"])],
        ignore_index=True,
    )
    topk_ic = pd.concat(
        [topk_ic, summarize_variant_average(topk_ic, ["top_k"])],
        ignore_index=True,
    )
    bucket_ic = pd.concat(
        [
            bucket_ic,
            summarize_variant_average(
                bucket_ic,
                ["bucket_width", "bucket", "start_rank", "end_rank"],
            ),
        ],
        ignore_index=True,
    )
    window_ic = pd.concat(
        [
            window_ic,
            summarize_variant_average(
                window_ic,
                ["window_width", "start_rank", "end_rank"],
            ),
        ],
        ignore_index=True,
    )

    write_frame(output_dir / "bucket_width_distribution_summary.csv", bucket_summary)
    write_frame(output_dir / "bucket_width_distribution_month_summary.csv", bucket_month)
    write_frame(output_dir / "topk_shape_summary.csv", shape_summary)
    write_frame(output_dir / "topk_internal_ic_summary.csv", topk_ic)
    write_frame(output_dir / "bucket_width_within_ic_summary.csv", bucket_ic)
    write_frame(output_dir / "local_ic_window_summary.csv", window_ic)
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
