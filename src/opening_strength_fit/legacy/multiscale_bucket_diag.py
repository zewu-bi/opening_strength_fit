from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.model_metrics import array_corr
from opening_strength_fit.stock_pool import load_stock_pool

CANONICAL_WEIGHTING = "decision_group_equal"
GROUP_COLS = ["date", "decision_target_timestamp"]
PREDICTION_COLS = ["date", "symbol", "decision_target_timestamp", "prediction"]
LABEL_COLS = ["date", "symbol", "decision_target_timestamp", "alpha_return_next_close"]


@dataclass(frozen=True)
class MultiscaleBucketDiagConfig:
    prediction_root: Path
    next_label_root: Path
    pool_path: str
    output_dir: Path
    run_ids: dict[str, str]
    months: list[str]
    bucket_widths: list[int]
    top_k: list[int]
    window_widths: list[int]
    window_stride: int = 50
    top_n: int = 1000


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


def normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def load_label_for_year(path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(path, columns=LABEL_COLS)
    labels["date"] = normalize_date(labels["date"])
    return labels


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


def spearman_rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    return array_corr(scores, outcomes, "spearman")


def fixed_score_spearman(outcomes: np.ndarray) -> float:
    return spearman_rank_ic(np.arange(len(outcomes), 0, -1, dtype="float64"), outcomes)


def summarize_monthly_stability(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    monthly = (
        frame.groupby(group_cols + ["month"], observed=True)[value_col]
        .mean()
        .reset_index(name=f"{value_col}_month_mean")
    )
    return (
        monthly.groupby(group_cols, observed=True)
        .agg(
            months=("month", "nunique"),
            positive_months=(f"{value_col}_month_mean", lambda s: int((s > 0).sum())),
            month_min=(f"{value_col}_month_mean", "min"),
            month_max=(f"{value_col}_month_mean", "max"),
        )
        .reset_index()
    )


def summarize_variant_average(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    excluded = set(key_cols + ["variant", "rank_slice", "window"])
    metric_cols = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not metric_cols:
        return pd.DataFrame()
    out = frame.groupby(key_cols, observed=True)[metric_cols].mean().reset_index()
    out.insert(0, "variant", "old3_mean")
    for label_col in ("rank_slice", "window"):
        if label_col in frame.columns:
            labels = frame.drop_duplicates(key_cols)[key_cols + [label_col]]
            out = out.merge(labels, on=key_cols, how="left")
    return out


def with_variant_average(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    return pd.concat(
        [frame, summarize_variant_average(frame, key_cols)],
        ignore_index=True,
    )


def add_bucket_summaries(
    top_rows: pd.DataFrame,
    *,
    variant: str,
    bucket_widths: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    month_summaries: list[pd.DataFrame] = []
    for width in bucket_widths:
        max_rank = (int(top_rows["top_n"].iloc[0]) // width) * width
        frame = top_rows.loc[top_rows["score_rank"] <= max_rank].copy()
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
        frame["variant"] = variant
        frame["bucket_width"] = width

        group_cols = ["variant", "bucket_width", "bucket", "rank_slice"]
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
        stability = summarize_monthly_stability(frame, group_cols, "excess_bps")
        summaries.append(summary.merge(stability, on=group_cols, how="left"))

        month_summaries.append(
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
        scores = group["prediction"].to_numpy(dtype="float64", copy=False)
        y = group["excess_bps"].to_numpy(dtype="float64", copy=False)
        month = group["month"].iloc[0]
        n = len(y)
        for k in top_k:
            if n >= k:
                topk_rows.append((variant, month, k, spearman_rank_ic(scores[:k], y[:k])))
        for width in bucket_widths:
            max_bucket = min(top_n, n) // width
            for bucket in range(1, max_bucket + 1):
                start = (bucket - 1) * width
                end = bucket * width
                bucket_ic_rows.append(
                    (
                        variant,
                        month,
                        width,
                        bucket,
                        start + 1,
                        end,
                        spearman_rank_ic(scores[start:end], y[start:end]),
                    )
                )
        for width in window_widths:
            if n < width:
                continue
            max_start = min(top_n, n) - width
            for start in range(0, max_start + 1, window_stride):
                end = start + width
                window_rows.append(
                    (
                        variant,
                        month,
                        width,
                        start + 1,
                        end,
                        spearman_rank_ic(scores[start:end], y[start:end]),
                    )
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
    stability = summarize_monthly_stability(frame, key_cols, "spearman_ic")
    return summary.merge(stability, on=key_cols, how="left")


def load_ranked_pool_shard(
    *,
    pred_path: Path,
    labels: pd.DataFrame | None,
    pool: pd.DataFrame,
    prediction_next_label_col: str = "",
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    columns = [*PREDICTION_COLS, prediction_next_label_col] if prediction_next_label_col else None
    pred = pd.read_parquet(pred_path, columns=columns or PREDICTION_COLS)
    if prediction_next_label_col:
        pred = pred.rename(columns={prediction_next_label_col: "alpha_return_next_close"})
    pred["date"] = normalize_date(pred["date"])
    pred = pred.loc[stock_pool_membership_mask(pred, pool)].copy()
    pred_rows = len(pred)
    duplicate_keys = int(pred.duplicated(PREDICTION_COLS[:3]).sum())
    if duplicate_keys:
        raise ValueError(f"invalid predictions for {pred_path}: duplicates={duplicate_keys}")
    if prediction_next_label_col:
        frame = pred
    else:
        if labels is None:
            raise ValueError("separate next-close labels are required")
        frame = pred.merge(labels, on=PREDICTION_COLS[:3], how="left", validate="one_to_one")
    missing_labels = int(frame["alpha_return_next_close"].isna().sum())
    if missing_labels:
        source = "embedded labels" if prediction_next_label_col else "prediction/label join"
        raise ValueError(f"invalid {source} for {pred_path}: missing_labels={missing_labels}")
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
    grouped = frame.groupby(GROUP_COLS, sort=False, observed=True)
    frame["score_rank"] = grouped.cumcount() + 1
    frame["group_size"] = grouped["symbol"].transform("size")
    frame["month"] = frame["date"].str.slice(0, 7)
    trace: dict[str, int | str] = {
        "prediction_pool_rows": pred_rows,
        "joined_rows": len(frame),
        "groups": int(grouped.ngroups),
        "duplicate_keys": duplicate_keys,
        "missing_labels": missing_labels,
    }
    if prediction_next_label_col:
        trace["next_label_source"] = f"prediction:{prediction_next_label_col}"
    return frame, trace


def ranked_pool_shards(
    prediction_root: Path,
    next_label_root: Path | None,
    prediction_next_label_col: str,
    pool_path: str,
    run_id: str,
    months: list[str],
    top_n: int,
    message: str,
    *,
    pool: pd.DataFrame | None = None,
    require_complete_groups: bool = True,
) -> Iterator[tuple[str, pd.DataFrame, dict[str, int | str]]]:
    pool = load_stock_pool(pool_path) if pool is None else pool
    labels: dict[str, pd.DataFrame] = {}
    if not prediction_next_label_col and next_label_root is None:
        raise ValueError("next_label_root is required without embedded prediction labels")
    for month in months:
        year = month[:4]
        if not prediction_next_label_col and year not in labels:
            assert next_label_root is not None
            labels = {year: load_label_for_year(next_label_path(next_label_root, year))}
        pred_path = prediction_path(prediction_root, run_id, month)
        print(f"{message} shard {month}: {pred_path}", flush=True)
        frame, trace = load_ranked_pool_shard(
            pred_path=pred_path,
            labels=labels.get(year),
            pool=pool,
            prediction_next_label_col=prediction_next_label_col,
        )
        if require_complete_groups:
            group_sizes = frame.groupby(GROUP_COLS, observed=True)["group_size"].first()
            complete_groups = group_sizes.loc[group_sizes.ge(top_n)]
            trace.update(
                {
                    "pool_group_size_min": int(group_sizes.min()),
                    "pool_groups_total": int(len(group_sizes)),
                    "pool_groups_with_top_n": int(len(complete_groups)),
                    "pool_groups_below_top_n": int(len(group_sizes) - len(complete_groups)),
                }
            )
            if complete_groups.empty:
                raise ValueError(f"no pool group has at least Top{top_n} rows")
            frame = frame.loc[frame["group_size"].ge(top_n)].copy()
        yield month, frame, trace


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_trace(
    prediction_root: Path,
    next_label_root: Path | None,
    pool_path: str,
    *,
    prediction_next_label_col: str | None = None,
    **details: object,
) -> dict[str, object]:
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root) if next_label_root is not None else None,
    }
    if prediction_next_label_col is not None:
        trace["prediction_next_label_col"] = prediction_next_label_col or None
    trace.update(pool_path=pool_path, **details)
    return trace


def run_multiscale_bucket_diagnostics(config: MultiscaleBucketDiagConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading stock pool", config.pool_path, flush=True)
    pool = load_stock_pool(config.pool_path)
    print(f"pool shape={pool.shape}", flush=True)

    all_bucket_summary: list[pd.DataFrame] = []
    all_bucket_month: list[pd.DataFrame] = []
    all_shape_summary: list[pd.DataFrame] = []
    all_topk_ic_rows: list[tuple] = []
    all_bucket_ic_rows: list[tuple] = []
    all_window_ic_rows: list[tuple] = []
    trace = run_trace(
        config.prediction_root,
        config.next_label_root,
        config.pool_path,
        canonical_weighting=CANONICAL_WEIGHTING,
        variants={},
        months=config.months,
        bucket_widths=config.bucket_widths,
        top_k=config.top_k,
        window_widths=config.window_widths,
        window_stride=config.window_stride,
        top_n=config.top_n,
        rank_ic_method="spearman_average_rank_ties",
    )

    for variant, run_id in config.run_ids.items():
        print(f"processing {variant} ({run_id})", flush=True)
        top_parts: list[pd.DataFrame] = []
        variant_trace = {"run_id": run_id, "months": {}}
        for month, frame, shard_trace in ranked_pool_shards(
            config.prediction_root,
            config.next_label_root,
            "",
            config.pool_path,
            run_id,
            config.months,
            config.top_n,
            " ",
            pool=pool,
            require_complete_groups=False,
        ):
            top = frame.loc[frame["score_rank"] <= config.top_n].copy()
            top["top_n"] = config.top_n
            shard_trace["top_rows"] = len(top)
            topk_rows, bucket_ic_rows, window_rows = add_ic_rows(
                top,
                variant=variant,
                top_k=config.top_k,
                bucket_widths=config.bucket_widths,
                window_widths=config.window_widths,
                window_stride=config.window_stride,
                top_n=config.top_n,
            )
            all_topk_ic_rows.extend(topk_rows)
            all_bucket_ic_rows.extend(bucket_ic_rows)
            all_window_ic_rows.extend(window_rows)
            top_parts.append(top)
            variant_trace["months"][month] = shard_trace

        top_rows = pd.concat(top_parts, ignore_index=True)
        bucket_summary, bucket_month = add_bucket_summaries(
            top_rows,
            variant=variant,
            bucket_widths=config.bucket_widths,
        )
        all_bucket_summary.append(bucket_summary)
        all_bucket_month.append(bucket_month)
        all_shape_summary.append(build_shape_summary(top_rows, variant=variant, top_k=config.top_k))
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
    bucket_ic["rank_slice"] = (
        bucket_ic["start_rank"].astype(str) + "-" + bucket_ic["end_rank"].astype(str)
    )
    window_ic = summarize_ic(
        all_window_ic_rows,
        ["variant", "month", "window_width", "start_rank", "end_rank", "spearman_ic"],
        ["variant", "window_width", "start_rank", "end_rank"],
    )
    window_ic["window"] = (
        window_ic["start_rank"].astype(str) + "-" + window_ic["end_rank"].astype(str)
    )

    bucket_summary = with_variant_average(
        bucket_summary,
        ["bucket_width", "bucket"],
    )
    shape_summary = with_variant_average(shape_summary, ["top_k"])
    topk_ic = with_variant_average(topk_ic, ["top_k"])
    bucket_ic = with_variant_average(
        bucket_ic,
        ["bucket_width", "bucket", "start_rank", "end_rank"],
    )
    window_ic = with_variant_average(
        window_ic,
        ["window_width", "start_rank", "end_rank"],
    )

    write_frame(config.output_dir / "bucket_width_distribution_summary.csv", bucket_summary)
    write_frame(config.output_dir / "bucket_width_distribution_month_summary.csv", bucket_month)
    write_frame(config.output_dir / "topk_shape_summary.csv", shape_summary)
    write_frame(config.output_dir / "topk_internal_ic_summary.csv", topk_ic)
    write_frame(config.output_dir / "bucket_width_within_ic_summary.csv", bucket_ic)
    write_frame(config.output_dir / "local_ic_window_summary.csv", window_ic)
    (config.output_dir / "trace.json").write_text(
        json.dumps(trace, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {config.output_dir}", flush=True)
