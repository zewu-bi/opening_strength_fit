from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.legacy import multiscale_bucket_diag as ms
from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
    TOP1000_RETURN_HISTOGRAM_Y_LIMITS,
    TOP1000_SCORE_BUCKETS,
    load_ranked_pool_shard,
)
from opening_strength_fit.stock_pool import load_stock_pool


def positive_return_histogram(values: pd.Series, *, bin_width_bps: int = 10) -> pd.DataFrame:
    positive = values.loc[values >= 0].astype("float64")
    lower = (np.floor(positive / bin_width_bps) * bin_width_bps).astype("int64")
    max_lower = int(lower.max())
    counts = lower.value_counts().reindex(
        np.arange(0, max_lower + bin_width_bps, bin_width_bps), fill_value=0
    )
    histogram = counts.rename_axis("lower_bps").reset_index(name="observations")
    histogram["upper_bps"] = histogram["lower_bps"] + bin_width_bps
    histogram["interval"] = (
        histogram["lower_bps"].astype(str) + "-" + histogram["upper_bps"].astype(str)
    )
    histogram["share_of_all_top100"] = histogram["observations"] / len(values)
    histogram["share_of_positive_top100"] = histogram["observations"] / len(positive)
    return histogram


def plot_positive_return_histogram(
    histogram: pd.DataFrame,
    *,
    total_observations: int,
    output_dir: Path,
    zoom_max_bps: int = 500,
) -> None:
    positive_observations = int(histogram["observations"].sum())
    zoom = histogram.loc[histogram["lower_bps"] < zoom_max_bps]
    tail_observations = int(
        histogram.loc[histogram["lower_bps"] >= zoom_max_bps, "observations"].sum()
    )
    nonzero = histogram.loc[histogram["observations"] > 0]

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.25, 1]})
    axes[0].plot(
        zoom["lower_bps"] + 5,
        zoom["observations"],
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=2,
    )
    axes[0].set_title(
        "auction_multiden Top100 positive excess-return counts",
        loc="left",
        fontweight="bold",
    )
    axes[0].set_ylabel("Stock-decision observations")
    axes[0].set_xlabel("Excess-return interval midpoint (bps; 10 bps per interval)")
    axes[0].set_xlim(0, zoom_max_bps)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        0.99,
        0.95,
        f">={zoom_max_bps} bps: {tail_observations:,} observations",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )

    axes[1].plot(
        nonzero["lower_bps"] + 5,
        nonzero["observations"],
        color="#d95f02",
        linewidth=1.5,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Observations (log scale)")
    axes[1].set_xlabel("Positive excess return (bps; full range, 10 bps per interval)")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.99,
        0.95,
        (
            f"positive: {positive_observations:,} / {total_observations:,} "
            f"({positive_observations / total_observations:.2%})"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"top100_positive_return_10bps_counts.{extension}", dpi=140)
    plt.close(fig)


def run_top100_positive_histogram(
    *,
    prediction_root: Path,
    next_label_root: Path,
    pool_path: str,
    output_dir: Path,
    variant: str,
    run_id: str,
    months: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(pool_path)
    labels: dict[str, pd.DataFrame] = {}
    value_parts: list[pd.Series] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root),
        "pool_path": pool_path,
        "variant": variant,
        "run_id": run_id,
        "months": {},
        "scope": "Top100 within pool_L",
        "value": "next internal excess bps",
        "bin_width_bps": 10,
        "observation_unit": "stock x decision cross-section",
    }
    for month in months:
        year = month[:4]
        if year not in labels:
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"Top100 histogram shard {month}: {pred_path}", flush=True)
        top, shard_trace = ms.process_shard(
            pred_path=pred_path,
            labels=labels[year],
            pool=pool,
            top_n=100,
        )
        value_parts.append(top["excess_bps"])
        trace["months"][month] = shard_trace

    values = pd.concat(value_parts, ignore_index=True).astype("float64")
    histogram = positive_return_histogram(values)
    positive = values.loc[values >= 0]
    summary = {
        "total_observations": len(values),
        "negative_observations": int((values < 0).sum()),
        "positive_or_zero_observations": len(positive),
        "positive_or_zero_rate": float((values >= 0).mean()),
        "mean_excess_bps": float(values.mean()),
        "median_excess_bps": float(values.median()),
        "positive_p50_bps": float(positive.quantile(0.50)),
        "positive_p90_bps": float(positive.quantile(0.90)),
        "positive_p95_bps": float(positive.quantile(0.95)),
        "positive_p99_bps": float(positive.quantile(0.99)),
        "maximum_excess_bps": float(values.max()),
    }
    ms.write_frame(output_dir / "top100_positive_return_10bps_counts.csv", histogram)
    plot_positive_return_histogram(
        histogram,
        total_observations=len(values),
        output_dir=output_dir,
    )
    trace["summary"] = summary
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


def full_return_histogram(values: pd.Series, *, bin_width_bps: int = 100) -> pd.DataFrame:
    if bin_width_bps <= 0:
        raise ValueError("bin_width_bps must be positive")
    values = values.astype("float64")
    lower = (np.floor(values / bin_width_bps) * bin_width_bps).astype("int64")
    min_lower = int(lower.min())
    max_lower = int(lower.max())
    counts = lower.value_counts().reindex(
        np.arange(min_lower, max_lower + bin_width_bps, bin_width_bps), fill_value=0
    )
    histogram = counts.rename_axis("lower_bps").reset_index(name="observations")
    histogram["upper_bps"] = histogram["lower_bps"] + bin_width_bps
    histogram["midpoint_bps"] = histogram["lower_bps"] + bin_width_bps / 2.0
    histogram["interval"] = (
        "[" + histogram["lower_bps"].astype(str) + ", " + histogram["upper_bps"].astype(str) + ")"
    )
    histogram["share_of_top100"] = histogram["observations"] / len(values)
    return histogram


def plot_full_return_histogram(
    histogram: pd.DataFrame,
    *,
    bin_width_bps: int,
    total_observations: int,
    output_dir: Path,
    zoom_abs_bps: int = 1000,
) -> None:
    zoom = histogram.loc[histogram["midpoint_bps"].between(-zoom_abs_bps, zoom_abs_bps)]
    negative = zoom.loc[zoom["upper_bps"] <= 0]
    positive = zoom.loc[zoom["lower_bps"] >= 0]
    nonzero = histogram.loc[histogram["observations"] > 0]

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.25, 1]})
    axes[0].plot(
        negative["midpoint_bps"],
        negative["observations"],
        color="#c44e52",
        marker="o",
        linewidth=2.2,
        label="Negative excess",
    )
    axes[0].plot(
        positive["midpoint_bps"],
        positive["observations"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        label="Non-negative excess",
    )
    axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_title(
        f"auction_multiden Top100 excess-return counts ({bin_width_bps} bps intervals)",
        loc="left",
        fontweight="bold",
    )
    axes[0].set_ylabel("Stock-decision observations")
    axes[0].set_xlabel("Next internal excess return (bps)")
    axes[0].set_xlim(-zoom_abs_bps, zoom_abs_bps)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        nonzero["midpoint_bps"],
        nonzero["observations"],
        color="#555555",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    axes[1].axvline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Observations (log scale)")
    axes[1].set_xlabel("Full excess-return range (bps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.99,
        0.95,
        f"total: {total_observations:,} stock-decision observations",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"top100_return_{bin_width_bps}bps_counts.{extension}", dpi=140)
    plt.close(fig)


def run_top100_return_histogram(
    *,
    prediction_root: Path,
    next_label_root: Path,
    pool_path: str,
    output_dir: Path,
    variant: str,
    run_id: str,
    months: list[str],
    bin_width_bps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(pool_path)
    labels: dict[str, pd.DataFrame] = {}
    value_parts: list[pd.Series] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root),
        "pool_path": pool_path,
        "variant": variant,
        "run_id": run_id,
        "months": {},
        "scope": "Top100 within pool_L",
        "value": "next internal excess bps",
        "bin_width_bps": bin_width_bps,
        "observation_unit": "stock x decision cross-section",
    }
    for month in months:
        year = month[:4]
        if year not in labels:
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"Top100 full histogram shard {month}: {pred_path}", flush=True)
        top, shard_trace = ms.process_shard(
            pred_path=pred_path,
            labels=labels[year],
            pool=pool,
            top_n=100,
        )
        value_parts.append(top["excess_bps"])
        trace["months"][month] = shard_trace

    values = pd.concat(value_parts, ignore_index=True).astype("float64")
    histogram = full_return_histogram(values, bin_width_bps=bin_width_bps)
    summary = {
        "total_observations": len(values),
        "negative_observations": int((values < 0).sum()),
        "negative_rate": float((values < 0).mean()),
        "positive_or_zero_observations": int((values >= 0).sum()),
        "positive_or_zero_rate": float((values >= 0).mean()),
        "mean_excess_bps": float(values.mean()),
        "median_excess_bps": float(values.median()),
        "p01_excess_bps": float(values.quantile(0.01)),
        "p05_excess_bps": float(values.quantile(0.05)),
        "p10_excess_bps": float(values.quantile(0.10)),
        "p90_excess_bps": float(values.quantile(0.90)),
        "p95_excess_bps": float(values.quantile(0.95)),
        "p99_excess_bps": float(values.quantile(0.99)),
        "minimum_excess_bps": float(values.min()),
        "maximum_excess_bps": float(values.max()),
    }
    ms.write_frame(output_dir / f"top100_return_{bin_width_bps}bps_counts.csv", histogram)
    plot_full_return_histogram(
        histogram,
        bin_width_bps=bin_width_bps,
        total_observations=len(values),
        output_dir=output_dir,
    )
    trace["summary"] = summary
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


def score_bucket_histogram(frame: pd.DataFrame, *, bin_width_bps: int = 100) -> pd.DataFrame:
    work = frame[["score_bucket", "excess_bps"]].copy()
    work["lower_bps"] = (np.floor(work["excess_bps"] / bin_width_bps) * bin_width_bps).astype(
        "int64"
    )
    min_lower = int(work["lower_bps"].min())
    max_lower = int(work["lower_bps"].max())
    buckets = np.arange(1, 11, dtype="int64")
    lower_edges = np.arange(min_lower, max_lower + bin_width_bps, bin_width_bps, dtype="int64")
    full_index = pd.MultiIndex.from_product(
        [buckets, lower_edges], names=["score_bucket", "lower_bps"]
    )
    histogram = (
        work.groupby(["score_bucket", "lower_bps"], observed=True)
        .size()
        .reindex(full_index, fill_value=0)
        .rename("observations")
        .reset_index()
    )
    histogram["rank_start"] = (histogram["score_bucket"] - 1) * 100 + 1
    histogram["rank_end"] = histogram["score_bucket"] * 100
    histogram["upper_bps"] = histogram["lower_bps"] + bin_width_bps
    histogram["midpoint_bps"] = histogram["lower_bps"] + bin_width_bps / 2.0
    histogram["interval"] = (
        "[" + histogram["lower_bps"].astype(str) + ", " + histogram["upper_bps"].astype(str) + ")"
    )
    totals = histogram.groupby("score_bucket", observed=True)["observations"].transform("sum")
    histogram["share_within_score_bucket"] = histogram["observations"] / totals
    return histogram


def plot_score_bucket_histograms(
    histogram: pd.DataFrame,
    *,
    bin_width_bps: int,
    output_dir: Path,
    variant: str,
) -> None:
    _plot_score_bucket_histograms(
        histogram,
        bin_width_bps=bin_width_bps,
        output_dir=output_dir,
        variant=variant,
        output_stem=f"top1000_score_bucket_return_{bin_width_bps}bps_counts",
        title_suffix="",
        x_limits=(
            -TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
            TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
        ),
        y_limits=TOP1000_RETURN_HISTOGRAM_Y_LIMITS,
    )


def plot_score_bucket_histograms_full_scale(
    histogram: pd.DataFrame,
    *,
    bin_width_bps: int,
    output_dir: Path,
    variant: str,
) -> None:
    nonzero = histogram.loc[histogram["observations"].gt(0)]
    if nonzero.empty:
        raise ValueError("histogram must contain at least one non-zero observation")
    x_extent = float(nonzero["midpoint_bps"].abs().max()) + float(bin_width_bps) / 2.0
    y_max = float(nonzero["observations"].max())
    _plot_score_bucket_histograms(
        histogram,
        bin_width_bps=bin_width_bps,
        output_dir=output_dir,
        variant=variant,
        output_stem=(f"top1000_score_bucket_return_{bin_width_bps}bps_counts_full_scale"),
        title_suffix=", full scale",
        x_limits=(-x_extent, x_extent),
        y_limits=(0.8, y_max * 1.25),
    )


def _plot_score_bucket_histograms(
    histogram: pd.DataFrame,
    *,
    bin_width_bps: int,
    output_dir: Path,
    variant: str,
    output_stem: str,
    title_suffix: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    required_columns = {"score_bucket", "midpoint_bps", "observations"}
    missing_columns = required_columns.difference(histogram.columns)
    if missing_columns:
        raise ValueError(f"histogram is missing columns: {sorted(missing_columns)}")
    buckets = tuple(sorted(histogram["score_bucket"].dropna().astype(int).unique()))
    if buckets != TOP1000_SCORE_BUCKETS:
        raise ValueError(
            f"histogram must contain score buckets {TOP1000_SCORE_BUCKETS}, got {buckets}"
        )
    if (histogram["observations"] < 0).any():
        raise ValueError("histogram observations must be non-negative")

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.get_cmap("tab10").colors
    for bucket in TOP1000_SCORE_BUCKETS:
        series = histogram.loc[
            histogram["score_bucket"].eq(bucket) & histogram["observations"].gt(0)
        ]
        start = (bucket - 1) * 100 + 1
        end = bucket * 100
        ax.plot(
            series["midpoint_bps"],
            series["observations"],
            color=colors[bucket - 1],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=f"Rank {start}-{end}",
        )
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.65)
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title(
        (f"{variant} Top1000 return distributions ({bin_width_bps} bps intervals{title_suffix})"),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Next internal excess return (bps)")
    ax.set_ylabel("Stock-decision observations (log scale)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(
            output_dir / f"{output_stem}.{extension}",
            dpi=140,
        )
    plt.close(fig)


def run_top1000_bucket_return_histogram(
    *,
    prediction_root: Path,
    next_label_root: Path | None,
    prediction_next_label_col: str,
    pool_path: str,
    output_dir: Path,
    variant: str,
    run_id: str,
    months: list[str],
    bin_width_bps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(pool_path)
    labels: dict[str, pd.DataFrame] = {}
    value_parts: list[pd.DataFrame] = []
    extreme_parts: list[pd.DataFrame] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root) if next_label_root is not None else None,
        "prediction_next_label_col": prediction_next_label_col or None,
        "pool_path": pool_path,
        "variant": variant,
        "run_id": run_id,
        "months": {},
        "scope": "Top1000 within pool_L, split into ten 100-name score buckets",
        "value": "next internal excess bps",
        "bin_width_bps": bin_width_bps,
        "observation_unit": "stock x decision cross-section",
    }
    for month in months:
        year = month[:4]
        if not prediction_next_label_col and year not in labels:
            if next_label_root is None:
                raise ValueError("next_label_root is required without embedded prediction labels")
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"Top1000 bucket histogram shard {month}: {pred_path}", flush=True)
        frame, shard_trace = load_ranked_pool_shard(
            pred_path=pred_path,
            labels=labels.get(year),
            pool=pool,
            prediction_next_label_col=prediction_next_label_col,
        )
        group_sizes = frame.groupby(ms.GROUP_COLS, observed=True)["group_size"].first()
        complete_groups = group_sizes.loc[group_sizes.ge(1000)]
        shard_trace["pool_group_size_min"] = int(group_sizes.min())
        shard_trace["pool_groups_total"] = int(len(group_sizes))
        shard_trace["pool_groups_with_top_n"] = int(len(complete_groups))
        shard_trace["pool_groups_below_top_n"] = int(len(group_sizes) - len(complete_groups))
        if complete_groups.empty:
            raise ValueError("no pool group has at least Top1000 rows")
        if len(complete_groups) != len(group_sizes):
            complete_index = pd.MultiIndex.from_frame(
                complete_groups.index.to_frame(index=False)[ms.GROUP_COLS]
            )
            frame_index = pd.MultiIndex.from_frame(frame[ms.GROUP_COLS])
            frame = frame.loc[frame_index.isin(complete_index)].copy()
        top = frame.loc[frame["score_rank"] <= 1000].copy()
        top["score_bucket"] = ((top["score_rank"] - 1) // 100 + 1).astype(int)
        value_parts.append(top[["score_bucket", "excess_bps"]])
        extreme = top.loc[
            top["excess_bps"].abs() >= 4000,
            [
                "date",
                "symbol",
                "decision_target_timestamp",
                "score_rank",
                "score_bucket",
                "prediction",
                "alpha_return_next_close",
                "pool_mean",
                "excess_bps",
            ],
        ].copy()
        extreme["raw_next_close_bps"] = extreme["alpha_return_next_close"] * 10000.0
        extreme["pool_mean_bps"] = extreme["pool_mean"] * 10000.0
        extreme_parts.append(extreme)
        shard_trace["top_rows"] = len(top)
        shard_trace["extreme_abs_4000_rows"] = len(extreme)
        trace["months"][month] = shard_trace

    values = pd.concat(value_parts, ignore_index=True)
    extremes = pd.concat(extreme_parts, ignore_index=True).sort_values(
        "excess_bps", ascending=False
    )
    histogram = score_bucket_histogram(values, bin_width_bps=bin_width_bps)
    summary_rows: list[dict[str, object]] = []
    for bucket, group in values.groupby("score_bucket", observed=True):
        summary_rows.append(
            {
                "score_bucket": bucket,
                "rank_start": (bucket - 1) * 100 + 1,
                "rank_end": bucket * 100,
                "observations": len(group),
                "mean_excess_bps": float(group["excess_bps"].mean()),
                "median_excess_bps": float(group["excess_bps"].median()),
                "positive_rate": float((group["excess_bps"] >= 0).mean()),
                "positive_4000_count": int((group["excess_bps"] >= 4000).sum()),
                "negative_4000_count": int((group["excess_bps"] <= -4000).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    ms.write_frame(
        output_dir / f"top1000_score_bucket_return_{bin_width_bps}bps_counts.csv",
        histogram,
    )
    ms.write_frame(output_dir / "top1000_score_bucket_distribution_summary.csv", summary)
    ms.write_frame(output_dir / "top1000_abs_4000bps_observations.csv", extremes)
    plot_score_bucket_histograms(
        histogram,
        bin_width_bps=bin_width_bps,
        output_dir=output_dir,
        variant=variant,
    )
    plot_score_bucket_histograms_full_scale(
        histogram,
        bin_width_bps=bin_width_bps,
        output_dir=output_dir,
        variant=variant,
    )
    trace["total_top1000_rows"] = len(values)
    trace["total_abs_4000_rows"] = len(extremes)
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)
