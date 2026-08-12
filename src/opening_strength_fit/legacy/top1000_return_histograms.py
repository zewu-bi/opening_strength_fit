from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.legacy import multiscale_bucket_diag as ms
from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
    TOP1000_RETURN_HISTOGRAM_Y_LIMITS,
    TOP1000_SCORE_BUCKETS,
    ranked_pool_shards,
    run_trace,
    save_figure,
)


def _top100_returns(
    prediction_root: Path,
    next_label_root: Path,
    pool_path: str,
    variant: str,
    run_id: str,
    months: list[str],
    bin_width_bps: int,
    message: str,
) -> tuple[pd.Series, dict[str, object]]:
    trace = run_trace(
        prediction_root,
        next_label_root,
        pool_path,
        variant=variant,
        run_id=run_id,
        months={},
        scope="Top100 within pool_L",
        value="next internal excess bps",
        bin_width_bps=bin_width_bps,
        observation_unit="stock x decision cross-section",
    )
    parts: list[pd.Series] = []
    for month, frame, shard_trace in ranked_pool_shards(
        prediction_root,
        next_label_root,
        "",
        pool_path,
        run_id,
        months,
        100,
        message,
        require_complete_groups=False,
    ):
        top = frame.loc[frame["score_rank"] <= 100]
        parts.append(top["excess_bps"])
        shard_trace["top_rows"] = len(top)
        trace["months"][month] = shard_trace
    return pd.concat(parts, ignore_index=True).astype("float64"), trace


def _return_histogram(
    values: pd.Series, *, bin_width_bps: int, positive_only: bool
) -> pd.DataFrame:
    if not positive_only and bin_width_bps <= 0:
        raise ValueError("bin_width_bps must be positive")
    selected = values.loc[values >= 0] if positive_only else values
    selected = selected.astype("float64")
    lower = (np.floor(selected / bin_width_bps) * bin_width_bps).astype("int64")
    min_lower = 0 if positive_only else int(lower.min())
    max_lower = int(lower.max())
    counts = lower.value_counts().reindex(
        np.arange(min_lower, max_lower + bin_width_bps, bin_width_bps), fill_value=0
    )
    histogram = counts.rename_axis("lower_bps").reset_index(name="observations")
    histogram["upper_bps"] = histogram["lower_bps"] + bin_width_bps
    if positive_only:
        histogram["interval"] = (
            histogram["lower_bps"].astype(str) + "-" + histogram["upper_bps"].astype(str)
        )
        histogram["share_of_all_top100"] = histogram["observations"] / len(values)
        histogram["share_of_positive_top100"] = histogram["observations"] / len(selected)
    else:
        histogram["midpoint_bps"] = histogram["lower_bps"] + bin_width_bps / 2.0
        histogram["interval"] = (
            "["
            + histogram["lower_bps"].astype(str)
            + ", "
            + histogram["upper_bps"].astype(str)
            + ")"
        )
        histogram["share_of_top100"] = histogram["observations"] / len(values)
    return histogram


positive_return_histogram = partial(_return_histogram, bin_width_bps=10, positive_only=True)
full_return_histogram = partial(_return_histogram, bin_width_bps=100, positive_only=False)


def _plot_top100_return_histogram(
    histogram: pd.DataFrame,
    *,
    bin_width_bps: int = 10,
    total_observations: int,
    output_dir: Path,
    zoom_max_bps: int = 500,
    zoom_abs_bps: int = 1000,
    positive_only: bool,
) -> None:
    nonzero = histogram.loc[histogram["observations"] > 0]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.25, 1]})
    if positive_only:
        positive_observations = int(histogram["observations"].sum())
        zoom = histogram.loc[histogram["lower_bps"] < zoom_max_bps]
        tail_observations = int(
            histogram.loc[histogram["lower_bps"] >= zoom_max_bps, "observations"].sum()
        )
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
        axes[0].set_xlabel("Excess-return interval midpoint (bps; 10 bps per interval)")
        axes[0].set_xlim(0, zoom_max_bps)
        axes[0].text(
            0.99,
            0.95,
            f">={zoom_max_bps} bps: {tail_observations:,} observations",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
        )
        second_x = nonzero["lower_bps"] + 5
        second_style = {"color": "#d95f02", "linewidth": 1.5}
        second_xlabel = "Positive excess return (bps; full range, 10 bps per interval)"
        annotation = (
            f"positive: {positive_observations:,} / {total_observations:,} "
            f"({positive_observations / total_observations:.2%})"
        )
        stem = "top100_positive_return_10bps_counts"
    else:
        zoom = histogram.loc[histogram["midpoint_bps"].between(-zoom_abs_bps, zoom_abs_bps)]
        for values, color, label in (
            (zoom.loc[zoom["upper_bps"] <= 0], "#c44e52", "Negative excess"),
            (zoom.loc[zoom["lower_bps"] >= 0], "#1f77b4", "Non-negative excess"),
        ):
            axes[0].plot(
                values["midpoint_bps"],
                values["observations"],
                color=color,
                marker="o",
                linewidth=2.2,
                label=label,
            )
        axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.6)
        axes[0].set_title(
            f"auction_multiden Top100 excess-return counts ({bin_width_bps} bps intervals)",
            loc="left",
            fontweight="bold",
        )
        axes[0].set_xlabel("Next internal excess return (bps)")
        axes[0].set_xlim(-zoom_abs_bps, zoom_abs_bps)
        axes[0].legend(loc="best")
        second_x = nonzero["midpoint_bps"]
        second_style = {
            "color": "#555555",
            "marker": "o",
            "markersize": 3,
            "linewidth": 1.5,
        }
        second_xlabel = "Full excess-return range (bps)"
        annotation = f"total: {total_observations:,} stock-decision observations"
        stem = f"top100_return_{bin_width_bps}bps_counts"
    axes[0].set_ylabel("Stock-decision observations")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(second_x, nonzero["observations"], **second_style)
    if not positive_only:
        axes[1].axvline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Observations (log scale)")
    axes[1].set_xlabel(second_xlabel)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.99,
        0.95,
        annotation,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
    )
    save_figure(fig, output_dir, stem)


plot_positive_return_histogram = partial(_plot_top100_return_histogram, positive_only=True)
plot_full_return_histogram = partial(_plot_top100_return_histogram, positive_only=False)


def _run_top100_return_histogram(
    *,
    prediction_root: Path,
    next_label_root: Path,
    pool_path: str,
    output_dir: Path,
    variant: str,
    run_id: str,
    months: list[str],
    bin_width_bps: int,
    positive_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values, trace = _top100_returns(
        prediction_root,
        next_label_root,
        pool_path,
        variant,
        run_id,
        months,
        bin_width_bps,
        "Top100 histogram" if positive_only else "Top100 full histogram",
    )
    histogram = _return_histogram(values, bin_width_bps=bin_width_bps, positive_only=positive_only)
    summary = _return_summary(values, positive_only=positive_only)
    if positive_only:
        name = "top100_positive_return_10bps_counts.csv"
        plot_positive_return_histogram(
            histogram, total_observations=len(values), output_dir=output_dir
        )
    else:
        name = f"top100_return_{bin_width_bps}bps_counts.csv"
        plot_full_return_histogram(
            histogram,
            bin_width_bps=bin_width_bps,
            total_observations=len(values),
            output_dir=output_dir,
        )
    ms.write_frame(output_dir / name, histogram)
    trace["summary"] = summary
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


run_top100_positive_histogram = partial(
    _run_top100_return_histogram, bin_width_bps=10, positive_only=True
)
run_top100_return_histogram = partial(_run_top100_return_histogram, positive_only=False)


def _return_summary(values: pd.Series, *, positive_only: bool) -> dict[str, int | float]:
    nonnegative = values.loc[values >= 0]
    summary: dict[str, int | float] = {
        "total_observations": len(values),
        "negative_observations": int((values < 0).sum()),
    }
    if not positive_only:
        summary["negative_rate"] = float((values < 0).mean())
    summary.update(
        positive_or_zero_observations=len(nonnegative),
        positive_or_zero_rate=float((values >= 0).mean()),
        mean_excess_bps=float(values.mean()),
        median_excess_bps=float(values.median()),
    )
    quantiles = (50, 90, 95, 99) if positive_only else (1, 5, 10, 90, 95, 99)
    sample, prefix = (nonnegative, "positive_p") if positive_only else (values, "p")
    summary.update(
        {f"{prefix}{q:02d}_excess_bps": float(sample.quantile(q / 100)) for q in quantiles}
    )
    if not positive_only:
        summary["minimum_excess_bps"] = float(values.min())
    summary["maximum_excess_bps"] = float(values.max())
    return summary


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
    full_scale: bool = False,
) -> None:
    if full_scale:
        nonzero = histogram.loc[histogram["observations"].gt(0)]
        if nonzero.empty:
            raise ValueError("histogram must contain at least one non-zero observation")
        x_extent = float(nonzero["midpoint_bps"].abs().max()) + float(bin_width_bps) / 2.0
        x_limits = (-x_extent, x_extent)
        y_limits = (0.8, float(nonzero["observations"].max()) * 1.25)
    else:
        x_limits = (
            -TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
            TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS,
        )
        y_limits = TOP1000_RETURN_HISTOGRAM_Y_LIMITS
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
        (
            f"{variant} Top1000 return distributions "
            f"({bin_width_bps} bps intervals{', full scale' if full_scale else ''})"
        ),
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Next internal excess return (bps)")
    ax.set_ylabel("Stock-decision observations (log scale)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    stem = f"top1000_score_bucket_return_{bin_width_bps}bps_counts"
    save_figure(fig, output_dir, f"{stem}_full_scale" if full_scale else stem)


plot_score_bucket_histograms_full_scale = partial(plot_score_bucket_histograms, full_scale=True)


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
    value_parts: list[pd.DataFrame] = []
    extreme_parts: list[pd.DataFrame] = []
    trace = run_trace(
        prediction_root,
        next_label_root,
        pool_path,
        prediction_next_label_col=prediction_next_label_col,
        variant=variant,
        run_id=run_id,
        months={},
        scope="Top1000 within pool_L, split into ten 100-name score buckets",
        value="next internal excess bps",
        bin_width_bps=bin_width_bps,
        observation_unit="stock x decision cross-section",
    )
    for month, frame, shard_trace in ranked_pool_shards(
        prediction_root,
        next_label_root,
        prediction_next_label_col,
        pool_path,
        run_id,
        months,
        1000,
        "Top1000 bucket histogram",
    ):
        top = frame.loc[frame["score_rank"] <= 1000].copy()
        top["score_bucket"] = ((top["score_rank"] - 1) // 100 + 1).astype(int)
        value_parts.append(top[["score_bucket", "excess_bps"]])
        extreme = top.loc[
            top["excess_bps"].abs() >= 4000,
            "date symbol decision_target_timestamp score_rank score_bucket prediction alpha_return_next_close pool_mean excess_bps".split(),
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
    summary = (
        values.groupby("score_bucket", observed=True)["excess_bps"]
        .agg(
            observations="size",
            mean_excess_bps="mean",
            median_excess_bps="median",
            positive_rate=lambda values: float((values >= 0).mean()),
            positive_4000_count=lambda values: int((values >= 4000).sum()),
            negative_4000_count=lambda values: int((values <= -4000).sum()),
        )
        .reset_index()
    )
    summary.insert(1, "rank_start", (summary["score_bucket"] - 1) * 100 + 1)
    summary.insert(2, "rank_end", summary["score_bucket"] * 100)
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
