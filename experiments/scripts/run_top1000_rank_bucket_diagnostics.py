from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.legacy import multiscale_bucket_diag as ms
from opening_strength_fit.legacy import rank_bucket_reaudit as rb
from opening_strength_fit.stock_pool import load_stock_pool

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
IC_METRICS = (
    "spearman_realized_rank_ic",
    "score_rank_raw_return_pearson_ic",
    "raw_score_raw_return_pearson_ic",
)
TOP1000_SCORE_BUCKETS = tuple(range(1, 11))
TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS = 100
TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS = 3000
TOP1000_RETURN_HISTOGRAM_Y_LIMITS = (1e2, 3e5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Top1000 rank, bucket, and economically aligned IC diagnostics."
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--next-label-root", type=Path, required=True)
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--top100-positive-histogram-only",
        action="store_true",
        help="Only count Top100 positive excess returns in 10 bps intervals.",
    )
    parser.add_argument(
        "--top100-return-histogram-only",
        action="store_true",
        help="Only count the full Top100 excess-return distribution.",
    )
    parser.add_argument(
        "--top1000-bucket-return-histogram-only",
        action="store_true",
        help="Compare 100-name score-bucket return distributions within Top1000.",
    )
    parser.add_argument(
        "--histogram-bin-width-bps",
        type=int,
        default=TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
    )
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, copy=False).rank(method="average").to_numpy(dtype="float64")


def pearson_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return math.nan
    x = scores[valid]
    y = outcomes[valid]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_group_ic(frame: pd.DataFrame, *, variant: str, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(ms.GROUP_COLS, sort=False, observed=True):
        scopes = {
            "pool_L": group,
            "top1000": group.iloc[:top_n],
            "top100": group.iloc[:100],
        }
        for scope, scoped in scopes.items():
            scores = scoped["prediction"].to_numpy(dtype="float64", copy=False)
            labels = scoped["alpha_return_next_close"].to_numpy(dtype="float64", copy=False)
            rank_ic = ms.spearman_rank_ic(scores, labels)
            rows.append(
                {
                    "variant": variant,
                    "date": keys[0],
                    "decision_target_timestamp": keys[1],
                    "month": scoped["month"].iloc[0],
                    "scope": scope,
                    "rows": len(scoped),
                    "spearman_realized_rank_ic": rank_ic,
                    "score_rank_raw_return_pearson_ic": pearson_ic(average_ranks(scores), labels),
                    "raw_score_raw_return_pearson_ic": pearson_ic(scores, labels),
                    "reverse_score_rank_ic": ms.spearman_rank_ic(-scores, labels),
                }
            )
    return pd.DataFrame(rows)


def hac_mean_t(values: pd.Series, max_lag: int = 5) -> tuple[float, float]:
    x = values.dropna().to_numpy(dtype="float64")
    n = len(x)
    centered = x - x.mean()
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        autocovariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    t_stat = float(x.mean() / standard_error) if standard_error else math.nan
    return standard_error, t_stat


def summarize_daily_ic(group_ic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        group_ic.groupby(["variant", "scope", "date"], observed=True)[list(IC_METRICS)]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, group in daily.groupby(["variant", "scope"], observed=True):
        for metric in IC_METRICS:
            values = group[metric].dropna().astype("float64")
            n = len(values)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            naive_se = std / math.sqrt(n)
            hac_se, hac_t = hac_mean_t(values)
            rows.append(
                {
                    "variant": keys[0],
                    "scope": keys[1],
                    "metric": metric,
                    "observations": n,
                    "mean": mean,
                    "std": std,
                    "positive_rate": float((values > 0).mean()),
                    "ordinary_t": mean / naive_se if naive_se else math.nan,
                    "hac_lag5_t": hac_t,
                    "hac_lag5_ci95_low": mean - 1.96 * hac_se,
                    "hac_lag5_ci95_high": mean + 1.96 * hac_se,
                    "naive_annualized_ir": mean / std * math.sqrt(252) if std else math.nan,
                }
            )
    return daily, pd.DataFrame(rows)


def summarize_group_ic(group_ic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in group_ic.groupby(["variant", "scope"], observed=True):
        for metric in IC_METRICS:
            values = group[metric].dropna()
            rows.append(
                {
                    "variant": keys[0],
                    "scope": keys[1],
                    "metric": metric,
                    "groups": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "positive_rate": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def exact_rank_part(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    top = frame.loc[frame["score_rank"] <= top_n]
    return (
        top.groupby("score_rank", observed=True)["excess_bps"]
        .agg(excess_sum="sum", observations="count")
        .reset_index()
    )


def finalize_exact_rank(parts: list[pd.DataFrame], *, variant: str) -> pd.DataFrame:
    curve = (
        pd.concat(parts, ignore_index=True)
        .groupby("score_rank", observed=True)[["excess_sum", "observations"]]
        .sum()
        .reset_index()
        .sort_values("score_rank")
    )
    curve["mean_excess_bps"] = curve["excess_sum"] / curve["observations"]
    curve.insert(0, "variant", variant)
    return curve


def exact_curve_summary(curve: pd.DataFrame) -> pd.DataFrame:
    score_order = -curve["score_rank"].astype("float64")
    outcomes = curve["mean_excess_bps"].astype("float64")
    row: dict[str, object] = {
        "variant": curve["variant"].iloc[0],
        "exact_rank_points": len(curve),
        "exact_rank_curve_spearman_ic": score_order.corr(outcomes, method="spearman"),
        "exact_rank_curve_pearson_ic": score_order.corr(outcomes),
        "mean_rank_1_100_bps": outcomes.iloc[:100].mean(),
        "mean_rank_901_1000_bps": outcomes.iloc[-100:].mean(),
        "top100_minus_rank901_1000_bps": outcomes.iloc[:100].mean() - outcomes.iloc[-100:].mean(),
    }
    for bucket_count in rb.BUCKET_COUNTS:
        size = len(curve) // bucket_count
        bucket = outcomes.groupby((curve["score_rank"] - 1) // size).mean()
        row[f"bucket{bucket_count}_curve_spearman_ic"] = pd.Series(
            -np.arange(1, bucket_count + 1), dtype="float64"
        ).corr(bucket.reset_index(drop=True), method="spearman")
    return pd.DataFrame([row])


def build_top_rank_subbuckets(
    curve: pd.DataFrame, *, top_k: int = 100, bucket_width: int = 10
) -> pd.DataFrame:
    top = curve.loc[curve["score_rank"] <= top_k].copy()
    top["subbucket"] = ((top["score_rank"] - 1) // bucket_width + 1).astype(int)
    summary = (
        top.groupby("subbucket", observed=True)
        .agg(
            start_rank=("score_rank", "min"),
            end_rank=("score_rank", "max"),
            rows=("observations", "sum"),
            mean_excess_bps=("mean_excess_bps", "mean"),
            min_exact_rank_mean_bps=("mean_excess_bps", "min"),
            max_exact_rank_mean_bps=("mean_excess_bps", "max"),
        )
        .reset_index()
    )
    summary.insert(0, "variant", curve["variant"].iloc[0])
    summary["rank_slice"] = (
        summary["start_rank"].astype(str) + "-" + summary["end_rank"].astype(str)
    )
    return summary


def plot_top100_exact_rank_curve(
    curve: pd.DataFrame, subbuckets: pd.DataFrame, *, variant: str, output_dir: Path
) -> None:
    top = curve.loc[curve["score_rank"] <= 100].copy()
    midpoint = (subbuckets["start_rank"] + subbuckets["end_rank"]) / 2.0
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        top["score_rank"],
        top["mean_excess_bps"],
        color="#8fb9dc",
        marker="o",
        markersize=3,
        linewidth=1.2,
        alpha=0.75,
        label="Exact score-rank mean",
    )
    ax.plot(
        midpoint,
        subbuckets["mean_excess_bps"],
        color="#d95f02",
        marker="o",
        markersize=7,
        linewidth=2.8,
        label="10-name subbucket mean",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"{variant} Top100 return by score rank", loc="left", fontweight="bold")
    ax.set_xlabel("Score rank within Top100")
    ax.set_ylabel("Mean pool_L next internal excess (bps)")
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"top100_exact_rank_returns.{extension}", dpi=140)
    plt.close(fig)


def plot_bucket_curves(curves: pd.DataFrame, *, variant: str, output_dir: Path) -> None:
    for scope in ("top1000", "pool_L"):
        scoped = curves.loc[curves["scope"].eq(scope)]
        fig, ax = plt.subplots(figsize=(14, 8))
        for bucket_count in rb.BUCKET_COUNTS:
            series = scoped.loc[scoped["bucket_count"].eq(bucket_count)].sort_values("bucket")
            ax.plot(
                series["x"],
                series["mean_excess_bps"],
                marker="o",
                linewidth=2,
                markersize=4,
                label=f"{bucket_count} buckets",
            )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(f"{variant} {scope} bucket returns", loc="left", fontweight="bold")
        ax.set_xlabel("Score rank" if scope == "top1000" else "Score percentile")
        ax.set_ylabel("Mean pool_L next internal excess (bps)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        for extension in ("svg", "png"):
            fig.savefig(output_dir / f"{scope}_bucket_returns.{extension}", dpi=140)
        plt.close(fig)


def keep_requested_variant(output_dir: Path, *, variant: str) -> None:
    """Remove the legacy synthetic average emitted for a single-variant run."""
    for name in (
        "bucket_width_distribution_summary.csv",
        "topk_shape_summary.csv",
        "topk_internal_ic_summary.csv",
        "bucket_width_within_ic_summary.csv",
        "local_ic_window_summary.csv",
    ):
        path = output_dir / name
        frame = pd.read_csv(path)
        ms.write_frame(path, frame.loc[frame["variant"].eq(variant)].copy())


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
    value_parts: list[pd.DataFrame] = []
    extreme_parts: list[pd.DataFrame] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root),
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
        if year not in labels:
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"Top1000 bucket histogram shard {month}: {pred_path}", flush=True)
        frame, shard_trace = ms.load_ranked_pool_shard(
            pred_path=pred_path,
            labels=labels[year],
            pool=pool,
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


def run_rank_bucket(
    *,
    prediction_root: Path,
    next_label_root: Path,
    pool_path: str,
    output_dir: Path,
    variant: str,
    run_id: str,
    months: list[str],
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(pool_path)
    labels: dict[str, pd.DataFrame] = {}
    group_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    exact_parts: list[pd.DataFrame] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_root": str(prediction_root),
        "next_label_root": str(next_label_root),
        "pool_path": pool_path,
        "variant": variant,
        "run_id": run_id,
        "months": {},
        "top_n": top_n,
        "inference_unit": "trading_day",
        "hac": "Bartlett/Newey-West, maximum lag 5",
    }
    for month in months:
        year = month[:4]
        if year not in labels:
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"rank/bucket shard {month}: {pred_path}", flush=True)
        frame, shard_trace = ms.load_ranked_pool_shard(
            pred_path=pred_path,
            labels=labels[year],
            pool=pool,
        )
        group_sizes = frame.groupby(ms.GROUP_COLS, observed=True)["group_size"].first()
        complete_groups = group_sizes.loc[group_sizes.ge(top_n)]
        shard_trace["pool_group_size_min"] = int(group_sizes.min())
        shard_trace["pool_groups_total"] = int(len(group_sizes))
        shard_trace["pool_groups_with_top_n"] = int(len(complete_groups))
        shard_trace["pool_groups_below_top_n"] = int(len(group_sizes) - len(complete_groups))
        if complete_groups.empty:
            raise ValueError(f"no pool group has at least Top{top_n} rows")
        if len(complete_groups) != len(group_sizes):
            complete_index = pd.MultiIndex.from_frame(
                complete_groups.index.to_frame(index=False)[ms.GROUP_COLS]
            )
            frame_index = pd.MultiIndex.from_frame(frame[ms.GROUP_COLS])
            frame = frame.loc[frame_index.isin(complete_index)].copy()
        group_parts.append(build_group_ic(frame, variant=variant, top_n=top_n))
        bucket_ic, curve_part = rb.build_bucket_diagnostics(frame, variant=variant)
        bucket_parts.append(bucket_ic)
        curve_parts.append(curve_part)
        exact_parts.append(exact_rank_part(frame, top_n=top_n))
        trace["months"][month] = shard_trace

    group_ic = pd.concat(group_parts, ignore_index=True)
    bucket_ic = pd.concat(bucket_parts, ignore_index=True)
    curves = rb.finalize_curve_data(pd.concat(curve_parts, ignore_index=True))
    exact_curve = finalize_exact_rank(exact_parts, variant=variant)
    top100_subbuckets = build_top_rank_subbuckets(exact_curve)
    daily_ic, daily_summary = summarize_daily_ic(group_ic)

    ms.write_frame(output_dir / "group_ic_values.csv", group_ic)
    ms.write_frame(output_dir / "group_ic_summary.csv", summarize_group_ic(group_ic))
    ms.write_frame(output_dir / "daily_ic_values.csv", daily_ic)
    ms.write_frame(output_dir / "daily_ic_summary.csv", daily_summary)
    ms.write_frame(output_dir / "bucket_group_rank_ic_values.csv", bucket_ic)
    ms.write_frame(
        output_dir / "bucket_group_rank_ic_summary.csv",
        rb.summarize_ic(
            bucket_ic,
            value="bucket_rank_ic",
            keys=["variant", "scope", "bucket_count"],
        ),
    )
    ms.write_frame(output_dir / "bucket_curve_plot_data.csv", curves)
    ms.write_frame(output_dir / "bucket_curve_rank_ic.csv", rb.curve_rank_ic(curves))
    ms.write_frame(output_dir / "top1000_exact_rank_curve.csv", exact_curve)
    ms.write_frame(
        output_dir / "top1000_exact_rank_curve_summary.csv", exact_curve_summary(exact_curve)
    )
    ms.write_frame(output_dir / "top100_rank10_summary.csv", top100_subbuckets)
    plot_bucket_curves(curves, variant=variant, output_dir=output_dir)
    plot_top100_exact_rank_curve(
        exact_curve,
        top100_subbuckets,
        variant=variant,
        output_dir=output_dir,
    )
    (output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    months = [month.strip() for month in args.months.split(",") if month.strip()]
    if args.top1000_bucket_return_histogram_only:
        run_top1000_bucket_return_histogram(
            prediction_root=args.prediction_root,
            next_label_root=args.next_label_root,
            pool_path=args.pool_path,
            output_dir=args.output_dir,
            variant=args.variant,
            run_id=args.run_id,
            months=months,
            bin_width_bps=args.histogram_bin_width_bps,
        )
        return
    if args.top100_return_histogram_only:
        run_top100_return_histogram(
            prediction_root=args.prediction_root,
            next_label_root=args.next_label_root,
            pool_path=args.pool_path,
            output_dir=args.output_dir,
            variant=args.variant,
            run_id=args.run_id,
            months=months,
            bin_width_bps=args.histogram_bin_width_bps,
        )
        return
    if args.top100_positive_histogram_only:
        run_top100_positive_histogram(
            prediction_root=args.prediction_root,
            next_label_root=args.next_label_root,
            pool_path=args.pool_path,
            output_dir=args.output_dir,
            variant=args.variant,
            run_id=args.run_id,
            months=months,
        )
        return
    rank_output = args.output_dir / "rank_bucket"
    multiscale_output = args.output_dir / "multiscale"
    run_rank_bucket(
        prediction_root=args.prediction_root,
        next_label_root=args.next_label_root,
        pool_path=args.pool_path,
        output_dir=rank_output,
        variant=args.variant,
        run_id=args.run_id,
        months=months,
        top_n=args.top_n,
    )
    ms.run_multiscale_bucket_diagnostics(
        ms.MultiscaleBucketDiagConfig(
            prediction_root=args.prediction_root,
            next_label_root=args.next_label_root,
            pool_path=args.pool_path,
            output_dir=multiscale_output,
            run_ids={args.variant: args.run_id},
            months=months,
            bucket_widths=[50, 100, 200],
            top_k=[50, 100, 150, 200, 500, 1000],
            window_widths=[50, 100, 200],
            window_stride=50,
            top_n=args.top_n,
        )
    )
    keep_requested_variant(multiscale_output, variant=args.variant)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
