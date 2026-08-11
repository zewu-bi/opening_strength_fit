"""Top1000 rank-bucket diagnostics and command orchestration."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.analysis import newey_west_mean_se
from opening_strength_fit.legacy import multiscale_bucket_diag as ms
from opening_strength_fit.legacy import rank_bucket_reaudit as rb
from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
    load_ranked_pool_shard,
)
from opening_strength_fit.model_metrics import array_corr
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Top1000 rank, bucket, and economically aligned IC diagnostics."
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--next-label-root", type=Path)
    parser.add_argument(
        "--prediction-next-label-col",
        default="",
        help=(
            "Read next-close labels directly from each prediction parquet column instead of "
            "loading and joining separate yearly label files."
        ),
    )
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--rank-bucket-only",
        action="store_true",
        help="Write rank/bucket diagnostics without the additional multiscale diagnostics.",
    )
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
                    "score_rank_raw_return_pearson_ic": array_corr(average_ranks(scores), labels),
                    "raw_score_raw_return_pearson_ic": array_corr(scores, labels),
                    "reverse_score_rank_ic": ms.spearman_rank_ic(-scores, labels),
                }
            )
    return pd.DataFrame(rows)


def hac_mean_t(values: pd.Series, max_lag: int = 5) -> tuple[float, float]:
    mean, standard_error = newey_west_mean_se(values, lag=max_lag)
    t_stat = mean / standard_error if standard_error else math.nan
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


def run_rank_bucket(
    *,
    prediction_root: Path,
    next_label_root: Path | None,
    prediction_next_label_col: str,
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
        "next_label_root": str(next_label_root) if next_label_root is not None else None,
        "prediction_next_label_col": prediction_next_label_col or None,
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
        if not prediction_next_label_col and year not in labels:
            if next_label_root is None:
                raise ValueError("next_label_root is required without embedded prediction labels")
            labels.clear()
            labels[year] = ms.load_label_for_year(ms.next_label_path(next_label_root, year))
        pred_path = ms.prediction_path(prediction_root, run_id, month)
        print(f"rank/bucket shard {month}: {pred_path}", flush=True)
        frame, shard_trace = load_ranked_pool_shard(
            pred_path=pred_path,
            labels=labels.get(year),
            pool=pool,
            prediction_next_label_col=prediction_next_label_col,
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
    from opening_strength_fit.legacy.top1000_return_histograms import (
        run_top100_positive_histogram,
        run_top100_return_histogram,
        run_top1000_bucket_return_histogram,
    )

    args = parse_args()
    months = [month.strip() for month in args.months.split(",") if month.strip()]
    if args.next_label_root is None and not args.prediction_next_label_col:
        raise SystemExit(
            "pass --next-label-root or read embedded labels with --prediction-next-label-col"
        )
    if args.prediction_next_label_col and (
        args.top100_positive_histogram_only or args.top100_return_histogram_only
    ):
        raise SystemExit(
            "embedded prediction labels are supported for rank-bucket and Top1000 bucket "
            "distribution diagnostics"
        )
    if (
        args.prediction_next_label_col
        and not args.rank_bucket_only
        and not args.top1000_bucket_return_histogram_only
    ):
        raise SystemExit(
            "embedded prediction labels currently require --rank-bucket-only; "
            "the multiscale diagnostic still requires --next-label-root"
        )
    if args.top1000_bucket_return_histogram_only:
        run_top1000_bucket_return_histogram(
            prediction_root=args.prediction_root,
            next_label_root=args.next_label_root,
            prediction_next_label_col=args.prediction_next_label_col,
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
        prediction_next_label_col=args.prediction_next_label_col,
        pool_path=args.pool_path,
        output_dir=rank_output,
        variant=args.variant,
        run_id=args.run_id,
        months=months,
        top_n=args.top_n,
    )
    if args.rank_bucket_only:
        print(f"wrote {args.output_dir}", flush=True)
        return
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
