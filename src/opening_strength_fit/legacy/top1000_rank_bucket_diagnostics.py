"""Top1000 rank-bucket diagnostics and command orchestration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.analysis import newey_west_mean_se
from opening_strength_fit.legacy import multiscale_bucket_diag as ms
from opening_strength_fit.legacy.top1000_rank_data import (
    DEFAULT_DIAGNOSTIC_MONTHS as DEFAULT_MONTHS,
)
from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
    ranked_pool_shards,
    run_trace,
    save_figure,
)
from opening_strength_fit.model_metrics import array_corr

IC_METRICS = tuple(
    "spearman_realized_rank_ic score_rank_raw_return_pearson_ic "
    "raw_score_raw_return_pearson_ic".split()
)
BUCKET_COUNTS = (10, 20, 50)


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
    modes = {
        "rank-bucket": "Write rank/bucket diagnostics without the additional multiscale diagnostics.",
        "top100-positive-histogram": "Only count Top100 positive excess returns in 10 bps intervals.",
        "top100-return-histogram": "Only count the full Top100 excess-return distribution.",
        "top1000-bucket-return-histogram": (
            "Compare 100-name score-bucket return distributions within Top1000."
        ),
    }
    for name, help_text in modes.items():
        parser.add_argument(f"--{name}-only", action="store_true", help=help_text)
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
            _, hac_se = newey_west_mean_se(values, lag=5)
            hac_t = mean / hac_se if hac_se else math.nan
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


def build_bucket_diagnostics(
    frame: pd.DataFrame,
    *,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, object]] = []
    curve_parts: list[pd.DataFrame] = []
    scopes = {
        "top1000": frame.loc[frame["score_rank"] <= 1000],
        "pool_L": frame,
    }
    for scope, scoped in scopes.items():
        work = scoped.copy()
        work["outcome_rank_pct"] = work.groupby(ms.GROUP_COLS, sort=False, observed=True)[
            "alpha_return_next_close"
        ].rank(method="average", pct=True)
        for bucket_count in BUCKET_COUNTS:
            divisor = 1000 if scope == "top1000" else work["group_size"]
            work["bucket"] = (
                ((work["score_rank"] - 1) * bucket_count // divisor + 1)
                .clip(upper=bucket_count)
                .astype(int)
            )
            group_bucket = (
                work.groupby(ms.GROUP_COLS + ["month", "bucket"], observed=True)
                .agg(
                    mean_excess_bps=("excess_bps", "mean"),
                    mean_outcome_rank_pct=("outcome_rank_pct", "mean"),
                )
                .reset_index()
            )
            curve_part = (
                group_bucket.groupby("bucket", observed=True)
                .agg(
                    excess_sum=("mean_excess_bps", "sum"),
                    outcome_rank_sum=("mean_outcome_rank_pct", "sum"),
                    groups=("mean_excess_bps", "count"),
                )
                .reset_index()
            )
            curve_part.insert(0, "bucket_count", bucket_count)
            curve_part.insert(0, "scope", scope)
            curve_part.insert(0, "variant", variant)
            curve_parts.append(curve_part)
            for keys, group in group_bucket.groupby(ms.GROUP_COLS, sort=False, observed=True):
                ic_rows.append(
                    {
                        "variant": variant,
                        "date": keys[0],
                        "decision_target_timestamp": keys[1],
                        "month": group["month"].iloc[0],
                        "scope": scope,
                        "bucket_count": bucket_count,
                        "bucket_rank_ic": ms.spearman_rank_ic(
                            -group["bucket"].to_numpy(dtype="float64", copy=False),
                            group["mean_excess_bps"].to_numpy(dtype="float64", copy=False),
                        ),
                    }
                )
    return pd.DataFrame(ic_rows), pd.concat(curve_parts, ignore_index=True)


def finalize_curve_data(parts: pd.DataFrame) -> pd.DataFrame:
    curves = (
        parts.groupby(["variant", "scope", "bucket_count", "bucket"], observed=True)[
            ["excess_sum", "outcome_rank_sum", "groups"]
        ]
        .sum()
        .reset_index()
    )
    curves["mean_excess_bps"] = curves.pop("excess_sum") / curves["groups"]
    curves["mean_within_group_outcome_rank_pct"] = curves.pop("outcome_rank_sum") / curves["groups"]
    top = curves["scope"].eq("top1000")
    curves["x"] = np.where(
        top,
        (curves["bucket"] - 0.5) * 1000 / curves["bucket_count"],
        (curves["bucket"] - 0.5) * 100.0 / curves["bucket_count"],
    )
    return curves


def summarize_ic(frame: pd.DataFrame, *, value: str, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True)[value]
        .agg(
            groups="count",
            mean="mean",
            median="median",
            positive_rate=lambda series: float((series > 0).mean()),
        )
        .reset_index()
    )


def curve_rank_ic(curves: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "scope", "bucket_count"]
    return (
        curves.groupby(keys, sort=False, observed=True)
        .apply(
            lambda group: ms.spearman_rank_ic(-group["bucket"], group["mean_excess_bps"]),
            include_groups=False,
        )
        .rename("curve_rank_ic")
        .reset_index()
    )


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
    for bucket_count in BUCKET_COUNTS:
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
    save_figure(fig, output_dir, "top100_exact_rank_returns")


def plot_bucket_curves(curves: pd.DataFrame, *, variant: str, output_dir: Path) -> None:
    for scope in ("top1000", "pool_L"):
        scoped = curves.loc[curves["scope"].eq(scope)]
        fig, ax = plt.subplots(figsize=(14, 8))
        for bucket_count in BUCKET_COUNTS:
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
        save_figure(fig, output_dir, f"{scope}_bucket_returns")


def keep_requested_variant(output_dir: Path, *, variant: str) -> None:
    """Remove the legacy synthetic average emitted for a single-variant run."""
    names = "bucket_width_distribution_summary.csv topk_shape_summary.csv topk_internal_ic_summary.csv bucket_width_within_ic_summary.csv local_ic_window_summary.csv"
    for name in names.split():
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
    group_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    exact_parts: list[pd.DataFrame] = []
    trace = run_trace(
        prediction_root,
        next_label_root,
        pool_path,
        prediction_next_label_col=prediction_next_label_col,
        variant=variant,
        run_id=run_id,
        months={},
        top_n=top_n,
        inference_unit="trading_day",
        hac="Bartlett/Newey-West, maximum lag 5",
    )
    for month, frame, shard_trace in ranked_pool_shards(
        prediction_root,
        next_label_root,
        prediction_next_label_col,
        pool_path,
        run_id,
        months,
        top_n,
        "rank/bucket",
    ):
        group_parts.append(build_group_ic(frame, variant=variant, top_n=top_n))
        bucket_ic, curve_part = build_bucket_diagnostics(frame, variant=variant)
        bucket_parts.append(bucket_ic)
        curve_parts.append(curve_part)
        exact_parts.append(exact_rank_part(frame, top_n=top_n))
        trace["months"][month] = shard_trace

    group_ic = pd.concat(group_parts, ignore_index=True)
    bucket_ic = pd.concat(bucket_parts, ignore_index=True)
    curves = finalize_curve_data(pd.concat(curve_parts, ignore_index=True))
    exact_curve = finalize_exact_rank(exact_parts, variant=variant)
    top100_subbuckets = build_top_rank_subbuckets(exact_curve)
    daily_ic, daily_summary = summarize_daily_ic(group_ic)

    outputs = {
        "group_ic_values.csv": group_ic,
        "group_ic_summary.csv": summarize_group_ic(group_ic),
        "daily_ic_values.csv": daily_ic,
        "daily_ic_summary.csv": daily_summary,
        "bucket_group_rank_ic_values.csv": bucket_ic,
        "bucket_group_rank_ic_summary.csv": summarize_ic(
            bucket_ic,
            value="bucket_rank_ic",
            keys=["variant", "scope", "bucket_count"],
        ),
        "bucket_curve_plot_data.csv": curves,
        "bucket_curve_rank_ic.csv": curve_rank_ic(curves),
        "top1000_exact_rank_curve.csv": exact_curve,
        "top1000_exact_rank_curve_summary.csv": exact_curve_summary(exact_curve),
        "top100_rank10_summary.csv": top100_subbuckets,
    }
    for name, frame in outputs.items():
        ms.write_frame(output_dir / name, frame)
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
    common = {
        "prediction_root": args.prediction_root,
        "next_label_root": args.next_label_root,
        "pool_path": args.pool_path,
        "variant": args.variant,
        "run_id": args.run_id,
        "months": months,
    }
    if args.top1000_bucket_return_histogram_only:
        run_top1000_bucket_return_histogram(
            prediction_next_label_col=args.prediction_next_label_col,
            output_dir=args.output_dir,
            bin_width_bps=args.histogram_bin_width_bps,
            **common,
        )
        return
    if args.top100_return_histogram_only:
        run_top100_return_histogram(
            output_dir=args.output_dir,
            bin_width_bps=args.histogram_bin_width_bps,
            **common,
        )
        return
    if args.top100_positive_histogram_only:
        run_top100_positive_histogram(output_dir=args.output_dir, **common)
        return
    rank_output = args.output_dir / "rank_bucket"
    multiscale_output = args.output_dir / "multiscale"
    run_rank_bucket(
        prediction_next_label_col=args.prediction_next_label_col,
        output_dir=rank_output,
        top_n=args.top_n,
        **common,
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
