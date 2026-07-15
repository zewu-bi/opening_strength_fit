from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    GROUP_COLS,
    load_label_for_year,
    load_ranked_pool_shard,
    next_label_path,
    prediction_path,
    spearman_rank_ic,
    write_frame,
)
from opening_strength_fit.stock_pool import load_stock_pool

BUCKET_COUNTS = (10, 20, 50)
TOP_N = 1000


@dataclass(frozen=True)
class RankBucketReauditConfig:
    prediction_root: Path
    next_label_root: Path
    pool_path: str
    output_dir: Path
    run_ids: dict[str, str]
    months: list[str]


def numpy_average_ranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks implemented without the production rank helper."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = ((starts + ends + 1.0) / 2.0)[np.cumsum(group_start) - 1]
    return ranks


def pearson_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return math.nan
    scores = scores[valid]
    outcomes = outcomes[valid]
    if np.ptp(scores) == 0 or np.ptp(outcomes) == 0:
        return math.nan
    return float(np.corrcoef(scores, outcomes)[0, 1])


def numpy_rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return math.nan
    return pearson_ic(
        numpy_average_ranks(scores[valid]),
        numpy_average_ranks(outcomes[valid]),
    )


def build_group_ic_diagnostics(frame: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(GROUP_COLS, sort=False, observed=True):
        for scope, scoped in {
            "pool_L": group,
            "top1000": group.iloc[:TOP_N],
            "top100": group.iloc[:100],
        }.items():
            scores = scoped["prediction"].to_numpy(dtype="float64", copy=False)
            labels = scoped["alpha_return_next_close"].to_numpy(dtype="float64", copy=False)
            excess = scoped["excess_bps"].to_numpy(dtype="float64", copy=False)
            rank_ic = spearman_rank_ic(scores, labels)
            independent_ic = numpy_rank_ic(scores, labels)
            reverse_ic = spearman_rank_ic(-scores, labels)
            excess_ic = spearman_rank_ic(scores, excess)
            rows.append(
                {
                    "variant": variant,
                    "date": keys[0],
                    "decision_target_timestamp": keys[1],
                    "month": scoped["month"].iloc[0],
                    "scope": scope,
                    "rows": len(scoped),
                    "rank_ic": rank_ic,
                    "independent_rank_ic": independent_ic,
                    "reverse_score_rank_ic": reverse_ic,
                    "excess_rank_ic": excess_ic,
                    "score_rank_raw_return_pearson_ic": pearson_ic(
                        numpy_average_ranks(scores), labels
                    ),
                    "implementation_abs_diff": abs(rank_ic - independent_ic),
                    "reverse_sign_abs_error": abs(rank_ic + reverse_ic),
                    "label_excess_abs_diff": abs(rank_ic - excess_ic),
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
        "top1000": frame.loc[frame["score_rank"] <= TOP_N],
        "pool_L": frame,
    }
    for scope, scoped in scopes.items():
        work = scoped.copy()
        work["outcome_rank_pct"] = work.groupby(
            GROUP_COLS, sort=False, observed=True
        )["alpha_return_next_close"].rank(method="average", pct=True)
        for bucket_count in BUCKET_COUNTS:
            divisor = TOP_N if scope == "top1000" else work["group_size"]
            work["bucket"] = (
                (work["score_rank"] - 1) * bucket_count // divisor + 1
            ).clip(upper=bucket_count).astype(int)
            group_bucket = (
                work.groupby(GROUP_COLS + ["month", "bucket"], observed=True)
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
            for keys, group in group_bucket.groupby(GROUP_COLS, sort=False, observed=True):
                ic_rows.append(
                    {
                        "variant": variant,
                        "date": keys[0],
                        "decision_target_timestamp": keys[1],
                        "month": group["month"].iloc[0],
                        "scope": scope,
                        "bucket_count": bucket_count,
                        "bucket_rank_ic": spearman_rank_ic(
                            -group["bucket"].to_numpy(dtype="float64", copy=False),
                            group["mean_excess_bps"].to_numpy(
                                dtype="float64", copy=False
                            ),
                        ),
                    }
                )
    return pd.DataFrame(ic_rows), pd.concat(curve_parts, ignore_index=True)


def finalize_curve_data(parts: pd.DataFrame) -> pd.DataFrame:
    curves = (
        parts.groupby(
            ["variant", "scope", "bucket_count", "bucket"], observed=True
        )[["excess_sum", "outcome_rank_sum", "groups"]]
        .sum()
        .reset_index()
    )
    curves["mean_excess_bps"] = curves.pop("excess_sum") / curves["groups"]
    curves["mean_within_group_outcome_rank_pct"] = (
        curves.pop("outcome_rank_sum") / curves["groups"]
    )
    top = curves["scope"].eq("top1000")
    curves["x"] = np.where(
        top,
        (curves["bucket"] - 0.5) * TOP_N / curves["bucket_count"],
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
            positive_rate=lambda s: float((s > 0).mean()),
        )
        .reset_index()
    )


def curve_rank_ic(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in curves.groupby(
        ["variant", "scope", "bucket_count"], sort=False, observed=True
    ):
        rows.append(
            {
                "variant": keys[0],
                "scope": keys[1],
                "bucket_count": keys[2],
                "curve_rank_ic": spearman_rank_ic(
                    -group["bucket"].to_numpy(dtype="float64", copy=False),
                    group["mean_excess_bps"].to_numpy(dtype="float64", copy=False),
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_return_curves(curves: pd.DataFrame, *, variant: str, output_dir: Path) -> None:
    for scope in ("top1000", "pool_L"):
        scoped = curves.loc[(curves["variant"] == variant) & (curves["scope"] == scope)]
        fig, ax = plt.subplots(figsize=(14, 8))
        for bucket_count in BUCKET_COUNTS:
            series = scoped.loc[scoped["bucket_count"] == bucket_count].sort_values("bucket")
            label = (
                f"{bucket_count} buckets ({TOP_N // bucket_count} names/bucket)"
                if scope == "top1000"
                else f"{bucket_count} equal-count buckets"
            )
            ax.plot(
                series["x"],
                series["mean_excess_bps"],
                marker="o",
                linewidth=2,
                markersize=4,
                label=label,
            )
            first = series.iloc[0]
            ax.annotate(
                f"{first['mean_excess_bps']:.2f}",
                (first["x"], first["mean_excess_bps"]),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=10,
            )
        last = scoped.loc[scoped["bucket_count"] == 10].sort_values("bucket").iloc[-1]
        ax.annotate(
            f"{last['mean_excess_bps']:.2f}",
            (last["x"], last["mean_excess_bps"]),
            xytext=(-35, 7),
            textcoords="offset points",
            fontsize=10,
        )
        scope_label = "Top1000" if scope == "top1000" else "Full-pool"
        ax.set_title(
            f"mech328 v2 {scope_label} Bucket Returns",
            loc="left",
            fontsize=18,
            fontweight="bold",
            pad=28,
        )
        ax.text(
            0.0,
            1.025,
            "pool_L next internal excess bps, decision-group equal weighting",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_xlabel(
            "Score rank within Top1000"
            if scope == "top1000"
            else "Score percentile within pool_L (high to low)"
        )
        ax.set_ylabel("Mean excess (bps)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        stem = f"mech328_v2_{scope.lower()}_bucket_returns"
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=140, bbox_inches="tight")
        plt.close(fig)


def run_rank_bucket_reaudit(config: RankBucketReauditConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_stock_pool(config.pool_path)
    labels: dict[str, pd.DataFrame] = {}
    group_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    trace: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pool_path": config.pool_path,
        "variants": {},
    }
    for variant, run_id in config.run_ids.items():
        variant_trace: dict[str, object] = {"run_id": run_id, "months": {}}
        for month in config.months:
            year = month[:4]
            if year not in labels:
                labels.clear()
                labels[year] = load_label_for_year(
                    next_label_path(config.next_label_root, year)
                )
            frame, shard_trace = load_ranked_pool_shard(
                pred_path=prediction_path(config.prediction_root, run_id, month),
                labels=labels[year],
                pool=pool,
            )
            if int(frame["group_size"].min()) < TOP_N:
                raise ValueError(f"pool group smaller than Top{TOP_N}")
            group_parts.append(build_group_ic_diagnostics(frame, variant=variant))
            bucket_ic, curve_part = build_bucket_diagnostics(frame, variant=variant)
            bucket_parts.append(bucket_ic)
            curve_parts.append(curve_part)
            variant_trace["months"][month] = shard_trace
        trace["variants"][variant] = variant_trace

    group_ic = pd.concat(group_parts, ignore_index=True)
    bucket_ic = pd.concat(bucket_parts, ignore_index=True)
    curves = finalize_curve_data(pd.concat(curve_parts, ignore_index=True))
    group_summary = summarize_ic(
        group_ic, value="rank_ic", keys=["variant", "scope"]
    )
    audit_errors = (
        group_ic.groupby(["variant", "scope"], observed=True)[
            [
                "implementation_abs_diff",
                "reverse_sign_abs_error",
                "label_excess_abs_diff",
            ]
        ]
        .max()
        .reset_index()
    )
    group_summary = group_summary.merge(audit_errors, on=["variant", "scope"])
    bucket_summary = summarize_ic(
        bucket_ic,
        value="bucket_rank_ic",
        keys=["variant", "scope", "bucket_count"],
    )
    write_frame(config.output_dir / "group_rank_ic_summary.csv", group_summary)
    write_frame(config.output_dir / "bucket_group_rank_ic_summary.csv", bucket_summary)
    write_frame(config.output_dir / "bucket_curve_plot_data.csv", curves)
    write_frame(config.output_dir / "bucket_curve_rank_ic.csv", curve_rank_ic(curves))
    for variant in config.run_ids:
        if "mech328_v2" in variant:
            plot_return_curves(curves, variant=variant, output_dir=config.output_dir)
    (config.output_dir / "trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
