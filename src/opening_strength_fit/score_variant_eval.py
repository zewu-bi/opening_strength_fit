from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    positive_count,
    positive_rate,
    selection_return_stats,
)


def score_variants(
    test: pd.DataFrame,
    *,
    month: str,
    variants: list[dict[str, object]],
    top_n: int,
    selection_mask_col: str = "",
) -> pd.DataFrame:
    if selection_mask_col and selection_mask_col not in test.columns:
        raise SystemExit(f"selection mask column does not exist: {selection_mask_col}")

    rows = []
    group_cols = ["date", "decision_target_timestamp"]
    for spec in variants:
        variant = str(spec["variant"])
        risk_model = str(spec.get("risk_model", "") or "").lower()
        penalty = float(spec.get("penalty", 0.0) or 0.0)
        candidate_min = float(spec.get("candidate_alpha_rank_min", 0.0) or 0.0)
        if risk_model == "gap":
            risk_col = "gap_risk_rank"
        elif risk_model == "binary":
            risk_col = "binary_risk_rank"
        else:
            risk_col = ""

        for (date, timestamp), full_group in test.groupby(group_cols, sort=True):
            alpha_candidate_mask = full_group["candidate_alpha_rank"].ge(candidate_min)
            alpha_candidates = full_group.loc[alpha_candidate_mask]
            if selection_mask_col:
                pool_mask = full_group[selection_mask_col].astype(bool)
                group = full_group.loc[alpha_candidate_mask & pool_mask].copy()
                stock_pool_candidate_rows = int(pool_mask.sum())
            else:
                group = alpha_candidates.copy()
                stock_pool_candidate_rows = float("nan")
            if len(group):
                risk_values = group[risk_col] if risk_col else 0.0
                group["final_score"] = group["candidate_alpha_rank"] - penalty * risk_values
                selected = group.sort_values("final_score", ascending=False).head(top_n)
            else:
                selected = group
            short_stats = selection_return_stats(
                full_group,
                selected,
                label_col="label",
                prefix="short",
            )
            next_stats = selection_return_stats(
                full_group,
                selected,
                label_col="alpha_return_next_close",
                prefix="next",
            )
            rows.append(
                {
                    "test_month": month,
                    "variant": variant,
                    "risk_model": risk_model,
                    "penalty": penalty,
                    "candidate_alpha_rank_min": candidate_min,
                    "date": str(date),
                    "decision_target_timestamp": pd.Timestamp(timestamp),
                    "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                    "rows": int(len(full_group)),
                    "alpha_candidate_rows": int(len(alpha_candidates)),
                    "stock_pool_candidate_rows": stock_pool_candidate_rows,
                    "candidate_rows": int(len(group)),
                    "selected_rows": int(len(selected)),
                    "selected_stock_pool_rows": (
                        int(selected[selection_mask_col].astype(bool).sum())
                        if selection_mask_col and len(selected)
                        else float("nan")
                    ),
                    "short_top_mean_bps": short_stats["short_top_mean_bps"],
                    "short_top_excess_bps": short_stats["short_top_excess_bps"],
                    "next_top_mean_bps": next_stats["next_top_mean_bps"],
                    "next_top_excess_bps": next_stats["next_top_excess_bps"],
                    "selected_gap_risk_rank": (
                        float(selected["gap_risk_rank"].mean())
                        if len(selected) and "gap_risk_rank" in selected
                        else float("nan")
                    ),
                    "selected_binary_risk_rank": (
                        float(selected["binary_risk_rank"].mean())
                        if len(selected) and "binary_risk_rank" in selected
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_group_metrics(group_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_metrics = group_metrics.copy()
    optional_defaults = {
        "alpha_candidate_rows": np.nan,
        "stock_pool_candidate_rows": np.nan,
        "selected_stock_pool_rows": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in group_metrics.columns:
            group_metrics[column] = default

    keys = ["variant", "risk_model", "penalty", "candidate_alpha_rank_min"]
    month_summary = (
        group_metrics.groupby(["test_month", *keys], as_index=False)
        .agg(
            groups=("date", "size"),
            alpha_candidate_rows=("alpha_candidate_rows", "mean"),
            stock_pool_candidate_rows=("stock_pool_candidate_rows", "mean"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            selected_stock_pool_rows=("selected_stock_pool_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_excess_positive_rate=("next_top_excess_bps", positive_rate),
            selected_gap_risk_rank=("selected_gap_risk_rank", "mean"),
            selected_binary_risk_rank=("selected_binary_risk_rank", "mean"),
        )
        .sort_values(["test_month", "next_top_excess_bps"], ascending=[True, False])
    )
    minute_positive = (
        group_metrics.groupby([*keys, "clock"])["next_top_excess_bps"]
        .mean()
        .reset_index()
        .groupby(keys)["next_top_excess_bps"]
        .apply(positive_count)
        .reset_index(name="next_positive_minute_count")
    )
    monthly_positive = (
        month_summary.groupby(keys)["next_top_excess_bps"]
        .apply(positive_count)
        .reset_index(name="next_positive_month_count")
    )
    summary = (
        group_metrics.groupby(keys, as_index=False)
        .agg(
            groups=("date", "size"),
            months=("test_month", "nunique"),
            alpha_candidate_rows=("alpha_candidate_rows", "mean"),
            stock_pool_candidate_rows=("stock_pool_candidate_rows", "mean"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            selected_stock_pool_rows=("selected_stock_pool_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_excess_positive_rate=("next_top_excess_bps", positive_rate),
            selected_gap_risk_rank=("selected_gap_risk_rank", "mean"),
            selected_binary_risk_rank=("selected_binary_risk_rank", "mean"),
        )
        .merge(minute_positive, on=keys, how="left")
        .merge(monthly_positive, on=keys, how="left")
        .sort_values(["next_top_excess_bps", "short_top_excess_bps"], ascending=[False, False])
    )
    return month_summary, summary
