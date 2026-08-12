from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    mean_aggregations,
    positive_count,
    positive_rate,
    selection_group_metrics,
)

GAP_P80_VARIANTS = tuple(
    {
        "variant": f"gap_penalty_{int(penalty * 100):03d}_p80",
        "risk_model": "gap",
        "penalty": penalty,
        "candidate_alpha_rank_min": 0.80,
    }
    for penalty in (0.30, 0.35)
)


def configured_score_variants(
    config: dict,
    section: str,
    defaults: tuple[dict[str, object], ...],
    *,
    default_risk_model: str = "",
) -> list[dict[str, object]]:
    configured = config.get(section, {}).get("variants", [])
    if not configured:
        return [dict(item) for item in defaults]
    return [
        {
            "variant": str(item.get("variant", "")).strip(),
            "risk_model": str(item.get("risk_model", default_risk_model) or "").strip().lower(),
            "penalty": float(item.get("penalty", 0.0) or 0.0),
            "candidate_alpha_rank_min": float(item.get("candidate_alpha_rank_min", 0.0) or 0.0),
        }
        for item in configured
        if str(item.get("variant", "")).strip()
    ]


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
            rows.append(
                {
                    "test_month": month,
                    "variant": variant,
                    "risk_model": risk_model,
                    "penalty": penalty,
                    "candidate_alpha_rank_min": candidate_min,
                    **selection_group_metrics(
                        full_group,
                        selected,
                        date=date,
                        timestamp=timestamp,
                        candidate_counts={
                            "alpha_candidate_rows": int(len(alpha_candidates)),
                            "stock_pool_candidate_rows": stock_pool_candidate_rows,
                            "candidate_rows": int(len(group)),
                        },
                        selection_counts={
                            "selected_stock_pool_rows": (
                                int(selected[selection_mask_col].astype(bool).sum())
                                if selection_mask_col and len(selected)
                                else float("nan")
                            )
                        },
                        include_all_mean=False,
                        include_win_rate=False,
                    ),
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
            **mean_aggregations(
                *"alpha_candidate_rows stock_pool_candidate_rows candidate_rows selected_rows "
                "selected_stock_pool_rows short_top_mean_bps short_top_excess_bps "
                "next_top_mean_bps next_top_excess_bps".split()
            ),
            next_excess_positive_rate=("next_top_excess_bps", positive_rate),
            **mean_aggregations("selected_gap_risk_rank", "selected_binary_risk_rank"),
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
            **mean_aggregations(
                *"alpha_candidate_rows stock_pool_candidate_rows candidate_rows selected_rows "
                "selected_stock_pool_rows short_top_mean_bps short_top_excess_bps "
                "next_top_mean_bps next_top_excess_bps".split()
            ),
            next_excess_positive_rate=("next_top_excess_bps", positive_rate),
            **mean_aggregations("selected_gap_risk_rank", "selected_binary_risk_rank"),
        )
        .merge(minute_positive, on=keys, how="left")
        .merge(monthly_positive, on=keys, how="left")
        .sort_values(["next_top_excess_bps", "short_top_excess_bps"], ascending=[False, False])
    )
    return month_summary, summary
