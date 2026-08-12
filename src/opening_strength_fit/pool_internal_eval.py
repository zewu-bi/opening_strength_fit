from __future__ import annotations

from functools import partial

import pandas as pd

from opening_strength_fit.model_metrics import corr
from opening_strength_fit.prediction_frames import clock_label

GROUP_COLS = ("date", "decision_target_timestamp")
POOL_INTERNAL_ROLLING_COLUMNS = (
    "pool_short_mean_bps",
    "selected_short_mean_bps",
    "short_internal_excess_bps",
    "pool_next_mean_bps",
    "selected_next_mean_bps",
    "next_internal_excess_bps",
    "short_rank_ic",
    "next_rank_ic",
)
POOL_INTERNAL_MEAN_AGGREGATIONS = {
    column: (column, "mean")
    for column in ("candidate_rows", "selected_rows", *POOL_INTERNAL_ROLLING_COLUMNS)
}


def _positive_period_aggregations(period: str) -> dict[str, tuple[str, object]]:
    return {
        f"{horizon}_positive_{period}": (
            f"{horizon}_internal_excess_bps",
            lambda value: int((value > 0).sum()),
        )
        for horizon in ("short", "next")
    }


POSITIVE_MONTH_AGGREGATIONS = _positive_period_aggregations("months")


def evaluate_pool(
    frame: pd.DataFrame,
    *,
    pool_name: str,
    score_col: str,
    short_label_col: str,
    next_label_col: str,
    top_n: int,
) -> pd.DataFrame:
    work = frame.dropna(subset=[score_col, short_label_col, next_label_col]).copy()
    if work.empty:
        return pd.DataFrame()
    work["_score_rank"] = work.groupby(list(GROUP_COLS), sort=False)[score_col].rank(
        ascending=False,
        method="first",
    )
    work["_selected"] = work["_score_rank"].le(top_n)

    group = work.groupby(list(GROUP_COLS), sort=False)
    base = group.agg(
        candidate_rows=(score_col, "size"),
        pool_short_mean=(short_label_col, "mean"),
        pool_next_mean=(next_label_col, "mean"),
    )
    selected = (
        work.loc[work["_selected"]]
        .groupby(list(GROUP_COLS), sort=False)
        .agg(
            selected_rows=(score_col, "size"),
            selected_short_mean=(short_label_col, "mean"),
            selected_next_mean=(next_label_col, "mean"),
        )
    )
    rank_ic = group.apply(
        lambda item: pd.Series(
            {
                "short_rank_ic": corr(item[score_col], item[short_label_col], "spearman"),
                "next_rank_ic": corr(item[score_col], item[next_label_col], "spearman"),
            }
        )
    )
    metrics = base.join(selected, how="left").join(rank_ic, how="left").reset_index()
    metrics["pool"] = pool_name
    metrics["test_month"] = pd.to_datetime(metrics["date"]).dt.to_period("M").astype(str)
    metrics["clock"] = clock_label(metrics["decision_target_timestamp"])
    metrics["short_internal_excess_bps"] = (
        metrics["selected_short_mean"] - metrics["pool_short_mean"]
    ) * 10_000.0
    metrics["next_internal_excess_bps"] = (
        metrics["selected_next_mean"] - metrics["pool_next_mean"]
    ) * 10_000.0
    for column in (
        "pool_short_mean",
        "selected_short_mean",
        "pool_next_mean",
        "selected_next_mean",
    ):
        metrics[f"{column}_bps"] = metrics[column] * 10_000.0
    return metrics[
        [
            "pool",
            "test_month",
            "date",
            "decision_target_timestamp",
            "clock",
            *POOL_INTERNAL_MEAN_AGGREGATIONS,
        ]
    ]


def summarize_groups(
    group_metrics: pd.DataFrame,
    by: list[str],
    *,
    month_col: str = "test_month",
) -> pd.DataFrame:
    if group_metrics.empty:
        return pd.DataFrame()
    out = (
        group_metrics.groupby(by, sort=False)
        .agg(
            groups=("short_internal_excess_bps", "size"),
            months=(month_col, "nunique"),
            **POOL_INTERNAL_MEAN_AGGREGATIONS,
        )
        .reset_index()
    )
    return out


def _positive_period_summary(summary: pd.DataFrame, period: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby("pool", sort=False)
        .agg(**_positive_period_aggregations(period))
        .reset_index()
    )


positive_month_summary = partial(_positive_period_summary, period="months")
positive_clock_summary = partial(_positive_period_summary, period="clocks")


def halfyear_summary(month_summary: pd.DataFrame) -> pd.DataFrame:
    if month_summary.empty:
        return pd.DataFrame()
    frame = month_summary.copy()
    frame["year"] = frame["test_month"].astype(str).str.slice(0, 4).astype(int)
    month_num = frame["test_month"].astype(str).str.slice(5, 7).astype(int)
    frame["half"] = month_num.map(lambda value: "H1" if value <= 6 else "H2")
    return (
        frame.groupby(["pool", "year", "half"], sort=False)
        .agg(
            months=("test_month", "nunique"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
            **POSITIVE_MONTH_AGGREGATIONS,
        )
        .reset_index()
    )


def year_summary(month_summary: pd.DataFrame) -> pd.DataFrame:
    if month_summary.empty:
        return pd.DataFrame()
    frame = month_summary.copy()
    frame["year"] = frame["test_month"].astype(str).str.slice(0, 4).astype(int)
    return (
        frame.groupby(["pool", "year"], sort=False)
        .agg(
            months=("test_month", "nunique"),
            **POOL_INTERNAL_MEAN_AGGREGATIONS,
            **POSITIVE_MONTH_AGGREGATIONS,
        )
        .reset_index()
    )


def quarter_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    if group_metrics.empty:
        return pd.DataFrame()
    frame = group_metrics.copy()
    frame["_source_month"] = frame["test_month"]
    frame["test_month"] = (
        pd.to_datetime(frame["date"], errors="coerce").dt.to_period("Q").astype(str)
    )
    return summarize_groups(frame, ["pool", "test_month"], month_col="_source_month")
