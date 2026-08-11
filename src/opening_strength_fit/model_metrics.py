from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.evaluation import resolve_group_cols


def corr(a: pd.Series, b: pd.Series, method: str, *, min_count: int = 2) -> float:
    values = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if (
        len(values) < min_count
        or values["a"].nunique(dropna=True) < 2
        or values["b"].nunique(dropna=True) < 2
    ):
        return float("nan")
    return float(values["a"].corr(values["b"], method=method))


def array_corr(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    method: str = "pearson",
) -> float:
    if len(a) != len(b):
        raise ValueError("correlation inputs must have the same length")
    return corr(pd.Series(a, copy=False), pd.Series(b, copy=False), method, min_count=3)


def ir(mean: float, std: float) -> float:
    if pd.isna(std) or std == 0:
        return float("nan")
    return mean / std


def daily_prediction_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    return (
        frame.groupby("date")
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "ic": corr(group[label_col], group[score_col], "pearson"),
                    "rank_ic": corr(group[label_col], group[score_col], "spearman"),
                    "mean_label": float(group[label_col].mean()),
                    "win_rate": float((group[label_col] > 0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def grouped_prediction_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date",),
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    available_group_cols = resolve_group_cols(frame, group_cols)
    if not available_group_cols:
        return pd.DataFrame(
            [
                {
                    "rows": len(frame),
                    "ic": corr(frame[label_col], frame[score_col], "pearson"),
                    "rank_ic": corr(frame[label_col], frame[score_col], "spearman"),
                    "mean_label": (float(frame[label_col].mean()) if len(frame) else float("nan")),
                    "win_rate": (
                        float((frame[label_col] > 0).mean()) if len(frame) else float("nan")
                    ),
                }
            ]
        )
    return (
        frame.groupby(list(available_group_cols))
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "ic": corr(group[label_col], group[score_col], "pearson"),
                    "rank_ic": corr(group[label_col], group[score_col], "spearman"),
                    "mean_label": float(group[label_col].mean()),
                    "win_rate": float((group[label_col] > 0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def evaluate_prediction_frame(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date",),
) -> dict[str, object]:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    resolved_group_cols = resolve_group_cols(frame, group_cols)
    daily = daily_prediction_metrics(frame, label_col=label_col, score_col=score_col)
    grouped = grouped_prediction_metrics(
        frame,
        label_col=label_col,
        score_col=score_col,
        group_cols=resolved_group_cols,
    )
    ic_mean = float(daily["ic"].mean())
    ic_std = float(daily["ic"].std())
    rank_ic_mean = float(daily["rank_ic"].mean())
    rank_ic_std = float(daily["rank_ic"].std())
    group_ic_mean = float(grouped["ic"].mean())
    group_ic_std = float(grouped["ic"].std())
    group_rank_ic_mean = float(grouped["rank_ic"].mean())
    group_rank_ic_std = float(grouped["rank_ic"].std())
    ic_grouping = ",".join(resolved_group_cols) if resolved_group_cols else "global"
    sample_grain = (
        "date x symbol x decision_time"
        if "decision_time" in frame.columns
        else "date x symbol x opening_tick"
    )
    return {
        "rows": len(frame),
        "dates": int(frame["date"].nunique()) if "date" in frame.columns else 0,
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "sample_grain": sample_grain,
        "ic_grouping": ic_grouping,
        "ic_groups": int(len(grouped)),
        "overall_ic": corr(frame[label_col], frame[score_col], "pearson"),
        "overall_rank_ic": corr(frame[label_col], frame[score_col], "spearman"),
        "group_ic_mean": group_ic_mean,
        "group_ic_std": group_ic_std,
        "group_ic_ir": ir(group_ic_mean, group_ic_std),
        "group_rank_ic_mean": group_rank_ic_mean,
        "group_rank_ic_std": group_rank_ic_std,
        "group_rank_ic_ir": ir(group_rank_ic_mean, group_rank_ic_std),
        "daily_ic_mean": ic_mean,
        "daily_ic_std": ic_std,
        "daily_ic_ir": ir(ic_mean, ic_std),
        "daily_rank_ic_mean": rank_ic_mean,
        "daily_rank_ic_std": rank_ic_std,
        "daily_rank_ic_ir": ir(rank_ic_mean, rank_ic_std),
        "mean_label": float(frame[label_col].mean()) if len(frame) else float("nan"),
        "win_rate": float((frame[label_col] > 0).mean()) if len(frame) else float("nan"),
    }
