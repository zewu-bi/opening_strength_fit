from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.evaluation import summarize_trades, top_score_trades


def summarize_top_score_strategy(
    predictions: pd.DataFrame,
    *,
    top_n: int = 20,
    label_col: str = "label",
    score_col: str = "prediction",
) -> dict[str, object]:
    trades = top_score_trades(
        predictions,
        top_n=top_n,
        label_col=label_col,
        score_col=score_col,
    )
    return summarize_trades(trades, label_col=label_col)


def load_backtest_series(path: Path, name: str | None = None) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = frame.iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    if name is not None:
        series.name = name
    return series.dropna().sort_index()


def summarize_daily_series(series: pd.Series) -> dict[str, float | str]:
    cumulative = series.cumsum()
    drawdown = cumulative - cumulative.cummax()
    summary: dict[str, float | str] = {}
    if not cumulative.empty:
        summary["start_date"] = str(cumulative.index.min().date())
        summary["end_date"] = str(cumulative.index.max().date())
    summary.update(
        {
            "days": int(series.shape[0]),
            "daily_mean": float(series.mean()),
            "daily_std": float(series.std()),
            "cumulative_end": float(cumulative.iloc[-1]),
        }
    )
    if not cumulative.empty:
        summary["best_day"] = float(series.max())
        summary["worst_day"] = float(series.min())
        summary["max_drawdown"] = float(drawdown.min())
    return summary
