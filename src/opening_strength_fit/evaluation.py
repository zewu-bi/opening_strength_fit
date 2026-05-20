from __future__ import annotations

import numpy as np
import pandas as pd


SELECTION_GROUPS = {
    "global": (),
    "daily": ("date",),
    "symbol_day": ("date", "symbol"),
    "cross_section": ("date", "decision_target_timestamp"),
}

GROUP_COLUMN_FALLBACKS = {
    "decision_target_timestamp": ("decision_time", "timestamp"),
}


def group_cols_for_mode(mode: str) -> tuple[str, ...]:
    try:
        return SELECTION_GROUPS[mode]
    except KeyError as exc:
        valid = ", ".join(sorted(SELECTION_GROUPS))
        raise SystemExit(f"unknown selection mode {mode!r}; expected one of: {valid}") from exc


def resolve_group_cols(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> tuple[str, ...]:
    resolved = []
    for column in group_cols:
        if column in frame.columns:
            resolved.append(column)
            continue
        fallback = next(
            (
                candidate
                for candidate in GROUP_COLUMN_FALLBACKS.get(column, ())
                if candidate in frame.columns
            ),
            None,
        )
        if fallback is not None:
            resolved.append(fallback)
    return tuple(dict.fromkeys(resolved))


def format_group_cols(group_cols: tuple[str, ...]) -> str:
    return ",".join(group_cols) if group_cols else "global"


def score_bucket_returns(
    df: pd.DataFrame,
    *,
    bins: int = 5,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date",),
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "rows", "mean_label", "win_rate"])

    resolved_group_cols = resolve_group_cols(frame, group_cols)
    if resolved_group_cols:
        rank_pct = frame.groupby(list(resolved_group_cols))[score_col].rank(
            method="first",
            pct=True,
        )
    else:
        rank_pct = frame[score_col].rank(method="first", pct=True)
    frame["bucket"] = np.ceil(rank_pct * bins).clip(1, bins).astype(int)
    return (
        frame.groupby("bucket")
        .agg(
            rows=(label_col, "size"),
            mean_label=(label_col, "mean"),
            median_label=(label_col, "median"),
            win_rate=(label_col, lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )


def top_score_trades(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date", "symbol"),
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    if frame.empty:
        return frame

    if not group_cols:
        return (
            frame.sort_values(score_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    resolved_group_cols = resolve_group_cols(frame, group_cols)
    if not resolved_group_cols:
        return (
            frame.sort_values(score_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    selected = []
    for _, group in frame.groupby(list(resolved_group_cols), sort=False):
        selected.append(group.sort_values(score_col, ascending=False).head(top_n))
    if not selected:
        return frame.iloc[:0].copy()
    return pd.concat(selected, ignore_index=True)


def summarize_trades(
    trades: pd.DataFrame,
    *,
    label_col: str = "label",
    group_cols: tuple[str, ...] = ("date", "symbol"),
) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "groups": 0,
            "group_cols": format_group_cols(group_cols),
            "mean_return": float("nan"),
            "median_return": float("nan"),
            "win_rate": float("nan"),
            "return_std": float("nan"),
        }
    resolved_group_cols = resolve_group_cols(trades, group_cols)
    group_count = 0
    if resolved_group_cols:
        group_count = int(trades.groupby(list(resolved_group_cols)).ngroups)
    elif not group_cols:
        group_count = 1
    return {
        "trades": len(trades),
        "groups": group_count,
        "group_cols": format_group_cols(resolved_group_cols),
        "mean_return": float(trades[label_col].mean()),
        "median_return": float(trades[label_col].median()),
        "win_rate": float((trades[label_col] > 0).mean()),
        "return_std": float(trades[label_col].std()),
    }
