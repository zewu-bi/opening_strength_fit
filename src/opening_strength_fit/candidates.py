from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import pandas as pd


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise SystemExit(f"candidate filter missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _float_map(values: Mapping[str, object] | None) -> dict[str, float]:
    if not values:
        return {}
    return {
        str(column): float(value)
        for column, value in values.items()
        if value not in (None, "")
    }


def _resolve_group_cols(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
) -> list[str]:
    resolved = []
    for column in group_cols:
        if column in frame.columns:
            resolved.append(column)
        elif column == "decision_target_timestamp":
            for fallback in ("decision_time", "timestamp"):
                if fallback in frame.columns:
                    resolved.append(fallback)
                    break
    return list(dict.fromkeys(resolved))


def filter_opening_candidates(
    frame: pd.DataFrame,
    *,
    min_values: Mapping[str, object] | None = None,
    max_values: Mapping[str, object] | None = None,
    rank_min_values: Mapping[str, object] | None = None,
    rank_max_values: Mapping[str, object] | None = None,
    rank_group_cols: Sequence[str] = ("date", "decision_target_timestamp"),
    rank_method: str = "first",
) -> pd.DataFrame:
    """Keep opening candidate rows using only same-tick or past features."""

    mask = opening_candidate_mask(
        frame,
        min_values=min_values,
        max_values=max_values,
        rank_min_values=rank_min_values,
        rank_max_values=rank_max_values,
        rank_group_cols=rank_group_cols,
        rank_method=rank_method,
    )
    return frame.loc[mask].copy()


def opening_candidate_mask(
    frame: pd.DataFrame,
    *,
    min_values: Mapping[str, object] | None = None,
    max_values: Mapping[str, object] | None = None,
    rank_min_values: Mapping[str, object] | None = None,
    rank_max_values: Mapping[str, object] | None = None,
    rank_group_cols: Sequence[str] = ("date", "decision_target_timestamp"),
    rank_method: str = "first",
) -> pd.Series:
    """Return a boolean visible-information candidate mask."""

    if frame.empty:
        return pd.Series(True, index=frame.index)

    mask = pd.Series(True, index=frame.index)
    for column, threshold in _float_map(min_values).items():
        values = _numeric_column(frame, column)
        mask &= values.notna() & values.ge(threshold)

    for column, threshold in _float_map(max_values).items():
        values = _numeric_column(frame, column)
        mask &= values.notna() & values.le(threshold)

    rank_min = _float_map(rank_min_values)
    rank_max = _float_map(rank_max_values)
    if rank_min or rank_max:
        group_cols = _resolve_group_cols(frame, rank_group_cols)
        if not group_cols:
            raise SystemExit(
                "candidate rank filters need at least one available group column"
            )
        allowed_rank_methods = {"average", "min", "max", "first", "dense"}
        if rank_method not in allowed_rank_methods:
            allowed = ", ".join(sorted(allowed_rank_methods))
            raise SystemExit(f"candidate rank_method must be one of: {allowed}")
        rank_cache: dict[str, pd.Series] = {}
        for column in sorted(set(rank_min) | set(rank_max)):
            values = _numeric_column(frame, column)
            rank_cache[column] = values.groupby([frame[col] for col in group_cols]).rank(
                method=rank_method,
                pct=True,
            )
        for column, threshold in rank_min.items():
            values = _numeric_column(frame, column)
            mask &= values.notna() & rank_cache[column].ge(threshold)
        for column, threshold in rank_max.items():
            values = _numeric_column(frame, column)
            mask &= values.notna() & rank_cache[column].le(threshold)

    return mask
