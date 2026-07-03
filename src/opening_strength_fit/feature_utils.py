from __future__ import annotations

import re

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    denominator = np.asarray(denominator, dtype="float64")
    numerator = np.asarray(numerator, dtype="float64")
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    out = np.full_like(numerator, np.nan, dtype="float64")
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=df.index)
    values = df[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    return values.sum(axis=1, min_count=1)


def _column_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return _numeric_series(df[column])


def _matching_columns(
    columns: pd.Index,
    *,
    include_columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
) -> list[str]:
    explicit = set(include_columns)
    compiled = [re.compile(pattern) for pattern in include_patterns]
    matched: list[str] = []
    for column in columns:
        name = str(column)
        if name in explicit:
            matched.append(name)
            continue
        if include_prefixes and name.startswith(include_prefixes):
            matched.append(name)
            continue
        if compiled and any(pattern.search(name) for pattern in compiled):
            matched.append(name)
    return list(dict.fromkeys(matched))


def _sum_present_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    return _sum_columns(df, present)


def _weighted_mean(
    values: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    weighted = values.astype("float64") * weights.astype("float64")
    return safe_divide(
        weighted.sum(axis=1, min_count=1),
        weights.sum(axis=1, min_count=1),
    )
