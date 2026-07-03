from __future__ import annotations

import re

import numpy as np
import pandas as pd

from opening_strength_fit.model_types import LEAKY_PREFIXES, NON_FEATURE_COLUMNS


def _match_patterns(column: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, column) for pattern in patterns)


def _match_prefixes(column: str, prefixes: tuple[str, ...]) -> bool:
    return bool(prefixes) and column.startswith(prefixes)


def feature_columns(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    include_columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    drop_columns: tuple[str, ...] = (),
    drop_prefixes: tuple[str, ...] = (),
    drop_patterns: tuple[str, ...] = (),
) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns
    include_columns_set = set(include_columns)
    drop_columns_set = set(drop_columns)
    has_include_filter = bool(include_columns_set or include_prefixes or include_patterns)
    features = []
    for column in numeric_columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in LEAKY_PREFIXES):
            continue
        if column in drop_columns_set:
            continue
        if _match_prefixes(str(column), drop_prefixes):
            continue
        if drop_patterns and _match_patterns(str(column), drop_patterns):
            continue
        if has_include_filter and not (
            column in include_columns_set
            or _match_prefixes(str(column), include_prefixes)
            or _match_patterns(str(column), include_patterns)
        ):
            continue
        features.append(str(column))
    if limit is not None:
        features = features[:limit]
    return features


def _clean_xy(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise SystemExit(f"missing model target column: {target_col}")
    target = pd.to_numeric(df[target_col], errors="coerce")
    mask = target.notna() & np.isfinite(target)
    if "valid_label" in df.columns:
        mask &= df["valid_label"].fillna(False).astype(bool)
    if not bool(mask.any()):
        raise SystemExit("empty labeled frame after filtering valid labels")
    x = df.loc[mask, features].replace([np.inf, -np.inf], np.nan)
    y = target.loc[mask].astype("float64")
    return x, y
