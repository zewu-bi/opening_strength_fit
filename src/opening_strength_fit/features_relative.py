from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.feature_utils import (
    _matching_columns,
    _numeric_series,
    safe_divide,
)


def _price_bucket_labels(edges: tuple[float, ...]) -> list[str]:
    if len(edges) == 2:
        return ["cheap", "mid", "expensive"]
    return [f"bucket{i}" for i in range(len(edges) + 1)]


def add_cross_sectional_relative_features(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    modes: tuple[str, ...] = ("zscore", "rank_centered"),
    prefix: str = "xs_rel_",
    rank_method: str = "average",
) -> pd.DataFrame:
    """Add within-cross-section relative versions of configured numeric columns."""

    available_group_cols = tuple(column for column in group_cols if column in frame.columns)
    if not available_group_cols:
        raise ValueError("cross-sectional relative features need at least one group column")

    source_columns = [
        column
        for column in _matching_columns(
            frame.columns,
            include_columns=columns,
            include_prefixes=include_prefixes,
            include_patterns=include_patterns,
        )
        if column not in available_group_cols
    ]
    source_columns = [
        column
        for column in source_columns
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column])
    ]
    if not source_columns:
        return frame.copy()

    normalized_modes = tuple(mode.strip().lower() for mode in modes if mode.strip())
    valid_modes = {"demean", "zscore", "rank_pct", "rank_centered", "rank"}
    unknown_modes = sorted(set(normalized_modes) - valid_modes)
    if unknown_modes:
        raise ValueError(f"unknown cross-sectional relative feature modes: {unknown_modes}")

    out = frame.copy()
    group_keys = [out[column] for column in available_group_cols]
    new_columns: dict[str, pd.Series] = {}
    for column in source_columns:
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        grouped = values.groupby(group_keys, sort=False)
        centered = None
        rank_pct = None
        for mode in normalized_modes:
            if mode in {"demean", "zscore"} and centered is None:
                centered = values - grouped.transform("mean")
            if mode == "demean":
                new_columns[f"{prefix}{column}_demean"] = centered
            elif mode == "zscore":
                std = grouped.transform("std")
                new_columns[f"{prefix}{column}_zscore"] = pd.Series(
                    safe_divide(centered, std),
                    index=out.index,
                )
            elif mode in {"rank_pct", "rank_centered", "rank"}:
                if rank_pct is None:
                    rank_pct = grouped.rank(method=rank_method, pct=True)
                if mode == "rank_pct":
                    new_columns[f"{prefix}{column}_rank_pct"] = rank_pct
                else:
                    new_columns[f"{prefix}{column}_rank_centered"] = rank_pct - 0.5

    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)


def add_price_scale_features(
    frame: pd.DataFrame,
    *,
    price_col: str = "ask_price_1",
    tick_size: float = 0.01,
    bucket_edges: tuple[float, ...] = (5.0, 20.0),
    interaction_columns: tuple[str, ...] = (),
    prefix: str = "price_scale_",
) -> pd.DataFrame:
    """Add price-regime and tick/bps scale features.

    These features make the A-share fixed 0.01 tick explicit, so low-price and
    high-price names do not have to share one raw orderbook interpretation.
    """

    if price_col not in frame.columns:
        return frame.copy()

    out = frame.copy()
    price = _numeric_series(out[price_col]).where(lambda s: s > 0.0)
    tick_size = float(tick_size)
    edges = tuple(sorted(float(edge) for edge in bucket_edges))
    labels = _price_bucket_labels(edges)

    bucket_code = pd.Series(np.nan, index=out.index, dtype="float64")
    valid = price.notna()
    if valid.any():
        bucket_code.loc[valid] = np.searchsorted(edges, price.loc[valid], side="right")

    new_columns: dict[str, pd.Series] = {
        f"{prefix}log_price": np.log(price),
        f"{prefix}tick_bps": safe_divide(tick_size, price) * 10_000.0,
        f"{prefix}bucket_code": bucket_code,
    }
    for code, label in enumerate(labels):
        new_columns[f"{prefix}bucket_{label}"] = bucket_code.eq(float(code)).astype("int8")

    if "spread_abs" in out.columns:
        new_columns[f"{prefix}spread_ticks"] = safe_divide(
            _numeric_series(out["spread_abs"]),
            tick_size,
        )
    elif {"ask_price_1", "bid_price_1"}.issubset(out.columns):
        new_columns[f"{prefix}spread_ticks"] = safe_divide(
            _numeric_series(out["ask_price_1"]) - _numeric_series(out["bid_price_1"]),
            tick_size,
        )

    if "ask_price_1" in out.columns:
        ask1 = _numeric_series(out["ask_price_1"])
        for level in range(2, 11):
            column = f"ask_price_{level}"
            if column in out.columns:
                new_columns[f"{prefix}ask_gap_{level}_ticks"] = safe_divide(
                    _numeric_series(out[column]) - ask1,
                    tick_size,
                )
    if "bid_price_1" in out.columns:
        bid1 = _numeric_series(out["bid_price_1"])
        for level in range(2, 11):
            column = f"bid_price_{level}"
            if column in out.columns:
                new_columns[f"{prefix}bid_gap_{level}_ticks"] = safe_divide(
                    bid1 - _numeric_series(out[column]),
                    tick_size,
                )

    for column in (
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_3",
        "bid_depth_3",
        "ask_depth_10",
        "bid_depth_10",
        "postopen_v2_ask_depth_3",
        "postopen_v2_bid_depth_3",
        "postopen_v2_ask_depth_10",
        "postopen_v2_bid_depth_10",
        "volume_diff_1t",
        "volume_diff_3t",
    ):
        if column in out.columns:
            new_columns[f"{prefix}{column}_notional"] = _numeric_series(out[column]) * price

    interaction_frame = pd.DataFrame(new_columns, index=out.index)
    for column in interaction_columns:
        if column not in out.columns:
            continue
        values = _numeric_series(out[column])
        for label in labels:
            bucket = interaction_frame[f"{prefix}bucket_{label}"].astype("float64")
            interaction_frame[f"{prefix}{column}_x_{label}"] = values * bucket

    return pd.concat([out, interaction_frame], axis=1)
