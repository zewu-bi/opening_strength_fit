from __future__ import annotations

import re

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
    valid_modes = {"demean", "zscore", "robust_zscore", "rank_pct", "rank_centered", "rank"}
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
            elif mode == "robust_zscore":
                median = grouped.transform("median")
                centered = values - median
                mad = centered.abs().groupby(group_keys, sort=False).transform("median")
                new_columns[f"{prefix}{column}_robust_zscore"] = pd.Series(
                    safe_divide(centered, mad * 1.4826),
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


def transform_cross_sectional_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    mode: str = "rank_centered",
    rank_method: str = "average",
) -> pd.DataFrame:
    """Replace configured feature values with within-cross-section transforms."""

    normalized_mode = str(mode).strip().lower().replace("-", "_")
    valid_modes = {"demean", "zscore", "robust_zscore", "rank_pct", "rank_centered", "rank"}
    if normalized_mode not in valid_modes:
        raise ValueError(f"unknown cross-sectional feature value transform: {mode!r}")

    available_group_cols = tuple(column for column in group_cols if column in frame.columns)
    if not available_group_cols:
        raise ValueError("cross-sectional feature value transform needs a group column")

    source_columns = [
        column
        for column in dict.fromkeys(str(column) for column in columns)
        if column in frame.columns and column not in available_group_cols
    ]
    source_columns = [
        column
        for column in source_columns
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column])
    ]
    if not source_columns:
        return frame.copy()

    out = frame.copy()
    group_keys = [out[column] for column in available_group_cols]
    for column in source_columns:
        values = (
            pd.to_numeric(out[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float64")
        )
        grouped = values.groupby(group_keys, sort=False)
        if normalized_mode == "demean":
            transformed = values - grouped.transform("mean")
        elif normalized_mode == "zscore":
            centered = values - grouped.transform("mean")
            transformed = pd.Series(
                safe_divide(centered, grouped.transform("std")),
                index=out.index,
            )
        elif normalized_mode == "robust_zscore":
            centered = values - grouped.transform("median")
            mad = centered.abs().groupby(group_keys, sort=False).transform("median")
            transformed = pd.Series(
                safe_divide(centered, mad * 1.4826),
                index=out.index,
            )
        elif normalized_mode == "rank":
            transformed = grouped.rank(method=rank_method, pct=False)
        else:
            rank_pct = grouped.rank(method=rank_method, pct=True)
            transformed = rank_pct if normalized_mode == "rank_pct" else rank_pct - 0.5
        out[column] = transformed.astype("float32")
    return out


_RAW_PRICE_COLUMNS = {
    "ask_price_1",
    "bid_price_1",
    "mid_price",
    "avg_ask_price",
    "avg_bid_price",
    "preopen_last_price",
    "preopen_price_min",
    "preopen_price_max",
}

_COUNT_COLUMNS = {
    "trade_num",
    "total_ask_count",
    "total_bid_count",
}

_DIMENSIONLESS_NAME_MARKERS = (
    "_bps",
    "bps_",
    "return",
    "imbalance",
    "ratio",
    "_rel_",
    "_from_open_rel",
    "_share_",
    "share_depth",
    "concentration",
    "position",
    "_flag",
    "_zscore",
    "_rank",
    "_slope",
    "replenish_vs_trade",
    "to_ask_depth10",
    "to_bid_depth10",
    "to_ask1",
    "to_bid1",
    "turnover_to_depth_notional",
)


def mechanismized_feature_value_reference_columns() -> tuple[str, ...]:
    """Columns used as references by mechanismized in-place value transforms."""

    base_columns = (
        "ask_price_1",
        "bid_price_1",
        "mid_price",
        "avg_ask_price",
        "avg_bid_price",
        "ask_depth_10",
        "bid_depth_10",
        "prev_close",
        "open_price",
        "preopen_last_price",
    )
    historical_ratio_bases = (
        "volume_diff_1t",
        "volume_diff_3t",
        "turnover_diff_1t",
        "turnover_diff_3t",
        "ask_depth_10",
        "bid_depth_10",
        "spread_bps",
        "trade_vwap_1t",
        "return_vs_open",
        "return_vs_prev_close",
        "postopen_v2_trade_turnover_to_depth_notional_1t",
        "postopen_v2_spread_compression_1m",
    )
    historical_ratio_columns = tuple(
        f"hist_surprise_{column}_{window}d_ratio"
        for column in historical_ratio_bases
        for window in (20, 60)
    )
    return (*base_columns, *historical_ratio_columns)


def _clean_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float64")
    )


def _positive(values: pd.Series) -> pd.Series:
    return values.where(values > 0.0)


def _signed_log1p(values) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    return np.sign(series) * np.log1p(np.abs(series))


def _log_positive_ratio(values) -> pd.Series:
    ratio = _positive(pd.Series(values, dtype="float64"))
    return np.log(ratio)


def _reference_mid(frame: pd.DataFrame) -> pd.Series:
    mid = _clean_values(frame, "mid_price")
    if mid.notna().any():
        return _positive(mid)
    ask1 = _clean_values(frame, "ask_price_1")
    bid1 = _clean_values(frame, "bid_price_1")
    fallback = (ask1 + bid1) / 2.0
    return _positive(fallback)


def _reference_price(frame: pd.DataFrame) -> pd.Series:
    for column in ("ask_price_1", "mid_price", "preopen_last_price", "avg_ask_price"):
        values = _positive(_clean_values(frame, column))
        if values.notna().any():
            return values
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _side_depth(frame: pd.DataFrame, column: str) -> pd.Series:
    if column.startswith(("ask_", "postopen_ask_", "postopen_v2_ask_", "total_ask_")):
        return _positive(_clean_values(frame, "ask_depth_10"))
    if column.startswith(("bid_", "postopen_bid_", "postopen_v2_bid_", "total_bid_")):
        return _positive(_clean_values(frame, "bid_depth_10"))
    ask_depth = _clean_values(frame, "ask_depth_10")
    bid_depth = _clean_values(frame, "bid_depth_10")
    return _positive((ask_depth + bid_depth) / 2.0)


def _historical_ratio(frame: pd.DataFrame, column: str) -> pd.Series | None:
    for window in (20, 60):
        ratio_column = f"hist_surprise_{column}_{window}d_ratio"
        if ratio_column not in frame.columns:
            continue
        ratio = _clean_values(frame, ratio_column)
        if ratio.notna().any():
            return _log_positive_ratio(ratio)
    return None


def _is_historical_ratio_name(name: str) -> bool:
    return name.startswith("hist_surprise_") and name.endswith("_ratio")


def _is_dimensionless_name(name: str) -> bool:
    return any(marker in name for marker in _DIMENSIONLESS_NAME_MARKERS)


def _is_count_name(name: str) -> bool:
    return (
        name in _COUNT_COLUMNS
        or name.startswith(("ask_count_", "bid_count_"))
        or name.startswith(("postopen_ask_count_", "postopen_bid_count_"))
        or name.startswith(("postopen_v2_ask_count_", "postopen_v2_bid_count_"))
    )


def _is_share_volume_name(name: str) -> bool:
    if name.startswith(
        (
            "ask_volume_",
            "bid_volume_",
            "ask_depth_",
            "bid_depth_",
            "volume_diff_",
            "postopen_volume_",
            "postopen_ask_volume_",
            "postopen_bid_volume_",
            "postopen_ask_depth_",
            "postopen_bid_depth_",
            "postopen_v2_volume_",
            "postopen_v2_ask_volume_",
            "postopen_v2_bid_volume_",
            "postopen_v2_ask_depth_",
            "postopen_v2_bid_depth_",
            "path_shape_bid_depth_",
            "path_shape_ask_depth_",
            "path_shape_depth_",
        )
    ):
        return True
    return name in {
        "preopen_volume",
        "total_ask_volume",
        "total_bid_volume",
    }


def _is_trade_share_volume_name(name: str) -> bool:
    return (
        name == "preopen_volume"
        or name.startswith(("volume_diff_", "postopen_volume_", "postopen_v2_volume_"))
    )


def _is_book_depth_name(name: str) -> bool:
    return (
        name in {"total_ask_volume", "total_bid_volume"}
        or name.startswith(("ask_depth_", "bid_depth_"))
        or name.startswith(("postopen_ask_depth_", "postopen_bid_depth_"))
        or name.startswith(("postopen_v2_ask_depth_", "postopen_v2_bid_depth_"))
        or name.startswith(("path_shape_bid_depth_", "path_shape_ask_depth_", "path_shape_depth_"))
    )


def _is_book_level_share_name(name: str) -> bool:
    return (
        name.startswith(("ask_volume_", "bid_volume_"))
        or name.startswith(("postopen_ask_volume_", "postopen_bid_volume_"))
        or name.startswith(("postopen_v2_ask_volume_", "postopen_v2_bid_volume_"))
    )


def _is_notional_name(name: str) -> bool:
    return (
        name == "preopen_turnover"
        or name.startswith(("turnover_diff_", "postopen_turnover_", "postopen_v2_turnover_"))
    )


def _is_price_diff_name(name: str) -> bool:
    if "_rel_" in name or name.endswith("_rel") or "_bps" in name:
        return False
    return bool(
        re.search(
            r"(?:^|_)(?:ask_price_1|bid_price_1|mid_price|price)(?:_from_open)?_diff",
            name,
        )
    )


def _is_raw_price_name(name: str) -> bool:
    if name in _RAW_PRICE_COLUMNS:
        return True
    if name.startswith("trade_vwap_"):
        return True
    if _is_price_diff_name(name):
        return True
    if "price" in name and not _is_dimensionless_name(name):
        return True
    return False


def _mechanismized_feature_series(
    frame: pd.DataFrame,
    column: str,
    *,
    tick_size: float,
) -> pd.Series:
    name = str(column).lower()
    values = _clean_values(frame, column)
    mid = _reference_mid(frame)
    price = _reference_price(frame)

    if name == "exch_time_offset_us":
        return values / 1_000_000.0

    if name == "spread_abs":
        return pd.Series(safe_divide(values, float(tick_size)), index=frame.index)

    if _is_count_name(name):
        return np.log1p(values.clip(lower=0.0))

    if _is_dimensionless_name(name):
        return values

    if _is_raw_price_name(name):
        if name == "mid_price":
            return pd.Series(safe_divide(float(tick_size), _positive(values)) * 10_000.0)
        if name == "ask_price_1":
            return pd.Series(safe_divide(values - mid, mid) * 10_000.0, index=frame.index)
        if name == "bid_price_1":
            return pd.Series(safe_divide(mid - values, mid) * 10_000.0, index=frame.index)
        if name.startswith("trade_vwap_"):
            return pd.Series(safe_divide(values - price, price) * 10_000.0, index=frame.index)
        if _is_price_diff_name(name):
            return pd.Series(safe_divide(values, price) * 10_000.0, index=frame.index)
        return pd.Series(safe_divide(values - mid, mid) * 10_000.0, index=frame.index)

    if _is_share_volume_name(name):
        base_depth = _side_depth(frame, name)
        if (
            name.startswith(("volume_diff_", "postopen_volume_", "postopen_v2_volume_"))
            or name == "preopen_volume"
        ):
            notional = values * price
            return _signed_log1p(notional)
        return pd.Series(safe_divide(values, base_depth), index=frame.index)

    if _is_notional_name(name):
        return _signed_log1p(values)

    return values


def _mechanismized_v2_feature_series(
    frame: pd.DataFrame,
    column: str,
    *,
    tick_size: float,
) -> pd.Series:
    name = str(column).lower()
    values = _clean_values(frame, column)
    mid = _reference_mid(frame)
    price = _reference_price(frame)

    if name == "exch_time_offset_us":
        return values / 1_000_000.0

    if _is_historical_ratio_name(name):
        return _log_positive_ratio(values)

    if name == "spread_abs":
        return pd.Series(safe_divide(values, float(tick_size)), index=frame.index)

    if _is_count_name(name):
        ratio = _historical_ratio(frame, column)
        return ratio if ratio is not None else np.log1p(values.clip(lower=0.0))

    if _is_raw_price_name(name):
        if name == "mid_price":
            return pd.Series(safe_divide(float(tick_size), _positive(values)) * 10_000.0)
        if name == "ask_price_1":
            return pd.Series(safe_divide(values - mid, mid) * 10_000.0, index=frame.index)
        if name == "bid_price_1":
            return pd.Series(safe_divide(mid - values, mid) * 10_000.0, index=frame.index)
        if name.startswith("trade_vwap_"):
            return pd.Series(safe_divide(values - price, price) * 10_000.0, index=frame.index)
        if _is_price_diff_name(name):
            return pd.Series(safe_divide(values, price) * 10_000.0, index=frame.index)
        return pd.Series(safe_divide(values - mid, mid) * 10_000.0, index=frame.index)

    if _is_book_depth_name(name):
        return _signed_log1p(values * price)

    if _is_book_level_share_name(name):
        return pd.Series(safe_divide(values, _side_depth(frame, name)), index=frame.index)

    if _is_dimensionless_name(name):
        return values

    if _is_trade_share_volume_name(name):
        ratio = _historical_ratio(frame, column)
        return ratio if ratio is not None else _signed_log1p(values)

    if _is_notional_name(name):
        return _signed_log1p(values)

    if _is_share_volume_name(name):
        return _signed_log1p(values)

    return values


def transform_mechanismized_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    cross_sectional_mode: str = "rank_centered",
) -> pd.DataFrame:
    """Replace configured values with mechanism-aware dimensionless values.

    The transform keeps feature names unchanged. Raw price levels become bps/tick
    quantities; share-volume features become notional/log pressure or relative
    book depth; amount/count fields become monotone scale-compressed values. A
    final within-cross-section transform makes the model input unit-free while
    preserving the selector's cross-sectional comparison.
    """

    source_columns = [
        column
        for column in dict.fromkeys(str(column) for column in columns)
        if column in frame.columns
    ]
    source_columns = [
        column
        for column in source_columns
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column])
    ]
    if not source_columns:
        return frame.copy()

    out = frame.copy()
    for column in source_columns:
        transformed = _mechanismized_feature_series(
            frame,
            column,
            tick_size=float(tick_size),
        )
        out[column] = (
            pd.to_numeric(transformed, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )

    normalized_cross_sectional_mode = str(cross_sectional_mode or "none").strip().lower()
    normalized_cross_sectional_mode = normalized_cross_sectional_mode.replace("-", "_")
    if normalized_cross_sectional_mode in {"", "none", "identity", "raw", "off", "false"}:
        return out
    return transform_cross_sectional_feature_values(
        out,
        columns=tuple(source_columns),
        group_cols=group_cols,
        mode=normalized_cross_sectional_mode,
        rank_method=rank_method,
    )


def transform_mechanismized_v2_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    cross_sectional_mode: str = "robust_zscore",
) -> pd.DataFrame:
    """Replace values with semantic-preserving dimensionless values.

    V2 keeps the same feature names but avoids changing feature meaning:
    share-volume flow stays share activity, turnover stays notional activity,
    aggregate book depth stays liquidity depth, and level queue sizes become
    structure shares of the current side depth.
    """

    source_columns = [
        column
        for column in dict.fromkeys(str(column) for column in columns)
        if column in frame.columns
    ]
    source_columns = [
        column
        for column in source_columns
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column])
    ]
    if not source_columns:
        return frame.copy()

    out = frame.copy()
    for column in source_columns:
        transformed = _mechanismized_v2_feature_series(
            frame,
            column,
            tick_size=float(tick_size),
        )
        out[column] = (
            pd.to_numeric(transformed, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )

    normalized_cross_sectional_mode = str(cross_sectional_mode or "robust_zscore").strip().lower()
    normalized_cross_sectional_mode = normalized_cross_sectional_mode.replace("-", "_")
    if normalized_cross_sectional_mode in {"", "none", "identity", "raw", "off", "false"}:
        return out
    return transform_cross_sectional_feature_values(
        out,
        columns=tuple(source_columns),
        group_cols=group_cols,
        mode=normalized_cross_sectional_mode,
        rank_method=rank_method,
    )


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
