from __future__ import annotations

import re
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from opening_strength_fit.feature_transforms.cross_sectional import (
    transform_cross_sectional_feature_values,
)
from opening_strength_fit.feature_utils import safe_divide

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
        "market_cap",
        "float_market_cap",
        "total_market_cap",
        "mkt_cap",
        "total_mkt_cap",
        "total_shares",
        "float_shares",
        "shares_outstanding",
        "total_share",
        "share_capital",
        "circulating_shares",
        "free_float_shares",
        "hist_avg_daily_volume_20d",
        "hist_avg_daily_volume_60d",
        "hist_avg_daily_volume_120d",
        "hist_avg_daily_turnover_20d",
        "hist_avg_daily_turnover_60d",
        "hist_avg_daily_turnover_120d",
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
    if "mid_price" in frame.columns:
        mid = _clean_values(frame, "mid_price")
        if mid.notna().any():
            return _positive(mid)
    ask1 = _clean_values(frame, "ask_price_1")
    bid1 = _clean_values(frame, "bid_price_1")
    fallback = (ask1 + bid1) / 2.0
    return _positive(fallback)


def _reference_price(frame: pd.DataFrame) -> pd.Series:
    for column in ("ask_price_1", "mid_price", "preopen_last_price", "avg_ask_price"):
        if column not in frame.columns:
            continue
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


def _reference_market_cap(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "market_cap",
        "total_market_cap",
        "mkt_cap",
        "total_mkt_cap",
        "float_market_cap",
    ):
        if column not in frame.columns:
            continue
        values = _positive(_clean_values(frame, column))
        if values.notna().any():
            return values

    historical_turnover = _historical_daily_activity_reference(frame, "turnover")
    if historical_turnover.notna().any():
        return historical_turnover

    shares = _reference_shares(frame)
    price = _reference_price(frame)
    fallback = shares * price
    return _positive(fallback)


def _reference_shares(frame: pd.DataFrame) -> pd.Series:
    for column in (
        "total_shares",
        "shares_outstanding",
        "total_share",
        "share_capital",
        "float_shares",
        "circulating_shares",
        "free_float_shares",
    ):
        if column not in frame.columns:
            continue
        values = _positive(_clean_values(frame, column))
        if values.notna().any():
            return values

    market_cap = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in (
        "market_cap",
        "total_market_cap",
        "mkt_cap",
        "total_mkt_cap",
        "float_market_cap",
    ):
        if column not in frame.columns:
            continue
        values = _positive(_clean_values(frame, column))
        if values.notna().any():
            market_cap = values
            break
    price = _reference_price(frame)
    market_cap_shares = _positive(pd.Series(safe_divide(market_cap, price), index=frame.index))
    if market_cap_shares.notna().any():
        return market_cap_shares

    return _historical_daily_activity_reference(frame, "volume")


def _historical_daily_activity_reference(frame: pd.DataFrame, kind: str) -> pd.Series:
    for window in (60, 20, 120):
        column = f"hist_avg_daily_{kind}_{window}d"
        if column not in frame.columns:
            continue
        values = _positive(_clean_values(frame, column))
        if values.notna().any():
            return values
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _total_book_count(frame: pd.DataFrame) -> pd.Series:
    total = _clean_values(frame, "total_ask_count") + _clean_values(frame, "total_bid_count")
    return _positive(total)


class _MechanismizedV3References:
    """Lazily build each full-length v3 denominator at most once."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    @cached_property
    def mid(self) -> pd.Series:
        return _reference_mid(self.frame)

    @cached_property
    def price(self) -> pd.Series:
        return _reference_price(self.frame)

    @cached_property
    def shares(self) -> pd.Series:
        return _reference_shares(self.frame)

    @cached_property
    def market_cap(self) -> pd.Series:
        return _reference_market_cap(self.frame)

    @cached_property
    def ask_depth(self) -> pd.Series:
        return _positive(_clean_values(self.frame, "ask_depth_10"))

    @cached_property
    def bid_depth(self) -> pd.Series:
        return _positive(_clean_values(self.frame, "bid_depth_10"))

    @cached_property
    def total_ask_count(self) -> pd.Series:
        return _positive(_clean_values(self.frame, "total_ask_count"))

    @cached_property
    def total_bid_count(self) -> pd.Series:
        return _positive(_clean_values(self.frame, "total_bid_count"))

    @cached_property
    def total_book_count(self) -> pd.Series:
        return _positive(self.total_ask_count + self.total_bid_count)

    def side_depth(self, column: str) -> pd.Series:
        if column.startswith(("ask_", "postopen_ask_", "postopen_v2_ask_", "total_ask_")):
            return self.ask_depth
        if column.startswith(("bid_", "postopen_bid_", "postopen_v2_bid_", "total_bid_")):
            return self.bid_depth
        return _positive((self.ask_depth + self.bid_depth) / 2.0)

    def same_side_count(self, column: str) -> pd.Series:
        if column.startswith(("ask_", "postopen_ask_", "postopen_v2_ask_", "total_ask_")):
            return self.total_ask_count
        if column.startswith(("bid_", "postopen_bid_", "postopen_v2_bid_", "total_bid_")):
            return self.total_bid_count
        return self.total_book_count


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
    return name == "preopen_volume" or name.startswith(
        ("volume_diff_", "postopen_volume_", "postopen_v2_volume_")
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
    return name == "preopen_turnover" or name.startswith(
        ("turnover_diff_", "postopen_turnover_", "postopen_v2_turnover_")
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


def mechanismized_v3_changed_feature_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return columns whose values are materially rewritten by the v3 transform."""

    changed: list[str] = []
    for column in dict.fromkeys(str(column) for column in columns):
        name = column.lower()
        if _is_dimensionless_name(name) or _is_historical_ratio_name(name):
            continue
        if (
            name in {"exch_time_offset_us", "postopen_minutes_since_0930", "spread_abs"}
            or _is_raw_price_name(name)
            or _is_book_level_share_name(name)
            or _is_book_depth_name(name)
            or _is_trade_share_volume_name(name)
            or _is_share_volume_name(name)
            or _is_notional_name(name)
            or _is_count_name(name)
        ):
            changed.append(column)
    return tuple(changed)


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


def _mechanismized_v3_feature_series(
    frame: pd.DataFrame,
    column: str,
    *,
    tick_size: float,
    references: _MechanismizedV3References | None = None,
) -> pd.Series:
    name = str(column).lower()
    values = _clean_values(frame, column)
    references = references or _MechanismizedV3References(frame)

    if name == "exch_time_offset_us":
        open_offset_us = (9 * 60 * 60 + 30 * 60) * 1_000_000.0
        ten_minutes_us = 10 * 60 * 1_000_000.0
        return (values - open_offset_us) / ten_minutes_us

    if name == "postopen_minutes_since_0930":
        return values / 10.0

    if _is_dimensionless_name(name) or _is_historical_ratio_name(name):
        return values

    if name == "spread_abs":
        return pd.Series(safe_divide(values, float(tick_size)), index=frame.index)

    if _is_raw_price_name(name):
        if name == "mid_price":
            return pd.Series(safe_divide(float(tick_size), _positive(values)) * 10_000.0)
        if name == "ask_price_1":
            return pd.Series(
                safe_divide(values - references.mid, references.mid) * 10_000.0,
                index=frame.index,
            )
        if name == "bid_price_1":
            return pd.Series(
                safe_divide(references.mid - values, references.mid) * 10_000.0,
                index=frame.index,
            )
        if name.startswith("trade_vwap_"):
            return pd.Series(
                safe_divide(values - references.price, references.price) * 10_000.0,
                index=frame.index,
            )
        if _is_price_diff_name(name):
            return pd.Series(
                safe_divide(values, references.price) * 10_000.0,
                index=frame.index,
            )
        return pd.Series(
            safe_divide(values - references.mid, references.mid) * 10_000.0,
            index=frame.index,
        )

    if _is_book_level_share_name(name):
        return pd.Series(safe_divide(values, references.side_depth(name)), index=frame.index)

    if _is_book_depth_name(name):
        by_shares = pd.Series(safe_divide(values, references.shares), index=frame.index)
        missing = by_shares.isna()
        if missing.any():
            by_shares.loc[missing] = safe_divide(
                values.loc[missing] * references.price.loc[missing],
                references.market_cap.loc[missing],
            )
        return by_shares

    if _is_trade_share_volume_name(name) or _is_share_volume_name(name):
        return pd.Series(safe_divide(values, references.shares), index=frame.index)

    if _is_notional_name(name):
        return pd.Series(safe_divide(values, references.market_cap), index=frame.index)

    if _is_count_name(name):
        if name in {"total_ask_count", "total_bid_count"}:
            return pd.Series(safe_divide(values, references.total_book_count), index=frame.index)
        if name == "trade_num":
            return pd.Series(safe_divide(values, references.total_book_count), index=frame.index)
        return pd.Series(safe_divide(values, references.same_side_count(name)), index=frame.index)

    return values


def _transform_mechanismized_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...],
    rank_method: str,
    cross_sectional_mode: str,
    transform: Callable[[str], pd.Series],
) -> pd.DataFrame:
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
        out[column] = (
            pd.to_numeric(transform(column), errors="coerce")
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


def transform_mechanismized_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    cross_sectional_mode: str = "rank_centered",
) -> pd.DataFrame:
    """Replace configured values with mechanism-aware dimensionless values."""

    return _transform_mechanismized_values(
        frame,
        columns=columns,
        group_cols=group_cols,
        rank_method=rank_method,
        cross_sectional_mode=cross_sectional_mode,
        transform=lambda column: _mechanismized_feature_series(
            frame, column, tick_size=float(tick_size)
        ),
    )


def transform_mechanismized_v2_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    cross_sectional_mode: str = "none",
) -> pd.DataFrame:
    """Replace values with semantic-preserving dimensionless values."""

    return _transform_mechanismized_values(
        frame,
        columns=columns,
        group_cols=group_cols,
        rank_method=rank_method,
        cross_sectional_mode=cross_sectional_mode,
        transform=lambda column: _mechanismized_v2_feature_series(
            frame, column, tick_size=float(tick_size)
        ),
    )


def transform_mechanismized_v3_feature_values(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    cross_sectional_mode: str = "none",
) -> pd.DataFrame:
    """Replace values with strict ratio-style dimensionless values."""

    references = _MechanismizedV3References(frame)
    return _transform_mechanismized_values(
        frame,
        columns=columns,
        group_cols=group_cols,
        rank_method=rank_method,
        cross_sectional_mode=cross_sectional_mode,
        transform=lambda column: _mechanismized_v3_feature_series(
            frame,
            column,
            tick_size=float(tick_size),
            references=references,
        ),
    )


__all__ = [
    "mechanismized_v3_changed_feature_columns",
    "mechanismized_feature_value_reference_columns",
    "transform_mechanismized_feature_values",
    "transform_mechanismized_v2_feature_values",
    "transform_mechanismized_v3_feature_values",
]
