from __future__ import annotations

import pandas as pd

from opening_strength_fit.feature_utils import _numeric_series, _sum_columns, safe_divide
from opening_strength_fit.schema import (
    PREOPEN_END,
    PREOPEN_START,
    ask_price_col,
    ask_volume_col,
    available_depth_levels,
    bid_price_col,
    bid_volume_col,
    ensure_timestamp_columns,
    filter_time_range,
)


def add_order_book_features(ticks: pd.DataFrame) -> pd.DataFrame:
    out = ticks.copy()
    levels = available_depth_levels(out)

    if "bid_price_1" in out.columns and "ask_price_1" in out.columns:
        bid1 = _numeric_series(out["bid_price_1"])
        ask1 = _numeric_series(out["ask_price_1"])
        out["mid_price"] = (bid1 + ask1) / 2.0
        out["spread_abs"] = ask1 - bid1
        out["spread_bps"] = safe_divide(out["spread_abs"], out["mid_price"]) * 10_000

    bid_volume_columns = [bid_volume_col(level) for level in levels]
    ask_volume_columns = [ask_volume_col(level) for level in levels]
    out["bid_depth_10"] = _sum_columns(out, bid_volume_columns)
    out["ask_depth_10"] = _sum_columns(out, ask_volume_columns)
    out["depth_imbalance_10"] = safe_divide(
        out["bid_depth_10"] - out["ask_depth_10"],
        out["bid_depth_10"] + out["ask_depth_10"],
    )

    if "bid_volume_1" in out.columns and "ask_volume_1" in out.columns:
        bid_volume_1 = _numeric_series(out["bid_volume_1"])
        ask_volume_1 = _numeric_series(out["ask_volume_1"])
        out["depth_imbalance_1"] = safe_divide(
            bid_volume_1 - ask_volume_1,
            bid_volume_1 + ask_volume_1,
        )

    if "limit_up_price" in out.columns and "ask_price_1" in out.columns:
        limit_up = _numeric_series(out["limit_up_price"])
        ask1 = _numeric_series(out["ask_price_1"])
        out["ask1_to_limit_up_bps"] = safe_divide(limit_up - ask1, ask1) * 10_000

    for level in levels:
        bid_price = bid_price_col(level)
        ask_price = ask_price_col(level)
        if level > 1 and bid_price in out.columns and "bid_price_1" in out.columns:
            bid1 = _numeric_series(out["bid_price_1"])
            bid_level = _numeric_series(out[bid_price])
            out[f"bid_gap_{level}_bps"] = safe_divide(bid1 - bid_level, bid1) * 10_000
        if level > 1 and ask_price in out.columns and "ask_price_1" in out.columns:
            ask1 = _numeric_series(out["ask_price_1"])
            ask_level = _numeric_series(out[ask_price])
            out[f"ask_gap_{level}_bps"] = safe_divide(ask_level - ask1, ask1) * 10_000
    return out


def add_trade_features(
    ticks: pd.DataFrame,
    *,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    out = ticks.sort_values(["date", "symbol", "timestamp"]).copy()
    group = out.groupby(["date", "symbol"], sort=False)

    if volume_col in out.columns:
        out["volume_diff_1t"] = group[volume_col].diff()
    if turnover_col in out.columns:
        out["turnover_diff_1t"] = group[turnover_col].diff()
    if volume_col in out.columns and turnover_col in out.columns:
        denominator = out["volume_diff_1t"] * float(volume_unit_multiplier)
        out["trade_vwap_1t"] = safe_divide(out["turnover_diff_1t"], denominator)
        for window in (3, 10, 30):
            out[f"volume_diff_{window}t"] = group[volume_col].diff(window)
            out[f"turnover_diff_{window}t"] = group[turnover_col].diff(window)
            out[f"trade_vwap_{window}t"] = safe_divide(
                out[f"turnover_diff_{window}t"],
                out[f"volume_diff_{window}t"] * float(volume_unit_multiplier),
            )
    return out


def add_momentum_features(ticks: pd.DataFrame) -> pd.DataFrame:
    out = ticks.sort_values(["date", "symbol", "timestamp"]).copy()
    group = out.groupby(["date", "symbol"], sort=False)
    price_col = None
    for candidate in ("last_price", "mid_price", "ask_price_1"):
        if candidate in out.columns:
            price_col = candidate
            break
    if price_col is None:
        return out

    if "prev_close" in out.columns:
        out["return_vs_prev_close"] = safe_divide(
            out[price_col] - out["prev_close"],
            out["prev_close"],
        )
    if "open_price" in out.columns:
        out["return_vs_open"] = safe_divide(
            out[price_col] - out["open_price"],
            out["open_price"],
        )

    for periods in (1, 3, 10, 30):
        out[f"return_{periods}t"] = group[price_col].pct_change(periods)
    return out


def build_preopen_features(
    ticks: pd.DataFrame,
    *,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    price_mode: str = "legacy_last_price",
    match_time: str = "09:25:00",
) -> pd.DataFrame:
    preopen = filter_time_range(
        ticks,
        PREOPEN_START,
        PREOPEN_END,
        include_end=True,
    )
    if preopen.empty:
        return pd.DataFrame(columns=["date", "symbol"])

    normalized_price_mode = str(price_mode).strip().lower().replace("-", "_")
    if normalized_price_mode not in {"legacy_last_price", "indicative_quote_v2"}:
        raise ValueError(
            "preopen price_mode must be legacy_last_price or indicative_quote_v2"
        )

    if normalized_price_mode == "indicative_quote_v2":
        keys = ["date", "symbol"]
        base = preopen[keys].drop_duplicates().reset_index(drop=True)
        agg_spec: dict[str, tuple[str, str]] = {}
        if volume_col in preopen.columns:
            agg_spec["preopen_volume"] = (volume_col, "max")
        if turnover_col in preopen.columns:
            agg_spec["preopen_turnover"] = (turnover_col, "max")
        if agg_spec:
            base = base.merge(
                preopen.groupby(keys, as_index=False).agg(**agg_spec),
                on=keys,
                how="left",
                validate="one_to_one",
            )

        timestamp = pd.to_datetime(preopen["timestamp"], errors="coerce")
        match_timestamp = pd.to_datetime(
            preopen["date"].astype(str) + f" {match_time}",
            errors="coerce",
        )
        indicative = preopen.loc[timestamp.lt(match_timestamp)].copy()
        if {"bid_price_1", "ask_price_1"}.issubset(indicative.columns):
            bid1 = pd.to_numeric(indicative["bid_price_1"], errors="coerce").where(
                lambda values: values > 0.0
            )
            ask1 = pd.to_numeric(indicative["ask_price_1"], errors="coerce").where(
                lambda values: values > 0.0
            )
            both_sides = bid1.notna() & ask1.notna()
            indicative_price = ((bid1 + ask1) / 2.0).where(
                both_sides,
                ask1.combine_first(bid1),
            )
            indicative = indicative.assign(_preopen_indicative_price=indicative_price)
            price_path = indicative.groupby(keys, as_index=False).agg(
                _preopen_indicative_last=("_preopen_indicative_price", "last"),
                preopen_price_min=("_preopen_indicative_price", "min"),
                preopen_price_max=("_preopen_indicative_price", "max"),
            )
            base = base.merge(
                price_path,
                on=keys,
                how="left",
                validate="one_to_one",
            )

        match = preopen.loc[timestamp.ge(match_timestamp)].copy()
        if "last_price" in match.columns:
            match_price = pd.to_numeric(match["last_price"], errors="coerce").where(
                lambda values: values > 0.0
            )
            match = match.assign(_preopen_match_price=match_price)
            match_prices = match.groupby(keys, as_index=False).agg(
                _preopen_match_price=("_preopen_match_price", "last")
            )
            base = base.merge(
                match_prices,
                on=keys,
                how="left",
                validate="one_to_one",
            )

        if "_preopen_match_price" in base.columns:
            fallback = base.get(
                "_preopen_indicative_last",
                pd.Series(float("nan"), index=base.index),
            )
            base["preopen_last_price"] = base["_preopen_match_price"].combine_first(
                fallback
            )
        elif "_preopen_indicative_last" in base.columns:
            base["preopen_last_price"] = base["_preopen_indicative_last"]

        if "depth_imbalance_10" in indicative.columns:
            imbalance = indicative.groupby(keys, as_index=False).agg(
                preopen_depth_imbalance_10=("depth_imbalance_10", "last")
            )
            base = base.merge(
                imbalance,
                on=keys,
                how="left",
                validate="one_to_one",
            )
        return base.drop(
            columns=["_preopen_indicative_last", "_preopen_match_price"],
            errors="ignore",
        )

    price_col = None
    for candidate in ("last_price", "mid_price", "ask_price_1"):
        if candidate in preopen.columns:
            price_col = candidate
            break

    agg_spec: dict[str, tuple[str, str]] = {}
    if volume_col in preopen.columns:
        agg_spec["preopen_volume"] = (volume_col, "max")
    if turnover_col in preopen.columns:
        agg_spec["preopen_turnover"] = (turnover_col, "max")
    if price_col is not None:
        agg_spec["preopen_last_price"] = (price_col, "last")
        agg_spec["preopen_price_min"] = (price_col, "min")
        agg_spec["preopen_price_max"] = (price_col, "max")
    if "depth_imbalance_10" in preopen.columns:
        agg_spec["preopen_depth_imbalance_10"] = ("depth_imbalance_10", "last")

    if not agg_spec:
        return preopen[["date", "symbol"]].drop_duplicates()
    return preopen.groupby(["date", "symbol"], as_index=False).agg(**agg_spec)


def build_feature_frame(
    ticks: pd.DataFrame,
    *,
    include_preopen: bool = True,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
    preopen_price_mode: str = "legacy_last_price",
    preopen_match_time: str = "09:25:00",
) -> pd.DataFrame:
    out = ensure_timestamp_columns(ticks)
    out = out.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    out = add_order_book_features(out)
    out = add_trade_features(
        out,
        volume_col=volume_col,
        turnover_col=turnover_col,
        volume_unit_multiplier=volume_unit_multiplier,
    )
    out = add_momentum_features(out)

    if include_preopen:
        preopen = build_preopen_features(
            out,
            volume_col=volume_col,
            turnover_col=turnover_col,
            price_mode=preopen_price_mode,
            match_time=preopen_match_time,
        )
        if not preopen.empty:
            out = out.merge(preopen, on=["date", "symbol"], how="left")
            if "prev_close" in out.columns and "preopen_last_price" in out.columns:
                out["preopen_return_vs_prev_close"] = safe_divide(
                    out["preopen_last_price"] - out["prev_close"],
                    out["prev_close"],
                )
            if {
                "preopen_price_min",
                "preopen_price_max",
                "preopen_last_price",
            }.issubset(out.columns):
                price_range = out["preopen_price_max"] - out["preopen_price_min"]
                reference_price = (
                    out["prev_close"] if "prev_close" in out.columns else out["preopen_last_price"]
                )
                out["auction_price_range_bps"] = safe_divide(price_range, reference_price) * 10_000
                out["auction_last_position_in_range"] = safe_divide(
                    out["preopen_last_price"] - out["preopen_price_min"],
                    price_range,
                )
    return out
