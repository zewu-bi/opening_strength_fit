from __future__ import annotations

import re

import numpy as np
import pandas as pd

from opening_strength_fit.labels import safe_price_return
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

NEXT_SESSION_OPEN_CLOSE_LABEL_COL = "alpha_return_next_session_open_close"
NEXT_SESSION_OPEN_CLOSE_COLUMNS = (
    "date",
    "symbol",
    "target_date",
    NEXT_SESSION_OPEN_CLOSE_LABEL_COL,
)
CLOSE_TO_NEXT_CLOSE_LABEL_COL = "alpha_return_close_to_next_close"
CLOSE_TO_NEXT_CLOSE_COLUMNS = (
    "date",
    "symbol",
    "target_date",
    CLOSE_TO_NEXT_CLOSE_LABEL_COL,
)


def _normalize_daily_bars(
    daily_bars: pd.DataFrame,
    *,
    price_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {"date", "symbol", *price_columns}
    missing = sorted(required.difference(daily_bars.columns))
    if missing:
        raise SystemExit(f"daily-bar input missing columns: {missing}")

    bars = daily_bars.loc[:, sorted(required)].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bars["symbol"] = bars["symbol"].astype(str)
    for column in price_columns:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["date", "symbol"])

    duplicate_keys = bars.duplicated(["date", "symbol"], keep=False)
    if duplicate_keys.any():
        examples = (
            bars.loc[duplicate_keys, ["date", "symbol"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise RuntimeError(f"daily-bar input has duplicate date-symbol keys: {examples}")
    return bars


def _next_session_mapping(
    bars: pd.DataFrame,
    *,
    feature_start_date: str,
    feature_end_date: str,
) -> dict[str, str]:
    trading_dates = sorted(bars["date"].unique())
    return {
        trading_dates[index]: trading_dates[index + 1]
        for index in range(len(trading_dates) - 1)
        if feature_start_date <= trading_dates[index] <= feature_end_date
    }


def build_next_session_open_close_labels(
    daily_bars: pd.DataFrame,
    *,
    feature_start_date: str,
    feature_end_date: str,
    fee_bps: float = 0.0,
    symbol_regex: str = DEFAULT_A_SHARE_SYMBOL_REGEX,
) -> pd.DataFrame:
    """Relabel next-session open-to-close returns by the preceding feature date."""

    bars = _normalize_daily_bars(
        daily_bars,
        price_columns=("open_price", "close_price"),
    )
    next_session = _next_session_mapping(
        bars,
        feature_start_date=feature_start_date,
        feature_end_date=feature_end_date,
    )
    feature_by_target = {target: feature for feature, target in next_session.items()}
    if not feature_by_target:
        return pd.DataFrame(columns=NEXT_SESSION_OPEN_CLOSE_COLUMNS)

    out = bars.loc[bars["date"].isin(feature_by_target)].copy()
    out["target_date"] = out["date"]
    out["date"] = out["target_date"].map(feature_by_target)
    if symbol_regex:
        pattern = re.compile(symbol_regex)
        out = out.loc[out["symbol"].str.fullmatch(pattern, na=False)].copy()

    out[NEXT_SESSION_OPEN_CLOSE_LABEL_COL] = safe_price_return(
        out["close_price"],
        out["open_price"],
        fee_bps=fee_bps,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[NEXT_SESSION_OPEN_CLOSE_LABEL_COL])
    if not out["target_date"].gt(out["date"]).all():
        raise RuntimeError("next-session label target_date must be strictly after feature date")

    return (
        out.loc[:, NEXT_SESSION_OPEN_CLOSE_COLUMNS]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_close_to_next_close_labels(
    daily_bars: pd.DataFrame,
    *,
    feature_start_date: str,
    feature_end_date: str,
    fee_bps: float = 0.0,
    symbol_regex: str = DEFAULT_A_SHARE_SYMBOL_REGEX,
) -> pd.DataFrame:
    """Build adjusted D-close to D+1-close returns keyed by feature date D."""

    bars = _normalize_daily_bars(
        daily_bars,
        price_columns=("close_price", "preclose_price"),
    )
    next_session = _next_session_mapping(
        bars,
        feature_start_date=feature_start_date,
        feature_end_date=feature_end_date,
    )
    if not next_session:
        return pd.DataFrame(columns=CLOSE_TO_NEXT_CLOSE_COLUMNS)

    feature_rows = bars.loc[
        bars["date"].isin(next_session),
        ["date", "symbol", "close_price"],
    ].rename(columns={"close_price": "_entry_close_price"})
    feature_rows["target_date"] = feature_rows["date"].map(next_session)
    target_rows = bars.loc[
        bars["date"].isin(set(next_session.values())),
        ["date", "symbol", "preclose_price", "close_price"],
    ].rename(
        columns={
            "date": "target_date",
            "preclose_price": "_target_preclose_price",
            "close_price": "_target_close_price",
        }
    )
    out = feature_rows.merge(
        target_rows,
        on=["target_date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    out = out.loc[np.isfinite(out["_entry_close_price"]) & out["_entry_close_price"].gt(0)].copy()
    if symbol_regex:
        pattern = re.compile(symbol_regex)
        out = out.loc[out["symbol"].str.fullmatch(pattern, na=False)].copy()

    # PreClosePrice on D+1 is the exchange's ex-right/ex-dividend reference for
    # the close held from D, and avoids treating corporate actions as alpha.
    out[CLOSE_TO_NEXT_CLOSE_LABEL_COL] = safe_price_return(
        out["_target_close_price"],
        out["_target_preclose_price"],
        fee_bps=fee_bps,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[CLOSE_TO_NEXT_CLOSE_LABEL_COL])
    if not out["target_date"].gt(out["date"]).all():
        raise RuntimeError("close-to-next-close target_date must be after feature date")

    return (
        out.loc[:, CLOSE_TO_NEXT_CLOSE_COLUMNS]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
