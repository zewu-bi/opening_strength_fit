from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import validate_table_name

DEFAULT_DAILY_MARKET_REFERENCE_TABLE = "stock.daily_bar_jy"
MARKET_REFERENCE_COLUMNS = (
    "total_market_cap",
    "float_market_cap",
    "total_shares",
    "float_shares",
    "free_float_shares",
)
MARKET_REFERENCE_CONTEXT_COLUMNS = (
    "market_cap_reference_date",
    "market_cap_reference_lag_sessions",
)


def _empty_reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "symbol",
            *MARKET_REFERENCE_COLUMNS,
            *MARKET_REFERENCE_CONTEXT_COLUMNS,
        ]
    )


def _strict_lagged_session_sql(table: str, lag_sessions: int) -> str:
    table = validate_table_name(table)
    if lag_sessions < 1:
        raise ValueError("daily market reference lag_sessions must be >= 1")
    offset = lag_sessions - 1
    return f"""select TradingDay as reference_day
from (
    select distinct TradingDay
    from {table}
    where TradingDay < {{trading_day:Date}}
)
order by TradingDay desc
limit 1 offset {offset}"""


def _market_reference_sql(table: str) -> str:
    table = validate_table_name(table)
    return f"""select
    Symbol,
    TotalMarketValue,
    TotalFloatMarketValue,
    TotalShareToday,
    FloatAShare,
    FreeShareToday
from {table}
where TradingDay = {{reference_day:Date}}
  and Symbol in {{symbols:Array(String)}}
order by Symbol"""


def query_lagged_daily_market_reference(
    client,
    *,
    trading_day: str,
    symbols: Sequence[str],
    table: str = DEFAULT_DAILY_MARKET_REFERENCE_TABLE,
    lag_sessions: int = 1,
    market_cap_unit_multiplier: float = 10_000.0,
    share_unit_multiplier: float = 10_000.0,
) -> pd.DataFrame:
    """Load a strictly lagged daily market-cap/share reference for one tick day."""

    normalized_symbols = sorted({str(symbol) for symbol in symbols if str(symbol)})
    if not normalized_symbols:
        return _empty_reference_frame()
    if market_cap_unit_multiplier <= 0 or share_unit_multiplier <= 0:
        raise ValueError("daily market reference unit multipliers must be positive")

    target_day = pd.Timestamp(trading_day).normalize()
    session = client.query_df(
        _strict_lagged_session_sql(table, int(lag_sessions)),
        parameters={"trading_day": str(target_day.date())},
    )
    if session.empty or "reference_day" not in session.columns:
        raise RuntimeError(f"no lagged daily market reference session before {trading_day}")
    reference_day = pd.to_datetime(session["reference_day"], errors="coerce").iloc[0]
    if pd.isna(reference_day) or reference_day.normalize() >= target_day:
        raise RuntimeError(
            "daily market reference must be strictly earlier than the tick trading day"
        )

    raw = client.query_df(
        _market_reference_sql(table),
        parameters={
            "reference_day": str(reference_day.date()),
            "symbols": normalized_symbols,
        },
    )
    if raw.empty:
        return _empty_reference_frame()
    required_source_columns = {
        "Symbol",
        "TotalMarketValue",
        "TotalFloatMarketValue",
        "TotalShareToday",
        "FloatAShare",
        "FreeShareToday",
    }
    missing_source_columns = sorted(required_source_columns.difference(raw.columns))
    if missing_source_columns:
        raise RuntimeError(
            "daily market reference query missing columns: "
            + ", ".join(missing_source_columns)
        )

    symbol = raw["Symbol"].astype(str)
    if symbol.duplicated().any():
        duplicates = sorted(symbol.loc[symbol.duplicated(keep=False)].unique())
        raise RuntimeError(
            "daily market reference has duplicate symbol rows: " + ", ".join(duplicates[:5])
        )

    def positive_scaled(column: str, multiplier: float) -> pd.Series:
        values = pd.to_numeric(raw[column], errors="coerce").astype("float64")
        values = values.where(np.isfinite(values) & values.gt(0.0))
        return values * float(multiplier)

    out = pd.DataFrame(
        {
            "date": str(target_day.date()),
            "symbol": symbol,
            "total_market_cap": positive_scaled(
                "TotalMarketValue", market_cap_unit_multiplier
            ),
            "float_market_cap": positive_scaled(
                "TotalFloatMarketValue", market_cap_unit_multiplier
            ),
            "total_shares": positive_scaled("TotalShareToday", share_unit_multiplier),
            "float_shares": positive_scaled("FloatAShare", share_unit_multiplier),
            "free_float_shares": positive_scaled(
                "FreeShareToday", share_unit_multiplier
            ),
            "market_cap_reference_date": reference_day.normalize(),
            "market_cap_reference_lag_sessions": int(lag_sessions),
        }
    )
    return out.reset_index(drop=True)


def attach_daily_market_reference(
    ticks: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a one-row-per-day-symbol market reference without overwriting tick data."""

    if ticks.empty:
        return ticks.copy()
    required_tick = {"date", "symbol"}
    missing_tick = sorted(required_tick.difference(ticks.columns))
    if missing_tick:
        raise ValueError(
            "daily market reference merge needs tick columns: " + ", ".join(missing_tick)
        )
    if reference.empty:
        out = ticks.copy()
        for column in MARKET_REFERENCE_COLUMNS:
            out[column] = np.nan
        out["market_cap_reference_date"] = pd.NaT
        out["market_cap_reference_lag_sessions"] = pd.Series(
            pd.NA, index=out.index, dtype="Int64"
        )
        return out

    required_reference = {
        "date",
        "symbol",
        *MARKET_REFERENCE_COLUMNS,
        *MARKET_REFERENCE_CONTEXT_COLUMNS,
    }
    missing_reference = sorted(required_reference.difference(reference.columns))
    if missing_reference:
        raise ValueError(
            "daily market reference merge missing columns: "
            + ", ".join(missing_reference)
        )
    overlap = sorted(
        (set(MARKET_REFERENCE_COLUMNS) | set(MARKET_REFERENCE_CONTEXT_COLUMNS))
        & set(ticks.columns)
    )
    if overlap:
        raise ValueError(
            "daily market reference would overwrite tick columns: " + ", ".join(overlap)
        )

    ref = reference.copy()
    ref["date"] = ref["date"].astype(str)
    ref["symbol"] = ref["symbol"].astype(str)
    if ref.duplicated(["date", "symbol"]).any():
        raise ValueError("daily market reference must be unique by date and symbol")
    out = ticks.copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    return out.merge(ref, on=["date", "symbol"], how="left", validate="many_to_one")
