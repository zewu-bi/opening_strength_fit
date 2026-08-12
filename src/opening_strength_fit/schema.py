from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

PREOPEN_START = "09:15:00"
PREOPEN_END = "09:25:30"
OPEN_SAMPLE_START = "09:30:00"
OPEN_SAMPLE_END = "09:40:00"
PRICE_LEVELS = tuple(range(1, 11))
EXCHANGE_OFFSET_US_COL = "exch_time_offset_us"
CLOCK_PATTERN = r"(\d{1,2}:\d{2}(?::\d{2})?)"
DECISION_KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")


def normalize_text_series(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        )
    )


def normalize_date_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def bid_price_col(level: int) -> str:
    return f"bid_price_{level}"


def ask_price_col(level: int) -> str:
    return f"ask_price_{level}"


def bid_volume_col(level: int) -> str:
    return f"bid_volume_{level}"


def ask_volume_col(level: int) -> str:
    return f"ask_volume_{level}"


def normalize_clock_time(value: str) -> str:
    parts = str(value).strip().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"invalid clock time: {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"invalid clock time: {value!r}")
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def normalize_clock_series(values: pd.Series) -> pd.Series:
    """Normalize clock-like strings to ``HH:MM:SS``."""

    extracted = values.astype(str).str.extract(CLOCK_PATTERN, expand=False).fillna("")
    return extracted.map(lambda value: normalize_clock_time(value) if value else "")


def frame_clock_series(
    frame: pd.DataFrame,
    *,
    clock_col: str = "decision_time",
    timestamp_cols: tuple[str, ...] = ("decision_target_timestamp", "timestamp"),
) -> pd.Series:
    """Resolve the canonical decision clock from a research frame."""

    if clock_col in frame.columns:
        return normalize_clock_series(frame[clock_col])
    for column in timestamp_cols:
        if column in frame.columns:
            return pd.to_datetime(frame[column], errors="coerce").dt.strftime("%H:%M:%S").fillna("")
    expected = ", ".join((clock_col, *timestamp_cols))
    raise ValueError(f"frame has no clock column; expected one of: {expected}")


def normalize_decision_keys(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = DECISION_KEY_COLUMNS,
    drop_missing: bool = True,
    require_unique: bool = False,
    context: str | None = None,
) -> pd.DataFrame:
    """Normalize the shared date/symbol/decision timestamp join keys."""

    if context is not None:
        missing = [column for column in key_columns if column not in frame.columns]
        if missing:
            raise SystemExit(f"{context} is missing join keys {missing}")
    out = frame.copy()
    if "date" in out:
        out["date"] = normalize_date_series(out["date"])
    if "symbol" in out:
        out["symbol"] = normalize_text_series(out["symbol"])
    if "decision_target_timestamp" in out:
        out["decision_target_timestamp"] = pd.to_datetime(
            out["decision_target_timestamp"], errors="coerce"
        ).dt.tz_localize(None)
    if drop_missing:
        out = out.dropna(subset=list(key_columns)).copy()
    elif (
        context is not None and (invalid := out.loc[:, list(key_columns)].isna().any(axis=1)).any()
    ):
        raise SystemExit(f"{context} has {int(invalid.sum())} null-key rows")
    if require_unique:
        duplicate = out.duplicated(list(key_columns), keep=False)
        if duplicate.any():
            raise SystemExit(f"{context} has {int(duplicate.sum())} duplicate-key rows")
    return out


def _depth_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for level in PRICE_LEVELS:
        aliases.update(
            {
                f"bid{level}": bid_price_col(level),
                f"ask{level}": ask_price_col(level),
                f"bid_price{level}": bid_price_col(level),
                f"ask_price{level}": ask_price_col(level),
                f"bid_vol{level}": bid_volume_col(level),
                f"ask_vol{level}": ask_volume_col(level),
                f"bid_volume{level}": bid_volume_col(level),
                f"ask_volume{level}": ask_volume_col(level),
                f"b{level}": bid_price_col(level),
                f"a{level}": ask_price_col(level),
                f"bv{level}": bid_volume_col(level),
                f"av{level}": ask_volume_col(level),
            }
        )
    return aliases


def _clickhouse_tick_aliases() -> dict[str, str]:
    aliases = {
        "TradingDay": "date",
        "Symbol": "symbol",
        "ExchTimeOffsetUs": EXCHANGE_OFFSET_US_COL,
        "HighPrice": "high_price",
        "LowPrice": "low_price",
        "LastPrice": "last_price",
        "TradeNum": "trade_num",
        "Volume": "volume",
        "Turnover": "turnover",
        "Status": "status",
        "AvgAskPrice": "avg_ask_price",
        "TotalAskVolume": "total_ask_volume",
        "TotalAskCount": "total_ask_count",
        "AvgBidPrice": "avg_bid_price",
        "TotalBidVolume": "total_bid_volume",
        "TotalBidCount": "total_bid_count",
        "IOPV": "iopv",
        "LocalTimeStamp": "local_timestamp",
    }
    for level in PRICE_LEVELS:
        aliases.update(
            {
                f"AskPrice{level}": ask_price_col(level),
                f"AskVolume{level}": ask_volume_col(level),
                f"AskCount{level}": f"ask_count_{level}",
                f"BidPrice{level}": bid_price_col(level),
                f"BidVolume{level}": bid_volume_col(level),
                f"BidCount{level}": f"bid_count_{level}",
            }
        )
    return aliases


DEFAULT_COLUMN_ALIASES = {
    "datetime": "timestamp",
    "ts": "timestamp",
    "ticker": "symbol",
    "code": "symbol",
    "security_id": "symbol",
    "amount": "turnover",
    "cum_amount": "turnover",
    "cum_turnover": "turnover",
    "cum_volume": "volume",
    "last": "last_price",
    "price": "last_price",
    "preclose": "prev_close",
    "pre_close": "prev_close",
    **_depth_aliases(),
    **_clickhouse_tick_aliases(),
}


def standardize_columns(
    df: pd.DataFrame,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    all_aliases = dict(DEFAULT_COLUMN_ALIASES)
    if aliases:
        all_aliases.update(aliases)

    lower_to_original = {str(column).lower(): column for column in df.columns}
    rename: dict[object, str] = {}
    for alias, target in all_aliases.items():
        original = lower_to_original.get(alias.lower())
        if original is not None and target not in df.columns:
            rename[original] = target
    return df.rename(columns=rename)


def ensure_timestamp_columns(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    time_col: str = "time",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    exchange_offset_col: str = EXCHANGE_OFFSET_US_COL,
) -> pd.DataFrame:
    out = df.copy()
    if symbol_col not in out.columns:
        raise SystemExit(f"missing required column: {symbol_col}")

    if timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    elif date_col in out.columns and time_col in out.columns:
        out[timestamp_col] = pd.to_datetime(
            out[date_col].astype(str) + " " + out[time_col].astype(str)
        )
    elif date_col in out.columns and exchange_offset_col in out.columns:
        trading_day = pd.to_datetime(out[date_col].astype(str))
        offset_us = pd.to_numeric(out[exchange_offset_col], errors="coerce")
        out[timestamp_col] = trading_day + pd.to_timedelta(offset_us, unit="us")
    else:
        raise SystemExit(
            "missing timestamp information: expected timestamp, date + time, "
            "or date + exchange offset columns"
        )

    if date_col not in out.columns:
        out[date_col] = out[timestamp_col].dt.strftime("%Y-%m-%d")
    else:
        out[date_col] = pd.to_datetime(out[date_col].astype(str)).dt.strftime("%Y-%m-%d")

    out[time_col] = out[timestamp_col].dt.strftime("%H:%M:%S")
    out[symbol_col] = out[symbol_col].astype(str)
    return out


def time_mask(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
    *,
    timestamp_col: str = "timestamp",
    include_end: bool = False,
) -> pd.Series:
    start_time = normalize_clock_time(start_time)
    end_time = normalize_clock_time(end_time)
    clock = df[timestamp_col].dt.strftime("%H:%M:%S")
    if include_end:
        return (clock >= start_time) & (clock <= end_time)
    return (clock >= start_time) & (clock < end_time)


def filter_time_range(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
    *,
    timestamp_col: str = "timestamp",
    include_end: bool = False,
) -> pd.DataFrame:
    return df.loc[
        time_mask(
            df,
            start_time,
            end_time,
            timestamp_col=timestamp_col,
            include_end=include_end,
        )
    ].copy()


def available_depth_levels(df: pd.DataFrame) -> list[int]:
    levels = []
    for level in PRICE_LEVELS:
        if (
            bid_price_col(level) in df.columns
            and ask_price_col(level) in df.columns
            and bid_volume_col(level) in df.columns
            and ask_volume_col(level) in df.columns
        ):
            levels.append(level)
    return levels
