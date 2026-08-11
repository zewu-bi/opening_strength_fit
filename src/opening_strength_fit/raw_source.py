from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.clickhouse_ticks import validate_table_name
from opening_strength_fit.config import config_int
from opening_strength_fit.feature_utils import finite_numeric
from opening_strength_fit.schema import normalize_date_series, normalize_text_series

RAW_SOURCE_SCHEMA_VERSION = "raw_source_v2"

TICK_COLUMNS = (
    "TradingDay",
    "Symbol",
    "ExchTimeOffsetUs",
    "HighPrice",
    "LowPrice",
    "LastPrice",
    "TradeNum",
    "Volume",
    "Turnover",
    "Status",
    "AvgAskPrice",
    "TotalAskVolume",
    "TotalAskCount",
    "AvgBidPrice",
    "TotalBidVolume",
    "TotalBidCount",
    "IOPV",
    *(
        column
        for level in range(1, 11)
        for column in (
            f"AskPrice{level}",
            f"AskVolume{level}",
            f"AskCount{level}",
            f"BidPrice{level}",
            f"BidVolume{level}",
            f"BidCount{level}",
        )
    ),
)

DAILY_REFERENCE_COLUMNS = (
    "TradingDay",
    "Symbol",
    "OpenPrice",
    "ClosePrice",
    "PreClosePrice",
    "TradeStatus",
    "STStatus",
    "UpdownLimitStatus",
    "TotalMarketValue",
    "TotalFloatMarketValue",
    "TotalShareToday",
    "FloatAShare",
    "FreeShareToday",
)

CLOSE_REFERENCE_COLUMNS = (
    "TradingDay",
    "Symbol",
    "ClosePrice",
    "CloseSourceOffsetUs",
)

CALENDAR_COLUMNS = ("TradingDay",)


def read_daily_limit_flags(
    raw_source_root: Path,
    year: int,
    *,
    output_column: str = "final_up_limit",
) -> pd.DataFrame:
    frame = pd.read_parquet(
        raw_source_root / f"year={year}" / "daily_reference.parquet",
        columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
    ).rename(columns={"TradingDay": "date", "Symbol": "symbol"})
    frame["date"] = normalize_date_series(frame["date"])
    frame["symbol"] = normalize_text_series(frame["symbol"])
    frame[output_column] = pd.to_numeric(frame["UpdownLimitStatus"], errors="coerce").eq(1)
    return frame.loc[
        frame["date"].str.startswith(str(year), na=False),
        ["date", "symbol", output_column],
    ].drop_duplicates(["date", "symbol"], keep="last")


def read_daily_market_events(raw_source_root: Path, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(
        raw_source_root / f"year={year}" / "daily_reference.parquet",
        columns=[
            "TradingDay",
            "Symbol",
            "PreClosePrice",
            "ClosePrice",
            "STStatus",
            "UpdownLimitStatus",
        ],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "PreClosePrice": "prev_close",
            "ClosePrice": "daily_close",
            "STStatus": "st_status",
            "UpdownLimitStatus": "limit_status",
        }
    )
    frame["date"] = normalize_date_series(frame["date"])
    frame["symbol"] = normalize_text_series(frame["symbol"])
    for column in ("prev_close", "daily_close", "st_status", "limit_status"):
        frame[column] = finite_numeric(frame[column])
    frame["daily_return_bps"] = (
        frame["daily_close"].div(frame["prev_close"].where(frame["prev_close"].gt(0))).sub(1)
        * 10_000.0
    )
    frame["limit_up"] = frame["limit_status"].eq(1)
    frame["limit_down"] = frame["limit_status"].lt(0)
    frame["limit_event"] = frame["limit_up"] | frame["limit_down"]
    frame["st_flag"] = frame["st_status"].fillna(0).ne(0)
    return frame.drop_duplicates(["date", "symbol"], keep="last")


def parse_tick_windows(config: dict) -> tuple[tuple[int, int], ...]:
    section = config.get("raw_source", {})
    raw_windows = section.get("tick_windows", []) if isinstance(section, dict) else []
    windows: list[tuple[int, int]] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, list | tuple) or len(raw_window) != 2:
            raise SystemExit("[raw_source].tick_windows entries must be [start_us, end_us]")
        start, end = (int(raw_window[0]), int(raw_window[1]))
        if start < 0 or end < start:
            raise SystemExit(f"invalid tick window: [{start}, {end}]")
        windows.append((start, end))
    if not windows:
        raise SystemExit("[raw_source].tick_windows must not be empty")
    ordered = tuple(sorted(windows))
    if any(
        current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise SystemExit("[raw_source].tick_windows must be disjoint")
    return ordered


def parse_raw_source_years(config: dict) -> tuple[int, ...]:
    section = config.get("raw_source", {})
    raw_years = section.get("years", []) if isinstance(section, dict) else []
    years = tuple(int(value) for value in raw_years)
    if not years:
        raise SystemExit("[raw_source].years must not be empty")
    if len(set(years)) != len(years):
        raise SystemExit("[raw_source].years must be unique")
    return years


def parse_short_label_horizons(config: dict) -> tuple[int, ...]:
    section = config.get("raw_source", {})
    raw_horizons = (
        section.get("short_label_horizons_seconds", []) if isinstance(section, dict) else []
    )
    horizons = tuple(sorted({int(value) for value in raw_horizons}))
    if not horizons or any(value <= 0 for value in horizons):
        raise SystemExit("[raw_source].short_label_horizons_seconds must contain positive seconds")
    return horizons


def label_coverage(config: dict) -> dict[str, object]:
    windows = parse_tick_windows(config)
    horizons = parse_short_label_horizons(config)
    decision_end_offset_us = config_int(config, "raw_source", "decision_end_offset_us", 0)
    entry_delay_seconds = config_int(config, "raw_source", "entry_delay_seconds", 0)
    sell_window_seconds = config_int(config, "raw_source", "short_label_sell_window_seconds", 0)
    if decision_end_offset_us <= 0:
        raise SystemExit("[raw_source].decision_end_offset_us must be positive")
    if entry_delay_seconds < 0 or sell_window_seconds <= 0:
        raise SystemExit(
            "entry_delay_seconds must be non-negative and "
            "short_label_sell_window_seconds must be positive"
        )
    required_end_offset_us = (
        decision_end_offset_us
        + (entry_delay_seconds + max(horizons) + sell_window_seconds) * 1_000_000
    )
    covering_windows = [
        window
        for window in windows
        if window[0] <= decision_end_offset_us and window[1] >= required_end_offset_us
    ]
    if not covering_windows:
        raise SystemExit(
            "tick_windows do not continuously cover the latest short label: "
            f"decision_end={decision_end_offset_us}, required_end={required_end_offset_us}"
        )
    available_end_offset_us = max(window[1] for window in covering_windows)
    return {
        "short_label_horizons_seconds": list(horizons),
        "entry_delay_seconds": entry_delay_seconds,
        "sell_window_seconds": sell_window_seconds,
        "decision_end_offset_us": decision_end_offset_us,
        "required_tick_end_offset_us": required_end_offset_us,
        "available_tick_end_offset_us": available_end_offset_us,
        "tail_buffer_seconds": (available_end_offset_us - required_end_offset_us) / 1_000_000,
        "next_close_reference": True,
    }


def tick_source_sql(
    table: str,
    windows: tuple[tuple[int, int], ...],
    *,
    output_columns: tuple[str, ...] = TICK_COLUMNS,
) -> str:
    table = validate_table_name(table)
    outer_projection = ",\n    ".join(output_columns)
    selection_columns = tuple(
        dict.fromkeys(
            (
                *output_columns,
                "LocalTimeStamp",
                "TradeNum",
                "Volume",
                "Turnover",
                "LastPrice",
                "AskPrice1",
                "BidPrice1",
            )
        )
    )
    projection = ",\n        ".join(selection_columns)
    window_sql = "\n        or ".join(
        "(ExchTimeOffsetUs >= "
        f"{{window_start_{index}:UInt64}} and ExchTimeOffsetUs <= "
        f"{{window_end_{index}:UInt64}})"
        for index, _ in enumerate(windows)
    )
    return f"""select
    {outer_projection}
from (
    select
        {projection}
    from {table}
    where TradingDay = {{trading_day:Date}}
      and match(Symbol, {{symbol_regex:String}})
      and (
        {window_sql}
      )
    order by
        Symbol,
        ExchTimeOffsetUs,
        arrayMax(mapValues(LocalTimeStamp)) desc,
        TradeNum desc,
        Volume desc,
        Turnover desc,
        LastPrice desc,
        AskPrice1 desc,
        BidPrice1 desc
    limit 1 by Symbol, ExchTimeOffsetUs
)
order by Symbol, ExchTimeOffsetUs
format Parquet"""


def daily_reference_sql(table: str) -> str:
    table = validate_table_name(table)
    projection = ",\n    ".join(DAILY_REFERENCE_COLUMNS)
    return f"""select
    {projection}
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
  and match(Symbol, {{symbol_regex:String}})
order by TradingDay, Symbol
format Parquet"""


def close_reference_sql(table: str) -> str:
    table = validate_table_name(table)
    return f"""select
    TradingDay,
    Symbol,
    argMax(
        LastPrice,
        tuple(
            ExchTimeOffsetUs,
            arrayMax(mapValues(LocalTimeStamp)),
            TradeNum,
            Volume,
            Turnover
        )
    ) as ClosePrice,
    max(ExchTimeOffsetUs) as CloseSourceOffsetUs
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
  and match(Symbol, {{symbol_regex:String}})
  and ExchTimeOffsetUs >= {{close_start_offset_us:UInt64}}
  and ExchTimeOffsetUs <= {{close_end_offset_us:UInt64}}
  and LastPrice > 0
group by TradingDay, Symbol
order by TradingDay, Symbol
format Parquet"""


def calendar_sql(table: str) -> str:
    table = validate_table_name(table)
    return f"""select distinct TradingDay
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
  and match(Symbol, {{symbol_regex:String}})
order by TradingDay
format Parquet"""


def trading_dates_sql(table: str) -> str:
    table = validate_table_name(table)
    return f"""select distinct TradingDay
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
  and match(Symbol, {{symbol_regex:String}})
order by TradingDay"""


def query_trading_dates(client, *, table: str, year: int, symbol_regex: str) -> list[str]:
    frame = client.query_df(
        trading_dates_sql(table),
        parameters={
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "symbol_regex": symbol_regex,
        },
    )
    if frame.empty or "TradingDay" not in frame:
        raise RuntimeError(f"no trading dates found for {year}")
    dates = pd.to_datetime(frame["TradingDay"], errors="coerce").dropna()
    normalized = sorted({str(value.date()) for value in dates})
    if not normalized:
        raise RuntimeError(f"no valid trading dates found for {year}")
    return normalized


def parquet_metadata(path: Path, expected_columns: tuple[str, ...]) -> dict[str, int]:
    parquet = pq.ParquetFile(path)
    actual_columns = tuple(parquet.schema_arrow.names)
    if actual_columns != expected_columns:
        raise RuntimeError(
            f"unexpected parquet schema for {path}: expected={expected_columns}, "
            f"actual={actual_columns}"
        )
    rows = int(parquet.metadata.num_rows)
    if rows <= 0:
        raise RuntimeError(f"empty parquet file: {path}")
    return {"rows": rows, "bytes": int(path.stat().st_size)}


def stream_parquet_atomic(
    client,
    *,
    query: str,
    parameters: dict[str, object],
    output_path: Path,
    expected_columns: tuple[str, ...],
    overwrite: bool,
) -> dict[str, int | bool]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        try:
            metadata = parquet_metadata(output_path, expected_columns)
        except (OSError, RuntimeError):
            pass
        else:
            return {**metadata, "reused": True}

    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    try:
        with client.raw_stream(query, parameters=parameters) as stream:
            with partial.open("wb") as file:
                for chunk in stream:
                    file.write(chunk)
                file.flush()
                os.fsync(file.fileno())
        metadata = parquet_metadata(partial, expected_columns)
        os.replace(partial, output_path)
        return {**metadata, "reused": False}
    finally:
        if partial.exists():
            partial.unlink()
