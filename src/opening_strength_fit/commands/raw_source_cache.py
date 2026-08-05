from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    get_tick_client,
    validate_table_name,
)
from opening_strength_fit.config import config_int, config_str, load_toml, run_id
from opening_strength_fit.io.json import write_json
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

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


def _parse_windows(config: dict) -> tuple[tuple[int, int], ...]:
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


def _parse_years(config: dict) -> tuple[int, ...]:
    section = config.get("raw_source", {})
    raw_years = section.get("years", []) if isinstance(section, dict) else []
    years = tuple(int(value) for value in raw_years)
    if not years:
        raise SystemExit("[raw_source].years must not be empty")
    if len(set(years)) != len(years):
        raise SystemExit("[raw_source].years must be unique")
    return years


def _parse_short_label_horizons(config: dict) -> tuple[int, ...]:
    section = config.get("raw_source", {})
    raw_horizons = (
        section.get("short_label_horizons_seconds", []) if isinstance(section, dict) else []
    )
    horizons = tuple(sorted({int(value) for value in raw_horizons}))
    if not horizons or any(value <= 0 for value in horizons):
        raise SystemExit("[raw_source].short_label_horizons_seconds must contain positive seconds")
    return horizons


def label_coverage(config: dict) -> dict[str, object]:
    windows = _parse_windows(config)
    horizons = _parse_short_label_horizons(config)
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
) -> str:
    table = validate_table_name(table)
    projection = ",\n        ".join(TICK_COLUMNS)
    outer_projection = ",\n    ".join(TICK_COLUMNS)
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
        {projection},
        LocalTimeStamp
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


def _date_bounds(year: int, before_days: int, after_days: int) -> tuple[str, str]:
    start = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=before_days)
    end = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=after_days)
    return str(start.date()), str(end.date())


def _tick_dates(client, *, table: str, year: int, symbol_regex: str) -> list[str]:
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


def build_year(
    client,
    *,
    config: dict,
    config_path: Path,
    year: int,
    output_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    tick_table = config_str(config, "raw_source", "tick_table", DEFAULT_CLICKHOUSE_TICK_TABLE)
    daily_table = config_str(config, "raw_source", "daily_table", "stock.daily_bar_jy")
    symbol_regex = config_str(config, "raw_source", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
    windows = _parse_windows(config)
    coverage = label_coverage(config)
    context_before_days = config_int(config, "raw_source", "context_before_days", 31)
    context_after_days = config_int(config, "raw_source", "context_after_days", 14)
    close_start_offset_us = config_int(
        config, "raw_source", "close_start_offset_us", 52_200_000_000
    )
    close_end_offset_us = config_int(config, "raw_source", "close_end_offset_us", 54_000_000_000)
    year_root = output_root / f"year={year}"
    success_path = year_root / "_SUCCESS"
    manifest_path = year_root / "manifest.json"
    if overwrite and success_path.exists():
        success_path.unlink()

    dates = _tick_dates(client, table=tick_table, year=year, symbol_regex=symbol_regex)
    tick_parameters: dict[str, object] = {"symbol_regex": symbol_regex}
    for index, (start, end) in enumerate(windows):
        tick_parameters[f"window_start_{index}"] = start
        tick_parameters[f"window_end_{index}"] = end

    tick_files: list[dict[str, object]] = []
    tick_rows = 0
    tick_bytes = 0
    for index, trading_day in enumerate(dates, start=1):
        output_path = year_root / "ticks" / f"date={trading_day}.parquet"
        metadata = stream_parquet_atomic(
            client,
            query=tick_source_sql(tick_table, windows),
            parameters={**tick_parameters, "trading_day": trading_day},
            output_path=output_path,
            expected_columns=TICK_COLUMNS,
            overwrite=overwrite,
        )
        tick_rows += int(metadata["rows"])
        tick_bytes += int(metadata["bytes"])
        tick_files.append(
            {
                "date": trading_day,
                "path": str(output_path.relative_to(output_root)),
                **metadata,
            }
        )
        print(
            f"[{run_id(config, config_path)}] year={year} tick={index}/{len(dates)} "
            f"date={trading_day} rows={metadata['rows']} reused={metadata['reused']}",
            flush=True,
        )

    context_start, context_end = _date_bounds(
        year, before_days=context_before_days, after_days=context_after_days
    )
    common_parameters = {
        "start_date": context_start,
        "end_date": context_end,
        "symbol_regex": symbol_regex,
    }
    daily_path = year_root / "daily_reference.parquet"
    daily = stream_parquet_atomic(
        client,
        query=daily_reference_sql(daily_table),
        parameters=common_parameters,
        output_path=daily_path,
        expected_columns=DAILY_REFERENCE_COLUMNS,
        overwrite=overwrite,
    )
    close_path = year_root / "close_reference.parquet"
    close = stream_parquet_atomic(
        client,
        query=close_reference_sql(tick_table),
        parameters={
            **common_parameters,
            "close_start_offset_us": close_start_offset_us,
            "close_end_offset_us": close_end_offset_us,
        },
        output_path=close_path,
        expected_columns=CLOSE_REFERENCE_COLUMNS,
        overwrite=overwrite,
    )
    calendar_path = year_root / "trading_calendar.parquet"
    calendar = stream_parquet_atomic(
        client,
        query=calendar_sql(tick_table),
        parameters=common_parameters,
        output_path=calendar_path,
        expected_columns=CALENDAR_COLUMNS,
        overwrite=overwrite,
    )

    manifest: dict[str, object] = {
        "schema_version": RAW_SOURCE_SCHEMA_VERSION,
        "run_id": run_id(config, config_path),
        "year": year,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": os.environ.get(
            "OPENING_STRENGTH_SOURCE_REVISION",
            os.environ.get("SOURCE_REVISION", "unknown"),
        ),
        "config_path": str(config_path),
        "contains_features": False,
        "contains_labels": False,
        "source_tables": {"ticks": tick_table, "daily_reference": daily_table},
        "symbol_regex": symbol_regex,
        "tick_windows_us": [list(window) for window in windows],
        "label_coverage": coverage,
        "tick_columns": list(TICK_COLUMNS),
        "tick_deduplication": {
            "key": ["TradingDay", "Symbol", "ExchTimeOffsetUs"],
            "selection": [
                "latest LocalTimeStamp",
                "TradeNum",
                "Volume",
                "Turnover",
                "LastPrice",
                "AskPrice1",
                "BidPrice1",
            ],
            "local_timestamp_persisted": False,
        },
        "trading_dates": {"count": len(dates), "first": dates[0], "last": dates[-1]},
        "tick_files": tick_files,
        "ticks": {"rows": tick_rows, "bytes": tick_bytes},
        "daily_reference": {
            "path": str(daily_path.relative_to(output_root)),
            "columns": list(DAILY_REFERENCE_COLUMNS),
            "context_start": context_start,
            "context_end": context_end,
            **daily,
        },
        "close_reference": {
            "path": str(close_path.relative_to(output_root)),
            "columns": list(CLOSE_REFERENCE_COLUMNS),
            "window_us": [close_start_offset_us, close_end_offset_us],
            "context_start": context_start,
            "context_end": context_end,
            **close,
        },
        "trading_calendar": {
            "path": str(calendar_path.relative_to(output_root)),
            "columns": list(CALENDAR_COLUMNS),
            "context_start": context_start,
            "context_end": context_end,
            **calendar,
        },
    }
    write_json(manifest_path, manifest, sort_keys=True, atomic=True)
    success_path.touch()
    print(
        f"[{run_id(config, config_path)}] complete year={year} dates={len(dates)} "
        f"tick_rows={tick_rows} tick_bytes={tick_bytes} root={year_root}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a label-free, feature-free raw ClickHouse source cache."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clickhouse-host", default=os.environ.get("CLICKHOUSE_HOST", ""))
    parser.add_argument(
        "--clickhouse-port",
        type=int,
        default=int(os.environ.get("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT)),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    years = _parse_years(config)
    selected_years = (args.year,) if args.year is not None else years
    unknown_years = sorted(set(selected_years).difference(years))
    if unknown_years:
        raise SystemExit(f"requested years are not configured: {unknown_years}")
    output_root = Path(
        args.output_root
        or config_str(config, "raw_source", "output_root", "output/raw_source_cache")
    )
    username = os.environ.get("CLICKHOUSE_USER", "")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    if not username or not password:
        raise SystemExit("ClickHouse credentials are missing; set CLICKHOUSE_USER/PASSWORD")
    host = args.clickhouse_host or DEFAULT_CLICKHOUSE_TICK_HOST
    client = get_tick_client(
        host=host,
        port=int(args.clickhouse_port),
        username=username,
        password=password,
    )
    try:
        for year in selected_years:
            build_year(
                client,
                config=config,
                config_path=config_path,
                year=int(year),
                output_root=output_root,
                overwrite=bool(args.overwrite),
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
