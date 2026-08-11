from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    managed_tick_client,
)
from opening_strength_fit.config import config_int, config_str, load_toml, run_id
from opening_strength_fit.io.json import write_json
from opening_strength_fit.raw_source import (
    CALENDAR_COLUMNS,
    CLOSE_REFERENCE_COLUMNS,
    DAILY_REFERENCE_COLUMNS,
    RAW_SOURCE_SCHEMA_VERSION,
    TICK_COLUMNS,
    calendar_sql,
    close_reference_sql,
    daily_reference_sql,
    label_coverage,
    parse_raw_source_years,
    parse_tick_windows,
    query_trading_dates,
    stream_parquet_atomic,
    tick_source_sql,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX


def _date_bounds(year: int, before_days: int, after_days: int) -> tuple[str, str]:
    start = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=before_days)
    end = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=after_days)
    return str(start.date()), str(end.date())


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
    windows = parse_tick_windows(config)
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

    dates = query_trading_dates(client, table=tick_table, year=year, symbol_regex=symbol_regex)
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
    years = parse_raw_source_years(config)
    selected_years = (args.year,) if args.year is not None else years
    unknown_years = sorted(set(selected_years).difference(years))
    if unknown_years:
        raise SystemExit(f"requested years are not configured: {unknown_years}")
    output_root = Path(
        args.output_root
        or config_str(config, "raw_source", "output_root", "output/raw_source_cache")
    )
    host = args.clickhouse_host or DEFAULT_CLICKHOUSE_TICK_HOST
    with managed_tick_client(host=host, port=int(args.clickhouse_port)) as client:
        for year in selected_years:
            build_year(
                client,
                config=config,
                config_path=config_path,
                year=int(year),
                output_root=output_root,
                overwrite=bool(args.overwrite),
            )


if __name__ == "__main__":
    main()
