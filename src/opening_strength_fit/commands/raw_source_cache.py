from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.config import config_int, config_str
from opening_strength_fit.io.json import write_json
from opening_strength_fit.raw_source import (
    CALENDAR_COLUMNS,
    CLOSE_REFERENCE_COLUMNS,
    DAILY_REFERENCE_COLUMNS,
    RAW_SOURCE_SCHEMA_VERSION,
    TICK_COLUMNS,
    build_tick_source_manifest,
    calendar_sql,
    close_reference_sql,
    daily_reference_sql,
    label_coverage,
    parse_tick_windows,
    raw_source_parser,
    run_raw_source_builder,
    stream_parquet_atomic,
)

_parse_windows = parse_tick_windows


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
    daily_table = config_str(config, "raw_source", "daily_table", "stock.daily_bar_jy")
    coverage = label_coverage(config)
    context_before_days = config_int(config, "raw_source", "context_before_days", 31)
    context_after_days = config_int(config, "raw_source", "context_after_days", 14)
    close_start_offset_us = config_int(
        config, "raw_source", "close_start_offset_us", 52_200_000_000
    )
    close_end_offset_us = config_int(config, "raw_source", "close_end_offset_us", 54_000_000_000)
    manifest, year_root = build_tick_source_manifest(
        client,
        config=config,
        config_path=config_path,
        year=year,
        output_root=output_root,
        columns=TICK_COLUMNS,
        overwrite=overwrite,
        schema_version=RAW_SOURCE_SCHEMA_VERSION,
    )
    tick_table = str(manifest.pop("source_table"))
    symbol_regex = str(manifest["symbol_regex"])

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

    manifest.update(
        {
            "source_tables": {"ticks": tick_table, "daily_reference": daily_table},
            "label_coverage": coverage,
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
    )
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    (year_root / "_SUCCESS").touch()
    dates = manifest["trading_dates"]
    ticks = manifest["ticks"]
    print(
        f"[{manifest['run_id']}] complete year={year} dates={dates['count']} "
        f"tick_rows={ticks['rows']} tick_bytes={ticks['bytes']} root={year_root}",
        flush=True,
    )
    return manifest


def main() -> None:
    args = raw_source_parser(
        "Extract a label-free, feature-free raw ClickHouse source cache."
    ).parse_args()
    run_raw_source_builder(
        args,
        default_output_root="output/raw_source_cache",
        build_year=build_year,
    )


if __name__ == "__main__":
    main()
