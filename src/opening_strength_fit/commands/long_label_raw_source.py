from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    managed_tick_client,
)
from opening_strength_fit.config import config_str, load_toml, run_id
from opening_strength_fit.io.json import write_json
from opening_strength_fit.raw_source import (
    parse_raw_source_years,
    parse_tick_windows,
    query_trading_dates,
    stream_parquet_atomic,
    tick_source_sql,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

RAW_SOURCE_SCHEMA_VERSION = "long_label_raw_source_v1"
TICK_COLUMNS = (
    "TradingDay",
    "Symbol",
    "ExchTimeOffsetUs",
    "Volume",
    "Turnover",
)


def build_year(
    client,
    *,
    config: dict,
    config_path: Path,
    year: int,
    output_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    table = config_str(config, "raw_source", "tick_table", DEFAULT_CLICKHOUSE_TICK_TABLE)
    symbol_regex = config_str(config, "raw_source", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
    windows = parse_tick_windows(config)
    year_root = output_root / f"year={year}"
    success_path = year_root / "_SUCCESS"
    if overwrite and success_path.exists():
        success_path.unlink()

    dates = query_trading_dates(client, table=table, year=year, symbol_regex=symbol_regex)
    parameters: dict[str, object] = {"symbol_regex": symbol_regex}
    for index, (start, end) in enumerate(windows):
        parameters[f"window_start_{index}"] = start
        parameters[f"window_end_{index}"] = end

    files = []
    rows = 0
    size = 0
    for index, trading_day in enumerate(dates, start=1):
        output_path = year_root / "ticks" / f"date={trading_day}.parquet"
        metadata = stream_parquet_atomic(
            client,
            query=tick_source_sql(table, windows, output_columns=TICK_COLUMNS),
            parameters={**parameters, "trading_day": trading_day},
            output_path=output_path,
            expected_columns=TICK_COLUMNS,
            overwrite=overwrite,
        )
        rows += int(metadata["rows"])
        size += int(metadata["bytes"])
        files.append(
            {"date": trading_day, "path": str(output_path.relative_to(output_root)), **metadata}
        )
        print(
            f"[{run_id(config, config_path)}] year={year} tick={index}/{len(dates)} "
            f"date={trading_day} rows={metadata['rows']} reused={metadata['reused']}",
            flush=True,
        )

    manifest: dict[str, object] = {
        "schema_version": RAW_SOURCE_SCHEMA_VERSION,
        "run_id": run_id(config, config_path),
        "year": int(year),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": os.environ.get(
            "OPENING_STRENGTH_SOURCE_REVISION",
            os.environ.get("SOURCE_REVISION", "unknown"),
        ),
        "config_path": str(config_path),
        "contains_features": False,
        "contains_labels": False,
        "source_table": table,
        "symbol_regex": symbol_regex,
        "tick_windows_us": [list(window) for window in windows],
        "tick_columns": list(TICK_COLUMNS),
        "tick_deduplication": {
            "key": ["TradingDay", "Symbol", "ExchTimeOffsetUs"],
            "selection": "same latest-local-timestamp tie-break as raw_source_v2",
        },
        "trading_dates": {"count": len(dates), "first": dates[0], "last": dates[-1]},
        "tick_files": files,
        "ticks": {"rows": rows, "bytes": size},
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a projected raw tick cache for long-horizon VWAP labels."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--output-root", default="")
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
    selected = (int(args.year),) if args.year is not None else years
    unknown = sorted(set(selected).difference(years))
    if unknown:
        raise SystemExit(f"requested years are not configured: {unknown}")
    output_root = Path(
        args.output_root
        or config_str(config, "raw_source", "output_root", "output/long_label_raw_source")
    )
    with managed_tick_client(
        host=args.clickhouse_host or DEFAULT_CLICKHOUSE_TICK_HOST,
        port=int(args.clickhouse_port),
    ) as client:
        for year in selected:
            build_year(
                client,
                config=config,
                config_path=config_path,
                year=year,
                output_root=output_root,
                overwrite=bool(args.overwrite),
            )


if __name__ == "__main__":
    main()
