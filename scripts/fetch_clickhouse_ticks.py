from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    DEFAULT_TICK_END_OFFSET_US,
    DEFAULT_TICK_START_OFFSET_US,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_window,
)
from opening_strength_fit.io import write_frame
from opening_strength_fit.reports import dataset_summary, print_mapping


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch one symbol/day tick window from ClickHouse."
    )
    parser.add_argument("--symbol", default="000925.SZ")
    parser.add_argument("--date", dest="trading_day", default="2021-09-22")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--host",
        default=os.getenv("CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT),
    )
    parser.add_argument("--user", default=os.getenv("CLICKHOUSE_USER"))
    parser.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD"))
    parser.add_argument(
        "--table",
        default=os.getenv("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
    )
    parser.add_argument("--start-offset-us", type=int, default=DEFAULT_TICK_START_OFFSET_US)
    parser.add_argument("--end-offset-us", type=int, default=DEFAULT_TICK_END_OFFSET_US)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="write raw ClickHouse column names instead of project-standard columns",
    )
    args = parser.parse_args()

    if not args.user or not args.password:
        raise SystemExit(
            "missing ClickHouse credentials: pass --user/--password or set "
            "CLICKHOUSE_USER and CLICKHOUSE_PASSWORD"
        )

    client = get_tick_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )
    ticks = query_tick_window(
        client,
        symbol=args.symbol,
        trading_day=args.trading_day,
        table=args.table,
        start_offset_us=args.start_offset_us,
        end_offset_us=args.end_offset_us,
    )
    if not args.raw:
        ticks = normalize_clickhouse_ticks(ticks)

    write_frame(ticks, args.output)
    print_mapping("clickhouse_ticks", dataset_summary(ticks))
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
