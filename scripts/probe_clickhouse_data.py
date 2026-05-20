from __future__ import annotations

import argparse
from collections import Counter
import os
import re
import shlex
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    DEFAULT_TICK_END_OFFSET_US,
    DEFAULT_TICK_START_OFFSET_US,
    field_description_frame,
    get_tick_client,
    normalize_clickhouse_ticks,
    validate_table_name,
)
from opening_strength_fit.schema import available_depth_levels
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX


INDEX_COLUMNS = ("TradingDay", "Symbol", "ExchTimeOffsetUs")
CORE_COLUMNS = (
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
    "LocalTimeStamp",
)
DEPTH_RE = re.compile(r"^(Ask|Bid)(Price|Volume|Count)(\d+)$")


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        parts = shlex.split(line, comments=True)
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def offset_to_time(offset_us) -> str:
    if offset_us is None or pd.isna(offset_us):
        return ""
    return str(pd.Timestamp("2000-01-01") + pd.to_timedelta(int(offset_us), unit="us"))[-8:]


def dtype_summary(dtype_counts: Counter[str]) -> str:
    if len(dtype_counts) == 1:
        dtype, count = next(iter(dtype_counts.items()))
        return f"dtype={dtype}, count={count:,}"
    return ", ".join(
        f"{dtype}={count:,}" for dtype, count in sorted(dtype_counts.items())
    )


def inline_fields(fields: list[tuple[str, str]]) -> str:
    return ", ".join(f"{name}={dtype}" for name, dtype in fields)


def time_to_offset_us(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    elif len(parts) == 3:
        hour, minute, second = parts
    else:
        raise argparse.ArgumentTypeError("time must be HH:MM or HH:MM:SS")
    return int(((hour * 60 + minute) * 60 + second) * 1_000_000)


def describe_table(client, table: str) -> pd.DataFrame:
    table = validate_table_name(table)
    return client.query_df(f"DESCRIBE TABLE {table}")


def where_clause(args: argparse.Namespace) -> str:
    clauses = [
        "TradingDay >= {start_date:String}",
        "TradingDay <= {end_date:String}",
    ]
    if args.start_offset_us is not None:
        clauses.append("ExchTimeOffsetUs >= {start_offset_us:UInt64}")
    if args.end_offset_us is not None:
        clauses.append("ExchTimeOffsetUs <= {end_offset_us:UInt64}")
    if args.symbol:
        clauses.append("Symbol in {symbols:Array(String)}")
    if args.symbol_regex:
        clauses.append("match(Symbol, {symbol_regex:String})")
    return " and ".join(clauses)


def query_params(args: argparse.Namespace) -> dict:
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "start_offset_us": (
            None if args.start_offset_us is None else int(args.start_offset_us)
        ),
        "end_offset_us": None if args.end_offset_us is None else int(args.end_offset_us),
        "symbols": args.symbol,
        "symbol_regex": args.symbol_regex,
    }


def query_dataset_summary(client, table: str, args: argparse.Namespace) -> dict:
    table = validate_table_name(table)
    health_columns = ""
    if args.data_health:
        health_columns = """,
    countIf(AskPrice1 > 0) as positive_ask1_rows,
    countIf(BidPrice1 > 0) as positive_bid1_rows,
    min(Volume) as volume_min,
    max(Volume) as volume_max,
    min(Turnover) as turnover_min,
    max(Turnover) as turnover_max"""
    frame = client.query_df(
        f"""select
    count() as rows,
    uniqExact(TradingDay) as trading_days,
    uniqExact(Symbol) as symbols,
    min(TradingDay) as date_min,
    max(TradingDay) as date_max,
    min(ExchTimeOffsetUs) as offset_min,
    max(ExchTimeOffsetUs) as offset_max{health_columns}
from {table}
where {where_clause(args)}""",
        parameters=query_params(args),
    )
    if frame.empty:
        return {"rows": 0}
    out = frame.iloc[0].to_dict()
    out["time_min"] = offset_to_time(out.get("offset_min"))
    out["time_max"] = offset_to_time(out.get("offset_max"))
    return out


def query_date_layout(client, table: str, args: argparse.Namespace) -> pd.DataFrame:
    table = validate_table_name(table)
    frame = client.query_df(
        f"""select
    TradingDay as date,
    count() as rows,
    uniqExact(Symbol) as symbols,
    min(ExchTimeOffsetUs) as offset_min,
    max(ExchTimeOffsetUs) as offset_max
from {table}
where {where_clause(args)}
group by TradingDay
order by TradingDay""",
        parameters=query_params(args),
    )
    if not frame.empty:
        frame["time_min"] = frame["offset_min"].map(offset_to_time)
        frame["time_max"] = frame["offset_max"].map(offset_to_time)
    return frame


def query_symbol_layout(client, table: str, args: argparse.Namespace) -> pd.DataFrame:
    table = validate_table_name(table)
    frame = client.query_df(
        f"""select
    Symbol as symbol,
    count() as rows,
    uniqExact(TradingDay) as trading_days,
    min(TradingDay) as date_min,
    max(TradingDay) as date_max,
    min(ExchTimeOffsetUs) as offset_min,
    max(ExchTimeOffsetUs) as offset_max
from {table}
where {where_clause(args)}
group by Symbol
order by rows desc
limit {{top_symbols:UInt32}}""",
        parameters={**query_params(args), "top_symbols": int(args.top_symbols)},
    )
    if not frame.empty:
        frame["time_min"] = frame["offset_min"].map(offset_to_time)
        frame["time_max"] = frame["offset_max"].map(offset_to_time)
    return frame


def query_preview(client, table: str, args: argparse.Namespace) -> pd.DataFrame:
    if args.preview_rows <= 0:
        return pd.DataFrame()
    table = validate_table_name(table)
    return client.query_df(
        f"""select *
from {table}
where {where_clause(args)}
order by TradingDay, Symbol, ExchTimeOffsetUs
limit {{preview_rows:UInt32}}""",
        parameters={**query_params(args), "preview_rows": int(args.preview_rows)},
    )


def query_year_layout(client, table: str, args: argparse.Namespace) -> pd.DataFrame:
    table = validate_table_name(table)
    frame = client.query_df(
        f"""select
    toYear(TradingDay) as year,
    count() as rows,
    uniqExact(TradingDay) as trading_days,
    uniqExact(Symbol) as symbols,
    min(TradingDay) as date_min,
    max(TradingDay) as date_max
from {table}
where {where_clause(args)}
group by year
order by year""",
        parameters=query_params(args),
    )
    return frame


def schema_fields(schema: pd.DataFrame) -> list[tuple[str, str]]:
    fields = []
    for _, row in schema.iterrows():
        name = row.get("name", row.get("Name", ""))
        dtype = row.get("type", row.get("Type", ""))
        fields.append((str(name), str(dtype)))
    return fields


def depth_group_name(name: str) -> str | None:
    match = DEPTH_RE.match(name)
    if not match:
        return None
    side, kind, _level = match.groups()
    return f"{side.lower()}_{kind.lower()}"


def print_dataset_overview(
    *,
    table: str,
    schema: pd.DataFrame,
    summary: dict,
    preview: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    normalized_preview = normalize_clickhouse_ticks(preview) if not preview.empty else preview
    print(f"\n{table}")
    print("  clickhouse_connect: ok")
    print(f"  requested_date_range: {args.start_date} -> {args.end_date}")
    if args.start_offset_us is None and args.end_offset_us is None:
        print("  requested_time_window: <all>")
    else:
        print(
            "  requested_time_window: "
            f"{offset_to_time(args.start_offset_us)} -> "
            f"{offset_to_time(args.end_offset_us)}"
        )
    print(f"  requested_symbols: {','.join(args.symbol) if args.symbol else '<all>'}")
    print(f"  rows: {int(summary.get('rows') or 0):,}")
    print(f"  columns: {len(schema):,}")
    print(f"  trading_days: {int(summary.get('trading_days') or 0):,}")
    print(f"  symbols: {int(summary.get('symbols') or 0):,}")
    print(f"  date_min: {summary.get('date_min', '')}")
    print(f"  date_max: {summary.get('date_max', '')}")
    print(f"  time_min: {summary.get('time_min', '')}")
    print(f"  time_max: {summary.get('time_max', '')}")
    print("  row_grain: TradingDay x Symbol x ExchTimeOffsetUs")
    if "positive_ask1_rows" in summary:
        print(f"  positive_ask1_rows: {int(summary.get('positive_ask1_rows') or 0):,}")
        print(f"  positive_bid1_rows: {int(summary.get('positive_bid1_rows') or 0):,}")
        print(
            "  volume_range: "
            f"{float(summary.get('volume_min') or 0):,.0f} -> "
            f"{float(summary.get('volume_max') or 0):,.0f}"
        )
        print(
            "  turnover_range: "
            f"{float(summary.get('turnover_min') or 0):,.0f} -> "
            f"{float(summary.get('turnover_max') or 0):,.0f}"
        )
    if not normalized_preview.empty:
        print(
            "  normalized_preview_columns: "
            f"count={len(normalized_preview.columns):,}, "
            f"depth_levels={available_depth_levels(normalized_preview)}"
        )


def print_date_layout(frame: pd.DataFrame) -> None:
    print("  date_layout:")
    if frame.empty:
        print("    <empty>")
        return
    for _, row in frame.iterrows():
        print(
            f"    {row['date']}: "
            f"rows={int(row['rows']):,}, "
            f"symbols={int(row['symbols']):,}, "
            f"time_min={row['time_min']}, "
            f"time_max={row['time_max']}"
        )


def print_year_layout(frame: pd.DataFrame) -> None:
    print("  year_layout:")
    if frame.empty:
        print("    <empty>")
        return
    for _, row in frame.iterrows():
        print(
            f"    {int(row['year'])}: "
            f"rows={int(row['rows']):,}, "
            f"trading_days={int(row['trading_days']):,}, "
            f"symbols={int(row['symbols']):,}, "
            f"date_min={row['date_min']}, "
            f"date_max={row['date_max']}"
        )


def print_symbol_layout(frame: pd.DataFrame) -> None:
    print("  symbol_layout_top:")
    if frame.empty:
        print("    <empty>")
        return
    for _, row in frame.iterrows():
        print(
            f"    {row['symbol']}: "
            f"rows={int(row['rows']):,}, "
            f"trading_days={int(row['trading_days']):,}, "
            f"date_min={row['date_min']}, "
            f"date_max={row['date_max']}, "
            f"time_min={row['time_min']}, "
            f"time_max={row['time_max']}"
        )


def print_column_layout(schema: pd.DataFrame, preview: pd.DataFrame) -> None:
    fields = schema_fields(schema)
    field_map = dict(fields)
    index_fields = [(name, field_map[name]) for name in INDEX_COLUMNS if name in field_map]
    core_fields = [(name, field_map[name]) for name in CORE_COLUMNS if name in field_map]
    data_fields = [(name, dtype) for name, dtype in fields if name not in INDEX_COLUMNS]
    depth_counts: dict[str, int] = {}
    for name, _dtype in fields:
        group = depth_group_name(name)
        if group:
            depth_counts[group] = depth_counts.get(group, 0) + 1

    print("  column_layout:")
    if index_fields:
        print(f"    index_columns: {inline_fields(index_fields)}")
    print(f"    raw_columns: {dtype_summary(Counter(dtype for _, dtype in fields))}")
    print(f"    data_columns: {dtype_summary(Counter(dtype for _, dtype in data_fields))}")
    if core_fields:
        print(f"    core_columns: {inline_fields(core_fields)}")
    if depth_counts:
        print("    depth_columns:")
        for group in sorted(depth_counts):
            print(f"      {group}: {depth_counts[group]:,}")


def print_schema(schema: pd.DataFrame) -> None:
    print("  all_columns:")
    for name, dtype in schema_fields(schema):
        print(f"    {name}: {dtype}")


def print_field_notes() -> None:
    fields = field_description_frame()
    keep = {
        "TradingDay",
        "Symbol",
        "ExchTimeOffsetUs",
        "LastPrice",
        "Volume",
        "Turnover",
        "Status",
        "AskPrice1",
        "AskVolume1",
        "BidPrice1",
        "BidVolume1",
    }
    print("\nfield_notes:")
    for _, row in fields.loc[fields["field"].isin(keep)].iterrows():
        print(f"  {row['field']}: {row['description']}")


def print_preview(preview: pd.DataFrame) -> None:
    if preview.empty:
        return
    ticks = normalize_clickhouse_ticks(preview)
    preview_columns = [
        column
        for column in (
            "date",
            "symbol",
            "time",
            "status",
            "last_price",
            "volume",
            "turnover",
            "bid_price_1",
            "bid_volume_1",
            "ask_price_1",
            "ask_volume_1",
        )
        if column in ticks.columns
    ]
    print("\npreview:")
    print(ticks.loc[:, preview_columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe ClickHouse stock.tick universe layout."
    )
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--end-date", default="2100-12-31")
    parser.add_argument(
        "--symbol",
        nargs="*",
        default=[],
        help="Optional symbol filter. Omit to probe all symbols in the date range.",
    )
    parser.add_argument(
        "--a-share-only",
        action="store_true",
        help="Restrict probes to 00/30 SZ and 60/68 SH stock symbols.",
    )
    parser.add_argument("--symbol-regex", default="")
    parser.add_argument("--top-symbols", type=int, default=0)
    parser.add_argument("--preview-rows", type=int, default=0)
    parser.add_argument(
        "--year-layout",
        action="store_true",
        help="Print yearly row/date/symbol layout.",
    )
    parser.add_argument(
        "--date-layout",
        action="store_true",
        help="Print per-date layout. Useful only for narrowed date ranges.",
    )
    parser.add_argument(
        "--opening-window",
        action="store_true",
        help="Use the project default 09:15-09:45 opening query window.",
    )
    parser.add_argument(
        "--data-health",
        action="store_true",
        help="Also scan ask/bid positivity and volume/turnover ranges.",
    )
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--field-notes", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-env-file", action="store_true")
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
    parser.add_argument("--start-offset-us", type=int, default=None)
    parser.add_argument("--end-offset-us", type=int, default=None)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    args = parser.parse_args()

    if args.a_share_only and not args.symbol_regex:
        args.symbol_regex = DEFAULT_A_SHARE_SYMBOL_REGEX

    if args.opening_window:
        args.start_offset_us = DEFAULT_TICK_START_OFFSET_US
        args.end_offset_us = DEFAULT_TICK_END_OFFSET_US
    if args.start_time:
        args.start_offset_us = time_to_offset_us(args.start_time)
    if args.end_time:
        args.end_offset_us = time_to_offset_us(args.end_time)

    if not args.no_env_file:
        load_env_file(args.env_file)
        args.user = args.user or os.getenv("CLICKHOUSE_USER")
        args.password = args.password or os.getenv("CLICKHOUSE_PASSWORD")
        args.host = os.getenv("CLICKHOUSE_HOST", args.host)
        args.port = _env_int("CLICKHOUSE_PORT", args.port)
        args.table = os.getenv("CLICKHOUSE_TICK_TABLE", args.table)

    if not args.user or not args.password:
        raise SystemExit(
            "missing ClickHouse credentials: set CLICKHOUSE_USER/CLICKHOUSE_PASSWORD, "
            "source .env, or pass --user/--password"
        )

    client = get_tick_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )
    schema = describe_table(client, args.table)
    summary = query_dataset_summary(client, args.table, args)
    year_layout = query_year_layout(client, args.table, args) if args.year_layout else pd.DataFrame()
    date_layout = query_date_layout(client, args.table, args) if args.date_layout else pd.DataFrame()
    symbol_layout = (
        query_symbol_layout(client, args.table, args)
        if args.top_symbols and args.top_symbols > 0
        else pd.DataFrame()
    )
    preview = query_preview(client, args.table, args)

    print_dataset_overview(
        table=args.table,
        schema=schema,
        summary=summary,
        preview=preview,
        args=args,
    )
    if args.year_layout:
        print_year_layout(year_layout)
    if args.date_layout:
        print_date_layout(date_layout)
    if args.top_symbols and args.top_symbols > 0:
        print_symbol_layout(symbol_layout)
    print_column_layout(schema, preview)
    print_preview(preview)

    if args.field_notes:
        print_field_notes()
    if args.schema:
        print_schema(schema)


if __name__ == "__main__":
    main()
