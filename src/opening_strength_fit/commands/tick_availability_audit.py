from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    get_tick_client,
    validate_table_name,
)
from opening_strength_fit.sampling import parse_clock_times
from opening_strength_fit.stock_pool import load_env_file_if_present
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX


def _clock_offset_us(clock: str) -> int:
    value = pd.Timestamp(f"2000-01-01 {clock}")
    midnight = value.normalize()
    return int((value - midnight) / pd.Timedelta(microseconds=1))


def _monthly_sample_dates(client, *, table: str, start_date: str, end_date: str) -> list[str]:
    frame = client.query_df(
        f"""select min(TradingDay) as sample_date
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
group by toYYYYMM(TradingDay)
order by sample_date""",
        parameters={"start_date": start_date, "end_date": end_date},
    )
    return pd.to_datetime(frame["sample_date"]).dt.strftime("%Y-%m-%d").tolist()


def _selected_states_for_day(
    client,
    *,
    table: str,
    trading_day: str,
    decision_offsets_us: list[int],
    symbol_regex: str,
) -> pd.DataFrame:
    # This tuple reproduces raw_source_cache.tick_source_sql followed by
    # clock-state sampling: latest revision of the latest exchange state <= clock.
    order_tuple = """tuple(
            ExchTimeOffsetUs,
            arrayMax(mapValues(LocalTimeStamp)),
            TradeNum,
            Volume,
            Turnover,
            LastPrice,
            AskPrice1,
            BidPrice1
        )"""
    sql = f"""select
    Symbol,
    decision_offset_us,
    argMax(ExchTimeOffsetUs, {order_tuple}) as source_offset_us,
    argMax(arrayMax(mapValues(LocalTimeStamp)), {order_tuple}) as receipt_epoch_raw
from {table}
array join {{decision_offsets_us:Array(UInt64)}} as decision_offset_us
where TradingDay = {{trading_day:Date}}
  and match(Symbol, {{symbol_regex:String}})
  and ExchTimeOffsetUs >= {{start_offset_us:UInt64}}
  and ExchTimeOffsetUs <= decision_offset_us
group by Symbol, decision_offset_us"""
    return client.query_df(
        sql,
        parameters={
            "decision_offsets_us": decision_offsets_us,
            "trading_day": trading_day,
            "symbol_regex": symbol_regex,
            "start_offset_us": min(33_300_000_000, min(decision_offsets_us)),
        },
    )


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    if values.empty:
        return {name: None for name in ("p50", "p95", "p99", "max")}
    return {
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def normalize_receipt_epoch_us(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Normalize observed LocalTimeStamp epochs while retaining unit provenance.

    The source table contains zero before availability timestamps were populated,
    epoch microseconds in early 2025, and epoch nanoseconds later in 2025.
    """

    raw = pd.to_numeric(values, errors="coerce").astype("float64")
    unit = pd.Series("unknown", index=values.index, dtype="object")
    normalized = pd.Series(float("nan"), index=values.index, dtype="float64")
    missing = raw.isna() | raw.le(0)
    milliseconds = raw.ge(1e11) & raw.lt(1e14)
    microseconds = raw.ge(1e14) & raw.lt(1e17)
    nanoseconds = raw.ge(1e17) & raw.lt(1e20)
    unit.loc[missing] = "missing"
    unit.loc[milliseconds] = "milliseconds"
    unit.loc[microseconds] = "microseconds"
    unit.loc[nanoseconds] = "nanoseconds"
    normalized.loc[milliseconds] = raw.loc[milliseconds] * 1_000
    normalized.loc[microseconds] = raw.loc[microseconds]
    normalized.loc[nanoseconds] = raw.loc[nanoseconds] / 1_000
    return normalized, unit


def summarize_availability(frame: pd.DataFrame) -> dict[str, object]:
    covered = frame.loc[frame["receipt_timestamp_covered"]]
    delay = covered["receipt_after_decision_seconds"]
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["Symbol"].nunique()),
        "receipt_timestamp_unit_rows": {
            str(unit): int(count)
            for unit, count in frame["receipt_timestamp_unit"].value_counts().items()
        },
        "receipt_timestamp_rows": int(len(covered)),
        "receipt_timestamp_coverage": float(len(covered) / len(frame)) if len(frame) else 0.0,
        "receipt_after_decision_rows": int(delay.gt(0).sum()),
        "receipt_after_decision_fraction_of_covered": (
            float(delay.gt(0).mean()) if len(covered) else None
        ),
        "receipt_after_decision_gt_2s_fraction_of_covered": (
            float(delay.gt(2).mean()) if len(covered) else None
        ),
        "receipt_after_decision_gt_6s_fraction_of_covered": (
            float(delay.gt(6).mean()) if len(covered) else None
        ),
        "receipt_after_decision_seconds": _quantiles(delay),
        "exchange_state_age_seconds": _quantiles(frame["exchange_state_age_seconds"]),
    }


def build_report(
    frame: pd.DataFrame,
    *,
    dates: list[str],
    clocks: list[str],
    source_cutoff_seconds: int,
) -> dict[str, object]:
    by_date = {
        str(date): summarize_availability(group)
        for date, group in frame.groupby("trading_day", sort=True)
    }
    by_clock = {
        str(clock): summarize_availability(group)
        for clock, group in frame.groupby("decision_time", sort=True)
    }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": (
            "latest local-receipt revision of latest exchange timestamp <= "
            "logical decision clock minus source_cutoff_seconds"
        ),
        "source_cutoff_seconds": int(source_cutoff_seconds),
        "dates": dates,
        "decision_times": clocks,
        "overall": summarize_availability(frame),
        "by_date": by_date,
        "by_decision_time": by_clock,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether selected ClickHouse tick states were locally received by decision time."
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Explicit trading dates; default is the first available date of every month.",
    )
    parser.add_argument("--decision-times", nargs="*", default=["09:31:00-09:40:00"])
    parser.add_argument("--source-cutoff-seconds", type=int, default=0)
    parser.add_argument("--symbol-regex", default=DEFAULT_A_SHARE_SYMBOL_REGEX)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_env_file_if_present()
    client = get_tick_client(
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        host=os.environ.get("CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST),
        port=int(os.environ.get("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT)),
    )
    table = validate_table_name(
        os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE)
    )
    dates = list(args.dates or []) or _monthly_sample_dates(
        client,
        table=table,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    clocks = parse_clock_times(args.decision_times)
    offsets = [_clock_offset_us(clock) for clock in clocks]
    if args.source_cutoff_seconds < 0:
        raise SystemExit("--source-cutoff-seconds must be >= 0")
    cutoff_offsets = [offset - int(args.source_cutoff_seconds) * 1_000_000 for offset in offsets]
    parts: list[pd.DataFrame] = []
    for trading_day in dates:
        day = _selected_states_for_day(
            client,
            table=table,
            trading_day=trading_day,
            decision_offsets_us=cutoff_offsets,
            symbol_regex=args.symbol_regex,
        )
        if day.empty:
            continue
        midnight_epoch_us = (
            pd.Timestamp(trading_day, tz="Asia/Shanghai").tz_convert("UTC").value // 1_000
        )
        logical_offset_by_cutoff = dict(zip(cutoff_offsets, offsets, strict=True))
        logical_decision_offset_us = day["decision_offset_us"].map(logical_offset_by_cutoff)
        decision_epoch_us = midnight_epoch_us + logical_decision_offset_us.astype("int64")
        day["trading_day"] = trading_day
        day["decision_time"] = day["decision_offset_us"].map(
            dict(zip(cutoff_offsets, clocks, strict=True))
        )
        day["exchange_state_age_seconds"] = (
            logical_decision_offset_us - day["source_offset_us"]
        ) / 1_000_000
        receipt_epoch_us, receipt_unit = normalize_receipt_epoch_us(day["receipt_epoch_raw"])
        day["receipt_timestamp_unit"] = receipt_unit
        day["receipt_timestamp_covered"] = receipt_epoch_us.notna()
        day["receipt_after_decision_seconds"] = (receipt_epoch_us - decision_epoch_us) / 1_000_000
        day.loc[~day["receipt_timestamp_covered"], "receipt_after_decision_seconds"] = pd.NA
        parts.append(day)
    if not parts:
        raise SystemExit("availability audit returned no selected tick states")

    report = build_report(
        pd.concat(parts, ignore_index=True),
        dates=dates,
        clocks=clocks,
        source_cutoff_seconds=args.source_cutoff_seconds,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
