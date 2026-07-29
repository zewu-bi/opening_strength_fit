from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    get_tick_client,
    validate_table_name,
)
from opening_strength_fit.config import (
    config_float,
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.daily_return_labels import (
    CLOSE_TO_NEXT_CLOSE_COLUMNS,
    CLOSE_TO_NEXT_CLOSE_LABEL_COL,
    NEXT_SESSION_OPEN_CLOSE_COLUMNS,
    NEXT_SESSION_OPEN_CLOSE_LABEL_COL,
    build_close_to_next_close_labels,
    build_next_session_open_close_labels,
)
from opening_strength_fit.io import frame_columns, write_frame_atomic, write_json
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

DEFAULT_DAILY_BAR_TABLE = "stock.daily_bar_jy"
SUPPORTED_LABEL_KINDS = {"next_session_open_close", "close_to_next_close"}


def query_daily_open_close_bars(
    client,
    *,
    table: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    table = validate_table_name(table)
    return client.query_df(
        f"""select
    toString(TradingDay) as date,
    Symbol as symbol,
    OpenPrice as open_price,
    ClosePrice as close_price,
    PreClosePrice as preclose_price
from {table}
where TradingDay >= {{start_date:Date}}
  and TradingDay <= {{end_date:Date}}
order by TradingDay, Symbol""",
        parameters={"start_date": start_date, "end_date": end_date},
    )


def _year_bounds(
    year: int,
    *,
    feature_start_date: str,
    feature_end_date: str,
) -> tuple[str, str] | None:
    start = max(pd.Timestamp(feature_start_date), pd.Timestamp(f"{year}-01-01"))
    end = min(pd.Timestamp(feature_end_date), pd.Timestamp(f"{year}-12-31"))
    if start > end:
        return None
    return str(start.date()), str(end.date())


def _label_definition(
    label_kind: str,
) -> tuple[tuple[str, ...], str, object, str]:
    if label_kind == "next_session_open_close":
        return (
            NEXT_SESSION_OPEN_CLOSE_COLUMNS,
            NEXT_SESSION_OPEN_CLOSE_LABEL_COL,
            build_next_session_open_close_labels,
            "daily_next_session_open_close_labels_v1",
        )
    if label_kind == "close_to_next_close":
        return (
            CLOSE_TO_NEXT_CLOSE_COLUMNS,
            CLOSE_TO_NEXT_CLOSE_LABEL_COL,
            build_close_to_next_close_labels,
            "daily_close_to_next_close_labels_v1",
        )
    allowed = ", ".join(sorted(SUPPORTED_LABEL_KINDS))
    raise SystemExit(f"unsupported daily label_kind={label_kind!r}; expected one of {allowed}")


def _validate_existing_shard(label_path: Path, *, expected_columns: tuple[str, ...]) -> None:
    expected = set(expected_columns)
    actual = frame_columns(label_path)
    if actual != expected:
        raise SystemExit(
            f"existing daily label shard has unexpected schema: {label_path}; "
            f"expected={sorted(expected)} actual={sorted(actual)}"
        )


def build_daily_return_label_cache(
    *,
    output_root: Path,
    label_kind: str = "next_session_open_close",
    feature_start_date: str,
    feature_end_date: str,
    calendar_days_after: int,
    fee_bps: float,
    symbol_regex: str,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    output_columns, label_col, builder, schema_version = _label_definition(label_kind)
    if not username or not password:
        raise SystemExit(
            "daily return labels need ClickHouse credentials; set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD"
        )
    start = pd.Timestamp(feature_start_date)
    end = pd.Timestamp(feature_end_date)
    if start > end:
        raise SystemExit("feature_start_date must be <= feature_end_date")
    if calendar_days_after < 1:
        raise SystemExit("calendar_days_after must be positive")

    client = get_tick_client(
        host=host or DEFAULT_CLICKHOUSE_TICK_HOST,
        port=int(port),
        username=username,
        password=password,
    )
    summaries: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for year in range(start.year, end.year + 1):
        bounds = _year_bounds(
            year,
            feature_start_date=feature_start_date,
            feature_end_date=feature_end_date,
        )
        if bounds is None:
            continue
        year_start, year_end = bounds
        shard_dir = output_root / f"year={year}"
        label_path = shard_dir / "labels.parquet"
        summary_path = shard_dir / "summary.json"
        if label_path.exists() and summary_path.exists() and not overwrite:
            _validate_existing_shard(label_path, expected_columns=output_columns)
            summaries.append(
                {
                    "year": year,
                    "action": "reused",
                    "output": str(label_path),
                }
            )
            continue
        if label_path.exists() and not overwrite:
            raise SystemExit(f"incomplete existing shard, pass --overwrite: {shard_dir}")

        lookup_end = str(
            (pd.Timestamp(year_end) + pd.Timedelta(days=int(calendar_days_after))).date()
        )
        bars = query_daily_open_close_bars(
            client,
            table=table,
            start_date=year_start,
            end_date=lookup_end,
        )
        labels = builder(
            bars,
            feature_start_date=year_start,
            feature_end_date=year_end,
            fee_bps=fee_bps,
            symbol_regex=symbol_regex,
        )
        duplicate_keys = int(labels.duplicated(["date", "symbol"]).sum())
        causal_violations = int((labels["target_date"] <= labels["date"]).sum())
        if duplicate_keys or causal_violations:
            raise RuntimeError(
                "daily label validation failed: "
                f"duplicate_keys={duplicate_keys} causal_violations={causal_violations}"
            )
        summary = {
            "year": year,
            "action": "built",
            "feature_start_date": year_start,
            "feature_end_date": year_end,
            "lookup_end_date": lookup_end,
            "source_rows": int(len(bars)),
            "label_rows": int(len(labels)),
            "symbols": int(labels["symbol"].nunique()),
            "feature_dates": int(labels["date"].nunique()),
            "target_dates": int(labels["target_date"].nunique()),
            "label_non_null": int(labels[label_col].notna().sum()),
            "duplicate_keys": duplicate_keys,
            "causal_date_violations": causal_violations,
            "output": str(label_path),
            "schema": list(output_columns),
        }
        write_frame_atomic(labels, label_path)
        write_json(summary_path, summary, atomic=True)
        summaries.append(summary)
        print_mapping(f"daily_label_year_{year}", summary)

    write_json(
        output_root / "manifest.json",
        {
            "schema_version": schema_version,
            "label_kind": label_kind,
            "feature_start_date": feature_start_date,
            "feature_end_date": feature_end_date,
            "fee_bps": fee_bps,
            "label_column": label_col,
            "schema": list(output_columns),
            "shards": summaries,
        },
        atomic=True,
    )
    (output_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build calendar-aligned daily return labels from ClickHouse."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config)
    run_name = run_id(config, args.config)
    output_root_raw = config_str(config, "daily_return_labels", "output_root", "").strip()
    if not output_root_raw:
        raise SystemExit("missing [daily_return_labels].output_root")
    output_root = Path(output_root_raw)
    feature_start_date = config_str(
        config,
        "daily_return_labels",
        "feature_start_date",
        "2019-01-02",
    )
    feature_end_date = config_str(
        config,
        "daily_return_labels",
        "feature_end_date",
        "2025-12-31",
    )
    calendar_days_after = config_int(
        config,
        "daily_return_labels",
        "calendar_days_after",
        14,
    )
    fee_bps = config_float(config, "daily_return_labels", "fee_bps", 0.0)
    label_kind = config_str(
        config,
        "daily_return_labels",
        "label_kind",
        "next_session_open_close",
    ).strip()
    _, label_col, _, _ = _label_definition(label_kind)
    symbol_regex = config_str(
        config,
        "universe",
        "symbol_regex",
        DEFAULT_A_SHARE_SYMBOL_REGEX,
    )
    host = (
        config_str(config, "clickhouse", "host", "")
        or os.environ.get("CLICKHOUSE_HOST", "")
        or DEFAULT_CLICKHOUSE_TICK_HOST
    )
    port = int(
        config_str(config, "clickhouse", "port", "")
        or os.environ.get("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT)
    )
    table = (
        config_str(config, "clickhouse", "table", "")
        or os.environ.get("CLICKHOUSE_DAILY_BAR_TABLE", "")
        or DEFAULT_DAILY_BAR_TABLE
    )
    username = os.environ.get("CLICKHOUSE_USER", "")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    trace_dir = Path(
        args.output_dir
        or config_str(
            config,
            "output",
            "local_dir",
            f"output/artifacts/cache_builds/{run_name}",
        )
    )
    trace_dir.mkdir(parents=True, exist_ok=True)

    print_mapping(
        "daily_return_labels",
        {
            "run_id": run_name,
            "output_root": str(output_root),
            "feature_start_date": feature_start_date,
            "feature_end_date": feature_end_date,
            "table": table,
            "label_kind": label_kind,
            "label_column": label_col,
        },
    )
    summaries = build_daily_return_label_cache(
        output_root=output_root,
        label_kind=label_kind,
        feature_start_date=feature_start_date,
        feature_end_date=feature_end_date,
        calendar_days_after=calendar_days_after,
        fee_bps=fee_bps,
        symbol_regex=symbol_regex,
        host=host,
        port=port,
        username=username,
        password=password,
        table=table,
        overwrite=args.overwrite,
    )
    trace = {
        "run_id": run_name,
        "output_root": str(output_root),
        "feature_start_date": feature_start_date,
        "feature_end_date": feature_end_date,
        "label_kind": label_kind,
        "label_column": label_col,
        "shards": summaries,
    }
    write_json(trace_dir / "daily_return_label_cache_trace.json", trace, atomic=True)
    write_json(trace_dir / "daily_return_label_cache_summary.json", trace, atomic=True)
    print_mapping(
        "daily_return_label_cache_summary",
        {
            "shards": len(summaries),
            "built": sum(item.get("action") == "built" for item in summaries),
            "reused": sum(item.get("action") == "reused" for item in summaries),
            "output_root": str(output_root),
        },
    )


if __name__ == "__main__":
    main()
