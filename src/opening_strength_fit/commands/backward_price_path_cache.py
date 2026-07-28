from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.backward_price_paths import (
    BACKWARD_PRICE_PATH_SCHEMA_VERSION,
    DEFAULT_VALID_PRICE_STATUSES,
    HORIZON_MINUTES,
    assemble_backward_price_sequence,
)
from opening_strength_fit.clickhouse_ticks import get_tick_client, validate_table_name
from opening_strength_fit.config import (
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import write_json
from opening_strength_fit.stock_pool import load_stock_pool
from opening_strength_fit.temporal_analysis import TARGET_COLUMN, write_sequence_npz
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

_OPEN_REFERENCE_START_US = 33_300_000_000
_MORNING_OPEN_US = 34_200_000_000
_MORNING_CLOSE_US = 41_400_000_000
_AFTERNOON_OPEN_US = 46_800_000_000
_AFTERNOON_CLOSE_US = 54_000_000_000
_MINUTE_US = 60_000_000


def endpoint_price_state_sql(table: str) -> str:
    table = validate_table_name(table)
    ordering = "tuple(ExchTimeOffsetUs, arrayMax(mapValues(LocalTimeStamp)))"
    return f"""
SELECT symbol, state_index, price, status, source_offset_us
FROM
(
    SELECT
        Symbol AS symbol,
        toInt16(0) AS state_index,
        argMax(LastPrice, {ordering}) AS price,
        argMax(Status, {ordering}) AS status,
        max(ExchTimeOffsetUs) AS source_offset_us
    FROM {table}
    WHERE TradingDay = {{trading_day:Date}}
      AND ExchTimeOffsetUs >= {{open_reference_start_us:Int64}}
      AND ExchTimeOffsetUs <= {{morning_open_us:Int64}}
      AND match(Symbol, {{symbol_regex:String}})
    GROUP BY Symbol

    UNION ALL

    SELECT
        Symbol AS symbol,
        toInt16(
            1 + if(
                ExchTimeOffsetUs < {{afternoon_open_us:Int64}},
                intDiv(ExchTimeOffsetUs - {{morning_open_us:Int64}}, {{minute_us:Int64}}),
                120 + intDiv(
                    ExchTimeOffsetUs - {{afternoon_open_us:Int64}},
                    {{minute_us:Int64}}
                )
            )
        ) AS state_index,
        argMax(LastPrice, {ordering}) AS price,
        argMax(Status, {ordering}) AS status,
        max(ExchTimeOffsetUs) AS source_offset_us
    FROM {table}
    WHERE TradingDay = {{trading_day:Date}}
      AND (
        (
            ExchTimeOffsetUs >= {{morning_open_us:Int64}}
            AND ExchTimeOffsetUs < {{morning_close_us:Int64}}
        )
        OR
        (
            ExchTimeOffsetUs >= {{afternoon_open_us:Int64}}
            AND ExchTimeOffsetUs < {{afternoon_close_us:Int64}}
        )
      )
      AND match(Symbol, {{symbol_regex:String}})
    GROUP BY Symbol, state_index
)
ORDER BY symbol, state_index
"""


def query_endpoint_price_states(
    client,
    *,
    trading_day: str,
    table: str,
    symbol_regex: str,
) -> pd.DataFrame:
    return client.query_df(
        endpoint_price_state_sql(table),
        parameters={
            "trading_day": trading_day,
            "symbol_regex": symbol_regex,
            "open_reference_start_us": _OPEN_REFERENCE_START_US,
            "morning_open_us": _MORNING_OPEN_US,
            "morning_close_us": _MORNING_CLOSE_US,
            "afternoon_open_us": _AFTERNOON_OPEN_US,
            "afternoon_close_us": _AFTERNOON_CLOSE_US,
            "minute_us": _MINUTE_US,
        },
    )


def _load_year_labels(label_root: Path, year: int) -> pd.DataFrame:
    path = label_root / f"year={year}" / "labels.parquet"
    if not path.exists():
        raise SystemExit(f"missing daily label shard: {path}")
    labels = pd.read_parquet(path, columns=["date", "symbol", TARGET_COLUMN])
    labels["date"] = labels["date"].astype(str)
    return labels


def _pool_symbols(pool: pd.DataFrame, date: str) -> set[str]:
    if date not in pool.index:
        return set()
    row = pool.loc[date]
    return set(row.index[row.to_numpy(dtype=bool, copy=False)].astype(str))


def _year_from_args(args, config: dict) -> int:
    if args.year is not None:
        return int(args.year)
    start_year = config_int(config, "backward_price_paths", "start_year", 2019)
    index_raw = os.environ.get("JOB_COMPLETION_INDEX")
    if index_raw is None:
        raise SystemExit("pass --year or set JOB_COMPLETION_INDEX")
    return start_year + int(index_raw)


def _maybe_finalize_root(
    output_root: Path,
    *,
    start_year: int,
    end_year: int,
    current_run_id: str,
) -> None:
    year_success = [
        output_root / f"year={year}" / "_SUCCESS" for year in range(start_year, end_year + 1)
    ]
    if not all(path.exists() for path in year_success):
        return
    summaries = []
    for year in range(start_year, end_year + 1):
        summary_path = output_root / f"year={year}" / "summary.json"
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    write_json(
        output_root / "manifest.json",
        {
            "run_id": current_run_id,
            "schema_version": BACKWARD_PRICE_PATH_SCHEMA_VERSION,
            "feature_definition": "P(t) / P(t-h trading minutes) - 1",
            "price_column": "LastPrice",
            "horizons_minutes": list(HORIZON_MINUTES),
            "years": summaries,
            "days": sum(int(item["days"]) for item in summaries),
        },
        atomic=True,
    )
    write_json(
        output_root / "_SUCCESS",
        {
            "run_id": current_run_id,
            "schema_version": BACKWARD_PRICE_PATH_SCHEMA_VERSION,
        },
        atomic=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build endpoint-aligned backward-looking price-return sequences."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    config = load_toml(args.config)
    current_run_id = run_id(config, args.config)
    year = _year_from_args(args, config)
    start_year = config_int(config, "backward_price_paths", "start_year", 2019)
    end_year = config_int(config, "backward_price_paths", "end_year", 2025)
    if not start_year <= year <= end_year:
        raise SystemExit(f"year {year} is outside configured range {start_year}..{end_year}")

    output_root = Path(config_str(config, "backward_price_paths", "sequence_root", ""))
    label_root = Path(config_str(config, "backward_price_paths", "label_root", ""))
    if not str(output_root) or not str(label_root):
        raise SystemExit("[backward_price_paths] requires sequence_root and label_root")
    year_root = output_root / f"year={year}"
    year_root.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    host = config_str(config, "clickhouse", "host", "")
    port = config_int(config, "clickhouse", "port", 8123)
    table = config_str(config, "clickhouse", "table", "stock.tick")
    username = os.environ.get("CLICKHOUSE_USER", "")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    if not username or not password:
        raise SystemExit("missing CLICKHOUSE_USER or CLICKHOUSE_PASSWORD")
    client = get_tick_client(
        host=host,
        port=port,
        username=username,
        password=password,
    )

    labels = _load_year_labels(label_root, year)
    pool = load_stock_pool(
        config_str(config, "stock_pool", "path", "lml.bzw@ssd/data/pool_L.parquet")
    )
    dates = sorted(labels["date"].unique())
    symbol_regex = config_str(
        config,
        "universe",
        "symbol_regex",
        DEFAULT_A_SHARE_SYMBOL_REGEX,
    )
    valid_statuses = config_list(
        config,
        "backward_price_paths",
        "valid_statuses",
        list(DEFAULT_VALID_PRICE_STATUSES),
    )

    summaries: list[dict[str, object]] = []
    for completed, date in enumerate(dates, start=1):
        sequence_path = year_root / f"date={date}" / "sequence.npz"
        if sequence_path.exists():
            with np.load(sequence_path, allow_pickle=False) as loaded:
                summaries.append(
                    {
                        "date": date,
                        "action": "resumed",
                        "symbols": int(len(loaded["symbols"])),
                        "target_valid": int(np.isfinite(loaded["target"]).sum()),
                    }
                )
            continue

        states = query_endpoint_price_states(
            client,
            trading_day=date,
            table=table,
            symbol_regex=symbol_regex,
        )
        day_labels = labels.loc[
            labels["date"].eq(date),
            ["symbol", TARGET_COLUMN],
        ]
        arrays = assemble_backward_price_sequence(
            states,
            day_labels,
            pool_symbols=_pool_symbols(pool, date),
            valid_statuses=valid_statuses,
        )
        write_sequence_npz(sequence_path, arrays)
        valid = arrays["valid"]
        summary = {
            "date": date,
            "action": "built",
            "state_rows": int(len(states)),
            "symbols": int(len(arrays["symbols"])),
            "target_valid": int(np.isfinite(arrays["target"]).sum()),
            "pool_members": int(arrays["pool_member"].sum()),
            "valid_1m": int(valid[:, 0, :].sum()),
            "valid_10m": int(valid[:, 1, :].sum()),
            "valid_60m": int(valid[:, 2, :].sum()),
        }
        summaries.append(summary)
        if completed == 1 or completed % 10 == 0 or completed == len(dates):
            print(
                f"year={year} completed={completed}/{len(dates)} "
                f"date={date} symbols={summary['symbols']}",
                flush=True,
            )

    year_summary = {
        "year": year,
        "days": len(summaries),
        "built_days": sum(item["action"] == "built" for item in summaries),
        "resumed_days": sum(item["action"] == "resumed" for item in summaries),
        "mean_symbols": float(np.mean([int(item["symbols"]) for item in summaries])),
        "target_valid": sum(int(item["target_valid"]) for item in summaries),
    }
    write_json(year_root / "summary.json", year_summary, atomic=True)
    write_json(
        year_root / "_SUCCESS",
        {
            "run_id": current_run_id,
            "year": year,
            "schema_version": BACKWARD_PRICE_PATH_SCHEMA_VERSION,
        },
        atomic=True,
    )
    _maybe_finalize_root(
        output_root,
        start_year=start_year,
        end_year=end_year,
        current_run_id=current_run_id,
    )
    write_json(
        Path(args.output_dir) / f"year={year}_trace.json",
        year_summary,
        atomic=True,
    )
    print(f"completed backward price paths for year={year}: {year_summary}", flush=True)


if __name__ == "__main__":
    main()
