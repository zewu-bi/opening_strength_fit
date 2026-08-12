from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import (
    load_or_fetch_next_close_labels as load_or_fetch_label_cache,
)
from opening_strength_fit.analysis import normalize_next_close_labels
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    DEFAULT_CLOSE_LOOKBACK_SECONDS,
    DEFAULT_CLOSE_OFFSET_US,
    compute_clickhouse_close_labels,
)
from opening_strength_fit.horizons import HorizonSpec
from opening_strength_fit.schema import DECISION_KEY_COLUMNS


def add_next_close_label_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_connection: bool,
    calendar_days_after: int = 10,
) -> None:
    parser.add_argument("--next-close-label-input", default="")
    if include_connection:
        parser.add_argument("--clickhouse-host", default=os.environ.get("CLICKHOUSE_HOST", ""))
        parser.add_argument(
            "--clickhouse-port",
            type=int,
            default=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        )
        parser.add_argument("--clickhouse-user", default=os.environ.get("CLICKHOUSE_USER", ""))
        parser.add_argument(
            "--clickhouse-password", default=os.environ.get("CLICKHOUSE_PASSWORD", "")
        )
        parser.add_argument(
            "--clickhouse-table",
            default=os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
        )
    parser.add_argument("--close-offset-us", type=int, default=DEFAULT_CLOSE_OFFSET_US)
    parser.add_argument(
        "--close-lookback-seconds", type=int, default=DEFAULT_CLOSE_LOOKBACK_SECONDS
    )
    parser.add_argument("--calendar-days-after", type=int, default=calendar_days_after)


def fetch_next_close_labels(
    base: pd.DataFrame,
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    close_offset_us: int = DEFAULT_CLOSE_OFFSET_US,
    close_lookback_seconds: int = DEFAULT_CLOSE_LOOKBACK_SECONDS,
    calendar_days_after: int = 14,
    fee_bps: float = 0.0,
    compute_labels=None,
) -> pd.DataFrame:
    label_builder = compute_labels or compute_clickhouse_close_labels
    labels = label_builder(
        base[[*DECISION_KEY_COLUMNS, "buy_price"]].copy(),
        [HorizonSpec(name="next_close", label="next close", seconds=None)],
        host=host or DEFAULT_CLICKHOUSE_TICK_HOST,
        port=int(port),
        username=username,
        password=password,
        table=table,
        close_offset_us=int(close_offset_us),
        close_lookback_seconds=int(close_lookback_seconds),
        calendar_days_after=int(calendar_days_after),
        fee_bps=float(fee_bps),
    )
    return normalize_next_close_labels(labels, key_columns=DECISION_KEY_COLUMNS)


def load_or_fetch_next_close_labels_from_args(
    predictions: pd.DataFrame,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    def fetch(base: pd.DataFrame) -> pd.DataFrame:
        if "buy_price" not in base.columns:
            raise SystemExit(
                "next-close labels not found and prediction input has no buy_price "
                "column for ClickHouse label generation."
            )
        if not args.clickhouse_user or not args.clickhouse_password:
            raise SystemExit(
                "next-close labels not found. Pass --next-close-label-input or set "
                "CLICKHOUSE_USER and CLICKHOUSE_PASSWORD."
            )
        return fetch_next_close_labels(
            base,
            host=args.clickhouse_host or DEFAULT_CLICKHOUSE_TICK_HOST,
            port=args.clickhouse_port,
            username=args.clickhouse_user,
            password=args.clickhouse_password,
            table=args.clickhouse_table,
            close_offset_us=args.close_offset_us,
            close_lookback_seconds=args.close_lookback_seconds,
            calendar_days_after=args.calendar_days_after,
        )

    return load_or_fetch_label_cache(
        predictions,
        output_dir=output_dir,
        label_input=args.next_close_label_input,
        fetch_labels=fetch,
        key_columns=DECISION_KEY_COLUMNS,
    )
