from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from opening_strength_fit.analysis import load_or_fetch_next_close_labels
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.config import config_optional_int, config_str
from opening_strength_fit.next_close_labels import fetch_next_close_labels
from opening_strength_fit.schema import DECISION_KEY_COLUMNS


@dataclass(frozen=True)
class NextCloseLabelRequest:
    label_input: str
    host: str
    port: int
    username: str
    password: str
    table: str
    close_offset_us: int
    close_lookback_seconds: int
    calendar_days_after: int


def next_close_label_request(args: Any, config: dict) -> NextCloseLabelRequest:
    return NextCloseLabelRequest(
        label_input=(
            str(getattr(args, "next_close_label_input", ""))
            or config_str(config, "risk_layer", "next_close_label_input", "")
        ),
        host=(
            str(getattr(args, "clickhouse_host", ""))
            or config_str(config, "clickhouse", "host", DEFAULT_CLICKHOUSE_TICK_HOST)
        ),
        port=int(
            getattr(args, "clickhouse_port", 0)
            or config_optional_int(config, "clickhouse", "port", None)
            or os.environ.get("CLICKHOUSE_PORT", "8123")
        ),
        username=(
            str(getattr(args, "clickhouse_user", ""))
            or config_str(config, "clickhouse", "user", "")
            or os.environ.get("CLICKHOUSE_USER", "")
        ),
        password=(
            str(getattr(args, "clickhouse_password", ""))
            or config_str(config, "clickhouse", "password", "")
            or os.environ.get("CLICKHOUSE_PASSWORD", "")
        ),
        table=(
            str(getattr(args, "clickhouse_table", ""))
            or config_str(
                config,
                "clickhouse",
                "table",
                os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
            )
        ),
        close_offset_us=int(
            config_optional_int(config, "risk_layer", "close_offset_us", None)
            or args.close_offset_us
        ),
        close_lookback_seconds=int(
            config_optional_int(config, "risk_layer", "close_lookback_seconds", None)
            or args.close_lookback_seconds
        ),
        calendar_days_after=int(
            config_optional_int(config, "risk_layer", "calendar_days_after", None)
            or args.calendar_days_after
        ),
    )


def load_risk_next_close_labels(
    labeled: pd.DataFrame,
    *,
    request: NextCloseLabelRequest,
    output_dir: Path,
) -> pd.DataFrame:
    def fetch(base: pd.DataFrame) -> pd.DataFrame:
        if not request.username or not request.password:
            raise SystemExit(
                "bad-tail risk labels need next-close labels. Pass "
                "--next-close-label-input or set ClickHouse credentials."
            )
        label_base = base[[*DECISION_KEY_COLUMNS, "buy_price"]].drop_duplicates(
            list(DECISION_KEY_COLUMNS)
        )
        return fetch_next_close_labels(
            label_base,
            host=request.host,
            port=request.port,
            username=request.username,
            password=request.password,
            table=request.table,
            close_offset_us=request.close_offset_us,
            close_lookback_seconds=request.close_lookback_seconds,
            calendar_days_after=request.calendar_days_after,
            fee_bps=0.0,
        )

    return load_or_fetch_next_close_labels(
        labeled,
        output_dir=output_dir,
        label_input=request.label_input,
        fetch_labels=fetch,
        key_columns=DECISION_KEY_COLUMNS,
    )
