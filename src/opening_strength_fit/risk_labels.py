from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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
from opening_strength_fit.labels import finite_numeric_series
from opening_strength_fit.next_close_labels import fetch_next_close_labels
from opening_strength_fit.schema import DECISION_KEY_COLUMNS

RISK_RANK_MIN = {
    "ask_depth_10": 0.40,
    "depth_imbalance_10": 0.20,
}
RISK_RANK_MAX = {
    "spread_bps": 0.80,
    "turnover_diff_10t": 0.80,
    "return_10t": 0.70,
    "depth_imbalance_10": 0.70,
}


def rank_risk_components(
    frame: pd.DataFrame,
    *,
    rank_min: Mapping[str, float] = RISK_RANK_MIN,
    rank_max: Mapping[str, float] = RISK_RANK_MAX,
    group_columns: Sequence[str] = ("date", "decision_target_timestamp"),
    context: str = "risk input",
) -> pd.DataFrame:
    """Return bounded cross-sectional risk components for configured rank tails."""
    groupers = [frame[column] for column in group_columns]
    components: dict[str, pd.Series] = {}
    for column in sorted(set(rank_min) | set(rank_max)):
        if column not in frame.columns:
            raise SystemExit(f"{context} missing required column: {column}")
        rank = (
            finite_numeric_series(frame[column]).groupby(groupers).rank(method="average", pct=True)
        )
        risks = []
        if column in rank_min:
            threshold = float(rank_min[column])
            risks.append(((threshold - rank) / threshold).clip(lower=0.0, upper=1.0))
        if column in rank_max:
            threshold = float(rank_max[column])
            risks.append(((rank - threshold) / (1.0 - threshold)).clip(lower=0.0, upper=1.0))
        components[column] = pd.concat(risks, axis=1).max(axis=1).fillna(0.0)
    return pd.DataFrame(components, index=frame.index, dtype="float64")


def short_next_ranks(
    labeled: pd.DataFrame,
    *,
    context: str = "risk target",
) -> tuple[pd.Series, pd.Series]:
    if "alpha_return_next_close" not in labeled.columns:
        raise SystemExit(f"{context} requires alpha_return_next_close")
    groupers = [labeled["date"], labeled["decision_target_timestamp"]]
    return tuple(
        finite_numeric_series(labeled[column]).groupby(groupers).rank(method="average", pct=True)
        for column in ("label", "alpha_return_next_close")
    )


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


def next_close_label_request(
    args: Any,
    config: dict,
    *,
    section: str = "risk_layer",
) -> NextCloseLabelRequest:
    def configured(arg_name: str, key: str, default: str = "", source: str = "clickhouse") -> str:
        return str(getattr(args, arg_name, "")) or config_str(config, source, key, default)

    def configured_int(name: str) -> int:
        return int(config_optional_int(config, section, name, None) or getattr(args, name))

    table_default = os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE)
    return NextCloseLabelRequest(
        label_input=configured("next_close_label_input", "next_close_label_input", source=section),
        host=configured("clickhouse_host", "host", DEFAULT_CLICKHOUSE_TICK_HOST),
        port=int(
            getattr(args, "clickhouse_port", 0)
            or config_optional_int(config, "clickhouse", "port", None)
            or os.environ.get("CLICKHOUSE_PORT", "8123")
        ),
        username=configured("clickhouse_user", "user") or os.environ.get("CLICKHOUSE_USER", ""),
        password=configured("clickhouse_password", "password")
        or os.environ.get("CLICKHOUSE_PASSWORD", ""),
        table=configured("clickhouse_table", "table", table_default),
        close_offset_us=configured_int("close_offset_us"),
        close_lookback_seconds=configured_int("close_lookback_seconds"),
        calendar_days_after=configured_int("calendar_days_after"),
    )


def load_risk_next_close_labels(
    labeled: pd.DataFrame,
    *,
    request: NextCloseLabelRequest,
    output_dir: Path,
    context: str = "bad-tail risk labels",
) -> pd.DataFrame:
    def fetch(base: pd.DataFrame) -> pd.DataFrame:
        if not request.username or not request.password:
            raise SystemExit(
                f"{context} need next-close labels. Pass "
                "--next-close-label-input or set ClickHouse credentials."
            )
        return fetch_next_close_labels(
            base[[*DECISION_KEY_COLUMNS, "buy_price"]].drop_duplicates(list(DECISION_KEY_COLUMNS)),
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
