from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_ENTRY_DEPTH_LEVELS = 10
TIME_PREDICTION_COLUMNS = ("decision_target_timestamp", "timestamp")


class ScenarioLike(Protocol):
    tradable_statuses: tuple[str, ...]
    require_entry_status: bool
    max_decision_lag_seconds: float | None
    max_entry_tick_gap_seconds: float | None
    max_spread_bps: float | None
    min_limit_up_room_bps: float | None
    min_ask_volume_1: float | None
    min_bid_volume_1: float | None
    min_capacity_notional: float
    max_participation_rate: float
    ask_depth_levels: int


def delay_entry_ticks(delay: str) -> int:
    token = str(delay).strip().lower()
    if not token.startswith("delay"):
        raise SystemExit(f"cannot infer entry tick delay from {delay!r}")
    return int(token.removeprefix("delay"))


def validate_context_delay(
    context: pd.DataFrame,
    *,
    delay: str,
) -> None:
    if context.empty:
        return
    expected = float(delay_entry_ticks(delay))
    column = "entry_delay_ticks"
    if column not in context.columns:
        raise SystemExit(
            f"{delay}: context input is missing {column!r}; cannot verify that "
            "the replay context matches this delay branch. Use raw tick context "
            "or a prebuilt labeled context/cache for the same delay."
        )
    values = pd.to_numeric(context[column], errors="coerce").dropna()
    if values.empty:
        raise SystemExit(
            f"{delay}: context input has no non-null {column!r}; cannot verify "
            "the context delay branch."
        )
    bad = values.ne(expected)
    if bool(bad.any()):
        observed = sorted(float(value) for value in values.drop_duplicates().head(8))
        raise SystemExit(
            f"{delay}: context delay mismatch; expected {column}={expected:g}, "
            f"observed sample={observed}. Use a per-delay labeled context or raw "
            "tick context so replay can derive labels for each delay branch."
        )


def replay_required_columns(
    scenarios: list[ScenarioLike],
    *,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    allow_decision_depth_fallback: bool,
) -> set[str]:
    required: set[str] = {"date", "symbol", "prediction", "label", "entry_delay_ticks"}
    for scenario in scenarios:
        if scenario.tradable_statuses:
            required.add("status")
            if scenario.require_entry_status:
                required.add("entry_status")
        if scenario.max_decision_lag_seconds is not None:
            required.add("decision_lag_seconds")
        if scenario.max_entry_tick_gap_seconds is not None:
            required.add("entry_max_tick_gap_seconds")
        if scenario.max_spread_bps is not None:
            required.add("spread_bps")
        if scenario.min_limit_up_room_bps is not None:
            required.add("ask1_to_limit_up_bps")
        if scenario.min_ask_volume_1 is not None:
            required.add("ask_volume_1")
        if scenario.min_bid_volume_1 is not None:
            required.add("bid_volume_1")
        if scenario.min_capacity_notional > 0 or scenario.max_participation_rate > 0:
            if capacity_notional_col:
                required.add(capacity_notional_col)
            elif capacity_volume_col:
                required.update({capacity_volume_col, capacity_price_col})
        if scenario.ask_depth_levels > 0 and not allow_decision_depth_fallback:
            required_depth_levels = max(
                int(scenario.ask_depth_levels),
                DEFAULT_ENTRY_DEPTH_LEVELS,
            )
            for level in range(1, required_depth_levels + 1):
                required.update(
                    {
                        f"entry_ask_price_{level}",
                        f"entry_ask_volume_{level}",
                    }
                )
    return required


def validate_prediction_interface(
    path: Path,
    *,
    delay: str,
    scenarios: list[ScenarioLike],
    context_columns: set[str] | None = None,
    capacity_notional_col: str = "turnover_diff_30t",
    capacity_volume_col: str = "",
    capacity_price_col: str = "ask_price_1",
    allow_decision_depth_fallback: bool = False,
) -> dict[str, object]:
    try:
        prediction_columns = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        raise SystemExit(f"{path}: cannot read prediction parquet schema: {exc}") from exc

    context_columns = set(context_columns or set())
    if not prediction_columns.intersection(TIME_PREDICTION_COLUMNS):
        raise SystemExit(
            f"{path}: missing a decision timestamp column; expected one of "
            f"{list(TIME_PREDICTION_COLUMNS)}"
        )

    available = prediction_columns | context_columns
    required = replay_required_columns(
        scenarios,
        capacity_notional_col=capacity_notional_col,
        capacity_volume_col=capacity_volume_col,
        capacity_price_col=capacity_price_col,
        allow_decision_depth_fallback=allow_decision_depth_fallback,
    )
    missing = sorted(required - available)
    if missing:
        source = "prediction/context" if context_columns else "prediction"
        raise SystemExit(
            f"{delay}: {path} missing replay interface columns in {source}: "
            f"{missing}. Fetch CPU LightGBM predictions generated from the "
            "delay labeled cache, or provide a matching --context-input."
        )

    if "entry_delay_ticks" in prediction_columns:
        values = pd.read_parquet(path, columns=["entry_delay_ticks"])["entry_delay_ticks"]
        values = pd.to_numeric(values, errors="coerce").dropna()
        expected = float(delay_entry_ticks(delay))
        if values.empty:
            raise SystemExit(f"{delay}: {path} has no non-null 'entry_delay_ticks'")
        if bool(values.ne(expected).any()):
            observed = sorted(float(value) for value in values.drop_duplicates().head(8))
            raise SystemExit(
                f"{delay}: prediction delay mismatch in {path}; expected "
                f"entry_delay_ticks={expected:g}, observed sample={observed}"
            )
        delay_source = "prediction"
    else:
        delay_source = "context"

    return {
        "path": str(path),
        "prediction_columns": len(prediction_columns),
        "context_columns": len(context_columns),
        "required_columns": sorted(required),
        "delay_source": delay_source,
    }


def missing_prediction_paths(runs: list[tuple[str, Path]]) -> list[Path]:
    return [path for _, path in runs if not path.exists()]
