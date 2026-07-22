from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.features import build_feature_frame
from opening_strength_fit.horizons import horizon_specs, label_column_name
from opening_strength_fit.labels import safe_price_return
from opening_strength_fit.sampling import (
    require_entry_after_cross_section_ready,
    select_decision_points,
)
from opening_strength_fit.schema import PRICE_LEVELS, ask_price_col, ask_volume_col
from opening_strength_fit.trading_sessions import (
    DEFAULT_A_SHARE_SESSIONS,
    TradingSession,
    coerce_trading_sessions,
    shift_series_by_trading_seconds,
)


def _align_clock_state(
    source: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    target_timestamp_col: str,
    value_columns: Sequence[str],
    suffix: str,
) -> pd.DataFrame:
    """Read the last source state at each logical target without looking forward."""

    out = pd.DataFrame(index=targets.index)
    out[f"target_timestamp_{suffix}"] = pd.to_datetime(
        targets[target_timestamp_col], errors="coerce"
    )
    out[f"source_timestamp_{suffix}"] = pd.NaT
    out[f"state_age_seconds_{suffix}"] = np.nan
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(pd.NA, index=targets.index, dtype="object")

    source_groups = {
        keys: group[["timestamp", *value_columns]]
        for keys, group in source.groupby(["date", "symbol"], sort=False, observed=True)
    }
    aligned_parts: list[pd.DataFrame] = []
    for keys, left_group in targets.groupby(["date", "symbol"], sort=False, observed=True):
        left = pd.DataFrame(
            {
                "_row": left_group.index,
                "_target_ts": pd.to_datetime(
                    left_group[target_timestamp_col], errors="coerce"
                ).to_numpy(),
            }
        ).dropna(subset=["_target_ts"])
        right = source_groups.get(keys, pd.DataFrame()).copy()
        right = right.dropna(subset=["timestamp"]).rename(columns={"timestamp": "_source_ts"})
        if left.empty or right.empty:
            continue
        left["_target_ts"] = pd.to_datetime(left["_target_ts"]).astype("datetime64[ns]")
        right["_source_ts"] = pd.to_datetime(right["_source_ts"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            left.sort_values("_target_ts", kind="mergesort"),
            right.sort_values("_source_ts", kind="mergesort"),
            left_on="_target_ts",
            right_on="_source_ts",
            direction="backward",
        )
        aligned_parts.append(merged.set_index("_row"))

    if not aligned_parts:
        return out
    aligned = pd.concat(aligned_parts).sort_index()
    out.loc[aligned.index, f"source_timestamp_{suffix}"] = aligned["_source_ts"]
    out.loc[aligned.index, f"state_age_seconds_{suffix}"] = (
        aligned["_target_ts"] - aligned["_source_ts"]
    ) / pd.Timedelta(seconds=1)
    for column in value_columns:
        out.loc[aligned.index, f"{column}_{suffix}"] = aligned[column]
    return out


def _status_allowed(values: pd.Series, allowed: set[str]) -> pd.Series:
    return values.astype(str).str.upper().isin(allowed)


def build_full_day_temporal_labels(
    ticks: pd.DataFrame,
    *,
    decision_times: Iterable[str],
    horizons: Iterable[str] = ("5m", "30m"),
    sessions: Iterable[str] | Sequence[TradingSession] = DEFAULT_A_SHARE_SESSIONS,
    decision_max_lag_seconds: int | None = 5,
    entry_clock_delay_seconds: int = 6,
    entry_tick_delay_audit: int = 2,
    sell_window_trading_seconds: int = 60,
    buy_price_col: str = "ask_price_1",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
    fee_bps: float = 0.0,
    include_preopen: bool = True,
    preopen_price_mode: str = "legacy_last_price",
    preopen_match_time: str = "09:25:00",
    tradable_statuses: Sequence[str] | None = None,
    require_cross_section_ready_entry: bool = True,
) -> pd.DataFrame:
    """Build minute-decision, causally auditable labels across the trading day.

    Timed horizons are measured in exchange trading seconds. The entry target is
    anchored to the actual sampled feature timestamp, preserving fixed-clock v4.
    """

    specs = horizon_specs(horizons)
    unsupported = [spec.name for spec in specs if spec.seconds is None]
    if unsupported:
        raise ValueError(
            "close-like horizons must be attached by the cache workflow: " + ", ".join(unsupported)
        )
    if not specs:
        raise ValueError("at least one timed horizon is required")
    parsed_sessions = coerce_trading_sessions(sessions)

    features = build_feature_frame(
        ticks,
        include_preopen=include_preopen,
        volume_col=volume_col,
        turnover_col=turnover_col,
        volume_unit_multiplier=volume_unit_multiplier,
        preopen_price_mode=preopen_price_mode,
        preopen_match_time=preopen_match_time,
    )
    missing = [
        column
        for column in (buy_price_col, volume_col, turnover_col)
        if column not in features.columns
    ]
    if missing:
        raise SystemExit(f"missing required columns for temporal labels: {missing}")
    features = features.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    sampled = select_decision_points(
        features,
        decision_times=decision_times,
        max_lag_seconds=decision_max_lag_seconds,
    ).reset_index(drop=True)
    if sampled.empty:
        return sampled

    sampled["entry_timestamp"] = pd.to_datetime(sampled["timestamp"]) + pd.Timedelta(
        seconds=int(entry_clock_delay_seconds)
    )
    entry_values = [buy_price_col]
    for level in PRICE_LEVELS:
        entry_values.extend(
            column
            for column in (ask_price_col(level), ask_volume_col(level))
            if column in features.columns
        )
    if "status" in features.columns:
        entry_values.append("status")
    entry_values = list(dict.fromkeys(entry_values))
    entry = _align_clock_state(
        features,
        sampled,
        target_timestamp_col="entry_timestamp",
        value_columns=entry_values,
        suffix="entry",
    )
    sampled["entry_source_timestamp"] = pd.to_datetime(
        entry["source_timestamp_entry"], errors="coerce"
    )
    sampled["entry_state_age_seconds"] = pd.to_numeric(
        entry["state_age_seconds_entry"], errors="coerce"
    )
    sampled["entry_delay_seconds"] = np.where(
        sampled["entry_source_timestamp"].notna(), float(entry_clock_delay_seconds), np.nan
    )
    sampled["entry_delay_ticks"] = np.where(
        sampled["entry_source_timestamp"].notna(), float(entry_tick_delay_audit), np.nan
    )
    sampled["entry_max_tick_gap_seconds"] = np.nan
    sampled["buy_price"] = pd.to_numeric(entry[f"{buy_price_col}_entry"], errors="coerce")
    for level in PRICE_LEVELS:
        for source_col in (ask_price_col(level), ask_volume_col(level)):
            aligned_col = f"{source_col}_entry"
            if aligned_col in entry.columns:
                sampled[f"entry_{source_col}"] = pd.to_numeric(entry[aligned_col], errors="coerce")
    if "status_entry" in entry.columns:
        sampled["entry_status"] = entry["status_entry"]

    if require_cross_section_ready_entry:
        sampled = require_entry_after_cross_section_ready(sampled)
    else:
        sampled["cross_section_ready_timestamp"] = pd.NaT
        sampled["entry_after_cross_section_ready"] = True

    allowed = {str(value).upper() for value in tradable_statuses or ()}
    decision_status_valid = pd.Series(True, index=sampled.index)
    entry_status_valid = pd.Series(True, index=sampled.index)
    if allowed and "status" in sampled.columns:
        decision_status_valid = _status_allowed(sampled["status"], allowed)
    if allowed and "entry_status" in sampled.columns:
        entry_status_valid = _status_allowed(sampled["entry_status"], allowed)
    sampled["valid_entry"] = (
        sampled["entry_source_timestamp"].notna()
        & sampled["buy_price"].gt(0)
        & sampled["entry_after_cross_section_ready"].fillna(False)
        & decision_status_valid
        & entry_status_valid
    )

    for spec in specs:
        assert spec.seconds is not None
        name = spec.name
        start_target_col = f"sell_start_target_timestamp_{name}"
        end_target_col = f"sell_end_target_timestamp_{name}"
        sampled[start_target_col] = shift_series_by_trading_seconds(
            sampled["entry_timestamp"], int(spec.seconds), sessions=parsed_sessions
        )
        sampled[end_target_col] = shift_series_by_trading_seconds(
            sampled[start_target_col],
            int(sell_window_trading_seconds),
            sessions=parsed_sessions,
        )
        start = _align_clock_state(
            features,
            sampled,
            target_timestamp_col=start_target_col,
            value_columns=[volume_col, turnover_col],
            suffix=f"sell_start_{name}",
        )
        end = _align_clock_state(
            features,
            sampled,
            target_timestamp_col=end_target_col,
            value_columns=[volume_col, turnover_col],
            suffix=f"sell_end_{name}",
        )
        for side, aligned in (("sell_start", start), ("sell_end", end)):
            suffix = f"{side}_{name}"
            sampled[f"{side}_source_timestamp_{name}"] = pd.to_datetime(
                aligned[f"source_timestamp_{suffix}"], errors="coerce"
            )
            sampled[f"{side}_state_age_seconds_{name}"] = pd.to_numeric(
                aligned[f"state_age_seconds_{suffix}"], errors="coerce"
            )
            sampled[f"{volume_col}_{side}_{name}"] = pd.to_numeric(
                aligned[f"{volume_col}_{suffix}"], errors="coerce"
            )
            sampled[f"{turnover_col}_{side}_{name}"] = pd.to_numeric(
                aligned[f"{turnover_col}_{suffix}"], errors="coerce"
            )

        sell_volume = (
            sampled[f"{volume_col}_sell_end_{name}"] - sampled[f"{volume_col}_sell_start_{name}"]
        )
        sell_turnover = (
            sampled[f"{turnover_col}_sell_end_{name}"]
            - sampled[f"{turnover_col}_sell_start_{name}"]
        )
        sampled[f"sell_volume_{name}"] = sell_volume
        sampled[f"sell_turnover_{name}"] = sell_turnover
        denominator = sell_volume * float(volume_unit_multiplier)
        sampled[f"sell_vwap_{name}"] = np.where(
            denominator > 0, sell_turnover / denominator, np.nan
        )
        label_col = label_column_name(name)
        sampled[f"gross_{label_col}"] = safe_price_return(
            sampled[f"sell_vwap_{name}"], sampled["buy_price"]
        )
        sampled[label_col] = sampled[f"gross_{label_col}"] - float(fee_bps) / 10_000.0
        sampled[f"valid_{label_col}"] = (
            sampled[label_col].notna()
            & np.isfinite(sampled[label_col])
            & sell_volume.gt(0)
            & sell_turnover.gt(0)
            & sampled["valid_entry"]
            & sampled[start_target_col].notna()
            & sampled[end_target_col].notna()
        )

    return sampled.sort_values(["date", "decision_target_timestamp", "symbol"]).reset_index(
        drop=True
    )
