from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from opening_strength_fit.features import build_feature_frame
from opening_strength_fit.io import read_frame
from opening_strength_fit.labels import build_trade_labels
from opening_strength_fit.sampling import (
    DEFAULT_DECISION_TIMES,
    normalize_decision_alignment,
    require_entry_after_cross_section_ready,
    sample_labeled_frame,
)
from opening_strength_fit.schema import (
    OPEN_SAMPLE_END,
    OPEN_SAMPLE_START,
    ensure_timestamp_columns,
    normalize_clock_time,
    standardize_columns,
)
from opening_strength_fit.universe import filter_symbol_universe


def load_ticks(
    path: str,
    *,
    columns: list[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    ticks = read_frame(path, columns=columns)
    ticks = standardize_columns(ticks, aliases)
    return ensure_timestamp_columns(ticks)


def _is_decision_point_mode(mode: str) -> bool:
    return str(mode).strip().lower() in {
        "decision",
        "decision_point",
        "decision_points",
    }


def _add_clock_seconds(value: str, seconds: int | None) -> str:
    if not seconds:
        return normalize_clock_time(value)
    timestamp = pd.Timestamp(f"2000-01-01 {normalize_clock_time(value)}")
    return (timestamp + pd.Timedelta(seconds=int(seconds))).strftime("%H:%M:%S")


def build_labeled_feature_frame(
    ticks: pd.DataFrame,
    *,
    buy_price_col: str = "ask_price_1",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    hold_seconds: int = 60,
    sell_window_seconds: int = 60,
    volume_unit_multiplier: float = 1.0,
    fee_bps: float = 0.0,
    entry_tick_delay: int = 0,
    entry_alignment: str = "tick_offset",
    entry_clock_delay_seconds: int | None = None,
    entry_max_gap_seconds: int | None = None,
    sample_start_time: str = OPEN_SAMPLE_START,
    sample_end_time: str = OPEN_SAMPLE_END,
    include_preopen: bool = True,
    preopen_price_mode: str = "legacy_last_price",
    preopen_match_time: str = "09:25:00",
    future_alignment: str = "next_tick",
    max_future_gap_seconds: int | None = None,
    tradable_statuses: Sequence[str] | None = None,
    universe_regex: str | None = None,
    universe_symbols: set[str] | None = None,
    sample_mode: str = "all_ticks",
    decision_times: list[str] | tuple[str, ...] = DEFAULT_DECISION_TIMES,
    decision_max_lag_seconds: int | None = 5,
    decision_alignment: str = "next_tick",
    decision_max_state_age_seconds: int | None = None,
    require_cross_section_ready_entry: bool = False,
) -> pd.DataFrame:
    if universe_regex or universe_symbols:
        ticks = filter_symbol_universe(
            ticks,
            symbol_regex=universe_regex,
            symbols=universe_symbols,
        )

    features = build_feature_frame(
        ticks,
        include_preopen=include_preopen,
        volume_col=volume_col,
        turnover_col=turnover_col,
        volume_unit_multiplier=volume_unit_multiplier,
        preopen_price_mode=preopen_price_mode,
        preopen_match_time=preopen_match_time,
    )
    normalized_decision_alignment = normalize_decision_alignment(decision_alignment)
    if _is_decision_point_mode(sample_mode) and normalized_decision_alignment == "clock_state":
        sampled_features = sample_labeled_frame(
            features,
            mode=sample_mode,
            decision_times=decision_times,
            max_lag_seconds=decision_max_lag_seconds,
            alignment=decision_alignment,
            max_state_age_seconds=decision_max_state_age_seconds,
        )
        sampled = build_trade_labels(
            sampled_features,
            buy_price_col=buy_price_col,
            volume_col=volume_col,
            turnover_col=turnover_col,
            hold_seconds=hold_seconds,
            sell_window_seconds=sell_window_seconds,
            volume_unit_multiplier=volume_unit_multiplier,
            fee_bps=fee_bps,
            entry_tick_delay=entry_tick_delay,
            entry_alignment=entry_alignment,
            entry_clock_delay_seconds=entry_clock_delay_seconds,
            entry_max_gap_seconds=entry_max_gap_seconds,
            sample_start_time=sample_start_time,
            sample_end_time=sample_end_time,
            future_alignment=future_alignment,
            max_future_gap_seconds=max_future_gap_seconds,
            tradable_statuses=tradable_statuses,
            state_ticks=features,
            entry_target_timestamp_col="decision_target_timestamp",
        )
    else:
        label_sample_end_time = (
            _add_clock_seconds(sample_end_time, decision_max_lag_seconds)
            if _is_decision_point_mode(sample_mode)
            else sample_end_time
        )
        labeled = build_trade_labels(
            features,
            buy_price_col=buy_price_col,
            volume_col=volume_col,
            turnover_col=turnover_col,
            hold_seconds=hold_seconds,
            sell_window_seconds=sell_window_seconds,
            volume_unit_multiplier=volume_unit_multiplier,
            fee_bps=fee_bps,
            entry_tick_delay=entry_tick_delay,
            entry_alignment=entry_alignment,
            entry_clock_delay_seconds=entry_clock_delay_seconds,
            entry_max_gap_seconds=entry_max_gap_seconds,
            sample_start_time=sample_start_time,
            sample_end_time=label_sample_end_time,
            future_alignment=future_alignment,
            max_future_gap_seconds=max_future_gap_seconds,
            tradable_statuses=tradable_statuses,
        )
        sampled = sample_labeled_frame(
            labeled,
            mode=sample_mode,
            decision_times=decision_times,
            max_lag_seconds=decision_max_lag_seconds,
            alignment=decision_alignment,
            max_state_age_seconds=decision_max_state_age_seconds,
        )
    if require_cross_section_ready_entry:
        sampled = require_entry_after_cross_section_ready(sampled)
    return sampled


def valid_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "valid_label" in frame.columns:
        return frame.loc[frame["valid_label"]].copy()
    return frame.loc[frame["label"].notna()].copy()
