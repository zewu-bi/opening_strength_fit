from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from opening_strength_fit.features import build_feature_frame
from opening_strength_fit.io import read_frame
from opening_strength_fit.labels import build_trade_labels
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, sample_labeled_frame
from opening_strength_fit.schema import (
    OPEN_SAMPLE_END,
    OPEN_SAMPLE_START,
    ensure_timestamp_columns,
    normalize_clock_time,
    standardize_columns,
)
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX, filter_symbol_universe


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
    entry_max_gap_seconds: int | None = None,
    sample_start_time: str = OPEN_SAMPLE_START,
    sample_end_time: str = OPEN_SAMPLE_END,
    include_preopen: bool = True,
    max_future_gap_seconds: int | None = None,
    tradable_statuses: Sequence[str] | None = None,
    universe_regex: str | None = None,
    universe_symbols: set[str] | None = None,
    sample_mode: str = "all_ticks",
    decision_times: list[str] | tuple[str, ...] = DEFAULT_DECISION_TIMES,
    decision_max_lag_seconds: int | None = 5,
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
    )
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
        entry_max_gap_seconds=entry_max_gap_seconds,
        sample_start_time=sample_start_time,
        sample_end_time=label_sample_end_time,
        max_future_gap_seconds=max_future_gap_seconds,
        tradable_statuses=tradable_statuses,
    )
    return sample_labeled_frame(
        labeled,
        mode=sample_mode,
        decision_times=decision_times,
        max_lag_seconds=decision_max_lag_seconds,
    )


def build_opening_research_frame(
    ticks: pd.DataFrame,
    *,
    universe_regex: str | None = DEFAULT_A_SHARE_SYMBOL_REGEX,
    universe_symbols: set[str] | None = None,
    sample_mode: str = "decision_points",
    decision_times: list[str] | tuple[str, ...] = DEFAULT_DECISION_TIMES,
    decision_max_lag_seconds: int | None = 5,
    **label_kwargs,
) -> pd.DataFrame:
    return build_labeled_feature_frame(
        ticks,
        universe_regex=universe_regex,
        universe_symbols=universe_symbols,
        sample_mode=sample_mode,
        decision_times=decision_times,
        decision_max_lag_seconds=decision_max_lag_seconds,
        **label_kwargs,
    )


def valid_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "valid_label" in frame.columns:
        return frame.loc[frame["valid_label"]].copy()
    return frame.loc[frame["label"].notna()].copy()
