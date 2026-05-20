from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.schema import (
    OPEN_SAMPLE_END,
    OPEN_SAMPLE_START,
    ensure_timestamp_columns,
    filter_time_range,
)


def _future_values(
    frame: pd.DataFrame,
    *,
    seconds: int,
    value_columns: Sequence[str],
    suffix: str,
    group_columns: Sequence[str] = ("date", "symbol"),
    timestamp_col: str = "timestamp",
    max_gap_seconds: int | None = None,
) -> pd.DataFrame:
    tolerance = (
        pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds is not None else None
    )
    aligned_parts = []

    for _, group in frame.groupby(list(group_columns), sort=False, observed=True):
        group = group.sort_values(timestamp_col)
        left = pd.DataFrame(
            {
                "_row": group.index.to_numpy(),
                "_target_ts": group[timestamp_col]
                + pd.to_timedelta(seconds, unit="s"),
            }
        )
        right = (
            group[[timestamp_col, *value_columns]]
            .rename(columns={timestamp_col: "_future_ts"})
            .sort_values("_future_ts")
        )
        merged = pd.merge_asof(
            left.sort_values("_target_ts"),
            right,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        )
        merged = merged.sort_values("_row").set_index("_row")
        aligned_parts.append(merged)

    out = pd.DataFrame(index=frame.index)
    if not aligned_parts:
        out[f"timestamp_{suffix}"] = pd.NaT
        for column in value_columns:
            out[f"{column}_{suffix}"] = np.nan
        return out

    aligned = pd.concat(aligned_parts).sort_index()
    out[f"timestamp_{suffix}"] = aligned["_future_ts"]
    for column in value_columns:
        out[f"{column}_{suffix}"] = aligned[column]
    return out


def build_trade_labels(
    ticks: pd.DataFrame,
    *,
    buy_price_col: str = "ask_price_1",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    hold_seconds: int = 60,
    sell_window_seconds: int = 60,
    volume_unit_multiplier: float = 1.0,
    fee_bps: float = 0.0,
    sample_start_time: str = OPEN_SAMPLE_START,
    sample_end_time: str = OPEN_SAMPLE_END,
    max_future_gap_seconds: int | None = None,
    tradable_statuses: Sequence[str] | None = None,
) -> pd.DataFrame:
    missing = [
        column
        for column in (buy_price_col, volume_col, turnover_col)
        if column not in ticks.columns
    ]
    if missing:
        raise SystemExit(f"missing required columns for labels: {missing}")

    work = ensure_timestamp_columns(ticks)
    work = work.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    value_columns = [volume_col, turnover_col]
    sell_start = _future_values(
        work,
        seconds=hold_seconds,
        value_columns=value_columns,
        suffix="sell_start",
        max_gap_seconds=max_future_gap_seconds,
    )
    sell_end = _future_values(
        work,
        seconds=hold_seconds + sell_window_seconds,
        value_columns=value_columns,
        suffix="sell_end",
        max_gap_seconds=max_future_gap_seconds,
    )
    work = pd.concat([work, sell_start, sell_end], axis=1)

    start_volume = pd.to_numeric(
        work[f"{volume_col}_sell_start"],
        errors="coerce",
    ).astype("float64")
    end_volume = pd.to_numeric(
        work[f"{volume_col}_sell_end"],
        errors="coerce",
    ).astype("float64")
    start_turnover = pd.to_numeric(
        work[f"{turnover_col}_sell_start"],
        errors="coerce",
    ).astype("float64")
    end_turnover = pd.to_numeric(
        work[f"{turnover_col}_sell_end"],
        errors="coerce",
    ).astype("float64")
    work["sell_volume"] = end_volume - start_volume
    work["sell_turnover"] = end_turnover - start_turnover

    denominator = work["sell_volume"] * float(volume_unit_multiplier)
    work["sell_vwap"] = np.where(
        denominator > 0,
        work["sell_turnover"] / denominator,
        np.nan,
    )
    work["buy_price"] = work[buy_price_col].astype("float64")
    work["gross_label"] = np.where(
        work["buy_price"] > 0,
        work["sell_vwap"] / work["buy_price"] - 1.0,
        np.nan,
    )
    work["label"] = work["gross_label"] - float(fee_bps) / 10_000.0
    work["valid_label"] = (
        work["label"].notna()
        & np.isfinite(work["label"])
        & (work["sell_volume"] > 0)
        & (work["sell_turnover"] > 0)
        & (work["buy_price"] > 0)
    )
    if tradable_statuses and "status" in work.columns:
        allowed = {str(status).upper() for status in tradable_statuses}
        work["valid_label"] &= work["status"].astype(str).str.upper().isin(allowed)

    return filter_time_range(
        work,
        sample_start_time,
        sample_end_time,
        include_end=True,
    )
