from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.schema import (
    OPEN_SAMPLE_END,
    OPEN_SAMPLE_START,
    PRICE_LEVELS,
    ask_price_col,
    ask_volume_col,
    ensure_timestamp_columns,
    filter_time_range,
)


def finite_numeric_series(
    values: object,
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    """Coerce numeric values and mask all non-finite values as NaN."""
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values, errors="coerce")
        if index is not None and not out.index.equals(index):
            out = out.reindex(index)
    else:
        out = pd.to_numeric(pd.Series(values, index=index), errors="coerce")
    return out.astype("float64").replace([np.inf, -np.inf], np.nan)


def safe_price_return(
    exit_price: object,
    buy_price: object,
    *,
    fee_bps: float = 0.0,
) -> pd.Series:
    """Return exit / buy - 1 with one shared validity rule for all labels."""
    exit_values = finite_numeric_series(exit_price)
    buy_values = finite_numeric_series(buy_price, index=exit_values.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = exit_values / buy_values - 1.0 - float(fee_bps) / 10_000.0
    valid = (
        np.isfinite(returns)
        & exit_values.gt(0)
        & buy_values.gt(0)
    )
    return returns.where(valid, np.nan)


def normalize_return_label_frame(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    label_col: str,
) -> pd.DataFrame:
    """Normalize an external return-label frame using the same finite rule."""
    required = [*key_columns, label_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"return label input missing columns: {missing}")
    out = frame[required].copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)
    if "decision_target_timestamp" in out.columns:
        out["decision_target_timestamp"] = pd.to_datetime(
            out["decision_target_timestamp"],
            errors="coerce",
        )
    out[label_col] = finite_numeric_series(out[label_col])
    drop_subset = [label_col]
    if "decision_target_timestamp" in out.columns:
        drop_subset.append("decision_target_timestamp")
    return out.dropna(subset=drop_subset).drop_duplicates(list(key_columns))


def _future_values(
    frame: pd.DataFrame,
    *,
    seconds: int,
    value_columns: Sequence[str],
    suffix: str,
    group_columns: Sequence[str] = ("date", "symbol"),
    timestamp_col: str = "timestamp",
    target_timestamp_col: str | None = None,
    max_gap_seconds: int | None = None,
) -> pd.DataFrame:
    tolerance = (
        pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds is not None else None
    )
    aligned_parts = []
    target_col = target_timestamp_col or timestamp_col

    out = pd.DataFrame(index=frame.index)
    out[f"timestamp_{suffix}"] = pd.NaT
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(pd.NA, index=frame.index, dtype="object")

    for _, group in frame.groupby(list(group_columns), sort=False, observed=True):
        group = group.sort_values(timestamp_col)
        left = pd.DataFrame(
            {
                "_row": group.index.to_numpy(),
                "_target_ts": group[target_col]
                + pd.to_timedelta(seconds, unit="s"),
            }
        ).dropna(subset=["_target_ts"])
        left["_target_ts"] = pd.to_datetime(left["_target_ts"]).astype("datetime64[ns]")
        right = (
            group[[timestamp_col, *value_columns]]
            .dropna(subset=[timestamp_col])
            .rename(columns={timestamp_col: "_future_ts"})
            .sort_values("_future_ts")
        )
        right["_future_ts"] = pd.to_datetime(right["_future_ts"]).astype("datetime64[ns]")
        if left.empty or right.empty:
            continue
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

    if not aligned_parts:
        return out

    aligned = pd.concat(aligned_parts).sort_index()
    out.loc[aligned.index, f"timestamp_{suffix}"] = aligned["_future_ts"]
    for column in value_columns:
        out.loc[aligned.index, f"{column}_{suffix}"] = aligned[column]
    return out


def _future_tick_values(
    frame: pd.DataFrame,
    *,
    offset_ticks: int,
    value_columns: Sequence[str],
    suffix: str,
    group_columns: Sequence[str] = ("date", "symbol"),
    timestamp_col: str = "timestamp",
    max_gap_seconds: int | None = None,
) -> pd.DataFrame:
    if offset_ticks < 0:
        raise SystemExit("entry_tick_delay must be non-negative")

    timing_columns = (
        f"{suffix}_delay_ticks",
        f"{suffix}_delay_seconds",
        f"{suffix}_max_tick_gap_seconds",
    )
    out = pd.DataFrame(index=frame.index)
    out[f"timestamp_{suffix}"] = pd.NaT
    for column in timing_columns:
        out[column] = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(pd.NA, index=frame.index, dtype="object")

    aligned_parts = []
    for _, group in frame.groupby(list(group_columns), sort=False, observed=True):
        group = group.sort_values(timestamp_col)
        shifted = group[[timestamp_col, *value_columns]].shift(-offset_ticks)
        shifted.index = group.index
        timestamps = pd.to_datetime(group[timestamp_col], errors="coerce").astype(
            "datetime64[ns]"
        )
        entry_timestamps = pd.to_datetime(
            shifted[timestamp_col],
            errors="coerce",
        ).astype("datetime64[ns]")
        valid_entry = timestamps.notna() & entry_timestamps.notna()
        shifted[f"_{suffix}_delay_ticks"] = np.where(
            valid_entry,
            float(offset_ticks),
            np.nan,
        )
        shifted[f"_{suffix}_delay_seconds"] = (
            entry_timestamps - timestamps
        ) / pd.Timedelta(seconds=1)
        if offset_ticks == 0:
            max_tick_gap = pd.Series(np.nan, index=group.index, dtype="float64")
            max_tick_gap.loc[valid_entry] = 0.0
        else:
            step_gap = timestamps.diff() / pd.Timedelta(seconds=1)
            path_gaps = pd.concat(
                [step_gap.shift(-step) for step in range(1, offset_ticks + 1)],
                axis=1,
            )
            complete_path = path_gaps.notna().all(axis=1)
            max_tick_gap = pd.Series(np.nan, index=group.index, dtype="float64")
            max_tick_gap.loc[complete_path] = path_gaps.loc[complete_path].max(axis=1)
        shifted[f"_{suffix}_max_tick_gap_seconds"] = max_tick_gap
        aligned_parts.append(shifted)

    if not aligned_parts:
        return out

    aligned = pd.concat(aligned_parts).sort_index()
    out.loc[aligned.index, f"timestamp_{suffix}"] = aligned[timestamp_col]
    for column in timing_columns:
        out.loc[aligned.index, column] = pd.to_numeric(
            aligned[f"_{column}"],
            errors="coerce",
        )
    for column in value_columns:
        out.loc[aligned.index, f"{column}_{suffix}"] = aligned[column]

    if max_gap_seconds is not None:
        entry_gap = out[f"{suffix}_max_tick_gap_seconds"]
        valid_gap = entry_gap.notna() & entry_gap.ge(0) & entry_gap.le(max_gap_seconds)
        out.loc[~valid_gap, f"timestamp_{suffix}"] = pd.NaT
        for column in timing_columns:
            out.loc[~valid_gap, column] = np.nan
        for column in value_columns:
            out.loc[~valid_gap, f"{column}_{suffix}"] = np.nan
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
    entry_tick_delay: int = 0,
    entry_max_gap_seconds: int | None = None,
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

    entry_value_columns = [buy_price_col]
    for level in PRICE_LEVELS:
        for column in (ask_price_col(level), ask_volume_col(level)):
            if column in work.columns:
                entry_value_columns.append(column)
    if "status" in work.columns:
        entry_value_columns.append("status")
    entry_value_columns = list(dict.fromkeys(entry_value_columns))
    entry = _future_tick_values(
        work,
        offset_ticks=int(entry_tick_delay),
        value_columns=entry_value_columns,
        suffix="entry",
        max_gap_seconds=entry_max_gap_seconds,
    )
    work["entry_timestamp"] = pd.to_datetime(
        entry["timestamp_entry"],
        errors="coerce",
    ).astype("datetime64[ns]")
    work["entry_delay_ticks"] = pd.to_numeric(
        entry["entry_delay_ticks"],
        errors="coerce",
    ).astype("float64")
    work["entry_delay_seconds"] = pd.to_numeric(
        entry["entry_delay_seconds"],
        errors="coerce",
    ).astype("float64")
    work["entry_max_tick_gap_seconds"] = pd.to_numeric(
        entry["entry_max_tick_gap_seconds"],
        errors="coerce",
    ).astype("float64")
    work["buy_price"] = pd.to_numeric(
        entry[f"{buy_price_col}_entry"],
        errors="coerce",
    ).astype("float64")
    for level in PRICE_LEVELS:
        price_col = ask_price_col(level)
        volume_col_name = ask_volume_col(level)
        if f"{price_col}_entry" in entry.columns:
            work[f"entry_{price_col}"] = pd.to_numeric(
                entry[f"{price_col}_entry"],
                errors="coerce",
            ).astype("float64")
        if f"{volume_col_name}_entry" in entry.columns:
            work[f"entry_{volume_col_name}"] = pd.to_numeric(
                entry[f"{volume_col_name}_entry"],
                errors="coerce",
            ).astype("float64")
    if "status_entry" in entry.columns:
        work["entry_status"] = entry["status_entry"]

    value_columns = [volume_col, turnover_col]
    sell_start = _future_values(
        work,
        seconds=hold_seconds,
        value_columns=value_columns,
        suffix="sell_start",
        target_timestamp_col="entry_timestamp",
        max_gap_seconds=max_future_gap_seconds,
    )
    sell_end = _future_values(
        work,
        seconds=hold_seconds + sell_window_seconds,
        value_columns=value_columns,
        suffix="sell_end",
        target_timestamp_col="entry_timestamp",
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
    work["gross_label"] = safe_price_return(work["sell_vwap"], work["buy_price"])
    work["label"] = work["gross_label"] - float(fee_bps) / 10_000.0
    work["valid_label"] = (
        work["label"].notna()
        & np.isfinite(work["label"])
        & (work["sell_volume"] > 0)
        & (work["sell_turnover"] > 0)
        & (work["buy_price"] > 0)
        & work["entry_timestamp"].notna()
    )
    if tradable_statuses and "status" in work.columns:
        allowed = {str(status).upper() for status in tradable_statuses}
        work["valid_label"] &= work["status"].astype(str).str.upper().isin(allowed)
        if "entry_status" in work.columns:
            work["valid_label"] &= (
                work["entry_status"].astype(str).str.upper().isin(allowed)
            )

    return filter_time_range(
        work,
        sample_start_time,
        sample_end_time,
        include_end=True,
    )
