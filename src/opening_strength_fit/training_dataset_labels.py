from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.io import read_frame
from opening_strength_fit.schema import DECISION_KEY_COLUMNS
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

# Dataset builders, label builders, training readers, and acceptance workflows all
# join on the same causal sample key. Keep the alias while older callers migrate to
# schema.DECISION_KEY_COLUMNS.
KEY_COLUMNS = DECISION_KEY_COLUMNS
RAW_LABEL_TICK_COLUMNS = (
    "Symbol",
    "ExchTimeOffsetUs",
    "Volume",
    "Turnover",
    "AskPrice1",
    "Status",
)


def normalize_dataset_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical, timezone-naive decision keys."""

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"], errors="coerce"
    ).dt.tz_localize(None)
    missing = out[list(KEY_COLUMNS)].isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"dataset has {int(missing.sum())} rows with missing keys")
    return out


def filter_decision_clocks(frame: pd.DataFrame, clocks: tuple[str, ...]) -> pd.DataFrame:
    """Normalize a dataset and retain only the declared decision clocks."""

    out = normalize_dataset_keys(frame)
    observed = out["decision_target_timestamp"].dt.strftime("%H:%M:%S")
    return out.loc[observed.isin(set(clocks))].copy()


def validate_dataset_keys(frame: pd.DataFrame, clocks: tuple[str, ...]) -> None:
    """Enforce key uniqueness and complete clock coverage before publication."""

    duplicate = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(f"output has {int(duplicate.sum())} duplicate key rows")
    observed = set(frame["decision_target_timestamp"].dt.strftime("%H:%M:%S").unique())
    missing_clocks = sorted(set(clocks).difference(observed))
    if missing_clocks:
        raise SystemExit(f"output is missing decision clocks: {missing_clocks}")


def _entry_offset_us(timestamp: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamp, errors="coerce")
    offset = (values - values.dt.normalize()) / pd.Timedelta(microseconds=1)
    return offset.to_numpy(dtype="float64")


def _clock_offset_us(clock: str) -> int:
    timestamp = pd.Timestamp(f"2000-01-01 {clock}")
    return int((timestamp - timestamp.normalize()) / pd.Timedelta(microseconds=1))


def build_label_base(
    raw_ticks: pd.DataFrame,
    *,
    trading_day: str,
    decision_times: tuple[str, ...],
    feature_tick_start_offset_us: int,
    entry_delay_seconds: int,
) -> pd.DataFrame:
    """Build point-in-time decision and delayed-entry states for one day."""

    raw = raw_ticks.reset_index(drop=True)
    offset = pd.to_numeric(raw["ExchTimeOffsetUs"], errors="coerce")
    eligible = offset.ge(int(feature_tick_start_offset_us)) & offset.notna()
    raw = raw.loc[eligible].reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                *KEY_COLUMNS,
                "timestamp",
                "entry_timestamp",
                "buy_price",
                "status",
                "entry_status",
                "entry_after_cross_section_ready",
            ]
        )
    symbol = decode_clickhouse_text(raw["Symbol"])
    offset = pd.to_numeric(raw["ExchTimeOffsetUs"], errors="coerce")
    ask = pd.to_numeric(raw["AskPrice1"], errors="coerce")
    status = decode_clickhouse_text(raw["Status"])
    targets = np.asarray([_clock_offset_us(clock) for clock in decision_times], dtype="int64")
    entry_targets = targets + int(entry_delay_seconds) * 1_000_000
    day = pd.Timestamp(trading_day)
    parts = []
    for name, positions_raw in symbol.groupby(symbol, sort=False).indices.items():
        positions = np.asarray(positions_raw, dtype="int64")
        offsets = offset.iloc[positions].to_numpy(dtype="int64")
        if len(offsets) > 1 and bool(np.any(offsets[1:] < offsets[:-1])):
            order = np.argsort(offsets, kind="stable")
            positions = positions[order]
            offsets = offsets[order]
        decision_index = np.searchsorted(offsets, targets, side="right") - 1
        entry_index = np.searchsorted(offsets, entry_targets, side="right") - 1
        matched = (decision_index >= 0) & (entry_index >= 0)
        if not matched.any():
            continue
        selected_decision = positions[decision_index[matched]]
        selected_entry = positions[entry_index[matched]]
        selected_targets = targets[matched]
        selected_entry_targets = entry_targets[matched]
        parts.append(
            pd.DataFrame(
                {
                    "date": trading_day,
                    "symbol": str(name),
                    "decision_target_timestamp": day + pd.to_timedelta(selected_targets, unit="us"),
                    "timestamp": day
                    + pd.to_timedelta(offset.iloc[selected_decision].to_numpy(), unit="us"),
                    "entry_timestamp": day + pd.to_timedelta(selected_entry_targets, unit="us"),
                    "buy_price": ask.iloc[selected_entry].to_numpy(dtype="float64"),
                    "status": status.iloc[selected_decision].to_numpy(),
                    "entry_status": status.iloc[selected_entry].to_numpy(),
                }
            )
        )
    if not parts:
        raise SystemExit(f"raw source produced no decision rows for {trading_day}")
    base = pd.concat(parts, ignore_index=True)
    group_keys = [base["date"], base["decision_target_timestamp"]]
    ready_timestamp = base["timestamp"].groupby(group_keys, sort=False).transform("max")
    base["entry_after_cross_section_ready"] = (
        base["entry_timestamp"].notna()
        & ready_timestamp.notna()
        & base["entry_timestamp"].ge(ready_timestamp)
    )
    return base.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def compute_clock_vwap_label_set(
    base: pd.DataFrame,
    raw_ticks: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    sell_window_seconds: int,
    volume_unit_multiplier: float,
    fee_bps: float,
    tradable_statuses: tuple[str, ...],
) -> pd.DataFrame:
    """Compute clock-state VWAP labels for several horizons in one tick pass."""

    out = base[list(KEY_COLUMNS)].copy()
    symbols = base["symbol"].astype(str).to_numpy()
    entry_offset = _entry_offset_us(base["entry_timestamp"])
    buy_price = pd.to_numeric(base["buy_price"], errors="coerce").to_numpy(dtype="float64")
    raw = raw_ticks.reset_index(drop=True)
    raw_symbol = decode_clickhouse_text(raw["Symbol"])
    raw_offset = pd.to_numeric(raw["ExchTimeOffsetUs"], errors="coerce")
    raw_volume = pd.to_numeric(raw["Volume"], errors="coerce")
    raw_turnover = pd.to_numeric(raw["Turnover"], errors="coerce")
    raw_groups = raw_symbol.groupby(raw_symbol, sort=False).indices
    base_groups = pd.Series(symbols).groupby(pd.Series(symbols), sort=False).indices

    allowed = {status.upper() for status in tradable_statuses}
    status_valid = np.ones(len(base), dtype=bool)
    for column in ("status", "entry_status"):
        if allowed and column in base:
            status_valid &= base[column].astype(str).str.upper().isin(allowed).to_numpy()
    if "entry_after_cross_section_ready" in base:
        status_valid &= base["entry_after_cross_section_ready"].fillna(False).to_numpy(dtype=bool)

    for horizon in horizons:
        label = np.full(len(base), np.nan, dtype="float64")
        valid = np.zeros(len(base), dtype=bool)
        for symbol, positions_raw in base_groups.items():
            positions = np.asarray(positions_raw, dtype="int64")
            tick_positions = raw_groups.get(str(symbol))
            if tick_positions is None:
                continue
            tick_positions = np.asarray(tick_positions, dtype="int64")
            offsets = raw_offset.iloc[tick_positions].to_numpy(dtype="int64")
            volumes = raw_volume.iloc[tick_positions].to_numpy(dtype="float64")
            turnovers = raw_turnover.iloc[tick_positions].to_numpy(dtype="float64")
            if len(offsets) > 1 and bool(np.any(offsets[1:] < offsets[:-1])):
                order = np.argsort(offsets, kind="stable")
                offsets = offsets[order]
                volumes = volumes[order]
                turnovers = turnovers[order]
            start_targets = entry_offset[positions] + int(horizon) * 1_000_000
            end_targets = start_targets + int(sell_window_seconds) * 1_000_000
            start_index = np.searchsorted(offsets, start_targets, side="right") - 1
            end_index = np.searchsorted(offsets, end_targets, side="right") - 1
            matched = (
                np.isfinite(start_targets)
                & np.isfinite(end_targets)
                & (start_index >= 0)
                & (end_index >= 0)
            )
            if not matched.any():
                continue
            matched_positions = positions[matched]
            start_index = start_index[matched]
            end_index = end_index[matched]
            sell_volume = volumes[end_index] - volumes[start_index]
            sell_turnover = turnovers[end_index] - turnovers[start_index]
            with np.errstate(divide="ignore", invalid="ignore"):
                sell_vwap = sell_turnover / (sell_volume * float(volume_unit_multiplier))
                values = sell_vwap / buy_price[matched_positions] - 1.0
                values -= float(fee_bps) / 10_000.0
            row_valid = (
                np.isfinite(values)
                & np.isfinite(buy_price[matched_positions])
                & (buy_price[matched_positions] > 0)
                & (sell_volume > 0)
                & (sell_turnover > 0)
                & status_valid[matched_positions]
            )
            label[matched_positions[row_valid]] = values[row_valid]
            valid[matched_positions[row_valid]] = True
        minutes = int(horizon) // 60
        out[f"label_short_{minutes}m"] = label
        out[f"valid_short_{minutes}m"] = valid
    return out


def next_close_label(base: pd.DataFrame, raw_year_root: Path) -> pd.DataFrame:
    """Build next-session close return labels using the trading calendar."""

    calendar = read_frame(raw_year_root / "trading_calendar.parquet", columns=["TradingDay"])
    dates = sorted(normalize_clickhouse_date(calendar["TradingDay"]).dropna().unique())
    next_date = {date: dates[index + 1] for index, date in enumerate(dates[:-1])}
    close = read_frame(
        raw_year_root / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    ).rename(columns={"TradingDay": "_next_date", "Symbol": "symbol", "ClosePrice": "_close"})
    close["_next_date"] = normalize_clickhouse_date(close["_next_date"])
    close["symbol"] = decode_clickhouse_text(close["symbol"])
    work = base[[*KEY_COLUMNS, "buy_price"]].copy()
    work["_next_date"] = work["date"].map(next_date)
    work = work.merge(close, on=["_next_date", "symbol"], how="left", validate="many_to_one")
    buy = pd.to_numeric(work["buy_price"], errors="coerce")
    price = pd.to_numeric(work["_close"], errors="coerce")
    work["label_next_close"] = (price / buy - 1.0).where((price > 0) & (buy > 0))
    work["valid_next_close"] = work["label_next_close"].notna()
    return work[[*KEY_COLUMNS, "label_next_close", "valid_next_close"]]


def mixed_target_label(
    frame: pd.DataFrame,
    *,
    weight: float,
    min_group_size: int,
) -> tuple[pd.Series, pd.Series]:
    """Build the causal cross-sectional short plus next-close training target."""

    short = pd.to_numeric(frame["label_short_1m"], errors="coerce")
    long = pd.to_numeric(frame["label_next_close"], errors="coerce")
    valid = frame["valid_short_1m"].astype(bool) & frame["valid_next_close"].astype(bool)
    keys = [frame["date"], frame["decision_target_timestamp"]]
    short_valid = short.where(valid)
    long_valid = long.where(valid)
    short_group = short_valid.groupby(keys, sort=False)
    long_group = long_valid.groupby(keys, sort=False)
    count = short_group.transform("count")
    short_std = short_group.transform(lambda values: values.std(ddof=0))
    long_std = long_group.transform(lambda values: values.std(ddof=0))
    usable = valid & count.ge(int(min_group_size)) & short_std.gt(1e-12) & long_std.gt(1e-12)
    short_z = (short - short_group.transform("mean")) / short_std
    long_z = (long - long_group.transform("mean")) / long_std
    mixed = (short_z + float(weight) * long_z).where(usable)
    return mixed, mixed.notna()


# Compatibility name retained for historical tests and archived callers. The
# implementation is horizon-agnostic and is owned by this domain module.
compute_short_label_set = compute_clock_vwap_label_set
