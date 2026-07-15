from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from opening_strength_fit.schema import ensure_timestamp_columns, normalize_clock_time


def minute_decision_times(
    start_time: str = "09:30:00",
    end_time: str = "09:40:00",
) -> tuple[str, ...]:
    start = pd.Timestamp(f"2000-01-01 {normalize_clock_time(start_time)}")
    end = pd.Timestamp(f"2000-01-01 {normalize_clock_time(end_time)}")
    if end < start:
        raise ValueError("decision minute range end must be >= start")
    return tuple(
        timestamp.strftime("%H:%M:%S") for timestamp in pd.date_range(start, end, freq="min")
    )


DEFAULT_DECISION_TIMES = minute_decision_times()


def _expand_clock_token(token: str) -> list[str]:
    token = token.strip()
    for separator in ("..", "-"):
        if separator in token:
            start, end = token.split(separator, 1)
            return list(minute_decision_times(start, end))
    return [normalize_clock_time(token)]


def parse_clock_times(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_DECISION_TIMES)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    times = []
    for part in parts:
        if part and part.strip():
            times.extend(_expand_clock_token(part))
    return times


def select_decision_points(
    frame: pd.DataFrame,
    *,
    decision_times: Iterable[str] = DEFAULT_DECISION_TIMES,
    max_lag_seconds: int | None = 5,
) -> pd.DataFrame:
    """Select the first sample tick at or after each configured clock time."""

    times = parse_clock_times(decision_times)
    if not times:
        raise SystemExit("decision point sampling needs at least one decision time")

    work = ensure_timestamp_columns(frame)
    if work.empty:
        return work.copy()

    tolerance = pd.Timedelta(seconds=max_lag_seconds) if max_lag_seconds is not None else None
    selected_parts = []
    base_dates = work[["date"]].drop_duplicates().sort_values("date")

    for _, date_row in base_dates.iterrows():
        trading_day = str(date_row["date"])
        date_targets = pd.DataFrame(
            {
                "decision_time": times,
                "_target_ts": pd.to_datetime([f"{trading_day} {clock}" for clock in times]),
            }
        ).sort_values("_target_ts")

        day = work.loc[work["date"] == trading_day]
        for symbol, group in day.groupby("symbol", sort=False, observed=True):
            right = (
                group.reset_index(names="_source_index")
                .sort_values("timestamp")
                .loc[:, ["_source_index", "timestamp"]]
            )
            if right.empty:
                continue
            merged = pd.merge_asof(
                date_targets,
                right,
                left_on="_target_ts",
                right_on="timestamp",
                direction="forward",
                tolerance=tolerance,
            ).dropna(subset=["_source_index"])
            if merged.empty:
                continue
            chosen = work.loc[merged["_source_index"].astype(int).to_numpy()].copy()
            chosen["decision_time"] = merged["decision_time"].to_numpy()
            chosen["decision_target_timestamp"] = merged["_target_ts"].to_numpy()
            chosen["decision_lag_seconds"] = (
                chosen["timestamp"].to_numpy() - chosen["decision_target_timestamp"]
            ) / pd.Timedelta(seconds=1)
            chosen["symbol"] = str(symbol)
            selected_parts.append(chosen)

    if not selected_parts:
        return work.iloc[:0].copy()

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)


def sample_labeled_frame(
    frame: pd.DataFrame,
    *,
    mode: str = "decision_points",
    decision_times: Iterable[str] = DEFAULT_DECISION_TIMES,
    max_lag_seconds: int | None = 5,
) -> pd.DataFrame:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "all", "all_ticks", "tick"}:
        return frame.copy()
    if normalized_mode in {"decision", "decision_point", "decision_points"}:
        return select_decision_points(
            frame,
            decision_times=decision_times,
            max_lag_seconds=max_lag_seconds,
        )
    valid = "all_ticks, decision_points"
    raise SystemExit(f"unknown sample mode {mode!r}; expected one of: {valid}")


def require_entry_after_cross_section_ready(
    frame: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
) -> pd.DataFrame:
    """Invalidate labels whose entry precedes the complete sampled cross-section."""

    missing = [
        column
        for column in (*group_cols, "timestamp", "entry_timestamp")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "cross-section entry readiness requires columns: " + ", ".join(missing)
        )

    out = frame.copy()
    feature_timestamp = pd.to_datetime(out["timestamp"], errors="coerce")
    entry_timestamp = pd.to_datetime(out["entry_timestamp"], errors="coerce")
    group_keys = [out[column] for column in group_cols]
    ready_timestamp = feature_timestamp.groupby(group_keys, sort=False).transform("max")
    ready = entry_timestamp.notna() & ready_timestamp.notna() & entry_timestamp.ge(
        ready_timestamp
    )
    out["cross_section_ready_timestamp"] = ready_timestamp
    out["entry_after_cross_section_ready"] = ready
    if "valid_label" in out.columns:
        out["valid_label"] = out["valid_label"].fillna(False).astype(bool) & ready
    return out
