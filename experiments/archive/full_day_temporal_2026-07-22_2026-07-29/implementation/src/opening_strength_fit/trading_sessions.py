from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from opening_strength_fit.schema import normalize_clock_time

DEFAULT_A_SHARE_SESSIONS = (
    "09:30:00-11:30:00",
    "13:00:00-15:00:00",
)


@dataclass(frozen=True)
class TradingSession:
    start: str
    end: str

    @property
    def start_delta(self) -> pd.Timedelta:
        return _clock_delta(self.start)

    @property
    def end_delta(self) -> pd.Timedelta:
        return _clock_delta(self.end)


def _clock_delta(value: str) -> pd.Timedelta:
    normalized = normalize_clock_time(value)
    hours, minutes, seconds = (int(part) for part in normalized.split(":"))
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def parse_trading_sessions(
    values: Iterable[str] | None = None,
) -> tuple[TradingSession, ...]:
    raw_values = tuple(values or DEFAULT_A_SHARE_SESSIONS)
    sessions: list[TradingSession] = []
    for raw in raw_values:
        token = str(raw).strip()
        if "-" not in token:
            raise ValueError(f"invalid trading session {raw!r}; expected HH:MM:SS-HH:MM:SS")
        start, end = token.split("-", 1)
        session = TradingSession(normalize_clock_time(start), normalize_clock_time(end))
        if session.end_delta <= session.start_delta:
            raise ValueError(f"trading session end must be after start: {raw!r}")
        sessions.append(session)
    sessions.sort(key=lambda item: item.start_delta)
    for previous, current in zip(sessions, sessions[1:], strict=False):
        if current.start_delta < previous.end_delta:
            raise ValueError("trading sessions must not overlap")
    if not sessions:
        raise ValueError("at least one trading session is required")
    return tuple(sessions)


def coerce_trading_sessions(
    values: Sequence[TradingSession] | Iterable[str] = DEFAULT_A_SHARE_SESSIONS,
) -> tuple[TradingSession, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("at least one trading session is required")
    if all(isinstance(item, TradingSession) for item in raw):
        return raw  # type: ignore[return-value]
    if any(isinstance(item, TradingSession) for item in raw):
        raise ValueError("trading sessions must be all strings or all TradingSession values")
    return parse_trading_sessions(raw)  # type: ignore[arg-type]


def shift_by_trading_seconds(
    timestamp: pd.Timestamp | str | None,
    seconds: int,
    *,
    sessions: Sequence[TradingSession] | Iterable[str] = DEFAULT_A_SHARE_SESSIONS,
) -> pd.Timestamp:
    """Advance inside same-day sessions, skipping breaks and preserving sub-seconds.

    A target exactly at the final session close is representable. Any target beyond
    that close is ``NaT`` so callers can mark its label unavailable.
    """

    value = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(value):
        return pd.NaT
    seconds = int(seconds)
    if seconds < 0:
        raise ValueError("trading-second shift must be non-negative")
    parsed = coerce_trading_sessions(sessions)
    day = value.normalize()
    windows = [(day + item.start_delta, day + item.end_delta) for item in parsed]

    for index, (start, end) in enumerate(windows):
        if value < start:
            return pd.NaT
        if value == end and seconds == 0:
            return value
        if not (start <= value < end):
            continue
        remaining = pd.Timedelta(seconds=seconds)
        available = end - value
        if remaining < available:
            return value + remaining
        if remaining == available:
            if index + 1 < len(windows):
                return windows[index + 1][0]
            return end
        remaining -= available
        for next_index in range(index + 1, len(windows)):
            next_start, next_end = windows[next_index]
            duration = next_end - next_start
            if remaining < duration:
                return next_start + remaining
            if remaining == duration:
                if next_index + 1 < len(windows):
                    return windows[next_index + 1][0]
                return next_end
            remaining -= duration
        return pd.NaT
    return pd.NaT


def shift_series_by_trading_seconds(
    timestamps: pd.Series,
    seconds: int,
    *,
    sessions: Sequence[TradingSession] | Iterable[str] = DEFAULT_A_SHARE_SESSIONS,
) -> pd.Series:
    parsed = coerce_trading_sessions(sessions)
    return pd.to_datetime(timestamps, errors="coerce").map(
        lambda value: shift_by_trading_seconds(value, seconds, sessions=parsed)
    )
