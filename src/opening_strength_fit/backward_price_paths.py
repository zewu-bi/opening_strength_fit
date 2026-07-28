from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from opening_strength_fit.temporal_analysis import TARGET_COLUMN

BACKWARD_PRICE_PATH_SCHEMA_VERSION = "full_day_backward_price_paths_sequence_v1"
HORIZON_MINUTES = (1, 10, 60)
STATE_COUNT = 241
RETURN_CLOCK_COUNT = 240

DEFAULT_VALID_PRICE_STATUSES = (
    "T0",
    "20",
    "TRADE",
    "O0",
    "10",
    "OCALL",
)

_MORNING_OPEN_SECONDS = 9 * 3600 + 30 * 60
_MORNING_CLOSE_SECONDS = 11 * 3600 + 30 * 60
_AFTERNOON_OPEN_SECONDS = 13 * 3600


def state_endpoint_seconds() -> np.ndarray:
    """Return the 09:30 boundary plus 240 trading-minute endpoints."""

    morning = np.arange(
        _MORNING_OPEN_SECONDS,
        _MORNING_CLOSE_SECONDS + 60,
        60,
        dtype=np.int32,
    )
    afternoon = np.arange(
        _AFTERNOON_OPEN_SECONDS + 60,
        15 * 3600 + 60,
        60,
        dtype=np.int32,
    )
    endpoints = np.concatenate([morning, afternoon])
    if len(endpoints) != STATE_COUNT:
        raise RuntimeError(f"unexpected endpoint count: {len(endpoints)}")
    return endpoints


def return_clock_seconds() -> np.ndarray:
    return state_endpoint_seconds()[1:]


def state_endpoint_offsets_us() -> np.ndarray:
    return state_endpoint_seconds().astype(np.int64) * 1_000_000


def _normalize_states(states: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "state_index", "price", "status", "source_offset_us"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise SystemExit(f"endpoint price states missing columns: {missing}")

    out = states.loc[:, sorted(required)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["state_index"] = pd.to_numeric(out["state_index"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["source_offset_us"] = pd.to_numeric(out["source_offset_us"], errors="coerce")
    out["status"] = out["status"].astype(str).str.upper()
    out = out.dropna(subset=["symbol", "state_index"])
    out["state_index"] = out["state_index"].astype(np.int16)

    invalid_index = ~out["state_index"].between(0, STATE_COUNT - 1)
    if invalid_index.any():
        examples = out.loc[invalid_index, ["symbol", "state_index"]].head(5).to_dict("records")
        raise RuntimeError(f"endpoint price states have invalid state_index: {examples}")
    duplicates = out.duplicated(["symbol", "state_index"], keep=False)
    if duplicates.any():
        examples = (
            out.loc[duplicates, ["symbol", "state_index"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise RuntimeError(f"duplicate symbol × state_index rows: {examples}")

    endpoint_offsets = state_endpoint_offsets_us()[out["state_index"].to_numpy()]
    future = out["source_offset_us"].to_numpy(dtype=np.float64) > endpoint_offsets
    if future.any():
        examples = out.loc[
            future,
            ["symbol", "state_index", "source_offset_us"],
        ].head(5)
        raise RuntimeError(f"price state uses a future tick: {examples.to_dict('records')}")
    return out


def _state_matrices(
    states: pd.DataFrame,
    *,
    valid_statuses: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalize_states(states)
    symbols = pd.Index(sorted(normalized["symbol"].unique()))
    symbol_codes = symbols.get_indexer(normalized["symbol"])
    state_codes = normalized["state_index"].to_numpy(dtype=np.intp)

    prices = np.full((len(symbols), STATE_COUNT), np.nan, dtype=np.float32)
    prices[symbol_codes, state_codes] = normalized["price"].to_numpy(dtype=np.float32)
    status_valid = np.zeros((len(symbols), STATE_COUNT), dtype=bool)
    allowed = {str(value).upper() for value in valid_statuses}
    status_valid[symbol_codes, state_codes] = normalized["status"].isin(allowed).to_numpy()
    state_valid = np.isfinite(prices) & (prices > 0) & status_valid
    return symbols.to_numpy(dtype="U16"), prices, state_valid


def assemble_backward_price_sequence(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    pool_symbols: set[str] | None = None,
    horizons: Iterable[int] = HORIZON_MINUTES,
    valid_statuses: Iterable[str] = DEFAULT_VALID_PRICE_STATUSES,
) -> dict[str, np.ndarray]:
    """Build endpoint-aligned P(t) / P(t-h) - 1 paths for one trading day."""

    requested = tuple(int(value) for value in horizons)
    if requested != HORIZON_MINUTES:
        raise ValueError(f"horizons must be exactly {HORIZON_MINUTES}, got {requested}")
    required_labels = {"symbol", TARGET_COLUMN}
    missing_labels = sorted(required_labels - set(labels.columns))
    if missing_labels:
        raise SystemExit(f"daily labels missing columns: {missing_labels}")

    symbols, prices, state_valid = _state_matrices(
        states,
        valid_statuses=valid_statuses,
    )
    values = np.full(
        (len(symbols), len(HORIZON_MINUTES), RETURN_CLOCK_COUNT),
        np.nan,
        dtype=np.float32,
    )
    for channel, horizon in enumerate(HORIZON_MINUTES):
        numerator = prices[:, horizon:]
        denominator = prices[:, :-horizon]
        valid = state_valid[:, horizon:] & state_valid[:, :-horizon]
        returns = np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan, dtype=np.float32),
            where=valid,
        )
        returns -= 1.0
        values[:, channel, horizon - 1 :] = returns

    label_frame = labels.loc[:, ["symbol", TARGET_COLUMN]].copy()
    label_frame["symbol"] = label_frame["symbol"].astype(str)
    if label_frame["symbol"].duplicated().any():
        raise RuntimeError("daily labels have duplicate symbols")
    target = pd.to_numeric(
        label_frame.set_index("symbol")[TARGET_COLUMN].reindex(symbols),
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    members = pool_symbols or set()
    pool_member = np.fromiter(
        (str(symbol) in members for symbol in symbols),
        dtype=np.bool_,
        count=len(symbols),
    )
    return {
        "symbols": symbols,
        "clock_seconds": return_clock_seconds(),
        "values": values,
        "valid": np.isfinite(values),
        "target": target,
        "pool_member": pool_member,
    }
