from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.temporal_analysis import (
    FEATURE_COLUMNS,
    cross_section_rank_values,
)

DEFAULT_LATEST_CLOCKS = {
    "1m": "14:47",
    "10m": "14:38",
    "60m": "13:48",
}


def time_to_seconds(value: str) -> int:
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError(f"invalid clock: {value!r}")
    hour, minute = parts[:2]
    second = parts[2] if len(parts) == 3 else 0
    return hour * 3600 + minute * 60 + second


def list_sequence_paths(
    sequence_root: Path,
    *,
    start_date: str,
    end_date: str,
) -> list[Path]:
    paths = sorted(sequence_root.glob("year=*/date=*/sequence.npz"))
    return [
        path for path in paths if start_date <= path.parent.name.removeprefix("date=") <= end_date
    ]


def load_sequence(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def aligned_sequence_validity(
    primary: Mapping[str, np.ndarray],
    mask_source: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Align another sequence's native validity by symbol and trading-minute slot."""

    primary_values = np.asarray(primary["values"])
    mask_values = np.asarray(mask_source["values"])
    if primary_values.ndim != 3 or mask_values.ndim != 3:
        raise ValueError("temporal values must have shape [symbols, channels, clocks]")
    if primary_values.shape[1:] != mask_values.shape[1:]:
        raise ValueError(
            "input mask sequence must match the primary channel and trading-minute slot counts"
        )

    primary_symbols = pd.Index(np.asarray(primary["symbols"]).astype(str))
    mask_symbols = pd.Index(np.asarray(mask_source["symbols"]).astype(str))
    if primary_symbols.has_duplicates or mask_symbols.has_duplicates:
        raise ValueError("temporal sequence symbols must be unique")
    mask_indices = mask_symbols.get_indexer(primary_symbols)
    matched = mask_indices >= 0
    aligned = np.zeros(primary_values.shape, dtype=bool)
    if matched.any():
        native_valid_source = mask_source.get("valid")
        native_valid = (
            np.asarray(native_valid_source, dtype=bool)
            if native_valid_source is not None
            else np.isfinite(mask_values)
        )
        aligned[matched] = native_valid[mask_indices[matched]]
    return aligned


def sequence_mask_path(primary_path: Path, mask_sequence_root: Path) -> Path:
    """Resolve the same year/date shard below another sequence root."""

    date_dir = primary_path.parent.name
    year_dir = primary_path.parent.parent.name
    if not date_dir.startswith("date=") or not year_dir.startswith("year="):
        raise ValueError(f"unexpected temporal sequence path layout: {primary_path}")
    return mask_sequence_root / year_dir / date_dir / primary_path.name


def eligible_feature_mask(
    arrays: Mapping[str, np.ndarray],
    *,
    latest_clocks: Mapping[str, str] | None = None,
    valid_override: np.ndarray | None = None,
) -> np.ndarray:
    latest = dict(DEFAULT_LATEST_CLOCKS)
    latest.update(latest_clocks or {})
    valid_source = valid_override if valid_override is not None else arrays.get("valid")
    valid = (
        np.asarray(valid_source, dtype=bool).copy()
        if valid_source is not None
        else np.isfinite(np.asarray(arrays["values"]))
    )
    clocks = np.asarray(arrays["clock_seconds"], dtype=np.int32)
    if valid.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("sequence channel count does not match FEATURE_COLUMNS")
    for channel, feature in enumerate(FEATURE_COLUMNS):
        horizon = feature.removeprefix("alpha_return_")
        cutoff = time_to_seconds(latest[horizon])
        valid[:, channel, clocks > cutoff] = False
    return valid


def prepare_day_inputs(
    arrays: Mapping[str, np.ndarray],
    *,
    value_mode: str,
    latest_clocks: Mapping[str, str] | None = None,
    raw_scale: float = 0.02,
    raw_scales: Mapping[str, float] | None = None,
    input_valid_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arrays["values"], dtype=np.float32)
    value_valid = eligible_feature_mask(arrays, latest_clocks=latest_clocks)
    input_valid = (
        value_valid
        if input_valid_override is None
        else eligible_feature_mask(
            arrays,
            latest_clocks=latest_clocks,
            valid_override=input_valid_override,
        )
    )
    normalized_mode = str(value_mode).strip().lower()
    if normalized_mode == "cross_section_rank":
        cached_rank = arrays.get("rank_values")
        transformed = (
            np.asarray(cached_rank, dtype=np.float32).copy()
            if cached_rank is not None
            else cross_section_rank_values(values, value_valid)
        )
        transformed[~value_valid] = 0.0
    elif normalized_mode in {"raw_tanh", "relative_tanh"}:
        configured_scales = {
            str(key).removeprefix("alpha_return_"): float(value)
            for key, value in (raw_scales or {}).items()
        }
        channel_scales = np.asarray(
            [
                configured_scales.get(
                    feature.removeprefix("alpha_return_"),
                    float(raw_scale),
                )
                for feature in FEATURE_COLUMNS
            ],
            dtype=np.float32,
        )
        if not np.isfinite(channel_scales).all() or (channel_scales <= 0).any():
            raise ValueError("raw scales must be finite and positive")
        source = values
        if normalized_mode == "relative_tanh":
            valid_values = np.where(value_valid, values, np.nan)
            counts = value_valid.sum(axis=0)
            sums = np.nansum(valid_values, axis=0)
            means = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums, dtype=np.float32),
                where=counts > 0,
            )
            source = values - means[None, :, :]
        transformed = np.tanh(source / channel_scales[None, :, None]).astype(np.float32)
        transformed[~value_valid] = 0.0
    else:
        raise ValueError(
            f"unsupported temporal value_mode={value_mode!r}; "
            "expected cross_section_rank, raw_tanh, or relative_tanh"
        )
    inputs = np.concatenate([transformed, input_valid.astype(np.float32)], axis=1)
    time_valid = input_valid.any(axis=1)
    return inputs, time_valid


def target_rank(
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
) -> np.ndarray:
    eligible = np.isfinite(target) & np.asarray(universe_mask, dtype=bool)
    out = np.full(len(target), np.nan, dtype=np.float32)
    if eligible.any():
        ranked = pd.Series(target[eligible]).rank(method="average")
        count = int(eligible.sum())
        denominator = (count - 1.0) / 2.0 if count > 1 else 1.0
        out[eligible] = (ranked.to_numpy(dtype=np.float32) - (count + 1.0) / 2.0) / denominator
    return out


def target_values(
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    mode: str,
    winsor_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Transform one day's target without changing its cross-sectional membership."""

    values = np.asarray(target, dtype=np.float32)
    eligible = np.isfinite(values) & np.asarray(universe_mask, dtype=bool)
    normalized = str(mode).strip().lower()
    if normalized == "rank":
        return target_rank(values, universe_mask=eligible)

    out = np.full(len(values), np.nan, dtype=np.float32)
    if not eligible.any():
        return out
    if normalized in {"raw", "raw_winsor"}:
        transformed = values[eligible].astype(np.float32, copy=True)
    elif normalized in {"market_relative", "market_relative_winsor"}:
        transformed = values[eligible] - float(np.mean(values[eligible], dtype=np.float64))
    else:
        raise ValueError(
            f"unsupported target mode={mode!r}; expected rank, raw, raw_winsor, "
            "market_relative, or market_relative_winsor"
        )
    if normalized.endswith("_winsor"):
        if winsor_bounds is None:
            raise ValueError(f"target mode {mode!r} requires winsor_bounds")
        lower, upper = (float(value) for value in winsor_bounds)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("winsor bounds must be finite and strictly increasing")
        transformed = np.clip(transformed, lower, upper)
    out[eligible] = transformed
    return out


def universe_mask(arrays: Mapping[str, np.ndarray], universe: str) -> np.ndarray:
    normalized = str(universe).strip().lower()
    target = np.asarray(arrays["target"])
    if normalized == "all_a":
        return np.ones(len(target), dtype=bool)
    if normalized == "pool_l":
        return np.asarray(arrays["pool_member"], dtype=bool)
    raise ValueError(f"unsupported universe={universe!r}; expected all_a or pool_l")


def month_bounds(month: str) -> tuple[str, str]:
    period = pd.Period(str(month), freq="M")
    return str(period.start_time.date()), str(period.end_time.date())


def rolling_date_bounds(
    *,
    test_start_month: str,
    test_end_month: str,
    train_months: int,
    validation_months: int,
) -> dict[str, str]:
    if train_months < 2:
        raise ValueError("train_months must be >= 2")
    if validation_months < 1 or validation_months >= train_months:
        raise ValueError("validation_months must be in [1, train_months)")
    test_start = pd.Period(test_start_month, freq="M")
    test_end = pd.Period(test_end_month, freq="M")
    train_start = test_start - train_months
    validation_start = test_start - validation_months
    train_end = validation_start - 1
    return {
        "train_start_date": str(train_start.start_time.date()),
        "train_end_date": str(train_end.end_time.date()),
        "validation_start_date": str(validation_start.start_time.date()),
        "validation_end_date": str((test_start - 1).end_time.date()),
        "test_start_date": str(test_start.start_time.date()),
        "test_end_date": str(test_end.end_time.date()),
    }


def daily_rank_ic(scores: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    eligible = np.asarray(mask, dtype=bool) & np.isfinite(scores) & np.isfinite(target)
    if eligible.sum() < 3:
        return float("nan")
    score_rank = pd.Series(scores[eligible]).rank(method="average")
    target_rank_values = pd.Series(target[eligible]).rank(method="average")
    return float(score_rank.corr(target_rank_values))


def top_n_excess(
    scores: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    top_n: int,
) -> float:
    eligible = np.asarray(mask, dtype=bool) & np.isfinite(scores) & np.isfinite(target)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return float("nan")
    count = min(max(1, int(top_n)), len(indices))
    selected = indices[np.argpartition(scores[indices], -count)[-count:]]
    return float(np.mean(target[selected]) - np.mean(target[indices]))
