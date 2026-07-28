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


def eligible_feature_mask(
    arrays: Mapping[str, np.ndarray],
    *,
    latest_clocks: Mapping[str, str] | None = None,
) -> np.ndarray:
    latest = dict(DEFAULT_LATEST_CLOCKS)
    latest.update(latest_clocks or {})
    valid_source = arrays.get("valid")
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
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(arrays["values"], dtype=np.float32)
    valid = eligible_feature_mask(arrays, latest_clocks=latest_clocks)
    normalized_mode = str(value_mode).strip().lower()
    if normalized_mode == "cross_section_rank":
        cached_rank = arrays.get("rank_values")
        transformed = (
            np.asarray(cached_rank, dtype=np.float32).copy()
            if cached_rank is not None
            else cross_section_rank_values(values, valid)
        )
        transformed[~valid] = 0.0
    elif normalized_mode == "raw_tanh":
        if raw_scale <= 0:
            raise ValueError("raw_scale must be positive")
        transformed = np.tanh(values / float(raw_scale)).astype(np.float32)
        transformed[~valid] = 0.0
    else:
        raise ValueError(
            f"unsupported temporal value_mode={value_mode!r}; "
            "expected cross_section_rank or raw_tanh"
        )
    inputs = np.concatenate([transformed, valid.astype(np.float32)], axis=1)
    time_valid = valid.any(axis=1)
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
