from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "alpha_return_1m",
    "alpha_return_10m",
    "alpha_return_60m",
)
TARGET_COLUMN = "alpha_return_close_to_next_close"
SEQUENCE_SCHEMA_VERSION = "full_day_return_paths_sequence_v1"


def clock_seconds(values: pd.Series | Sequence[object]) -> np.ndarray:
    raw = pd.Series(values, copy=False)
    clock_only = raw.astype(str).str.fullmatch(r"\d{2}:\d{2}(?::\d{2})?")
    if bool(clock_only.all()):
        parts = raw.astype(str).str.split(":", expand=True).astype(np.int32)
        seconds = parts[2].to_numpy(dtype=np.int32) if parts.shape[1] == 3 else 0
        return (
            parts[0].to_numpy(dtype=np.int32) * 3600
            + parts[1].to_numpy(dtype=np.int32) * 60
            + seconds
        )
    timestamps = pd.to_datetime(raw, errors="coerce")
    if isinstance(timestamps, pd.Series):
        hours = timestamps.dt.hour.to_numpy(dtype=np.int32)
        minutes = timestamps.dt.minute.to_numpy(dtype=np.int32)
        seconds = timestamps.dt.second.to_numpy(dtype=np.int32)
    return hours * 3600 + minutes * 60 + seconds


def assemble_day_sequence(
    feature_frame: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    clocks: Sequence[str],
    pool_symbols: set[str] | None = None,
) -> dict[str, np.ndarray]:
    required_features = {
        "symbol",
        "decision_target_timestamp",
        *FEATURE_COLUMNS,
    }
    missing_features = sorted(required_features - set(feature_frame.columns))
    if missing_features:
        raise SystemExit(f"temporal feature shard missing columns: {missing_features}")
    required_labels = {"symbol", TARGET_COLUMN}
    missing_labels = sorted(required_labels - set(labels.columns))
    if missing_labels:
        raise SystemExit(f"daily label shard missing columns: {missing_labels}")

    source = feature_frame.loc[:, ["symbol", "decision_target_timestamp", *FEATURE_COLUMNS]].copy()
    source["symbol"] = source["symbol"].astype(str)
    source["_clock"] = clock_seconds(source["decision_target_timestamp"])
    clock_values = clock_seconds(list(clocks))
    clock_index = pd.Index(clock_values)
    clock_codes = clock_index.get_indexer(source["_clock"])
    in_clock = clock_codes >= 0
    source = source.loc[in_clock].reset_index(drop=True)
    clock_codes = clock_codes[in_clock]

    symbols = pd.Index(sorted(source["symbol"].unique()))
    symbol_codes = symbols.get_indexer(source["symbol"])
    flat_codes = symbol_codes.astype(np.int64) * len(clock_values) + clock_codes
    if pd.Series(flat_codes).duplicated().any():
        raise SystemExit("temporal feature shard has duplicate symbol × clock rows")

    values = np.full(
        (len(symbols), len(FEATURE_COLUMNS), len(clock_values)),
        np.nan,
        dtype=np.float32,
    )
    for channel, column in enumerate(FEATURE_COLUMNS):
        values[symbol_codes, channel, clock_codes] = pd.to_numeric(
            source[column], errors="coerce"
        ).to_numpy(dtype=np.float32)

    label_frame = labels.loc[:, ["symbol", TARGET_COLUMN]].copy()
    label_frame["symbol"] = label_frame["symbol"].astype(str)
    if label_frame["symbol"].duplicated().any():
        raise SystemExit("daily label shard has duplicate symbols")
    aligned_labels = label_frame.set_index("symbol")[TARGET_COLUMN].reindex(symbols)
    target = pd.to_numeric(aligned_labels, errors="coerce").to_numpy(dtype=np.float32)
    pool_member = np.fromiter(
        (str(symbol) in (pool_symbols or set()) for symbol in symbols),
        dtype=np.bool_,
        count=len(symbols),
    )
    return {
        "symbols": symbols.to_numpy(dtype="U16"),
        "clock_seconds": clock_values.astype(np.int32),
        "values": values,
        "valid": np.isfinite(values),
        "target": target,
        "pool_member": pool_member,
    }


def cross_section_rank_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("values and valid must share shape [symbols, channels, clocks]")
    output = np.zeros(values.shape, dtype=np.float32)
    for channel in range(values.shape[1]):
        channel_valid = valid[:, channel, :]
        ranked = pd.DataFrame(np.where(channel_valid, values[:, channel, :], np.nan)).rank(
            axis=0,
            method="average",
        )
        counts = channel_valid.sum(axis=0).astype(np.float32)
        denominator = np.where(counts > 1.0, (counts - 1.0) / 2.0, 1.0)
        normalized = (
            ranked.to_numpy(dtype=np.float32) - (counts[None, :] + 1.0) / 2.0
        ) / denominator[None, :]
        normalized[~np.isfinite(normalized)] = 0.0
        output[:, channel, :] = normalized
    return output


def write_sequence_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in arrays.items() if key != "valid"}
    payload["values"] = np.asarray(arrays["values"], dtype=np.float16)
    payload["rank_values"] = cross_section_rank_values(
        np.asarray(arrays["values"], dtype=np.float32),
        np.asarray(arrays["valid"], dtype=bool),
    ).astype(np.float16)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **payload)
    os.replace(temporary, path)


def _column_metrics(
    values: np.ndarray,
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    top_n: int,
    tail_fraction: float,
) -> pd.DataFrame:
    if values.ndim != 2:
        raise ValueError("values must have shape [symbols, clocks]")
    eligible = np.asarray(universe_mask, dtype=bool) & np.isfinite(target)
    x = values[eligible].astype(np.float64, copy=False)
    y = target[eligible].astype(np.float64, copy=False)
    valid = np.isfinite(x)
    counts = valid.sum(axis=0).astype(np.int64)

    x_frame = pd.DataFrame(x)
    x_rank = x_frame.rank(axis=0, method="average").to_numpy(dtype=np.float64)
    y_matrix = np.broadcast_to(y[:, None], x.shape).copy()
    y_matrix[~valid] = np.nan
    y_rank = pd.DataFrame(y_matrix).rank(axis=0, method="average").to_numpy(dtype=np.float64)
    x_mean = np.divide(
        np.nansum(x_rank, axis=0),
        counts,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    y_mean = np.divide(
        np.nansum(y_rank, axis=0),
        counts,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    x_centered = x_rank - x_mean
    y_centered = y_rank - y_mean
    numerator = np.nansum(x_centered * y_centered, axis=0)
    denominator = np.sqrt(np.nansum(x_centered**2, axis=0) * np.nansum(y_centered**2, axis=0))
    rank_ic = np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=(counts >= 3) & (denominator > 0),
    )

    base_mean = np.divide(
        np.nansum(y_matrix, axis=0),
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    top_counts = np.minimum(counts, max(1, int(top_n)))
    top_mask = valid & (x_rank > (counts - top_counts)[None, :])
    top_mean = np.divide(
        np.nansum(np.where(top_mask, y_matrix, np.nan), axis=0),
        top_mask.sum(axis=0),
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=top_mask.sum(axis=0) > 0,
    )

    tail_counts = np.maximum(1, np.floor(counts * float(tail_fraction)).astype(np.int64))
    top_tail = valid & (x_rank > (counts - tail_counts)[None, :])
    bottom_tail = valid & (x_rank <= tail_counts[None, :])
    top_tail_mean = np.divide(
        np.nansum(np.where(top_tail, y_matrix, np.nan), axis=0),
        top_tail.sum(axis=0),
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=top_tail.sum(axis=0) > 0,
    )
    bottom_tail_mean = np.divide(
        np.nansum(np.where(bottom_tail, y_matrix, np.nan), axis=0),
        bottom_tail.sum(axis=0),
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=bottom_tail.sum(axis=0) > 0,
    )
    return pd.DataFrame(
        {
            "n": counts,
            "rank_ic": rank_ic,
            "base_return": base_mean,
            "top_n_return": top_mean,
            "top_n_excess": top_mean - base_mean,
            "top_tail_return": top_tail_mean,
            "bottom_tail_return": bottom_tail_mean,
            "head_tail_spread": top_tail_mean - bottom_tail_mean,
        }
    )


def analyze_day_sequence(
    arrays: dict[str, np.ndarray],
    *,
    date: str,
    top_n: int = 100,
    tail_fraction: float = 0.1,
) -> pd.DataFrame:
    values = arrays["values"]
    target = arrays["target"]
    clocks = arrays["clock_seconds"]
    pool_member = arrays["pool_member"].astype(bool)
    if values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("sequence channel count does not match FEATURE_COLUMNS")

    parts: list[pd.DataFrame] = []
    universes = {
        "all_a": np.ones(len(target), dtype=bool),
        "pool_l": pool_member,
    }
    for channel, feature_column in enumerate(FEATURE_COLUMNS):
        horizon = feature_column.removeprefix("alpha_return_")
        for universe, universe_mask in universes.items():
            metrics = _column_metrics(
                values[:, channel, :],
                target,
                universe_mask=universe_mask,
                top_n=top_n,
                tail_fraction=tail_fraction,
            )
            metrics.insert(0, "clock_seconds", clocks)
            metrics.insert(
                1,
                "clock",
                [f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}" for seconds in clocks],
            )
            metrics.insert(0, "horizon", horizon)
            metrics.insert(0, "universe", universe)
            metrics.insert(0, "date", str(date))
            parts.append(metrics)
    return pd.concat(parts, ignore_index=True)


def summarize_temporal_metrics(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = daily.copy()
    work["year"] = work["date"].astype(str).str[:4].astype(int)
    keys = ["universe", "horizon", "clock_seconds", "clock"]
    metric_columns = [
        "rank_ic",
        "top_n_excess",
        "head_tail_spread",
    ]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        output: dict[str, float | int] = {"days": int(group["date"].nunique())}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            output[f"mean_{column}"] = mean
            output[f"{column}_ir"] = (
                mean / std * np.sqrt(252.0) if np.isfinite(std) and std > 0 else np.nan
            )
            output[f"{column}_positive_fraction"] = float((values > 0).mean())
        output["mean_n"] = float(pd.to_numeric(group["n"], errors="coerce").mean())
        return pd.Series(output)

    overall = (
        work.groupby(keys, observed=True, sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    annual = (
        work.groupby(["year", *keys], observed=True, sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    return overall, annual
