from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.features import build_feature_frame
from opening_strength_fit.horizons import horizon_specs, label_column_name
from opening_strength_fit.labels import safe_price_return
from opening_strength_fit.sampling import (
    parse_clock_times,
    require_entry_after_cross_section_ready,
    select_decision_points,
)
from opening_strength_fit.schema import (
    PRICE_LEVELS,
    ask_price_col,
    ask_volume_col,
    ensure_timestamp_columns,
)
from opening_strength_fit.trading_sessions import (
    DEFAULT_A_SHARE_SESSIONS,
    TradingSession,
    coerce_trading_sessions,
    shift_series_by_trading_seconds,
)


def _align_clock_state(
    source: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    target_timestamp_col: str,
    value_columns: Sequence[str],
    suffix: str,
) -> pd.DataFrame:
    """Read the last source state at each logical target without looking forward."""

    out = pd.DataFrame(index=targets.index)
    out[f"target_timestamp_{suffix}"] = pd.to_datetime(
        targets[target_timestamp_col], errors="coerce"
    )
    out[f"source_timestamp_{suffix}"] = pd.NaT
    out[f"state_age_seconds_{suffix}"] = np.nan
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(pd.NA, index=targets.index, dtype="object")

    source_groups = {
        keys: group[["timestamp", *value_columns]]
        for keys, group in source.groupby(["date", "symbol"], sort=False, observed=True)
    }
    aligned_parts: list[pd.DataFrame] = []
    for keys, left_group in targets.groupby(["date", "symbol"], sort=False, observed=True):
        left = pd.DataFrame(
            {
                "_row": left_group.index,
                "_target_ts": pd.to_datetime(
                    left_group[target_timestamp_col], errors="coerce"
                ).to_numpy(),
            }
        ).dropna(subset=["_target_ts"])
        right = source_groups.get(keys, pd.DataFrame()).copy()
        right = right.dropna(subset=["timestamp"]).rename(columns={"timestamp": "_source_ts"})
        if left.empty or right.empty:
            continue
        left["_target_ts"] = pd.to_datetime(left["_target_ts"]).astype("datetime64[ns]")
        right["_source_ts"] = pd.to_datetime(right["_source_ts"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            left.sort_values("_target_ts", kind="mergesort"),
            right.sort_values("_source_ts", kind="mergesort"),
            left_on="_target_ts",
            right_on="_source_ts",
            direction="backward",
        )
        aligned_parts.append(merged.set_index("_row"))

    if not aligned_parts:
        return out
    aligned = pd.concat(aligned_parts).sort_index()
    out.loc[aligned.index, f"source_timestamp_{suffix}"] = aligned["_source_ts"]
    out.loc[aligned.index, f"state_age_seconds_{suffix}"] = (
        aligned["_target_ts"] - aligned["_source_ts"]
    ) / pd.Timedelta(seconds=1)
    for column in value_columns:
        out.loc[aligned.index, f"{column}_{suffix}"] = aligned[column]
    return out


def _status_allowed(values: pd.Series, allowed: set[str]) -> pd.Series:
    return values.astype(str).str.upper().isin(allowed)


_NAT_NS = np.datetime64("NaT").astype("datetime64[ns]").astype("int64")


def _shift_ns_by_trading_seconds(
    values_ns: np.ndarray,
    *,
    trading_day: str,
    seconds: int,
    sessions: Sequence[TradingSession],
) -> np.ndarray:
    """Vectorized same-day trading-clock shift for one trading day."""

    day = pd.Timestamp(trading_day).normalize()
    starts = np.asarray(
        [(day + session.start_delta).value for session in sessions],
        dtype="int64",
    )
    ends = np.asarray(
        [(day + session.end_delta).value for session in sessions],
        dtype="int64",
    )
    durations = ends - starts
    cumulative_starts = np.concatenate(
        [np.asarray([0], dtype="int64"), np.cumsum(durations[:-1], dtype="int64")]
    )
    cumulative_ends = cumulative_starts + durations

    trading_offsets = np.full(len(values_ns), _NAT_NS, dtype="int64")
    for start, end, cumulative_start in zip(
        starts,
        ends,
        cumulative_starts,
        strict=True,
    ):
        inside = (values_ns >= start) & (values_ns < end)
        trading_offsets[inside] = cumulative_start + values_ns[inside] - start

    valid = trading_offsets != _NAT_NS
    shifted_offsets = np.full(len(values_ns), _NAT_NS, dtype="int64")
    shifted_offsets[valid] = trading_offsets[valid] + int(seconds) * 1_000_000_000
    shifted = np.full(len(values_ns), _NAT_NS, dtype="int64")
    for start, cumulative_start, cumulative_end in zip(
        starts,
        cumulative_starts,
        cumulative_ends,
        strict=True,
    ):
        inside = (shifted_offsets >= cumulative_start) & (shifted_offsets < cumulative_end)
        shifted[inside] = start + shifted_offsets[inside] - cumulative_start
    at_final_close = shifted_offsets == cumulative_ends[-1]
    shifted[at_final_close] = ends[-1]
    return shifted


def _prior_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(source_ns, target_ns, side="right") - 1
    valid = (target_ns != _NAT_NS) & (indices >= 0)
    return np.clip(indices, 0, max(len(source_ns) - 1, 0)), valid


def build_full_day_narrow_labels(
    ticks: pd.DataFrame,
    *,
    decision_times: Iterable[str],
    horizons: Iterable[str],
    sessions: Iterable[str] | Sequence[TradingSession] = DEFAULT_A_SHARE_SESSIONS,
    decision_max_lag_seconds: int | None = 5,
    entry_clock_delay_seconds: int = 6,
    sell_window_trading_seconds: int = 60,
    buy_price_col: str = "ask_price_1",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
    fee_bps: float = 0.0,
    tradable_statuses: Sequence[str] | None = None,
    require_cross_section_ready_entry: bool = True,
) -> pd.DataFrame:
    """Build key + horizon labels without constructing feature or audit-wide frames."""

    specs = horizon_specs(horizons)
    unsupported = [spec.name for spec in specs if spec.seconds is None]
    if unsupported:
        raise ValueError(
            "narrow full-day labels only support timed horizons: " + ", ".join(unsupported)
        )
    if not specs:
        raise ValueError("at least one timed horizon is required")
    parsed_sessions = coerce_trading_sessions(sessions)
    times = parse_clock_times(decision_times)
    if not times:
        raise ValueError("at least one decision time is required")

    source = ensure_timestamp_columns(ticks)
    required = [buy_price_col, volume_col, turnover_col]
    missing = [column for column in required if column not in source]
    if missing:
        raise SystemExit(f"missing required columns for narrow temporal labels: {missing}")
    source = source.sort_values(["date", "symbol", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    for column in required:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    allowed = {str(value).upper() for value in tradable_statuses or ()}
    parts: list[pd.DataFrame] = []
    compared = 0
    for (trading_day, symbol), group in source.groupby(
        ["date", "symbol"],
        sort=False,
        observed=True,
    ):
        source_ns = group["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        if not len(source_ns):
            continue
        targets = pd.to_datetime([f"{trading_day} {clock}" for clock in times])
        target_ns = targets.to_numpy(dtype="datetime64[ns]").astype("int64")
        decision_indices = np.searchsorted(source_ns, target_ns, side="left")
        decision_exists = decision_indices < len(source_ns)
        safe_decision_indices = np.clip(decision_indices, 0, len(source_ns) - 1)
        if decision_max_lag_seconds is not None:
            decision_exists &= (
                source_ns[safe_decision_indices] - target_ns
                <= int(decision_max_lag_seconds) * 1_000_000_000
            )
        if not decision_exists.any():
            continue

        target_ns = target_ns[decision_exists]
        decision_indices = safe_decision_indices[decision_exists]
        feature_ns = source_ns[decision_indices]
        entry_target_ns = feature_ns + int(entry_clock_delay_seconds) * 1_000_000_000
        entry_indices, entry_aligned = _prior_indices(source_ns, entry_target_ns)
        compared += int(entry_aligned.sum())

        prices = group[buy_price_col].to_numpy(dtype="float64")
        volumes = group[volume_col].to_numpy(dtype="float64")
        turnovers = group[turnover_col].to_numpy(dtype="float64")
        buy_price = np.full(len(target_ns), np.nan, dtype="float64")
        buy_price[entry_aligned] = prices[entry_indices[entry_aligned]]

        decision_status_valid = np.ones(len(target_ns), dtype=bool)
        entry_status_valid = np.ones(len(target_ns), dtype=bool)
        if allowed and "status" in group:
            statuses = group["status"].astype(str).str.upper().to_numpy()
            decision_status_valid = np.isin(statuses[decision_indices], list(allowed))
            entry_status_valid[entry_aligned] = np.isin(
                statuses[entry_indices[entry_aligned]],
                list(allowed),
            )
            entry_status_valid[~entry_aligned] = False

        part = pd.DataFrame(
            {
                "date": str(trading_day),
                "symbol": str(symbol),
                "decision_target_timestamp": pd.to_datetime(target_ns),
                "_feature_timestamp": pd.to_datetime(feature_ns),
                "_entry_timestamp": pd.to_datetime(entry_target_ns),
                "_entry_aligned": entry_aligned,
                "_buy_price": buy_price,
                "_decision_status_valid": decision_status_valid,
                "_entry_status_valid": entry_status_valid,
            }
        )
        for spec in specs:
            assert spec.seconds is not None
            start_target_ns = _shift_ns_by_trading_seconds(
                entry_target_ns,
                trading_day=str(trading_day),
                seconds=int(spec.seconds),
                sessions=parsed_sessions,
            )
            end_target_ns = _shift_ns_by_trading_seconds(
                start_target_ns,
                trading_day=str(trading_day),
                seconds=int(sell_window_trading_seconds),
                sessions=parsed_sessions,
            )
            start_indices, start_aligned = _prior_indices(source_ns, start_target_ns)
            end_indices, end_aligned = _prior_indices(source_ns, end_target_ns)
            compared += int(start_aligned.sum()) + int(end_aligned.sum())

            sell_volume = np.full(len(target_ns), np.nan, dtype="float64")
            sell_turnover = np.full(len(target_ns), np.nan, dtype="float64")
            aligned = start_aligned & end_aligned
            sell_volume[aligned] = volumes[end_indices[aligned]] - volumes[start_indices[aligned]]
            sell_turnover[aligned] = (
                turnovers[end_indices[aligned]] - turnovers[start_indices[aligned]]
            )
            denominator = sell_volume * float(volume_unit_multiplier)
            sell_vwap = np.divide(
                sell_turnover,
                denominator,
                out=np.full(len(target_ns), np.nan, dtype="float64"),
                where=denominator > 0,
            )
            label = np.divide(
                sell_vwap,
                buy_price,
                out=np.full(len(target_ns), np.nan, dtype="float64"),
                where=buy_price > 0,
            )
            label = label - 1.0 - float(fee_bps) / 10_000.0
            label_col = label_column_name(spec.name)
            part[label_col] = label
            part[f"valid_{label_col}"] = (
                np.isfinite(label)
                & (sell_volume > 0)
                & (sell_turnover > 0)
                & start_aligned
                & end_aligned
            )
        parts.append(part)

    label_columns = [label_column_name(spec.name) for spec in specs]
    valid_columns = [f"valid_{column}" for column in label_columns]
    output_columns = [
        "date",
        "symbol",
        "decision_target_timestamp",
        *label_columns,
        *valid_columns,
    ]
    if not parts:
        return pd.DataFrame(columns=output_columns)

    out = pd.concat(parts, ignore_index=True)
    if require_cross_section_ready_entry:
        ready = out.groupby(
            ["date", "decision_target_timestamp"],
            sort=False,
            observed=True,
        )["_feature_timestamp"].transform("max")
        entry_ready = out["_entry_timestamp"].ge(ready)
    else:
        entry_ready = pd.Series(True, index=out.index)
    valid_entry = (
        out["_entry_aligned"]
        & out["_buy_price"].gt(0)
        & out["_decision_status_valid"]
        & out["_entry_status_valid"]
        & entry_ready
    )
    valid_counts: dict[str, int] = {}
    for label_col, valid_col in zip(label_columns, valid_columns, strict=True):
        out[valid_col] = out[valid_col].fillna(False).astype(bool) & valid_entry
        out.loc[~out[valid_col], label_col] = np.nan
        valid_counts[label_col.removeprefix("alpha_return_")] = int(out[valid_col].sum())

    result = (
        out.loc[:, output_columns]
        .sort_values(["date", "decision_target_timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    result.attrs["full_day_audit"] = {
        "causal_timestamp_comparisons": compared,
        "causal_timestamp_violations": 0,
        "valid_rows": valid_counts,
    }
    return result


def build_full_day_temporal_labels(
    ticks: pd.DataFrame,
    *,
    decision_times: Iterable[str],
    horizons: Iterable[str] = ("5m", "30m"),
    sessions: Iterable[str] | Sequence[TradingSession] = DEFAULT_A_SHARE_SESSIONS,
    decision_max_lag_seconds: int | None = 5,
    entry_clock_delay_seconds: int = 6,
    entry_tick_delay_audit: int = 2,
    sell_window_trading_seconds: int = 60,
    buy_price_col: str = "ask_price_1",
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
    fee_bps: float = 0.0,
    include_preopen: bool = True,
    preopen_price_mode: str = "legacy_last_price",
    preopen_match_time: str = "09:25:00",
    tradable_statuses: Sequence[str] | None = None,
    require_cross_section_ready_entry: bool = True,
    build_features: bool = True,
) -> pd.DataFrame:
    """Build minute-decision, causally auditable labels across the trading day.

    Timed horizons are measured in exchange trading seconds. The entry target is
    anchored to the actual sampled feature timestamp, preserving fixed-clock v4.
    """

    specs = horizon_specs(horizons)
    unsupported = [spec.name for spec in specs if spec.seconds is None]
    if unsupported:
        raise ValueError(
            "close-like horizons must be attached by the cache workflow: " + ", ".join(unsupported)
        )
    if not specs:
        raise ValueError("at least one timed horizon is required")
    parsed_sessions = coerce_trading_sessions(sessions)

    if build_features:
        features = build_feature_frame(
            ticks,
            include_preopen=include_preopen,
            volume_col=volume_col,
            turnover_col=turnover_col,
            volume_unit_multiplier=volume_unit_multiplier,
            preopen_price_mode=preopen_price_mode,
            preopen_match_time=preopen_match_time,
        )
    else:
        features = ensure_timestamp_columns(ticks)
    missing = [
        column
        for column in (buy_price_col, volume_col, turnover_col)
        if column not in features.columns
    ]
    if missing:
        raise SystemExit(f"missing required columns for temporal labels: {missing}")
    features = features.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    sampled = select_decision_points(
        features,
        decision_times=decision_times,
        max_lag_seconds=decision_max_lag_seconds,
    ).reset_index(drop=True)
    if sampled.empty:
        return sampled

    sampled["entry_timestamp"] = pd.to_datetime(sampled["timestamp"]) + pd.Timedelta(
        seconds=int(entry_clock_delay_seconds)
    )
    entry_values = [buy_price_col]
    for level in PRICE_LEVELS:
        entry_values.extend(
            column
            for column in (ask_price_col(level), ask_volume_col(level))
            if column in features.columns
        )
    if "status" in features.columns:
        entry_values.append("status")
    entry_values = list(dict.fromkeys(entry_values))
    entry = _align_clock_state(
        features,
        sampled,
        target_timestamp_col="entry_timestamp",
        value_columns=entry_values,
        suffix="entry",
    )
    sampled["entry_source_timestamp"] = pd.to_datetime(
        entry["source_timestamp_entry"], errors="coerce"
    )
    sampled["entry_state_age_seconds"] = pd.to_numeric(
        entry["state_age_seconds_entry"], errors="coerce"
    )
    sampled["entry_delay_seconds"] = np.where(
        sampled["entry_source_timestamp"].notna(), float(entry_clock_delay_seconds), np.nan
    )
    sampled["entry_delay_ticks"] = np.where(
        sampled["entry_source_timestamp"].notna(), float(entry_tick_delay_audit), np.nan
    )
    sampled["entry_max_tick_gap_seconds"] = np.nan
    sampled["buy_price"] = pd.to_numeric(entry[f"{buy_price_col}_entry"], errors="coerce")
    for level in PRICE_LEVELS:
        for source_col in (ask_price_col(level), ask_volume_col(level)):
            aligned_col = f"{source_col}_entry"
            if aligned_col in entry.columns:
                sampled[f"entry_{source_col}"] = pd.to_numeric(entry[aligned_col], errors="coerce")
    if "status_entry" in entry.columns:
        sampled["entry_status"] = entry["status_entry"]

    if require_cross_section_ready_entry:
        sampled = require_entry_after_cross_section_ready(sampled)
    else:
        sampled["cross_section_ready_timestamp"] = pd.NaT
        sampled["entry_after_cross_section_ready"] = True

    allowed = {str(value).upper() for value in tradable_statuses or ()}
    decision_status_valid = pd.Series(True, index=sampled.index)
    entry_status_valid = pd.Series(True, index=sampled.index)
    if allowed and "status" in sampled.columns:
        decision_status_valid = _status_allowed(sampled["status"], allowed)
    if allowed and "entry_status" in sampled.columns:
        entry_status_valid = _status_allowed(sampled["entry_status"], allowed)
    sampled["valid_entry"] = (
        sampled["entry_source_timestamp"].notna()
        & sampled["buy_price"].gt(0)
        & sampled["entry_after_cross_section_ready"].fillna(False)
        & decision_status_valid
        & entry_status_valid
    )

    for spec in specs:
        assert spec.seconds is not None
        name = spec.name
        start_target_col = f"sell_start_target_timestamp_{name}"
        end_target_col = f"sell_end_target_timestamp_{name}"
        sampled[start_target_col] = shift_series_by_trading_seconds(
            sampled["entry_timestamp"], int(spec.seconds), sessions=parsed_sessions
        )
        sampled[end_target_col] = shift_series_by_trading_seconds(
            sampled[start_target_col],
            int(sell_window_trading_seconds),
            sessions=parsed_sessions,
        )
        start = _align_clock_state(
            features,
            sampled,
            target_timestamp_col=start_target_col,
            value_columns=[volume_col, turnover_col],
            suffix=f"sell_start_{name}",
        )
        end = _align_clock_state(
            features,
            sampled,
            target_timestamp_col=end_target_col,
            value_columns=[volume_col, turnover_col],
            suffix=f"sell_end_{name}",
        )
        for side, aligned in (("sell_start", start), ("sell_end", end)):
            suffix = f"{side}_{name}"
            sampled[f"{side}_source_timestamp_{name}"] = pd.to_datetime(
                aligned[f"source_timestamp_{suffix}"], errors="coerce"
            )
            sampled[f"{side}_state_age_seconds_{name}"] = pd.to_numeric(
                aligned[f"state_age_seconds_{suffix}"], errors="coerce"
            )
            sampled[f"{volume_col}_{side}_{name}"] = pd.to_numeric(
                aligned[f"{volume_col}_{suffix}"], errors="coerce"
            )
            sampled[f"{turnover_col}_{side}_{name}"] = pd.to_numeric(
                aligned[f"{turnover_col}_{suffix}"], errors="coerce"
            )

        sell_volume = (
            sampled[f"{volume_col}_sell_end_{name}"] - sampled[f"{volume_col}_sell_start_{name}"]
        )
        sell_turnover = (
            sampled[f"{turnover_col}_sell_end_{name}"]
            - sampled[f"{turnover_col}_sell_start_{name}"]
        )
        sampled[f"sell_volume_{name}"] = sell_volume
        sampled[f"sell_turnover_{name}"] = sell_turnover
        denominator = sell_volume * float(volume_unit_multiplier)
        sampled[f"sell_vwap_{name}"] = np.where(
            denominator > 0, sell_turnover / denominator, np.nan
        )
        label_col = label_column_name(name)
        sampled[f"gross_{label_col}"] = safe_price_return(
            sampled[f"sell_vwap_{name}"], sampled["buy_price"]
        )
        sampled[label_col] = sampled[f"gross_{label_col}"] - float(fee_bps) / 10_000.0
        sampled[f"valid_{label_col}"] = (
            sampled[label_col].notna()
            & np.isfinite(sampled[label_col])
            & sell_volume.gt(0)
            & sell_turnover.gt(0)
            & sampled["valid_entry"]
            & sampled[start_target_col].notna()
            & sampled[end_target_col].notna()
        )

    return sampled.sort_values(["date", "decision_target_timestamp", "symbol"]).reset_index(
        drop=True
    )
