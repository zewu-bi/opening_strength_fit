from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.feature_utils import _numeric_series, safe_divide
from opening_strength_fit.schema import ensure_timestamp_columns, frame_clock_series


def _rolling_linear_slope(values: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    values = _numeric_series(values)
    rolling = values.rolling(window=window, min_periods=window)
    rolling_sum = rolling.sum()
    valid_window = rolling.count().eq(float(window))

    weighted_sum = pd.Series(0.0, index=values.index, dtype="float64")
    for offset in range(window - 1):
        weight = float(window - 1 - offset)
        weighted_sum = weighted_sum + values.shift(offset) * weight

    x_mean = float(window - 1) / 2.0
    denom = float(window * (window**2 - 1)) / 12.0
    slope = (weighted_sum - x_mean * rolling_sum) / denom
    return slope.where(valid_window)


def _grouped_rolling_linear_slope(
    values: pd.Series,
    group_keys: list[pd.Series],
    window: int,
) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    values = _numeric_series(values)
    grouped = values.groupby(group_keys, sort=False)
    rolling = grouped.rolling(window=window, min_periods=window)
    group_levels = list(range(len(group_keys)))
    rolling_sum = rolling.sum().reset_index(level=group_levels, drop=True)
    valid_window = rolling.count().reset_index(level=group_levels, drop=True).eq(float(window))

    weighted_sum = pd.Series(0.0, index=values.index, dtype="float64")
    for offset in range(window - 1):
        weight = float(window - 1 - offset)
        shifted = values if offset == 0 else grouped.shift(offset)
        weighted_sum = weighted_sum + shifted * weight

    x_mean = float(window - 1) / 2.0
    denom = float(window * (window**2 - 1)) / 12.0
    slope = (weighted_sum - x_mean * rolling_sum) / denom
    return slope.where(valid_window)


def _normalized_row_clock(frame: pd.DataFrame, time_col: str) -> pd.Series:
    return frame_clock_series(frame, timestamp_cols=(time_col,))


def add_path_shape_confirmation_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (2, 3, 5),
    prefix: str = "path_shape_",
    reference_name: str = "open",
) -> pd.DataFrame:
    """Add causal post-open path shape and confirmation features."""

    reference_name = reference_name.strip().lower()
    if reference_name not in {"open", "start"}:
        raise ValueError("path-shape reference_name must be 'open' or 'start'")
    out = ensure_timestamp_columns(frame)
    time_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in out.columns else "timestamp"
    )
    out = out.sort_values(["date", "symbol", time_col]).reset_index(drop=True)
    group_keys = [out["date"], out["symbol"]]
    group_cols = ["date", "symbol"]
    new_columns: dict[str, pd.Series] = {}

    if "mid_price" in out.columns:
        mid = _numeric_series(out["mid_price"])
        per_symbol = mid.groupby(group_keys, sort=False)
        expanding_high = per_symbol.cummax()
        expanding_low = per_symbol.cummin()
        first_mid = per_symbol.transform("first")
        mid_move_1 = mid - per_symbol.shift(1)
        mid_move_1_bps = pd.Series(
            safe_divide(mid_move_1, per_symbol.shift(1)) * 10_000,
            index=out.index,
        )
        positive_move = mid_move_1_bps.gt(0.0).astype("float64")

        new_columns[f"{prefix}mid_drawdown_from_{reference_name}_high_bps"] = (
            safe_divide(mid - expanding_high, expanding_high) * 10_000
        )
        new_columns[f"{prefix}mid_recovery_from_{reference_name}_low_bps"] = (
            safe_divide(mid - expanding_low, expanding_low) * 10_000
        )
        first_reference_column = (
            f"{prefix}mid_from_first_bps"
            if reference_name == "open"
            else f"{prefix}mid_from_start_bps"
        )
        new_columns[first_reference_column] = safe_divide(mid - first_mid, first_mid) * 10_000
        new_columns[f"{prefix}mid_new_high_flag"] = mid.ge(expanding_high).astype("int8")
        new_columns[f"{prefix}mid_new_low_flag"] = mid.le(expanding_low).astype("int8")
        new_columns[f"{prefix}return_positive_fraction"] = (
            positive_move.groupby(group_keys, sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        for window in windows:
            if window < 2:
                continue
            prior = per_symbol.shift(window - 1)
            move_bps = pd.Series(
                safe_divide(mid - prior, prior) * 10_000,
                index=out.index,
            )
            new_columns[f"{prefix}mid_move_{window}m_bps"] = move_bps
            new_columns[f"{prefix}return_accel_1m_vs_{window}m"] = mid_move_1_bps - safe_divide(
                move_bps, float(window - 1)
            )
            roll_high = (
                mid.groupby(group_keys, sort=False)
                .rolling(window=window, min_periods=window)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            roll_low = (
                mid.groupby(group_keys, sort=False)
                .rolling(window=window, min_periods=window)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )
            new_columns[f"{prefix}mid_position_roll{window}"] = safe_divide(
                mid - roll_low,
                roll_high - roll_low,
            )

        if "spread_bps" in out.columns:
            spread = _numeric_series(out["spread_bps"])
            spread_change = spread - spread.groupby(group_keys, sort=False).shift(1)
            new_columns[f"{prefix}spread_compress_after_upmove"] = -spread_change * positive_move
            new_columns[f"{prefix}spread_widen_after_upmove"] = spread_change * positive_move

        if "depth_imbalance_10" in out.columns:
            imbalance = _numeric_series(out["depth_imbalance_10"])
            for window in windows:
                if window < 2:
                    continue
                slope = _grouped_rolling_linear_slope(imbalance, group_keys, window)
                new_columns[f"{prefix}imbalance_slope_roll{window}"] = slope
                new_columns[f"{prefix}imbalance_slope_confirm_return_roll{window}"] = (
                    slope * move_positive_over_window(out, group_cols, mid, window)
                )

        if {"bid_depth_10", "ask_depth_10"}.issubset(out.columns):
            bid_depth = _numeric_series(out["bid_depth_10"])
            ask_depth = _numeric_series(out["ask_depth_10"])
            bid_change = bid_depth - bid_depth.groupby(group_keys, sort=False).shift(1)
            ask_change = ask_depth - ask_depth.groupby(group_keys, sort=False).shift(1)
            new_columns[f"{prefix}bid_depth_support_after_upmove"] = bid_change * positive_move
            new_columns[f"{prefix}ask_depth_fade_after_upmove"] = (-ask_change) * positive_move
            new_columns[f"{prefix}depth_support_imbalance_after_upmove"] = (
                bid_change - ask_change
            ) * positive_move

    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)


def move_positive_over_window(
    frame: pd.DataFrame,
    group_cols: list[str],
    values: pd.Series,
    window: int,
) -> pd.Series:
    group_keys = [frame[column] for column in group_cols]
    prior = values.groupby(group_keys, sort=False).shift(window - 1)
    return (values - prior).gt(0.0).astype("float64")


def add_historical_same_minute_surprise_features(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    windows: tuple[int, ...] = (20, 60),
    min_periods: int = 10,
    modes: tuple[str, ...] = ("zscore", "ratio"),
    prefix: str = "hist_surprise_",
) -> pd.DataFrame:
    """Add symbol x decision-clock historical surprise features using prior dates only."""

    out = ensure_timestamp_columns(frame)
    if "symbol" not in out.columns or "date" not in out.columns:
        raise ValueError("historical surprise features require date and symbol columns")
    time_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in out.columns else "timestamp"
    )
    clock = _normalized_row_clock(out, time_col)
    work = out.assign(_hist_clock=clock, _hist_orig_order=np.arange(len(out)))
    work = work.sort_values(["symbol", "_hist_clock", "date", time_col]).reset_index(drop=True)
    group_keys = [work["symbol"], work["_hist_clock"]]
    modes = tuple(mode.strip().lower() for mode in modes if mode.strip())
    valid_modes = {"zscore", "ratio", "diff"}
    unknown_modes = sorted(set(modes) - valid_modes)
    if unknown_modes:
        raise ValueError(f"unknown historical surprise modes: {unknown_modes}")

    selected = [
        column
        for column in columns
        if column in work.columns and pd.api.types.is_numeric_dtype(work[column])
    ]
    if not selected:
        return out.copy()

    min_periods = max(2, int(min_periods))
    new_sorted: dict[str, pd.Series] = {}
    for column in selected:
        values = pd.to_numeric(work[column], errors="coerce").astype("float64")
        past = values.groupby(group_keys, sort=False).shift(1)
        for window in windows:
            if window < 2:
                continue
            rolling = past.groupby(group_keys, sort=False).rolling(
                window=int(window),
                min_periods=min_periods,
            )
            mean = rolling.mean().reset_index(level=[0, 1], drop=True)
            std = rolling.std().reset_index(level=[0, 1], drop=True)
            if "zscore" in modes:
                new_sorted[f"{prefix}{column}_{window}d_zscore"] = pd.Series(
                    safe_divide(values - mean, std),
                    index=work.index,
                )
            if "ratio" in modes:
                new_sorted[f"{prefix}{column}_{window}d_ratio"] = pd.Series(
                    safe_divide(values, mean),
                    index=work.index,
                )
            if "diff" in modes:
                new_sorted[f"{prefix}{column}_{window}d_diff"] = values - mean

    if not new_sorted:
        return out.copy()
    features = pd.DataFrame(new_sorted, index=work.index)
    features["_hist_orig_order"] = work["_hist_orig_order"].to_numpy()
    features = features.sort_values("_hist_orig_order").drop(columns="_hist_orig_order")
    features.index = out.index
    return pd.concat([out.copy(), features], axis=1)


def add_historical_daily_activity_reference_features(
    frame: pd.DataFrame,
    *,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    windows: tuple[int, ...] = (60,),
    min_periods: int = 10,
    prefix: str = "hist_avg_daily_",
) -> pd.DataFrame:
    """Add prior-date rolling daily activity averages for dimensionless denominators.

    The daily activity available in the labeled cache is the max value observed
    within the cached decision rows for a symbol-date. Rolling averages are
    shifted by one date, so the current date never enters its own denominator.
    """

    out = frame.copy(deep=False)
    if "symbol" not in out.columns or "date" not in out.columns:
        raise ValueError("historical daily activity references require date and symbol columns")

    source_columns = list(
        dict.fromkeys(
            column
            for column in (volume_col, turnover_col)
            if column in out.columns and pd.api.types.is_numeric_dtype(out[column])
        )
    )
    if not source_columns:
        return out

    min_periods = max(1, int(min_periods))
    windows = tuple(int(window) for window in windows if int(window) >= 2)
    if not windows:
        return out

    hist_date = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    valid_key = out["symbol"].notna() & hist_date.notna()
    if not valid_key.any():
        return out

    work = pd.DataFrame(
        {
            "symbol": out.loc[valid_key, "symbol"].to_numpy(copy=False),
            "_hist_date": hist_date.loc[valid_key].to_numpy(copy=False),
        }
    )
    for column in source_columns:
        work[column] = pd.to_numeric(out.loc[valid_key, column], errors="coerce").to_numpy(
            copy=False
        )

    daily = (
        work.groupby(["symbol", "_hist_date"], sort=False, observed=True)[source_columns]
        .max()
        .sort_index()
    )
    del work
    if daily.empty:
        return out

    symbol_index = daily.index.levels[0]
    row_symbol_codes = symbol_index.get_indexer(out["symbol"].to_numpy(copy=False))
    row_days = hist_date.to_numpy(dtype="datetime64[D]").astype("int64", copy=False)
    daily_days = (
        daily.index.get_level_values("_hist_date")
        .to_numpy(
            dtype="datetime64[D]",
        )
        .astype("int64", copy=False)
    )
    valid_daily_days = daily_days[daily_days != np.datetime64("NaT", "D").astype("int64")]
    if len(valid_daily_days) == 0:
        return out
    min_day = int(valid_daily_days.min())
    day_span = int(valid_daily_days.max() - min_day + 1)
    daily_symbol_codes = daily.index.codes[0].astype("int64", copy=False)
    daily_keys = daily_symbol_codes * day_span + (daily_days - min_day)
    valid_rows = (
        (row_symbol_codes >= 0)
        & (row_days != np.datetime64("NaT", "D").astype("int64"))
        & (row_days >= min_day)
        & (row_days < min_day + day_span)
    )
    row_keys = row_symbol_codes[valid_rows].astype("int64", copy=False) * day_span + (
        row_days[valid_rows] - min_day
    )
    row_positions = np.flatnonzero(valid_rows)
    for column in source_columns:
        past = daily[column].groupby(level=0, sort=False).shift(1)
        for window in windows:
            averaged = (
                past.groupby(level=0, sort=False)
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )
            ref_col = f"{prefix}{column}_{window}d"
            values = np.full(len(out), np.nan, dtype="float32")
            positions = np.searchsorted(daily_keys, row_keys, side="left")
            in_bounds = positions < len(daily_keys)
            matched = in_bounds.copy()
            matched[in_bounds] = daily_keys[positions[in_bounds]] == row_keys[in_bounds]
            averaged_values = averaged.to_numpy(dtype="float32", copy=False)
            values[row_positions[matched]] = averaged_values[positions[matched]]
            out[ref_col] = values

    return out
