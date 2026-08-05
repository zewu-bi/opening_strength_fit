from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import TICK_FIELD_DESC, normalize_clickhouse_ticks
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_str,
)
from opening_strength_fit.feature_utils import safe_divide
from opening_strength_fit.features import build_feature_frame, build_preopen_features
from opening_strength_fit.io import read_frame
from opening_strength_fit.sampling import sample_labeled_frame
from opening_strength_fit.training_labeled import _apply_feature_transforms_from_config

RAW_FEATURE_TICK_COLUMNS = tuple(column for column in TICK_FIELD_DESC if column != "LocalTimeStamp")


def decode_clickhouse_text(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value.decode("utf-8") if isinstance(value, bytes | bytearray) else str(value)
    )


def normalize_clickhouse_date(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _positive_scaled(values: pd.Series, multiplier: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    return numeric.where(np.isfinite(numeric) & numeric.gt(0.0)) * float(multiplier)


def _daily_reference_for_day(
    raw_year_root: Path,
    trading_day: str,
    cache: dict[Path, pd.DataFrame],
    *,
    market_cap_multiplier: float,
    share_multiplier: float,
) -> pd.DataFrame:
    daily_path = raw_year_root / "daily_reference.parquet"
    if daily_path not in cache:
        daily = read_frame(daily_path)
        daily["TradingDay"] = normalize_clickhouse_date(daily["TradingDay"])
        daily["Symbol"] = decode_clickhouse_text(daily["Symbol"])
        cache[daily_path] = daily
    daily = cache[daily_path]
    calendar = sorted(daily["TradingDay"].dropna().unique())
    previous = [date for date in calendar if date < trading_day]
    if not previous:
        raise SystemExit(f"no lagged daily reference before {trading_day}: {daily_path}")
    reference_day = previous[-1]
    lagged = daily.loc[daily["TradingDay"].eq(reference_day)].copy()
    if lagged.empty:
        raise SystemExit(f"incomplete daily reference for {trading_day}: {daily_path}")

    lagged = lagged[
        [
            "Symbol",
            "TotalMarketValue",
            "TotalFloatMarketValue",
            "TotalShareToday",
            "FloatAShare",
            "FreeShareToday",
        ]
    ].rename(columns={"Symbol": "symbol"})
    lagged["total_market_cap"] = _positive_scaled(
        lagged.pop("TotalMarketValue"), market_cap_multiplier
    )
    lagged["float_market_cap"] = _positive_scaled(
        lagged.pop("TotalFloatMarketValue"), market_cap_multiplier
    )
    lagged["total_shares"] = _positive_scaled(lagged.pop("TotalShareToday"), share_multiplier)
    lagged["float_shares"] = _positive_scaled(lagged.pop("FloatAShare"), share_multiplier)
    lagged["free_float_shares"] = _positive_scaled(lagged.pop("FreeShareToday"), share_multiplier)
    lagged["market_cap_reference_date"] = pd.Timestamp(reference_day)
    lagged["market_cap_reference_lag_sessions"] = 1
    lagged["date"] = trading_day
    return lagged


def _attach_daily_reference(ticks: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    overlap = sorted((set(reference.columns) - {"date", "symbol"}) & set(ticks.columns))
    if overlap:
        raise SystemExit(f"daily reference would overwrite raw tick columns: {overlap}")
    return ticks.merge(reference, on=["date", "symbol"], how="left", validate="many_to_one")


def _attach_preopen_features(
    core: pd.DataFrame,
    all_ticks: pd.DataFrame,
    feature_config: dict,
) -> pd.DataFrame:
    if not config_bool(feature_config, "features", "include_preopen", True):
        return core
    preopen_state = build_feature_frame(
        all_ticks.loc[all_ticks["time"].le("09:25:30")].copy(),
        include_preopen=False,
        volume_col=config_str(feature_config, "labels", "volume_col", "volume"),
        turnover_col=config_str(feature_config, "labels", "turnover_col", "turnover"),
        volume_unit_multiplier=config_float(
            feature_config, "labels", "volume_unit_multiplier", 1.0
        ),
    )
    preopen = build_preopen_features(
        preopen_state,
        volume_col=config_str(feature_config, "labels", "volume_col", "volume"),
        turnover_col=config_str(feature_config, "labels", "turnover_col", "turnover"),
        price_mode=config_str(
            feature_config, "features", "preopen_price_mode", "legacy_last_price"
        ),
        match_time=config_str(feature_config, "features", "preopen_match_time", "09:25:00"),
    )
    out = core.merge(preopen, on=["date", "symbol"], how="left", validate="many_to_one")
    if "prev_close" in out and "preopen_last_price" in out:
        out["preopen_return_vs_prev_close"] = safe_divide(
            out["preopen_last_price"] - out["prev_close"], out["prev_close"]
        )
    required = {"preopen_price_min", "preopen_price_max", "preopen_last_price"}
    if required.issubset(out.columns):
        price_range = out["preopen_price_max"] - out["preopen_price_min"]
        reference_price = out["prev_close"] if "prev_close" in out else out["preopen_last_price"]
        out["auction_price_range_bps"] = safe_divide(price_range, reference_price) * 10_000
        out["auction_last_position_in_range"] = safe_divide(
            out["preopen_last_price"] - out["preopen_price_min"], price_range
        )
    return out


def build_raw_feature_day(
    raw_path: Path,
    trading_day: str,
    feature_config: dict,
    dataset_config: dict,
    daily_cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    """Compress one raw tick day into transformed context decision rows."""

    raw = read_frame(raw_path, columns=list(RAW_FEATURE_TICK_COLUMNS))
    raw["TradingDay"] = trading_day
    raw["Symbol"] = decode_clickhouse_text(raw["Symbol"])
    raw["Status"] = decode_clickhouse_text(raw["Status"])
    ticks = normalize_clickhouse_ticks(raw)
    reference = _daily_reference_for_day(
        raw_path.parents[1],
        trading_day,
        daily_cache,
        market_cap_multiplier=config_float(
            dataset_config, "dataset", "market_cap_unit_multiplier", 10_000.0
        ),
        share_multiplier=config_float(dataset_config, "dataset", "share_unit_multiplier", 10_000.0),
    )
    ticks = _attach_daily_reference(ticks, reference)
    feature_start = config_int(dataset_config, "dataset", "feature_tick_start_offset_us", 0)
    core_ticks = ticks.loc[ticks["exch_time_offset_us"].ge(feature_start)].copy()
    core = build_feature_frame(
        core_ticks,
        include_preopen=False,
        volume_col=config_str(feature_config, "labels", "volume_col", "volume"),
        turnover_col=config_str(feature_config, "labels", "turnover_col", "turnover"),
        volume_unit_multiplier=config_float(
            feature_config, "labels", "volume_unit_multiplier", 1.0
        ),
    )
    core = _attach_preopen_features(core, ticks, feature_config)
    sampled = sample_labeled_frame(
        core,
        mode="decision_points",
        decision_times=tuple(config_list(dataset_config, "dataset", "context_decision_times", [])),
        max_lag_seconds=None,
        alignment="clock_state",
        max_state_age_seconds=None,
    )
    return _apply_feature_transforms_from_config(
        sampled,
        feature_config,
        include_cross_sectional_relative=False,
        drop_features=False,
    )


__all__ = [
    "build_raw_feature_day",
    "decode_clickhouse_text",
    "normalize_clickhouse_date",
]
