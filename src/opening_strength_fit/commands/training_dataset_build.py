from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json
from opening_strength_fit.model_features import feature_columns
from opening_strength_fit.model_preprocessing import lightgbm_feature_value_frame
from opening_strength_fit.training_dataset_features import (
    build_raw_feature_day,
    decode_clickhouse_text,
    normalize_clickhouse_date,
)
from opening_strength_fit.training_labeled import (
    _apply_cross_sectional_relative_from_config,
    _apply_post_sample_feature_transforms_from_config,
    _drop_features_from_config,
    apply_candidate_filter_from_config,
)

KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
LABEL_COLUMNS = (
    "label_short_1m",
    "label_short_3m",
    "label_short_5m",
    "label_next_close",
    "label_mixed",
)
VALID_LABEL_COLUMNS = (
    "valid_short_1m",
    "valid_short_3m",
    "valid_short_5m",
    "valid_next_close",
    "valid_mixed",
)
RAW_LABEL_TICK_COLUMNS = (
    "Symbol",
    "ExchTimeOffsetUs",
    "Volume",
    "Turnover",
    "AskPrice1",
    "Status",
)


def _config_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _years(config: dict) -> tuple[int, ...]:
    years = tuple(int(value) for value in config_list(config, "dataset", "years", []))
    if not years:
        raise SystemExit("[dataset].years must not be empty")
    return years


def _raw_root(config: dict) -> Path:
    value = config_str(config, "dataset", "raw_source_root", "")
    if not value:
        raise SystemExit("missing [dataset].raw_source_root")
    return Path(value)


def _raw_year_root(config: dict, year: int) -> Path:
    return _raw_root(config) / f"year={year}"


def _output_root(config: dict, kind: str, override: str) -> Path:
    if override:
        return Path(override)
    key = "feature_output_root" if kind == "features" else "label_output_root"
    value = config_str(config, "dataset", key, "")
    if not value:
        raise SystemExit(f"missing [dataset].{key}")
    return Path(value)


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
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


def _filter_decision_clocks(frame: pd.DataFrame, clocks: tuple[str, ...]) -> pd.DataFrame:
    out = _normalize_keys(frame)
    observed = out["decision_target_timestamp"].dt.strftime("%H:%M:%S")
    return out.loc[observed.isin(set(clocks))].copy()


def _date_bounds(year: int, start: str, end: str) -> tuple[str, str]:
    resolved_start = start or f"{year}-01-01"
    resolved_end = end or f"{year}-12-31"
    if pd.Timestamp(resolved_start).year != year or pd.Timestamp(resolved_end).year != year:
        raise SystemExit("date overrides must stay inside the selected year")
    if resolved_end < resolved_start:
        raise SystemExit("date end must be >= date start")
    return resolved_start, resolved_end


def _raw_tick_dates(config: dict, *, start_date: str, end_date: str) -> list[tuple[str, Path]]:
    dates: list[tuple[str, Path]] = []
    for year in _years(config):
        year_root = _raw_year_root(config, year)
        if not (year_root / "_SUCCESS").exists():
            continue
        for path in sorted((year_root / "ticks").glob("date=*.parquet")):
            trading_day = path.stem.removeprefix("date=")
            if start_date <= trading_day <= end_date:
                dates.append((trading_day, path))
    return sorted(dates)


def _validate_output_keys(frame: pd.DataFrame, clocks: tuple[str, ...]) -> None:
    duplicate = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(f"output has {int(duplicate.sum())} duplicate key rows")
    observed = set(frame["decision_target_timestamp"].dt.strftime("%H:%M:%S").unique())
    missing_clocks = sorted(set(clocks).difference(observed))
    if missing_clocks:
        raise SystemExit(f"output is missing decision clocks: {missing_clocks}")


def build_feature_dataset(
    config: dict,
    config_path: Path,
    *,
    year: int,
    output_root: Path,
    start_date: str,
    end_date: str,
    context_days: int,
    overwrite: bool,
) -> dict[str, object]:
    canonical_path = Path(config_str(config, "dataset", "canonical_feature_config", ""))
    if not canonical_path.exists():
        raise SystemExit(f"canonical feature config does not exist: {canonical_path}")
    feature_config = load_toml(canonical_path)
    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    expected_features = config_int(config, "dataset", "expected_feature_count", 350)
    year_root = output_root / f"year={year}"
    output_path = year_root / "features.parquet"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"feature output exists, pass --overwrite: {output_path}")
    success_path = year_root / "_SUCCESS"
    if overwrite and success_path.exists():
        success_path.unlink()

    context_start = str((pd.Timestamp(start_date) - pd.Timedelta(days=int(context_days))).date())
    raw_dates = _raw_tick_dates(config, start_date=context_start, end_date=end_date)
    if not raw_dates:
        raise SystemExit(f"no raw tick days in {context_start}..{end_date}")
    daily_cache: dict[Path, pd.DataFrame] = {}
    parts = []
    for index, (trading_day, raw_path) in enumerate(raw_dates, start=1):
        part = build_raw_feature_day(
            raw_path,
            trading_day,
            feature_config,
            config,
            daily_cache,
        )
        if not part.empty:
            floats = part.select_dtypes(include=["float64"]).columns
            if len(floats):
                part[floats] = part[floats].astype("float32")
            parts.append(part)
        print(
            f"features year={year} day={index}/{len(raw_dates)} date={trading_day} "
            f"rows={len(part)}",
            flush=True,
        )
    if not parts:
        raise SystemExit("raw source produced no sampled feature rows")
    source = pd.concat(parts, ignore_index=True)
    transformed = _apply_post_sample_feature_transforms_from_config(source, feature_config)
    if config_bool(feature_config, "features", "include_cross_sectional_relative", False):
        transformed = _apply_cross_sectional_relative_from_config(transformed, feature_config)
    transformed = _drop_features_from_config(transformed, feature_config)
    transformed = apply_candidate_filter_from_config(transformed, feature_config)
    transformed = _normalize_keys(transformed)
    target_dates = transformed["date"].between(start_date, end_date)
    transformed = transformed.loc[target_dates].copy()
    transformed = _filter_decision_clocks(transformed, clocks)
    filters = feature_filters_from_config(feature_config)
    selected = feature_columns(transformed, None, **filters)
    if len(selected) != expected_features:
        raise SystemExit(
            f"canonical feature count mismatch: expected {expected_features}, got {len(selected)}"
        )
    value_transform = config_str(feature_config, "features", "feature_value_transform", "none")
    output, output_features = lightgbm_feature_value_frame(
        transformed,
        selected,
        feature_value_transform=value_transform,
        feature_value_transform_output="replace",
        group_cols=tuple(
            config_list(
                feature_config,
                "features",
                "feature_value_transform_group_cols",
                ["date", "decision_target_timestamp"],
            )
        ),
        rank_method=config_str(
            feature_config, "features", "feature_value_transform_rank_method", "average"
        ),
        tick_size=config_float(
            feature_config, "features", "feature_value_transform_tick_size", 0.01
        ),
        extra_columns=KEY_COLUMNS,
    )
    if output_features != selected:
        raise SystemExit("feature value transform changed the canonical feature names")
    output = output[[*KEY_COLUMNS, *selected]].copy()
    _validate_output_keys(output, clocks)
    float_columns = output.select_dtypes(include=["float64"]).columns
    if len(float_columns):
        output[float_columns] = output[float_columns].astype("float32")
    write_frame_atomic(output, output_path)
    parquet = pq.ParquetFile(output_path)
    manifest = {
        "schema_version": "opening_features_350_v1",
        "run_id": run_id(config, config_path),
        "kind": "features",
        "year": year,
        "date_start": start_date,
        "date_end": end_date,
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "feature_count": len(selected),
        "key_columns": list(KEY_COLUMNS),
        "feature_columns": selected,
        "decision_times": list(clocks),
        "source_raw_root": str(_raw_root(config)),
        "source_tick_files": len(raw_dates),
        "source_rows_with_context": int(len(source)),
        "context_days": context_days,
        "canonical_feature_config": str(canonical_path),
        "canonical_feature_config_fingerprint": _config_fingerprint(canonical_path),
        "feature_values": "model_ready",
        "feature_value_transform": value_transform,
        "training_feature_value_transform": "none",
        "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        "parquet_rows": int(parquet.metadata.num_rows),
        "contains_labels": False,
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    print(
        f"features complete year={year} rows={len(output)} features={len(selected)} "
        f"output={output_path}",
        flush=True,
    )
    return manifest


def _entry_offset_us(timestamp: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamp, errors="coerce")
    offset = (values - values.dt.normalize()) / pd.Timedelta(microseconds=1)
    return offset.to_numpy(dtype="float64")


def _clock_offset_us(clock: str) -> int:
    timestamp = pd.Timestamp(f"2000-01-01 {clock}")
    return int((timestamp - timestamp.normalize()) / pd.Timedelta(microseconds=1))


def _build_label_base(
    raw_ticks: pd.DataFrame,
    *,
    trading_day: str,
    decision_times: tuple[str, ...],
    feature_tick_start_offset_us: int,
    entry_delay_seconds: int,
) -> pd.DataFrame:
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


def compute_short_label_set(
    base: pd.DataFrame,
    raw_ticks: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    sell_window_seconds: int,
    volume_unit_multiplier: float,
    fee_bps: float,
    tradable_statuses: tuple[str, ...],
) -> pd.DataFrame:
    """Compute several clock-state VWAP labels with one pass over projected raw ticks."""

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


def _next_close_label(base: pd.DataFrame, raw_year_root: Path) -> pd.DataFrame:
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


def _mixed_label(
    frame: pd.DataFrame,
    *,
    weight: float,
    min_group_size: int,
) -> tuple[pd.Series, pd.Series]:
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


def build_label_dataset(
    config: dict,
    config_path: Path,
    *,
    year: int,
    output_root: Path,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> dict[str, object]:
    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    raw_year_root = _raw_year_root(config, year)
    manifest_path = raw_year_root / "manifest.json"
    if not (raw_year_root / "_SUCCESS").exists() or not manifest_path.exists():
        raise SystemExit(f"raw source year is incomplete: {raw_year_root}")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = raw_manifest.get("label_coverage", {})
    if coverage.get("short_label_horizons_seconds") != [60, 180, 300]:
        raise SystemExit(f"raw source does not declare 1m/3m/5m coverage: {manifest_path}")

    raw_dates = _raw_tick_dates(config, start_date=start_date, end_date=end_date)
    raw_dates = [item for item in raw_dates if pd.Timestamp(item[0]).year == year]
    if not raw_dates:
        raise SystemExit(f"no raw tick days in {start_date}..{end_date}")
    parts = []
    for index, (trading_day, tick_path) in enumerate(raw_dates, start=1):
        ticks = read_frame(tick_path, columns=list(RAW_LABEL_TICK_COLUMNS))
        day_base = _build_label_base(
            ticks,
            trading_day=trading_day,
            decision_times=clocks,
            feature_tick_start_offset_us=config_int(
                config, "dataset", "feature_tick_start_offset_us", 0
            ),
            entry_delay_seconds=config_int(config, "dataset", "entry_delay_seconds", 6),
        )
        labels = compute_short_label_set(
            day_base,
            ticks,
            horizons=(60, 180, 300),
            sell_window_seconds=config_int(config, "dataset", "sell_window_seconds", 60),
            volume_unit_multiplier=config_float(config, "dataset", "volume_unit_multiplier", 1.0),
            fee_bps=config_float(config, "dataset", "fee_bps", 0.0),
            tradable_statuses=tuple(
                config_list(config, "dataset", "tradable_statuses", ["T0", "20", "TRADE"])
            ),
        )
        parts.append(day_base.merge(labels, on=list(KEY_COLUMNS), validate="one_to_one"))
        print(
            f"labels year={year} day={index}/{len(raw_dates)} date={trading_day} "
            f"rows={len(labels)}",
            flush=True,
        )
    if not parts:
        raise SystemExit("raw source produced no label rows")
    base_and_short = pd.concat(parts, ignore_index=True)
    base = base_and_short[
        [
            *KEY_COLUMNS,
            "entry_timestamp",
            "buy_price",
            "status",
            "entry_status",
            "entry_after_cross_section_ready",
        ]
    ].copy()
    _validate_output_keys(base, clocks)
    output = base_and_short[[*KEY_COLUMNS, *LABEL_COLUMNS[:3], *VALID_LABEL_COLUMNS[:3]]]
    next_close = _next_close_label(base, raw_year_root)
    output = output.merge(next_close, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    output["valid_next_close"] = output["valid_next_close"].fillna(False).astype(bool)
    mixed, mixed_valid = _mixed_label(
        output,
        weight=config_float(config, "dataset", "mixed_next_close_weight", 0.30),
        min_group_size=config_int(config, "dataset", "mixed_min_group_size", 50),
    )
    output["label_mixed"] = mixed
    output["valid_mixed"] = mixed_valid
    output = output[[*KEY_COLUMNS, *LABEL_COLUMNS, *VALID_LABEL_COLUMNS]]
    _validate_output_keys(output, clocks)
    output[list(LABEL_COLUMNS)] = output[list(LABEL_COLUMNS)].astype("float32")

    year_root = output_root / f"year={year}"
    output_path = year_root / "labels.parquet"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"label output exists, pass --overwrite: {output_path}")
    success_path = year_root / "_SUCCESS"
    if overwrite and success_path.exists():
        success_path.unlink()
    write_frame_atomic(output, output_path)
    validity = {
        column: {
            "valid_rows": int(output[valid_column].sum()),
            "valid_ratio": float(output[valid_column].mean()),
        }
        for column, valid_column in zip(LABEL_COLUMNS, VALID_LABEL_COLUMNS, strict=True)
    }
    manifest = {
        "schema_version": "opening_labels_5_v1",
        "run_id": run_id(config, config_path),
        "kind": "labels",
        "year": year,
        "date_start": start_date,
        "date_end": end_date,
        "rows": int(len(output)),
        "columns": list(output.columns),
        "key_columns": list(KEY_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "valid_columns": list(VALID_LABEL_COLUMNS),
        "decision_times": list(clocks),
        "source_raw_root": str(raw_year_root),
        "definitions": {
            "entry": "decision_target_timestamp+6s clock state rebuilt from raw PVC ticks",
            "short_horizons_seconds": [60, 180, 300],
            "sell_window_seconds": config_int(config, "dataset", "sell_window_seconds", 60),
            "short_exit": "cumulative turnover delta / cumulative volume delta",
            "next_close": "next trading session close reference / buy_price - 1",
            "mixed": (
                "xs_zscore(short_1m) + "
                f"{config_float(config, 'dataset', 'mixed_next_close_weight', 0.30):g} "
                "* xs_zscore(next_close)"
            ),
        },
        "validity": validity,
        "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        "contains_features": False,
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    print(f"labels complete year={year} rows={len(output)} output={output_path}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build separated 350-feature or 5-label data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=("features", "labels"), required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--date-start", default="")
    parser.add_argument("--date-end", default="")
    parser.add_argument("--context-days", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    years = _years(config)
    if args.year not in years:
        raise SystemExit(f"year {args.year} is not configured: {list(years)}")
    start_date, end_date = _date_bounds(args.year, args.date_start, args.date_end)
    output_root = _output_root(config, args.kind, args.output_root)
    if args.kind == "features":
        build_feature_dataset(
            config,
            config_path,
            year=args.year,
            output_root=output_root,
            start_date=start_date,
            end_date=end_date,
            context_days=(
                args.context_days
                if args.context_days is not None
                else config_int(config, "dataset", "feature_context_days", 120)
            ),
            overwrite=args.overwrite,
        )
    else:
        build_label_dataset(
            config,
            config_path,
            year=args.year,
            output_root=output_root,
            start_date=start_date,
            end_date=end_date,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
