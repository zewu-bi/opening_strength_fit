from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
from opening_strength_fit.training_dataset_features import build_raw_feature_day
from opening_strength_fit.training_dataset_labels import (
    _build_label_base,
    _mixed_label,
    _next_close_label,
    compute_short_label_set,
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


def _load_canonical_feature_config(config: dict) -> tuple[Path, dict, bool]:
    canonical_path = Path(config_str(config, "dataset", "canonical_feature_config", ""))
    if not canonical_path.exists():
        raise SystemExit(f"canonical feature config does not exist: {canonical_path}")
    feature_config = load_toml(canonical_path)
    raw_feature_values = config_bool(config, "dataset", "raw_feature_values", False)
    if raw_feature_values:
        feature_config = {
            section: dict(values) if isinstance(values, dict) else values
            for section, values in feature_config.items()
        }
        features_section = dict(feature_config.get("features", {}))
        features_section["feature_value_transform"] = "none"
        feature_config["features"] = features_section
    return canonical_path, feature_config, raw_feature_values


def _sort_feature_source(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_keys(frame)
    return normalized.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _clock_partition(clock: str) -> str:
    parsed = pd.Timestamp(f"2000-01-01 {clock}")
    return parsed.strftime("%H%M%S")


def _feature_base_shard_root(output_root: Path, year: int, clock: str) -> Path:
    return output_root / f"year={year}" / f"clock={_clock_partition(clock)}"


def build_feature_base_shard(
    config: dict,
    config_path: Path,
    *,
    year: int,
    output_root: Path,
    output_decision_time: str,
    start_date: str,
    end_date: str,
    context_days: int,
    overwrite: bool,
) -> dict[str, object]:
    """Build one physical-cutoff clock shard before cross-day transforms."""

    canonical_path, feature_config, _ = _load_canonical_feature_config(config)
    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    if output_decision_time not in clocks:
        raise SystemExit(
            f"feature base clock {output_decision_time!r} is not configured: {list(clocks)}"
        )
    physical_cutoff_seconds = config_int(config, "dataset", "physical_tick_cutoff_seconds", -1)
    if physical_cutoff_seconds < 0:
        raise SystemExit("feature base shards require --physical-tick-cutoff-seconds >= 0")

    shard_root = _feature_base_shard_root(output_root, year, output_decision_time)
    output_path = shard_root / "features_base.parquet"
    success_path = shard_root / "_SUCCESS"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"feature base output exists, pass --overwrite: {output_path}")
    if overwrite and success_path.exists():
        success_path.unlink()

    context_start = str((pd.Timestamp(start_date) - pd.Timedelta(days=int(context_days))).date())
    raw_dates = _raw_tick_dates(config, start_date=context_start, end_date=end_date)
    if not raw_dates:
        raise SystemExit(f"no raw tick days in {context_start}..{end_date}")
    daily_cache: dict[Path, pd.DataFrame] = {}
    parts: list[pd.DataFrame] = []
    for index, (trading_day, raw_path) in enumerate(raw_dates, start=1):
        part = build_raw_feature_day(
            raw_path,
            trading_day,
            feature_config,
            config,
            daily_cache,
            output_decision_time=output_decision_time,
            source_cutoff_seconds=physical_cutoff_seconds,
        )
        if not part.empty:
            floats = part.select_dtypes(include=["float64"]).columns
            if len(floats):
                part[floats] = part[floats].astype("float32")
            parts.append(part)
        print(
            f"feature base year={year} clock={output_decision_time} "
            f"day={index}/{len(raw_dates)} date={trading_day} rows={len(part)}",
            flush=True,
        )
    if not parts:
        raise SystemExit("raw source produced no feature base rows")
    source = _sort_feature_source(pd.concat(parts, ignore_index=True))
    _validate_output_keys(source, (output_decision_time,))
    write_frame_atomic(source, output_path)
    parquet = pq.ParquetFile(output_path)
    manifest = {
        "schema_version": "opening_feature_base_clock_v1",
        "run_id": run_id(config, config_path),
        "kind": "feature_base_clock",
        "year": year,
        "output_decision_time": output_decision_time,
        "date_start": start_date,
        "date_end": end_date,
        "context_start": context_start,
        "context_days": context_days,
        "physical_tick_cutoff_seconds": physical_cutoff_seconds,
        "physical_per_decision_parquet_filter": True,
        "canonical_feature_config": str(canonical_path),
        "canonical_feature_config_fingerprint": _config_fingerprint(canonical_path),
        "source_raw_root": str(_raw_root(config)),
        "source_tick_files": len(raw_dates),
        "rows": int(len(source)),
        "columns": int(len(source.columns)),
        "contains_post_sample_transforms": False,
        "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        "parquet_rows": int(parquet.metadata.num_rows),
    }
    write_json(shard_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    print(
        f"feature base complete year={year} clock={output_decision_time} "
        f"rows={len(source)} output={output_path}",
        flush=True,
    )
    return manifest


def _load_feature_base_source(
    input_root: Path,
    *,
    year: int,
    clocks: tuple[str, ...],
    physical_cutoff_seconds: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for clock in clocks:
        shard_root = _feature_base_shard_root(input_root, year, clock)
        success_path = shard_root / "_SUCCESS"
        manifest_path = shard_root / "manifest.json"
        output_path = shard_root / "features_base.parquet"
        if not success_path.exists() or not manifest_path.exists() or not output_path.exists():
            raise SystemExit(f"incomplete feature base shard: {shard_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_decision_time") != clock:
            raise SystemExit(f"feature base clock mismatch: {manifest_path}")
        if manifest.get("physical_tick_cutoff_seconds") != physical_cutoff_seconds:
            raise SystemExit(f"feature base cutoff mismatch: {manifest_path}")
        parts.append(read_frame(output_path))
    source = _sort_feature_source(pd.concat(parts, ignore_index=True))
    duplicate = source.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(f"feature base source has {int(duplicate.sum())} duplicate key rows")
    _validate_output_keys(source, clocks)
    return source


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
    feature_base_input_root: Path | None = None,
) -> dict[str, object]:
    canonical_path, feature_config, raw_feature_values = _load_canonical_feature_config(config)
    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    expected_features = config_int(config, "dataset", "expected_feature_count", 350)
    year_root = output_root / f"year={year}"
    output_path = year_root / "features.parquet"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"feature output exists, pass --overwrite: {output_path}")
    success_path = year_root / "_SUCCESS"
    if overwrite and success_path.exists():
        success_path.unlink()

    physical_cutoff_seconds = config_int(config, "dataset", "physical_tick_cutoff_seconds", -1)
    if physical_cutoff_seconds < -1:
        raise SystemExit("[dataset].physical_tick_cutoff_seconds must be >= 0 when configured")
    context_start = str((pd.Timestamp(start_date) - pd.Timedelta(days=int(context_days))).date())
    raw_dates: list[tuple[str, Path]] = []
    if feature_base_input_root is not None:
        if physical_cutoff_seconds < 0:
            raise SystemExit("feature base reduction requires a physical tick cutoff")
        source = _load_feature_base_source(
            feature_base_input_root,
            year=year,
            clocks=clocks,
            physical_cutoff_seconds=physical_cutoff_seconds,
        )
    else:
        raw_dates = _raw_tick_dates(config, start_date=context_start, end_date=end_date)
        if not raw_dates:
            raise SystemExit(f"no raw tick days in {context_start}..{end_date}")
        daily_cache: dict[Path, pd.DataFrame] = {}
        parts = []
        for index, (trading_day, raw_path) in enumerate(raw_dates, start=1):
            if physical_cutoff_seconds >= 0:
                decision_parts = [
                    build_raw_feature_day(
                        raw_path,
                        trading_day,
                        feature_config,
                        config,
                        daily_cache,
                        output_decision_time=clock,
                        source_cutoff_seconds=physical_cutoff_seconds,
                    )
                    for clock in clocks
                ]
                part = pd.concat(decision_parts, ignore_index=True)
            else:
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
        source = _sort_feature_source(pd.concat(parts, ignore_index=True))
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
    output = _sort_feature_source(output[[*KEY_COLUMNS, *selected]].copy())
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
        "source_tick_files": len(raw_dates) if raw_dates else None,
        "source_feature_base_root": (
            str(feature_base_input_root) if feature_base_input_root is not None else None
        ),
        "source_feature_base_shards": len(clocks) if feature_base_input_root is not None else None,
        "source_rows_with_context": int(len(source)),
        "context_days": context_days,
        "canonical_feature_config": str(canonical_path),
        "canonical_feature_config_fingerprint": _config_fingerprint(canonical_path),
        "feature_values": "model_ready",
        "feature_value_transform": value_transform,
        "training_feature_value_transform": "none",
        "raw_feature_values": raw_feature_values,
        "physical_tick_cutoff_seconds": (
            physical_cutoff_seconds if physical_cutoff_seconds >= 0 else None
        ),
        "physical_per_decision_parquet_filter": physical_cutoff_seconds >= 0,
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
    parser.add_argument(
        "--physical-tick-cutoff-seconds",
        type=int,
        help=(
            "Leakage kill-test mode: read each decision clock separately with a Parquet "
            "filter ending this many seconds before the logical clock."
        ),
    )
    parser.add_argument(
        "--raw-feature-values",
        action="store_true",
        help="Leakage baseline mode: disable mechanismized/cross-sectional value transforms.",
    )
    parser.add_argument(
        "--feature-base-clock",
        default="",
        help=(
            "Map stage for leakage kill tests: build one physical-cutoff decision-clock "
            "feature shard before cross-day transforms."
        ),
    )
    parser.add_argument(
        "--feature-base-input-root",
        default="",
        help=(
            "Reduce stage for leakage kill tests: read all configured clock shards from "
            "this root, then compute history/cross-sectional/final feature transforms."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    if args.physical_tick_cutoff_seconds is not None:
        config = {
            section: dict(values) if isinstance(values, dict) else values
            for section, values in config.items()
        }
        dataset = dict(config.get("dataset", {}))
        dataset["physical_tick_cutoff_seconds"] = args.physical_tick_cutoff_seconds
        config["dataset"] = dataset
    if args.raw_feature_values:
        config = {
            section: dict(values) if isinstance(values, dict) else values
            for section, values in config.items()
        }
        dataset = dict(config.get("dataset", {}))
        dataset["raw_feature_values"] = True
        config["dataset"] = dataset
    years = _years(config)
    if args.year not in years:
        raise SystemExit(f"year {args.year} is not configured: {list(years)}")
    start_date, end_date = _date_bounds(args.year, args.date_start, args.date_end)
    output_root = _output_root(config, args.kind, args.output_root)
    if args.kind == "features":
        if args.feature_base_clock and args.feature_base_input_root:
            raise SystemExit(
                "--feature-base-clock and --feature-base-input-root are mutually exclusive"
            )
        context_days = (
            args.context_days
            if args.context_days is not None
            else config_int(config, "dataset", "feature_context_days", 120)
        )
        if args.feature_base_clock:
            build_feature_base_shard(
                config,
                config_path,
                year=args.year,
                output_root=output_root,
                output_decision_time=args.feature_base_clock,
                start_date=start_date,
                end_date=end_date,
                context_days=context_days,
                overwrite=args.overwrite,
            )
        else:
            build_feature_dataset(
                config,
                config_path,
                year=args.year,
                output_root=output_root,
                start_date=start_date,
                end_date=end_date,
                context_days=context_days,
                overwrite=args.overwrite,
                feature_base_input_root=(
                    Path(args.feature_base_input_root) if args.feature_base_input_root else None
                ),
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
