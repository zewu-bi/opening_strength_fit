from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.commands.arguments import add_arguments
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
)
from opening_strength_fit.training_dataset_labels import (
    KEY_COLUMNS,
    RAW_LABEL_TICK_COLUMNS,
    build_label_base,
    compute_clock_vwap_label_set,
    filter_decision_clocks,
    mixed_target_label,
    next_close_label,
    normalize_dataset_keys,
    validate_dataset_keys,
)
from opening_strength_fit.training_labeled import (
    _apply_cross_sectional_relative_from_config,
    _apply_post_sample_feature_transforms_from_config,
    _drop_features_from_config,
    apply_candidate_filter_from_config,
)

LABEL_COLUMNS = tuple(
    "label_short_1m label_short_3m label_short_5m label_next_close label_mixed".split()
)
VALID_LABEL_COLUMNS = tuple(
    "valid_short_1m valid_short_3m valid_short_5m valid_next_close valid_mixed".split()
)
_FEATURE_WORKER_CONFIG: tuple[dict, dict] | None = None
_FEATURE_WORKER_DAILY_CACHE: dict[Path, pd.DataFrame] = {}


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


def _initialize_feature_worker(feature_config: dict, dataset_config: dict) -> None:
    global _FEATURE_WORKER_CONFIG, _FEATURE_WORKER_DAILY_CACHE
    _FEATURE_WORKER_CONFIG = (feature_config, dataset_config)
    _FEATURE_WORKER_DAILY_CACHE = {}


def _build_feature_worker(item: tuple[str, Path]) -> pd.DataFrame:
    if _FEATURE_WORKER_CONFIG is None:
        raise RuntimeError("feature worker was not initialized")
    trading_day, raw_path = item
    feature_config, dataset_config = _FEATURE_WORKER_CONFIG
    return build_raw_feature_day(
        raw_path,
        trading_day,
        feature_config,
        dataset_config,
        _FEATURE_WORKER_DAILY_CACHE,
    )


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
    workers: int = 1,
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
    parts = []
    if workers < 1:
        raise SystemExit("feature workers must be >= 1")
    if workers == 1:
        daily_cache: dict[Path, pd.DataFrame] = {}
        built_parts = (
            build_raw_feature_day(
                raw_path,
                trading_day,
                feature_config,
                config,
                daily_cache,
            )
            for trading_day, raw_path in raw_dates
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_feature_worker,
            initargs=(feature_config, config),
        )
        built_parts = executor.map(_build_feature_worker, raw_dates, chunksize=1)
    try:
        for index, ((trading_day, _raw_path), part) in enumerate(
            zip(raw_dates, built_parts, strict=True), start=1
        ):
            if not part.empty:
                floats = part.select_dtypes(include=["float64"]).columns
                if len(floats):
                    part[floats] = part[floats].astype("float32")
                parts.append(part)
            print(
                f"features year={year} day={index}/{len(raw_dates)} date={trading_day} "
                f"rows={len(part)} workers={workers}",
                flush=True,
            )
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)
    if not parts:
        raise SystemExit("raw source produced no sampled feature rows")
    source = pd.concat(parts, ignore_index=True)
    transformed = _apply_post_sample_feature_transforms_from_config(source, feature_config)
    if config_bool(feature_config, "features", "include_cross_sectional_relative", False):
        transformed = _apply_cross_sectional_relative_from_config(transformed, feature_config)
    transformed = _drop_features_from_config(transformed, feature_config)
    transformed = apply_candidate_filter_from_config(transformed, feature_config)
    transformed = normalize_dataset_keys(transformed)
    target_dates = transformed["date"].between(start_date, end_date)
    transformed = transformed.loc[target_dates].copy()
    transformed = filter_decision_clocks(transformed, clocks)
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
    validate_dataset_keys(output, clocks)
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
        "build_workers": int(workers),
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
        day_base = build_label_base(
            ticks,
            trading_day=trading_day,
            decision_times=clocks,
            feature_tick_start_offset_us=config_int(
                config, "dataset", "feature_tick_start_offset_us", 0
            ),
            entry_delay_seconds=config_int(config, "dataset", "entry_delay_seconds", 6),
        )
        labels = compute_clock_vwap_label_set(
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
    validate_dataset_keys(base, clocks)
    output = base_and_short[[*KEY_COLUMNS, *LABEL_COLUMNS[:3], *VALID_LABEL_COLUMNS[:3]]]
    next_close = next_close_label(base, raw_year_root)
    output = output.merge(next_close, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    output["valid_next_close"] = output["valid_next_close"].fillna(False).astype(bool)
    mixed, mixed_valid = mixed_target_label(
        output,
        weight=config_float(config, "dataset", "mixed_next_close_weight", 0.30),
        min_group_size=config_int(config, "dataset", "mixed_min_group_size", 50),
    )
    output["label_mixed"] = mixed
    output["valid_mixed"] = mixed_valid
    output = output[[*KEY_COLUMNS, *LABEL_COLUMNS, *VALID_LABEL_COLUMNS]]
    validate_dataset_keys(output, clocks)
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
    add_arguments(parser, "output-root date-start date-end", default="")
    parser.add_argument("--context-days", type=int)
    parser.add_argument("--workers", type=int, default=1)
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
            workers=args.workers,
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
