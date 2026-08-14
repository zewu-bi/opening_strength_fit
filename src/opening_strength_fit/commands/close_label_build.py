from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.commands.long_horizon_labels import (
    _reuse_next_close,
    same_day_close_label,
)
from opening_strength_fit.config import (
    config_float,
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json
from opening_strength_fit.label_splitting import KEY_COLUMNS, OUTPUT_COLUMNS
from opening_strength_fit.labels import cross_sectional_mixed_target as mixed_target
from opening_strength_fit.training_dataset_labels import (
    RAW_LABEL_TICK_COLUMNS,
)
from opening_strength_fit.training_dataset_labels import (
    build_label_base as _build_label_base,
)
from opening_strength_fit.training_dataset_labels import (
    normalize_dataset_keys as _normalize_keys,
)
from opening_strength_fit.training_dataset_labels import (
    validate_dataset_keys as _validate_output_keys,
)


def _years(config: dict) -> tuple[int, ...]:
    years = tuple(int(value) for value in config_list(config, "dataset", "years", []))
    if not years or len(set(years)) != len(years):
        raise SystemExit("[dataset].years must contain unique years")
    return years


def _complete_year(root: Path, year: int) -> Path:
    year_root = root / f"year={year}"
    if not (year_root / "_SUCCESS").exists():
        raise SystemExit(f"source year is incomplete: {year_root}")
    return year_root


def build_close_label_year(
    config: dict,
    config_path: Path,
    *,
    year: int,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> dict[str, object]:
    raw_root_value = config_str(config, "dataset", "raw_source_root", "").strip()
    next_close_root_value = config_str(config, "dataset", "next_close_label_root", "").strip()
    output_root_value = config_str(config, "dataset", "label_output_root", "").strip()
    if not raw_root_value or not next_close_root_value or not output_root_value:
        raise SystemExit(
            "[dataset] requires raw_source_root, next_close_label_root, and label_output_root"
        )
    raw_root = Path(raw_root_value)
    next_close_root = Path(next_close_root_value)
    output_root = Path(output_root_value)

    raw_year_root = _complete_year(raw_root, year)
    next_close_year_root = _complete_year(next_close_root, year)
    next_close_path = next_close_year_root / "labels.parquet"
    if not next_close_path.exists():
        raise SystemExit(f"missing next-close source: {next_close_path}")

    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    if not clocks:
        raise SystemExit("[dataset].decision_times must not be empty")
    tradable = tuple(config_list(config, "dataset", "tradable_statuses", ["T0", "20", "TRADE"]))
    close_reference = read_frame(
        raw_year_root / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    )
    tick_paths = [
        path
        for path in sorted((raw_year_root / "ticks").glob("date=*.parquet"))
        if start_date <= path.stem.removeprefix("date=") <= end_date
    ]
    if not tick_paths:
        raise SystemExit(f"no raw tick days in {start_date}..{end_date}")

    parts: list[pd.DataFrame] = []
    for index, tick_path in enumerate(tick_paths, start=1):
        trading_day = tick_path.stem.removeprefix("date=")
        ticks = read_frame(tick_path, columns=list(RAW_LABEL_TICK_COLUMNS))
        base = _build_label_base(
            ticks,
            trading_day=trading_day,
            decision_times=clocks,
            feature_tick_start_offset_us=config_int(
                config, "dataset", "feature_tick_start_offset_us", 0
            ),
            entry_delay_seconds=config_int(config, "dataset", "entry_delay_seconds", 6),
        )
        parts.append(
            same_day_close_label(
                base,
                close_reference,
                tradable_statuses=tradable,
                fee_bps=config_float(config, "dataset", "fee_bps", 0.0),
            )
        )
        print(
            f"close labels year={year} day={index}/{len(tick_paths)} "
            f"date={trading_day} rows={len(base)}",
            flush=True,
        )

    close = _normalize_keys(pd.concat(parts, ignore_index=True))
    _validate_output_keys(close, clocks)
    reused = _reuse_next_close(close, next_close_path)
    source = close.merge(reused, on=list(KEY_COLUMNS), validate="one_to_one")
    weight = config_float(config, "dataset", "mixed_next_close_weight", 0.30)
    clip_std_multiple = config_float(config, "dataset", "mixed_clip_std_multiple", 0.0)
    output = source.loc[:, list(KEY_COLUMNS)].copy()
    output["label_short"] = pd.to_numeric(source["label_same_day_close"], errors="coerce")
    output["label_next_close"] = pd.to_numeric(source["label_next_close"], errors="coerce")
    output["target_label"] = mixed_target(
        source,
        short_column="label_same_day_close",
        weight=weight,
        min_group_size=config_int(config, "dataset", "mixed_min_group_size", 50),
        clip_std_multiple=clip_std_multiple,
    )
    output[["label_short", "label_next_close", "target_label"]] = output[
        ["label_short", "label_next_close", "target_label"]
    ].astype("float32")
    output = output.loc[:, list(OUTPUT_COLUMNS)]

    year_root = output_root / f"year={year}"
    output_path = year_root / "labels.parquet"
    success_path = year_root / "_SUCCESS"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"close label output exists, pass --overwrite: {output_path}")
    if overwrite and success_path.exists():
        success_path.unlink()
    write_frame_atomic(output, output_path)
    parquet = pq.ParquetFile(output_path)
    manifest = {
        "schema_version": "opening_close_horizon_labels_v1",
        "run_id": run_id(config, config_path),
        "kind": "close_horizon_labels",
        "year": int(year),
        "date_start": start_date,
        "date_end": end_date,
        "rows": int(len(output)),
        "columns": list(output.columns),
        "key_columns": list(KEY_COLUMNS),
        "label_columns": ["label_short", "label_next_close", "target_label"],
        "decision_times": list(clocks),
        "target_definition": (
            "xs_zscore(label_same_day_close)"
            if weight == 0.0
            else (
                f"xs_zscore(label_same_day_close) + {weight:g} * xs_zscore(reused label_next_close)"
            )
        ),
        "mixed_min_group_size": config_int(config, "dataset", "mixed_min_group_size", 50),
        "mixed_clip_std_multiple": clip_std_multiple,
        "source_raw_root": str(raw_year_root),
        "next_close_source": str(next_close_path),
        "non_null_rows": {
            column: int(output[column].notna().sum())
            for column in ("label_short", "label_next_close", "target_label")
        },
        "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        "parquet_rows": int(parquet.metadata.num_rows),
        "contains_features": False,
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    print(
        f"close labels complete year={year} rows={len(output)} "
        f"target_valid={output['target_label'].notna().sum()} output={output_path}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build model-ready same-day-close mixed labels from raw PVC sources."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    years = _years(config)
    if int(args.year) not in years:
        raise SystemExit(f"year {args.year} is not configured: {list(years)}")
    build_close_label_year(
        config,
        config_path,
        year=int(args.year),
        start_date=args.start_date or f"{args.year}-01-01",
        end_date=args.end_date or f"{args.year}-12-31",
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
