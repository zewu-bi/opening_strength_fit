from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from opening_strength_fit.cache_manifest import (
    build_cache_manifest,
    write_cache_manifest,
)
from opening_strength_fit.config import load_toml, run_id
from opening_strength_fit.io import frame_columns, read_frame
from opening_strength_fit.reports import print_mapping


SUMMARY_COLUMNS = (
    "date",
    "symbol",
    "timestamp",
    "decision_time",
    "decision_target_timestamp",
    "decision_lag_seconds",
    "entry_delay_ticks",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "label",
    "valid_label",
    "target_label",
)


def _parquet_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    if path.suffix.lower() == ".parquet":
        return [path]
    return []


def _parquet_metadata(path: Path) -> tuple[list[tuple[str, str]] | None, int | None, int]:
    files = _parquet_files(path)
    if not files:
        return None, None, 0

    import pyarrow.parquet as pq

    schema_columns: list[tuple[str, str]] | None = None
    row_count = 0
    for file in files:
        parquet_file = pq.ParquetFile(file)
        row_count += int(parquet_file.metadata.num_rows)
        if schema_columns is None:
            schema_columns = [
                (field.name, str(field.type))
                for field in parquet_file.schema_arrow
            ]
    return schema_columns, row_count, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a labeled cache and write a compact schema manifest."
    )
    parser.add_argument("--input", required=True, help="Parquet/csv cache file or dir.")
    parser.add_argument("--config", default="", help="Optional run TOML for config hash.")
    parser.add_argument("--output", default="", help="Optional manifest JSON output path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else input_path.stem

    schema_columns, row_count, file_count = _parquet_metadata(input_path)
    available_columns = (
        [name for name, _ in schema_columns]
        if schema_columns is not None
        else sorted(frame_columns(input_path))
    )
    summary_columns = [column for column in SUMMARY_COLUMNS if column in available_columns]
    frame = read_frame(input_path, columns=summary_columns or None)
    manifest = build_cache_manifest(
        frame,
        cache_path=input_path,
        config=config,
        run_name=run_name,
        config_path=args.config or "",
        schema_columns=schema_columns,
        row_count=row_count,
    )
    manifest["input"] = {
        "path": str(input_path),
        "parquet_files": int(file_count),
    }

    print_mapping(
        "labeled_cache_inspection",
        {
            "path": str(input_path),
            "rows": manifest["summary"].get("rows"),
            "columns": manifest["summary"].get("columns"),
            "date_min": manifest["summary"].get("date_min"),
            "date_max": manifest["summary"].get("date_max"),
            "missing_required": manifest["required_columns"]["missing"],
        },
    )

    if args.output:
        write_cache_manifest(manifest, args.output)
        print(f"\nwrote manifest: {args.output}")
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
