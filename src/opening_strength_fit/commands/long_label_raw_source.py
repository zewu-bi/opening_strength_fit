from __future__ import annotations

from pathlib import Path

from opening_strength_fit.io.json import write_json
from opening_strength_fit.raw_source import (
    build_tick_source_manifest,
    raw_source_parser,
    run_raw_source_builder,
)

RAW_SOURCE_SCHEMA_VERSION = "long_label_raw_source_v1"
TICK_COLUMNS = tuple("TradingDay Symbol ExchTimeOffsetUs Volume Turnover".split())


def build_year(
    client,
    *,
    config: dict,
    config_path: Path,
    year: int,
    output_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    manifest, year_root = build_tick_source_manifest(
        client,
        config=config,
        config_path=config_path,
        year=year,
        output_root=output_root,
        columns=TICK_COLUMNS,
        overwrite=overwrite,
        schema_version=RAW_SOURCE_SCHEMA_VERSION,
    )
    manifest["tick_deduplication"] = {
        "key": ["TradingDay", "Symbol", "ExchTimeOffsetUs"],
        "selection": "same latest-local-timestamp tie-break as raw_source_v2",
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    (year_root / "_SUCCESS").touch()
    return manifest


def main() -> None:
    args = raw_source_parser(
        "Build a projected raw tick cache for long-horizon VWAP labels."
    ).parse_args()
    run_raw_source_builder(
        args,
        default_output_root="output/long_label_raw_source",
        build_year=build_year,
    )


if __name__ == "__main__":
    main()
