from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.commands.horizon_label_split import (
    KEY_COLUMNS,
    OUTPUT_COLUMNS,
    mixed_target,
)
from opening_strength_fit.config import config_float, config_int, config_str, load_toml, run_id
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json


def _specs(config: dict) -> list[dict[str, str]]:
    section = config.get("dataset", {})
    raw = section.get("mixed_labels", []) if isinstance(section, dict) else []
    specs = []
    for item in raw:
        if not isinstance(item, dict):
            raise SystemExit("[dataset].mixed_labels entries must be tables")
        name = str(item.get("name", "")).strip()
        source_column = str(item.get("source_column", "")).strip()
        output_root = str(item.get("output_root", "")).strip()
        if not name or not source_column or not output_root:
            raise SystemExit("each mixed_labels entry needs name, source_column, output_root")
        specs.append(
            {"name": name, "source_column": source_column, "output_root": output_root}
        )
    if not specs or len({item["name"] for item in specs}) != len(specs):
        raise SystemExit("[dataset].mixed_labels must contain unique named entries")
    return specs


def split_label_year(
    config: dict,
    config_path: Path,
    *,
    year: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    source_root = Path(config_str(config, "dataset", "label_output_root", ""))
    source_year_root = source_root / f"year={year}"
    source_path = source_year_root / "labels.parquet"
    if not source_path.exists() or not (source_year_root / "_SUCCESS").exists():
        raise SystemExit(f"source label year is incomplete: {source_year_root}")
    specs = _specs(config)
    columns = [
        *KEY_COLUMNS,
        "label_next_close",
        *(item["source_column"] for item in specs),
    ]
    source = read_frame(source_path, columns=list(dict.fromkeys(columns)))
    duplicate = source.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(f"source has {int(duplicate.sum())} duplicate key rows: {source_path}")
    weight = config_float(config, "dataset", "mixed_next_close_weight", 0.30)
    min_group_size = config_int(config, "dataset", "mixed_min_group_size", 50)
    manifests = []

    for spec in specs:
        output_root = Path(spec["output_root"])
        year_root = output_root / f"year={year}"
        output_path = year_root / "labels.parquet"
        success_path = year_root / "_SUCCESS"
        if output_path.exists() and not overwrite:
            raise SystemExit(f"mixed label output exists, pass --overwrite: {output_path}")
        if overwrite and success_path.exists():
            success_path.unlink()

        output = source.loc[:, list(KEY_COLUMNS)].copy()
        output["label_short"] = pd.to_numeric(
            source[spec["source_column"]], errors="coerce"
        )
        output["label_next_close"] = pd.to_numeric(
            source["label_next_close"], errors="coerce"
        )
        output["target_label"] = mixed_target(
            source,
            short_column=spec["source_column"],
            weight=weight,
            min_group_size=min_group_size,
        )
        output[["label_short", "label_next_close", "target_label"]] = output[
            ["label_short", "label_next_close", "target_label"]
        ].astype("float32")
        output = output[list(OUTPUT_COLUMNS)]
        write_frame_atomic(output, output_path)

        manifest = {
            "schema_version": "opening_long_horizon_mixed_labels_v1",
            "run_id": run_id(config, config_path),
            "kind": "long_horizon_mixed_labels",
            "year": int(year),
            "horizon_name": spec["name"],
            "source_label_column": spec["source_column"],
            "rows": len(output),
            "columns": list(output.columns),
            "key_columns": list(KEY_COLUMNS),
            "label_columns": ["label_short", "label_next_close", "target_label"],
            "target_definition": (
                f"xs_zscore({spec['source_column']}) + {weight:g} "
                "* xs_zscore(reused label_next_close)"
            ),
            "mixed_min_group_size": int(min_group_size),
            "source": str(source_path),
            "non_null_rows": {
                column: int(output[column].notna().sum())
                for column in ("label_short", "label_next_close", "target_label")
            },
            "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        }
        write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
        success_path.touch()
        manifests.append(manifest)
        print(
            f"mixed labels complete year={year} horizon={spec['name']} "
            f"rows={len(output)} target_valid={output['target_label'].notna().sum()} "
            f"output={output_path}",
            flush=True,
        )
    return manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split combined long-horizon labels into model-ready mixed roots."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    split_label_year(
        config,
        config_path,
        year=int(args.year),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
