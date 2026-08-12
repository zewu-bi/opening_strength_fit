from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.config import config_float, config_int, config_str, run_id
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json
from opening_strength_fit.labels import cross_sectional_mixed_target as mixed_target
from opening_strength_fit.schema import DECISION_KEY_COLUMNS

KEY_COLUMNS = DECISION_KEY_COLUMNS
OUTPUT_COLUMNS = (*KEY_COLUMNS, "label_short", "label_next_close", "target_label")


def split_mixed_label_year(
    config: dict,
    config_path: Path,
    *,
    year: int,
    overwrite: bool,
    specs: list[dict[str, object]],
    schema_version: str,
    kind: str,
    common_manifest: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    source_root_value = config_str(config, "dataset", "label_output_root", "")
    if not source_root_value:
        raise SystemExit("missing [dataset].label_output_root")
    source_year_root = Path(source_root_value) / f"year={year}"
    source_path = source_year_root / "labels.parquet"
    if not source_path.exists() or not (source_year_root / "_SUCCESS").exists():
        raise SystemExit(f"source label year is incomplete: {source_year_root}")

    source_columns = [
        *KEY_COLUMNS,
        "label_next_close",
        *(str(spec["source_column"]) for spec in specs),
    ]
    source = read_frame(source_path, columns=list(dict.fromkeys(source_columns)))
    duplicate = source.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(f"source has {int(duplicate.sum())} duplicate key rows: {source_path}")

    weight = config_float(config, "dataset", "mixed_next_close_weight", 0.30)
    min_group_size = config_int(config, "dataset", "mixed_min_group_size", 50)
    manifests = []
    for spec in specs:
        source_column = str(spec["source_column"])
        output_root = Path(str(spec["output_root"]))
        year_root = output_root / f"year={year}"
        output_path = year_root / "labels.parquet"
        success_path = year_root / "_SUCCESS"
        if output_path.exists() and not overwrite:
            raise SystemExit(f"mixed label output exists, pass --overwrite: {output_path}")
        if overwrite and success_path.exists():
            success_path.unlink()

        output = source.loc[:, list(KEY_COLUMNS)].copy()
        output["label_short"] = pd.to_numeric(source[source_column], errors="coerce")
        output["label_next_close"] = pd.to_numeric(source["label_next_close"], errors="coerce")
        output["target_label"] = mixed_target(
            source,
            short_column=source_column,
            weight=weight,
            min_group_size=min_group_size,
        )
        value_columns = ["label_short", "label_next_close", "target_label"]
        output[value_columns] = output[value_columns].astype("float32")
        output = output.loc[:, list(OUTPUT_COLUMNS)]
        write_frame_atomic(output, output_path)

        manifest = {
            "schema_version": schema_version,
            "run_id": run_id(config, config_path),
            "kind": kind,
            "year": int(year),
            **dict(common_manifest or {}),
            **dict(spec.get("manifest", {})),
            "rows": int(len(output)),
            "columns": list(output.columns),
            "key_columns": list(KEY_COLUMNS),
            "label_columns": value_columns,
            "target_definition": str(spec["target_definition"]),
            "mixed_min_group_size": int(min_group_size),
            "source": str(source_path),
            "non_null_rows": {
                column: int(output[column].notna().sum()) for column in value_columns
            },
            "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        }
        write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
        success_path.touch()
        manifests.append(manifest)
        print(
            f"mixed labels complete year={year} {spec['log_label']} "
            f"rows={len(output)} target_valid={output['target_label'].notna().sum()} "
            f"output={output_path}",
            flush=True,
        )
    return manifests
