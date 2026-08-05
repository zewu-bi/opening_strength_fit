from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import (
    config_float,
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json

KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
HORIZON_COLUMNS = {
    1: "label_short_1m",
    3: "label_short_3m",
    5: "label_short_5m",
}
OUTPUT_COLUMNS = (*KEY_COLUMNS, "label_short", "label_next_close", "target_label")


def mixed_target(
    frame: pd.DataFrame,
    *,
    short_column: str,
    weight: float,
    min_group_size: int,
) -> pd.Series:
    short = pd.to_numeric(frame[short_column], errors="coerce")
    long = pd.to_numeric(frame["label_next_close"], errors="coerce")
    usable = short.notna() & long.notna()
    keys = [frame["date"], frame["decision_target_timestamp"]]
    short_valid = short.where(usable)
    long_valid = long.where(usable)
    short_group = short_valid.groupby(keys, sort=False)
    long_group = long_valid.groupby(keys, sort=False)
    count = short_group.transform("count")
    short_std = short_group.transform(lambda values: values.std(ddof=0))
    long_std = long_group.transform(lambda values: values.std(ddof=0))
    usable &= count.ge(int(min_group_size)) & short_std.gt(1e-12) & long_std.gt(1e-12)
    short_z = (short - short_group.transform("mean")) / short_std
    long_z = (long - long_group.transform("mean")) / long_std
    return (short_z + float(weight) * long_z).where(usable)


def _output_root(config: dict, horizon_minutes: int) -> Path:
    template = config_str(config, "dataset", "horizon_label_output_template", "")
    if not template:
        raise SystemExit("missing [dataset].horizon_label_output_template")
    try:
        return Path(template.format(horizon_minutes=int(horizon_minutes)))
    except KeyError as exc:
        raise SystemExit("horizon_label_output_template must accept {horizon_minutes}") from exc


def split_label_year(
    config: dict,
    config_path: Path,
    *,
    year: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    source_root = Path(config_str(config, "dataset", "label_output_root", ""))
    if not source_root:
        raise SystemExit("missing [dataset].label_output_root")
    source_year_root = source_root / f"year={year}"
    source_path = source_year_root / "labels.parquet"
    if not source_path.exists() or not (source_year_root / "_SUCCESS").exists():
        raise SystemExit(f"source label year is incomplete: {source_year_root}")

    source_columns = [*KEY_COLUMNS, *HORIZON_COLUMNS.values(), "label_next_close"]
    source = read_frame(source_path, columns=source_columns)
    weight = config_float(config, "dataset", "mixed_next_close_weight", 0.30)
    min_group_size = config_int(config, "dataset", "mixed_min_group_size", 50)
    manifests = []

    for horizon_minutes, short_column in HORIZON_COLUMNS.items():
        output_root = _output_root(config, horizon_minutes)
        year_root = output_root / f"year={year}"
        output_path = year_root / "labels.parquet"
        success_path = year_root / "_SUCCESS"
        if output_path.exists() and not overwrite:
            raise SystemExit(f"horizon label output exists, pass --overwrite: {output_path}")
        if overwrite and success_path.exists():
            success_path.unlink()

        output = source.loc[:, list(KEY_COLUMNS)].copy()
        output["label_short"] = pd.to_numeric(source[short_column], errors="coerce")
        output["label_next_close"] = pd.to_numeric(source["label_next_close"], errors="coerce")
        output["target_label"] = mixed_target(
            source,
            short_column=short_column,
            weight=weight,
            min_group_size=min_group_size,
        )
        output[["label_short", "label_next_close", "target_label"]] = output[
            ["label_short", "label_next_close", "target_label"]
        ].astype("float32")
        output = output.loc[:, list(OUTPUT_COLUMNS)]
        write_frame_atomic(output, output_path)

        manifest = {
            "schema_version": "opening_horizon_labels_v2",
            "run_id": run_id(config, config_path),
            "kind": "horizon_labels",
            "year": int(year),
            "horizon_minutes": int(horizon_minutes),
            "rows": int(len(output)),
            "columns": list(output.columns),
            "key_columns": list(KEY_COLUMNS),
            "label_columns": ["label_short", "label_next_close", "target_label"],
            "target_definition": (
                f"xs_zscore({short_column}) + {weight:g} * xs_zscore(label_next_close)"
            ),
            "mixed_min_group_size": int(min_group_size),
            "source": str(source_path),
            "non_null_rows": {
                column: int(output[column].notna().sum())
                for column in ("label_short", "label_next_close", "target_label")
            },
            "contains_validity_flags": False,
            "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        }
        write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
        success_path.touch()
        manifests.append(manifest)
        print(
            f"horizon labels complete year={year} horizon={horizon_minutes}m "
            f"rows={len(output)} target_valid={output['target_label'].notna().sum()} "
            f"output={output_path}",
            flush=True,
        )
        del output

    return manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split combined annual labels into 1m, 3m, and 5m training datasets."
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
