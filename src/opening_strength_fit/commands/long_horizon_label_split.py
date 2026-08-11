from __future__ import annotations

import argparse
from pathlib import Path

from opening_strength_fit.config import load_toml
from opening_strength_fit.label_splitting import split_mixed_label_year


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
        specs.append({"name": name, "source_column": source_column, "output_root": output_root})
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
    specs = _specs(config)
    weight = float(config.get("dataset", {}).get("mixed_next_close_weight", 0.30))
    normalized_specs: list[dict[str, object]] = []
    for spec in specs:
        source_column = spec["source_column"]
        normalized_specs.append(
            {
                **spec,
                "manifest": {
                    "horizon_name": spec["name"],
                    "source_label_column": source_column,
                },
                "target_definition": (
                    f"xs_zscore({source_column}) + {weight:g} * xs_zscore(reused label_next_close)"
                ),
                "log_label": f"horizon={spec['name']}",
            }
        )
    return split_mixed_label_year(
        config,
        config_path,
        year=year,
        overwrite=overwrite,
        specs=normalized_specs,
        schema_version="opening_long_horizon_mixed_labels_v1",
        kind="long_horizon_mixed_labels",
    )


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
