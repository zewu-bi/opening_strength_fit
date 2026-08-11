from __future__ import annotations

import argparse
from pathlib import Path

from opening_strength_fit.config import config_str, load_toml
from opening_strength_fit.label_splitting import split_mixed_label_year

HORIZON_COLUMNS = {
    1: "label_short_1m",
    3: "label_short_3m",
    5: "label_short_5m",
}


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
    weight = float(config.get("dataset", {}).get("mixed_next_close_weight", 0.30))
    specs = [
        {
            "source_column": short_column,
            "output_root": _output_root(config, horizon_minutes),
            "manifest": {"horizon_minutes": horizon_minutes},
            "target_definition": (
                f"xs_zscore({short_column}) + {weight:g} * xs_zscore(label_next_close)"
            ),
            "log_label": f"horizon={horizon_minutes}m",
        }
        for horizon_minutes, short_column in HORIZON_COLUMNS.items()
    ]
    return split_mixed_label_year(
        config,
        config_path,
        year=year,
        overwrite=overwrite,
        specs=specs,
        schema_version="opening_horizon_labels_v2",
        kind="horizon_labels",
        common_manifest={"contains_validity_flags": False},
    )


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
