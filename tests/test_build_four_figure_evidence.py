from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[1] / "experiments" / "scripts" / "build_four_figure_evidence.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_four_figure_evidence", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def test_build_bundle_copies_figures_and_compacts_cumulative_data(tmp_path: Path) -> None:
    for relative_source, _ in SCRIPT_MODULE.COPY_SPECS:
        source = tmp_path / relative_source
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source={relative_source}\n", encoding="utf-8")

    cumulative_source = tmp_path / SCRIPT_MODULE.CUMULATIVE_SOURCE
    cumulative_source.parent.mkdir(parents=True, exist_ok=True)
    with cumulative_source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*SCRIPT_MODULE.CUMULATIVE_COLUMNS, "row_level_detail"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "pool": "clock6_v4_multiden",
                "pool_label": "multi denominator",
                "week_start": "2022-01-03",
                "variant": "multi denominator",
                "next_cumulative_net_return_bps": "12.5",
                "next_cumulative_alpha_bps": "4.5",
                "row_level_detail": "must not be recorded",
            }
        )

    bucket_source = tmp_path / SCRIPT_MODULE.FULL_POOL_BUCKET_SOURCE
    with bucket_source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "scope", "bucket", "mean_excess_bps"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "clock6_v4_multiden",
                "scope": "top1000",
                "bucket": "1",
                "mean_excess_bps": "17.1",
            }
        )
        writer.writerow(
            {
                "variant": "clock6_v4_multiden",
                "scope": "pool_L",
                "bucket": "1",
                "mean_excess_bps": "10.5",
            }
        )

    journey_source = tmp_path / SCRIPT_MODULE.JOURNEY_CUMULATIVE_SOURCE
    journey_source.parent.mkdir(parents=True, exist_ok=True)
    with journey_source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*SCRIPT_MODULE.CUMULATIVE_COLUMNS, "row_level_detail"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "pool": "clock6_v4_multiden",
                "pool_label": "multi denominator",
                "week_start": "2022-01-03",
                "variant": "multi denominator",
                "next_cumulative_net_return_bps": "13.5",
                "next_cumulative_alpha_bps": "5.5",
                "row_level_detail": "must not be recorded",
            }
        )

    destination = SCRIPT_MODULE.build_bundle(tmp_path)

    compact = destination / SCRIPT_MODULE.CUMULATIVE_OUTPUT
    with compact.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == SCRIPT_MODULE.CUMULATIVE_COLUMNS
        assert list(reader)[0]["pool"] == "clock6_v4_multiden"

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_candidate"] == "clock6_v4_multiden"
    assert manifest["cumulative_rows"] == 1
    assert manifest["full_pool_bucket_rows"] == 1
    assert manifest["journey_cumulative_rows"] == 1
    assert "01_signal_acceptance.svg" in manifest["files"]
    assert "03b_full_pool_bucket_curve.svg" in manifest["files"]
    assert "04b_top1000_return_distribution_full_scale.svg" in manifest["files"]
    assert "05_model_journey_cumulative.svg" in manifest["files"]

    full_pool_rows = list(
        csv.DictReader(
            (destination / SCRIPT_MODULE.FULL_POOL_BUCKET_OUTPUT).open(
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert [row["scope"] for row in full_pool_rows] == ["pool_L"]
