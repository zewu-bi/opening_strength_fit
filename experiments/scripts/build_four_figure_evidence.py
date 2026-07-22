from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RUN_ID = (
    "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_"
    "grouped_gated_v2_mech_v3_gelu_mse_v1"
)
BUNDLE_RELATIVE_DIR = Path("experiments/evidence/backtests") / CANONICAL_RUN_ID

OVERLAY_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "optimization_overlay_acceptance_clock6_v4_control_multiden_vs_mech328_v2_2022_2025"
)
BUCKET_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "top1000_rank_bucket_diag_auction_multiden_2022_2025_v1/rank_bucket"
)
HISTOGRAM_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "top1000_score_bucket_return_histogram_auction_multiden_2022_2025_v1"
)

COPY_SPECS = (
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_overlay_acceptance.svg",
        "01_signal_acceptance.svg",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_overlay_acceptance_plot_data.csv",
        "01_signal_acceptance.csv",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative.svg",
        "02_top100_cumulative.svg",
    ),
    (
        BUCKET_RELATIVE_DIR / "top1000_bucket_returns.svg",
        "03_top1000_bucket_curve.svg",
    ),
    (
        BUCKET_RELATIVE_DIR / "bucket_curve_plot_data.csv",
        "03_top1000_bucket_curve.csv",
    ),
    (
        HISTOGRAM_RELATIVE_DIR / "top1000_score_bucket_return_100bps_counts.svg",
        "04_top1000_return_distribution.svg",
    ),
    (
        HISTOGRAM_RELATIVE_DIR / "top1000_score_bucket_return_100bps_counts.csv",
        "04_top1000_return_distribution.csv",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_trace.json",
        "trace_optimization.json",
    ),
    (BUCKET_RELATIVE_DIR / "trace.json", "trace_top1000_bucket.json"),
    (HISTOGRAM_RELATIVE_DIR / "trace.json", "trace_top1000_distribution.json"),
)

CUMULATIVE_SOURCE = (
    OVERLAY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative_plot_data.csv"
)
CUMULATIVE_OUTPUT = "02_top100_cumulative.csv"
CUMULATIVE_COLUMNS = (
    "pool",
    "pool_label",
    "week_start",
    "variant",
    "next_cumulative_net_return_bps",
    "next_cumulative_alpha_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required four-figure source is missing: {path}")


def _write_compact_cumulative(source: Path, destination: Path) -> int:
    _require(source)
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        missing = [
            column for column in CUMULATIVE_COLUMNS if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"{source}: missing cumulative columns: {missing}")
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            writer = csv.DictWriter(
                destination_handle,
                fieldnames=CUMULATIVE_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            row_count = 0
            for row in reader:
                writer.writerow({column: row[column] for column in CUMULATIVE_COLUMNS})
                row_count += 1
    return row_count


def build_bundle(root: Path = ROOT) -> Path:
    destination = root / BUNDLE_RELATIVE_DIR
    destination.mkdir(parents=True, exist_ok=True)

    sources: dict[str, str] = {}
    for relative_source, output_name in COPY_SPECS:
        source = root / relative_source
        _require(source)
        shutil.copyfile(source, destination / output_name)
        sources[output_name] = relative_source.as_posix()

    cumulative_source = root / CUMULATIVE_SOURCE
    cumulative_rows = _write_compact_cumulative(
        cumulative_source,
        destination / CUMULATIVE_OUTPUT,
    )
    sources[CUMULATIVE_OUTPUT] = CUMULATIVE_SOURCE.as_posix()

    output_names = sorted(sources)
    manifest = {
        "schema_version": 1,
        "canonical_candidate": "clock6_v4_multiden",
        "candidate_run_id": CANONICAL_RUN_ID,
        "control_role": "ablation baseline retained in figures 1 and 2",
        "cumulative_compact_columns": list(CUMULATIVE_COLUMNS),
        "cumulative_rows": cumulative_rows,
        "files": {
            name: {
                "source": sources[name],
                "sha256": _sha256(destination / name),
                "bytes": (destination / name).stat().st_size,
            }
            for name in output_names
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    destination = build_bundle()
    print(f"wrote four-figure evidence: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
