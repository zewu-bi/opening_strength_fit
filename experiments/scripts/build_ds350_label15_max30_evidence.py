from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
CASE_RUN_ID = "nn_ds350_w0931_0940_h1m_36m_grouped_gated_v2_mse_max30_v1"
BASELINE_RUN_ID = "nn_v6_w0931_0940_short1m_corrected_nextclose_36m_grouped_gated_v2_mse"
COMPARISON_RUN_ID = "ds350_max30_w0931_h1m_vs_v6_short1m_2022_2025_v1"
TOP1000_RUN_ID = "ds350_max30_w0931_h1m_top1000_acceptance_v1"

CASES = (
    "w0931_0940_h1m",
    "w0931_0940_h3m",
    "w0931_0940_h5m",
    "w0931_0940_h10m",
    "w0931_0940_h1h",
    "w0931_0940_hclose",
    "w1001_1010_h1m",
    "w1001_1010_h3m",
    "w1001_1010_h5m",
    "w1001_1010_h10m",
    "w1001_1010_h1h",
    "w1001_1010_hclose",
    "w1401_1410_h1m",
    "w1401_1410_h3m",
    "w1401_1410_h5m",
)
ARCHIVE_SOURCE = Path("output/artifacts/nn") / RUN_ID / "archive_source"
COMPARISON_DIR = Path("experiments/results/backtests") / COMPARISON_RUN_ID
CASE_BACKTEST_DIR = Path("experiments/results/backtests") / CASE_RUN_ID
TOP1000_DIR = Path("experiments/results/backtests") / TOP1000_RUN_ID
BUNDLE_DIR = Path("experiments/evidence/backtests") / RUN_ID

COPY_SPECS = (
    (
        COMPARISON_DIR / "optimization_directions_overlay_acceptance.svg",
        "01_signal_acceptance.svg",
    ),
    (
        COMPARISON_DIR / "optimization_directions_overlay_acceptance_plot_data.csv",
        "01_signal_acceptance.csv",
    ),
    (
        COMPARISON_DIR / "optimization_directions_net_alpha_cumulative.svg",
        "02_top100_cumulative.svg",
    ),
    (
        TOP1000_DIR / "top1000_rank/rank_bucket/top1000_bucket_returns.svg",
        "03_top1000_bucket_curve.svg",
    ),
    (
        TOP1000_DIR / "top1000_rank/rank_bucket/bucket_curve_plot_data.csv",
        "03_top1000_bucket_curve.csv",
    ),
    (
        TOP1000_DIR / "top1000_distribution/top1000_score_bucket_return_100bps_counts.svg",
        "04_top1000_return_distribution.svg",
    ),
    (
        TOP1000_DIR / "top1000_distribution/top1000_score_bucket_return_100bps_counts.csv",
        "04_top1000_return_distribution.csv",
    ),
    (
        TOP1000_DIR / "top1000_distribution/top1000_score_bucket_distribution_summary.csv",
        "top1000_distribution_summary.csv",
    ),
    (COMPARISON_DIR / "optimization_directions_trace.json", "trace_optimization.json"),
    (TOP1000_DIR / "top1000_rank/rank_bucket/trace.json", "trace_top1000_bucket.json"),
    (
        TOP1000_DIR / "top1000_distribution/trace.json",
        "trace_top1000_distribution.json",
    ),
    (CASE_BACKTEST_DIR / "pool_internal_summary.csv", "w0931_h1m_pool_internal_summary.csv"),
)
CUMULATIVE_SOURCE = COMPARISON_DIR / "optimization_directions_net_alpha_cumulative_plot_data.csv"
CUMULATIVE_COLUMNS = (
    "pool",
    "pool_label",
    "week_start",
    "variant",
    "next_cumulative_net_return_bps",
    "next_cumulative_alpha_bps",
)


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required max-30 evidence source is missing: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    _require(source)
    if source.suffix.lower() == ".svg":
        lines = source.read_text(encoding="utf-8").splitlines()
        destination.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n",
            encoding="utf-8",
        )
    else:
        shutil.copyfile(source, destination)


def _case_parts(case: str) -> tuple[str, str]:
    start, end, horizon = case.split("_", 2)
    start = start.removeprefix("w")
    return f"{start[:2]}:{start[2:]}-{end[:2]}:{end[2:]}", horizon.removeprefix("h")


def _training_fold_summary(source_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in CASES:
        window, horizon = _case_parts(case)
        for path in sorted((source_root / case).glob("month_*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            fold = path.parent.name.removeprefix("month_")
            stats = payload["train_stats_by_window"][fold]
            epoch_seconds = stats.get("training_epoch_seconds", [])
            reproducibility = payload["reproducibility"]
            rows.append(
                {
                    "case": case,
                    "window": window,
                    "horizon": horizon,
                    "test_fold": fold,
                    "epochs_trained": stats["epochs_trained"],
                    "best_epoch": stats["best_epoch"],
                    "train_loss": stats["train_loss"],
                    "validation_loss": stats["validation_loss"],
                    "train_rows": stats["rows"],
                    "features": stats["features"],
                    "gpu": stats["torch_device_name"],
                    "training_tensor_storage": stats["training_tensor_storage"],
                    "training_preparation_seconds": stats["training_preparation_seconds"],
                    "training_storage_transfer_seconds": stats["training_storage_transfer_seconds"],
                    "mean_epoch_seconds": (
                        sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None
                    ),
                    "config_sha256": reproducibility["config_sha256"],
                    "source_revision": reproducibility["source_revision"],
                    "feature_input": reproducibility["feature_input"],
                    "label_input": reproducibility["label_input"],
                }
            )
    frame = pd.DataFrame(rows)
    expected = len(CASES) * 8
    if len(frame) != expected:
        raise ValueError(f"expected {expected} training folds, found {len(frame)}")
    return frame


def _combined_case_csv(source_root: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case in CASES:
        window, horizon = _case_parts(case)
        if filename == "metrics_by_year.csv":
            paths = sorted((source_root / case).glob(f"month_*/{filename}"))
        else:
            paths = [source_root / case / "analysis/pool_internal_top100_horizon_v1" / filename]
        for path in paths:
            _require(path)
            frame = pd.read_csv(path)
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "window", window)
            frame.insert(0, "case", case)
            if filename == "metrics_by_year.csv":
                frame.insert(3, "test_fold", path.parent.name.removeprefix("month_"))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_compact_cumulative(source: Path, destination: Path) -> None:
    _require(source)
    frame = pd.read_csv(source)
    missing = sorted(set(CUMULATIVE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing cumulative columns: {missing}")
    frame.loc[:, list(CUMULATIVE_COLUMNS)].to_csv(destination, index=False)


def _matrix_table(pool_summary: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    universe = pool_summary.loc[pool_summary["pool"].eq("universe")].set_index("case")
    pool_l = pool_summary.loc[pool_summary["pool"].eq("pool_L")].set_index("case")
    epoch_mean = folds.groupby("case", observed=True)["best_epoch"].mean()
    rows = []
    for case in CASES:
        window, horizon = _case_parts(case)
        rows.append(
            {
                "case": case,
                "window": window,
                "horizon": horizon,
                "universe_short_rank_ic": universe.loc[case, "short_rank_ic"],
                "pool_L_short_excess_bps": pool_l.loc[case, "short_internal_excess_bps"],
                "pool_L_next_excess_bps": pool_l.loc[case, "next_internal_excess_bps"],
                "pool_L_next_rank_ic": pool_l.loc[case, "next_rank_ic"],
                "mean_best_epoch": epoch_mean.loc[case],
            }
        )
    return pd.DataFrame(rows)


def _write_readme(destination: Path, matrix: pd.DataFrame, folds: pd.DataFrame) -> None:
    display = matrix.copy().rename(
        columns={
            "universe_short_rank_ic": "universe short IC",
            "pool_L_short_excess_bps": "pool_L short excess bps",
            "pool_L_next_excess_bps": "pool_L next excess bps",
            "pool_L_next_rank_ic": "pool_L next IC",
            "mean_best_epoch": "mean best epoch",
        }
    )
    for column in (
        "universe short IC",
        "pool_L short excess bps",
        "pool_L next excess bps",
        "pool_L next IC",
        "mean best epoch",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    columns = list(display.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    table = "\n".join(table_lines)
    best_epochs = pd.to_numeric(folds["best_epoch"])
    readme = f"""# ds350 max-30 15-label result archive

This is the authoritative compact archive for `{RUN_ID}`. All 15 matrix cases and all
120 rolling OOS folds completed. The mean best epoch is `{best_epochs.mean():.2f}`;
`{int((best_epochs > 10).sum())}/120` folds selected an epoch after 10 and
`{int((best_epochs == 30).sum())}/120` selected epoch 30.

The standard four figures below focus on `09:31-09:40 / 1m`. Figures 1-2 compare the
authoritative max-30 result with the matching prior v6 1m / 10-epoch run. Figure 2 applies an
8 bps realized fee and shows next-close economic follow-through relative to each run's matching
`pool_L`. Figures 3-4 diagnose the current max-30 score head over Top1000 only.

![Signal acceptance](01_signal_acceptance.svg)

![Top100 cumulative](02_top100_cumulative.svg)

![Top1000 bucket curve](03_top1000_bucket_curve.svg)

![Top1000 return distribution](04_top1000_return_distribution.svg)

## Fifteen-label matrix

{table}

The aggregate training, OOS, and pool-internal sources are retained as compact CSVs. Large
predictions, model binaries, and row-level labels remain on the PVC and are addressed by the run
config, source revision, input lineage, and hashes in `manifest.json`.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")


def build_bundle(root: Path = ROOT) -> Path:
    source_root = root / ARCHIVE_SOURCE
    destination = root / BUNDLE_DIR
    destination.mkdir(parents=True, exist_ok=True)

    sources: dict[str, str] = {}
    for relative_source, output_name in COPY_SPECS:
        source = root / relative_source
        _copy(source, destination / output_name)
        sources[output_name] = relative_source.as_posix()

    _write_compact_cumulative(root / CUMULATIVE_SOURCE, destination / "02_top100_cumulative.csv")
    sources["02_top100_cumulative.csv"] = CUMULATIVE_SOURCE.as_posix()

    folds = _training_fold_summary(source_root)
    oos = _combined_case_csv(source_root, "metrics_by_year.csv")
    pool = _combined_case_csv(source_root, "pool_internal_summary.csv")
    halfyear = _combined_case_csv(source_root, "pool_internal_halfyear_summary.csv")
    quarter = _combined_case_csv(source_root, "pool_internal_quarter_summary.csv")
    matrix = _matrix_table(pool, folds)
    generated = {
        "training_fold_summary.csv": folds,
        "oos_metrics_by_fold.csv": oos,
        "pool_internal_summary.csv": pool,
        "pool_internal_halfyear_summary.csv": halfyear,
        "pool_internal_quarter_summary.csv": quarter,
        "matrix_summary.csv": matrix,
    }
    for name, frame in generated.items():
        frame.to_csv(destination / name, index=False)
        sources[name] = f"generated from {ARCHIVE_SOURCE.as_posix()}"

    _write_readme(destination, matrix, folds)
    sources["README.md"] = "generated archive guide"
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "completed",
        "cases": len(CASES),
        "training_folds": len(folds),
        "baseline_run_id": BASELINE_RUN_ID,
        "four_figure_case": "w0931_0940_h1m",
        "four_figure_comparison_run_id": COMPARISON_RUN_ID,
        "files": {
            name: {
                "source": sources[name],
                "sha256": _sha256(destination / name),
                "bytes": (destination / name).stat().st_size,
            }
            for name in sorted(sources)
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    destination = build_bundle()
    print(f"wrote ds350 max-30 evidence: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
