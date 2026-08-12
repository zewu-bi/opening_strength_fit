from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from opening_strength_fit.artifact_catalog import (
    CUMULATIVE_EVIDENCE_COLUMNS as CUMULATIVE_COLUMNS,
)
from opening_strength_fit.artifact_catalog import (
    artifact_file_manifest,
    copy_artifact_specs,
    copy_csv_columns,
    four_figure_artifact_specs,
    require_file,
)
from opening_strength_fit.io import write_json

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
    *four_figure_artifact_specs(
        COMPARISON_DIR,
        TOP1000_DIR / "top1000_rank/rank_bucket",
        TOP1000_DIR / "top1000_distribution",
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
            require_file(path)
            frame = pd.read_csv(path)
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "window", window)
            frame.insert(0, "case", case)
            if filename == "metrics_by_year.csv":
                frame.insert(3, "test_fold", path.parent.name.removeprefix("month_"))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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


def build_bundle(root: Path = ROOT) -> Path:
    source_root = root / ARCHIVE_SOURCE
    destination = root / BUNDLE_DIR
    destination.mkdir(parents=True, exist_ok=True)

    sources = copy_artifact_specs(root, destination, COPY_SPECS)

    copy_csv_columns(
        root / CUMULATIVE_SOURCE,
        destination / "02_top100_cumulative.csv",
        CUMULATIVE_COLUMNS,
    )
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

    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "completed",
        "cases": len(CASES),
        "training_folds": len(folds),
        "baseline_run_id": BASELINE_RUN_ID,
        "four_figure_case": "w0931_0940_h1m",
        "four_figure_comparison_run_id": COMPARISON_RUN_ID,
        "files": artifact_file_manifest(destination, sources),
    }
    write_json(destination / "manifest.json", manifest, sort_keys=True)
    return destination


def main() -> None:
    destination = build_bundle()
    print(f"wrote ds350 max-30 evidence: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
