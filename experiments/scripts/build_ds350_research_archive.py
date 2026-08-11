from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAX10_RUN_ID = "nn_ds350_label12_36m_grouped_gated_v2_mse_v1"
MAX10_ARCHIVE_ID = MAX10_RUN_ID
MAX30_RUN_ID = "nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
DIAGNOSTIC_ARCHIVE_ID = "ds350_long_label_2026h1_diagnostics_v1"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_parts(case: str) -> tuple[str, str]:
    start, end, horizon = case.split("_", 2)
    start = start.removeprefix("w")
    window = f"{start[:2]}:{start[2:]}-{end[:2]}:{end[2:]}"
    return window, horizon.removeprefix("h")


def _markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _write_manifest(destination: Path, archive_id: str, metadata: dict[str, object]) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(destination).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    payload = {
        "schema_version": 1,
        "archive_id": archive_id,
        **metadata,
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _training_fold_summary(source_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in CASES:
        window, label = _case_parts(case)
        for path in sorted((source_root / case).glob("month_*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            fold = path.parent.name.removeprefix("month_")
            stats = payload["train_stats_by_window"][fold]
            reproducibility = payload["reproducibility"]
            rows.append(
                {
                    "case": case,
                    "window": window,
                    "label": label,
                    "test_fold": fold,
                    "epochs_trained": stats["epochs_trained"],
                    "best_epoch": stats["best_epoch"],
                    "train_loss": stats["train_loss"],
                    "validation_loss": stats["validation_loss"],
                    "train_rows": stats["rows"],
                    "features": stats["features"],
                    "gpu": stats["torch_device_name"],
                    "config_sha256": reproducibility["config_sha256"],
                    "source_revision": reproducibility["source_revision"],
                    "feature_input": reproducibility["feature_input"],
                    "label_input": reproducibility["label_input"],
                }
            )
    result = pd.DataFrame(rows)
    expected = len(CASES) * 8
    if len(result) != expected:
        raise ValueError(f"expected {expected} max-10 folds, found {len(result)}")
    return result


def _combine_case_files(source_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for case in CASES:
        window, label = _case_parts(case)
        if filename == "metrics_by_year.csv":
            paths = sorted((source_root / case).glob(f"month_*/{filename}"))
        else:
            paths = [source_root / case / "analysis/pool_internal_top100_horizon_v1" / filename]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame.insert(0, "label", label)
            frame.insert(0, "window", window)
            frame.insert(0, "case", case)
            if filename == "metrics_by_year.csv":
                frame.insert(3, "test_fold", path.parent.name.removeprefix("month_"))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _matrix(pool_summary: pd.DataFrame, *, quarter_equal: bool) -> pd.DataFrame:
    source = pool_summary
    if quarter_equal:
        numeric = source.select_dtypes(include="number").columns
        source = source.groupby(["case", "window", "label", "pool"], as_index=False)[
            list(numeric)
        ].mean()
    universe = source.loc[source["pool"].eq("universe")].set_index("case")
    pool_l = source.loc[source["pool"].eq("pool_L")].set_index("case")
    rows = []
    for case in CASES:
        window, label = _case_parts(case)
        rows.append(
            {
                "case": case,
                "window": window,
                "label": label,
                "short_ic": universe.loc[case, "short_rank_ic"],
                "short_excess_bps": pool_l.loc[case, "short_internal_excess_bps"],
                "overnight_excess_bps": pool_l.loc[case, "next_internal_excess_bps"],
            }
        )
    return pd.DataFrame(rows)


def _build_max10(source: Path, repo_root: Path) -> Path:
    source_root = source / "nn" / MAX10_RUN_ID
    destination = repo_root / "experiments/evidence/backtests" / MAX10_ARCHIVE_ID
    destination.mkdir(parents=True, exist_ok=True)

    folds = _training_fold_summary(source_root)
    oos = _combine_case_files(source_root, "metrics_by_year.csv")
    pooled = _combine_case_files(source_root, "pool_internal_summary.csv")
    halfyear = _combine_case_files(source_root, "pool_internal_halfyear_summary.csv")
    quarter = _combine_case_files(source_root, "pool_internal_quarter_summary.csv")
    pooled_matrix = _matrix(pooled, quarter_equal=False)
    quarter_matrix = _matrix(quarter, quarter_equal=True)
    outputs = {
        "training_fold_summary.csv": folds,
        "oos_metrics_by_fold.csv": oos,
        "pool_internal_summary.csv": pooled,
        "pool_internal_halfyear_summary.csv": halfyear,
        "pool_internal_quarter_summary.csv": quarter,
        "matrix_group_pooled.csv": pooled_matrix,
        "matrix_quarter_equal.csv": quarter_matrix,
    }
    for name, frame in outputs.items():
        frame.to_csv(destination / name, index=False)

    trace_dir = destination / "traces"
    trace_dir.mkdir(exist_ok=True)
    for case in CASES:
        source_trace = (
            source_root / case / "analysis/pool_internal_top100_horizon_v1/pool_internal_trace.json"
        )
        shutil.copy2(source_trace, trace_dir / f"{case}.json")

    max30_quarter = pd.read_csv(
        repo_root
        / "experiments/evidence/backtests"
        / MAX30_RUN_ID
        / "pool_internal_quarter_summary.csv"
    )
    max30_quarter = max30_quarter.rename(columns={"horizon": "label"})
    max30_matrix = _matrix(max30_quarter, quarter_equal=True)
    comparison = quarter_matrix.merge(
        max30_matrix,
        on=["case", "window", "label"],
        suffixes=("_max10", "_max30"),
    )
    for metric in ("short_ic", "short_excess_bps", "overnight_excess_bps"):
        comparison[f"{metric}_max30_minus_max10"] = (
            comparison[f"{metric}_max30"] - comparison[f"{metric}_max10"]
        )
    comparison.to_csv(destination / "max10_vs_max30_quarter_equal.csv", index=False)

    report = quarter_matrix.loc[
        quarter_matrix["window"].isin(["09:31-09:40", "10:01-10:10"])
        & ~quarter_matrix["label"].eq("5m"),
        ["window", "label", "short_ic", "short_excess_bps", "overnight_excess_bps"],
    ]
    readme = f"""# ds350 historical max-10 archive

This bundle preserves the historical `{MAX10_RUN_ID}` result that produced the reported
`09:31-09:40 / close` quarter-equal values **short IC 0.03593, short excess 24.87 bps,
overnight excess 25.66 bps**. It is a completed 15-case × 8-fold rolling-OOS experiment,
but its model-selection budget is superseded by the separately archived max-30 run.

The distinction matters: 24.87 is not the max-30 result (20.85 bps under the later limit audit),
and neither number should be mixed with the 2026H1 strict holdout result.

## Historical report table (quarter-equal)

{_markdown_table(report, 5)}

`matrix_group_pooled.csv` and `matrix_quarter_equal.csv` retain both aggregation conventions.
`max10_vs_max30_quarter_equal.csv` is an explicit budget comparison. Fold metrics, year metrics,
half-year/quarter summaries and all 15 analysis traces are included; row-level predictions and
model binaries remain on the PVC.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    _write_manifest(
        destination,
        MAX10_ARCHIVE_ID,
        {
            "status": "superseded",
            "source_run_id": MAX10_RUN_ID,
            "superseded_by": MAX30_RUN_ID,
            "cases": len(CASES),
            "rolling_oos_folds": len(folds),
            "aggregation_note": "reported 24.87 bps is the equal mean of quarterly summaries",
        },
    )
    return destination


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _normalize_text_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _job_inventory(source: Path) -> pd.DataFrame:
    payload = json.loads((source / "k8s/jobs.json").read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        if "ds350" not in name and "stock-pool-object" not in name:
            continue
        status = item.get("status", {})
        conditions = status.get("conditions", [])
        failed = status.get("failed", 0) or 0
        succeeded = status.get("succeeded", 0) or 0
        if succeeded:
            state = "completed"
        elif failed or any(
            c.get("type") == "Failed" and c.get("status") == "True" for c in conditions
        ):
            state = "failed"
        else:
            state = "active_or_unknown"
        rows.append(
            {
                "job": name,
                "state": state,
                "succeeded_pods": succeeded,
                "failed_pods": failed,
                "start_time_utc": status.get("startTime"),
                "completion_time_utc": status.get("completionTime"),
                "completed_indexes": status.get("completedIndexes"),
            }
        )
    return pd.DataFrame(rows).sort_values("job").reset_index(drop=True)


def _model_comparison(source: Path) -> pd.DataFrame:
    path = source / "audits/ds350_2026h1_model_tradeability_compare_v1/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for model, result in payload["models"].items():
        if not isinstance(result, dict):
            continue
        performance = result["performance"]
        entry = result["selected_final_limit_tradeability"]
        rows.append(
            {
                "model": model,
                "own_label_rank_ic": performance["own_label_rank_ic"],
                "same_day_close_rank_ic": performance["same_day_close_rank_ic"],
                "same_day_close_excess_bps": performance["same_day_close_excess_bps"],
                "selected_limit_share_pct": performance["selected_limit_share_pct"],
                "limit_contribution_bps": performance["same_day_close_limit_contribution_bps"],
                "nonlimit_contribution_bps": performance[
                    "same_day_close_nonlimit_contribution_bps"
                ],
                "no_limit_reselect_excess_bps": performance[
                    "same_day_close_no_limit_reselect_excess_bps"
                ],
                "close_to_next_open_excess_bps": performance["close_to_next_open_excess_bps"],
                "next_open_to_next_close_excess_bps": performance[
                    "next_open_to_next_close_excess_bps"
                ],
                "close_to_next_close_excess_bps": performance["close_to_next_close_excess_bps"],
                "entry_return_median_pct": entry["entry_return_vs_prev_close_median_pct"],
                "entry_room_to_limit_median_pct": entry["entry_room_to_limit_median_pct"],
                "entry_ask1_notional_p10_cny": entry["entry_ask_notional_p10"],
                "entry_ask1_notional_median_cny": entry["entry_ask_notional_median"],
            }
        )
    return pd.DataFrame(rows)


def _loss_comparison(source: Path) -> pd.DataFrame:
    path = source / "audits/ds350_2026h1_close_loss_compare_v1/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for loss, result in payload["runs"].items():
        performance = result["performance"]
        rows.append(
            {
                "loss": loss,
                "rank_ic": performance["own_label_rank_ic"],
                "same_day_close_excess_bps": performance["same_day_close_excess_bps"],
                "selected_limit_share_pct": performance["selected_limit_share_pct"],
                "limit_contribution_bps": performance["same_day_close_limit_contribution_bps"],
                "nonlimit_contribution_bps": performance[
                    "same_day_close_nonlimit_contribution_bps"
                ],
                "no_limit_reselect_excess_bps": performance[
                    "same_day_close_no_limit_reselect_excess_bps"
                ],
                "close_to_next_close_excess_bps": performance["close_to_next_close_excess_bps"],
                "top100_overlap_with_mse_pct": payload["top100_overlap_with_mse_pct"][loss],
            }
        )
    return pd.DataFrame(rows)


def _capacity_summary(source: Path) -> pd.DataFrame:
    path = source / "audits/ds350_2026h1_top100_capacity_v2/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for model, result in payload["models"].items():
        all_segment = result["top100_segments"]["all"]
        fills = result["fixed_top100_fill_ratio"]
        limit_fill = result["eventual_limit_share_of_fillable_notional_pct"]
        rows.append(
            {
                "model": model,
                "ask1_notional_p10_cny": all_segment["ask1_notional_cny"]["p10"],
                "ask1_notional_median_cny": all_segment["ask1_notional_cny"]["median"],
                "ask10_notional_median_cny": all_segment["ask10_notional_cny"]["median"],
                "fixed_top100_fill_ask1_plus_turnover_pct": fills["plus_25pct_ask1"]["mean_pct"],
                "fixed_top100_fill_ask10_plus_turnover_pct": fills["plus_25pct_ask10"]["mean_pct"],
                "old_turnover_only_fill_pct": fills["old_turnover_rule_without_depth"]["mean_pct"],
                "eventual_limit_fill_share_ask1_pct": limit_fill["ask1_basis"],
                "eventual_limit_fill_share_ask10_pct": limit_fill["ask10_basis"],
            }
        )
    return pd.DataFrame(rows)


def _experiment_index(destination: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((destination / "raw").iterdir()):
        if not path.is_dir():
            continue
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        rows.append(
            {
                "section": path.name,
                "files": len(files),
                "bytes": sum(candidate.stat().st_size for candidate in files),
                "archive_path": path.relative_to(destination).as_posix(),
            }
        )
    return pd.DataFrame(rows)


def _build_diagnostics(source: Path, repo_root: Path) -> Path:
    destination = repo_root / "experiments/evidence/backtests" / DIAGNOSTIC_ARCHIVE_ID
    raw = destination / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    raw.mkdir(exist_ok=True)
    _copy_tree(source / "audits", raw / "audits")
    _copy_tree(
        source / "nn" / MAX30_RUN_ID / "analysis",
        raw / "max30_label_analyses",
    )
    _copy_tree(source / "runs/analyses/ds350_future_info_v1", raw / "future_info")
    _copy_tree(source / "nn/holdout", raw / "holdout_models")
    _copy_tree(source / "k8s/logs", raw / "k8s_logs")
    _normalize_text_tree(raw / "k8s_logs")

    jobs = _job_inventory(source)
    models = _model_comparison(source)
    losses = _loss_comparison(source)
    capacity = _capacity_summary(source)
    jobs.to_csv(destination / "k8s_job_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    models.to_csv(destination / "strict_2026h1_model_comparison.csv", index=False)
    losses.to_csv(destination / "strict_2026h1_loss_comparison.csv", index=False)
    capacity.to_csv(destination / "strict_2026h1_capacity_summary.csv", index=False)

    index = _experiment_index(destination)
    index.to_csv(destination / "experiment_index.csv", index=False)
    dataset = json.loads((raw / "audits/ds350_2026_holdout_dataset_v1/summary.json").read_text())
    readme = f"""# ds350 long-label and strict-2026H1 diagnostic archive

This compact archive closes the unarchived ds350 investigation chain through 2026-08-11. It
contains original summaries/traces, compact holdout metrics, and Kubernetes logs. Large row-level
predictions, labels, and model binaries remain on the PVC. Failed/superseded attempts are retained
as provenance: `top100_capacity_v1` has a turnover-unit bug and is superseded by v2; limit-nextclose
v3 is the corrected contribution decomposition; failed Top1000 v1/v2 jobs are visible in the job
inventory and v3 is the completed result.

## Strict 2026H1 dataset boundary

The strict holdout uses ClickHouse `stock.tick` plus `stock.daily_bar_jy`, trains on 2023-2025,
purges one session, and evaluates 110 days / 1,100 decision groups in 2026H1. The generated 2026
feature and label schemas match 2025, key order matches, and duplicate feature keys are zero.
There are {dataset["rows_2026"]["features"]:,} rows over {dataset["h1"]["sample_dates"]} tick dates.
Three missing tick dates and their predecessor dates are explicitly recorded in the dataset audit.
Pool L ends at 2025-12-31, so no strict 2026H1 Pool-L result exists; 2026H1 tables are all-A.

## Strict 2026H1 model comparison

{_markdown_table(models, 4)}

The close model's 36.51 bps is a different sample from the historical rolling-OOS max-10 24.87
bps and max-30 20.85 bps. Its 36.63 bps limit contribution and -0.12 bps non-limit contribution
show that the paper result is almost entirely a final-limit tail effect. The causal-selection,
purge-one-session and missing-return-as-zero audits in `raw/future_info/` do not make the historical
signal disappear, but retrained scores are not bit-identical to the original chain; the trace files
preserve that qualification.

## Loss-only comparison

{_markdown_table(losses, 4)}

Huber raises rank IC and reduces same-day tail dependence, but it does not establish a deployable
replacement by itself. The data/features/split/architecture/seed are held fixed in this comparison.

## Capacity check (corrected v2 units)

{_markdown_table(capacity, 2)}

The earlier turnover-only convention substantially overstates capacity. With no rank-101 refill,
25% displayed-depth participation, a 500k CNY/name cap and turnover participation, mean fixed-Top100
fill is only about 17.22%/14.60% on ask1 for the 1m/close models. This is an execution-risk result,
not evidence that the causal label or OOS split is invalid.

## Contents

{_markdown_table(index, 0)}

`strict_2026h1_*` files are presentation tables derived directly from the archived raw JSON.
`k8s_job_inventory.csv` records the cluster state observed at archive time. `manifest.json` hashes
every retained file.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    _write_manifest(
        destination,
        DIAGNOSTIC_ARCHIVE_ID,
        {
            "status": "completed",
            "archived_at_utc": "2026-08-11",
            "scope": "ds350 long-label causal, limit-tail, OOS, loss, and capacity diagnostics",
            "excluded": "project-governance work; large reconstructable PVC artifacts",
            "k8s_jobs_recorded": len(jobs),
        },
    )
    return destination


def build_archives(source: Path, repo_root: Path = ROOT) -> tuple[Path, Path]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    max10 = _build_max10(source, repo_root)
    diagnostics = _build_diagnostics(source, repo_root)
    return max10, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact ds350 research archives")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    for path in build_archives(args.source_root.resolve(), args.repo_root.resolve()):
        print(path.relative_to(args.repo_root.resolve()))


if __name__ == "__main__":
    main()
