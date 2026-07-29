from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opening_strength_fit.config import (
    config_clock_list,
    config_float,
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import write_frame_atomic, write_json
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.stock_pool import load_stock_pool
from opening_strength_fit.temporal_analysis import (
    FEATURE_COLUMNS,
    SEQUENCE_SCHEMA_VERSION,
    TARGET_COLUMN,
    analyze_day_sequence,
    assemble_day_sequence,
    summarize_temporal_metrics,
    write_sequence_npz,
)

_WORKER_LABELS: pd.DataFrame | None = None
_WORKER_POOL: pd.DataFrame | None = None
_WORKER_CLOCKS: list[str] = []
_WORKER_SEQUENCE_ROOT: Path | None = None
_WORKER_METRIC_ROOT: Path | None = None
_WORKER_TOP_N = 100
_WORKER_TAIL_FRACTION = 0.1


def _pool_symbols(date: str) -> set[str]:
    if _WORKER_POOL is None or date not in _WORKER_POOL.index:
        return set()
    row = _WORKER_POOL.loc[date]
    return set(row.index[row.to_numpy(dtype=bool, copy=False)].astype(str))


def _process_day(feature_path_raw: str) -> dict[str, object]:
    if _WORKER_LABELS is None:
        raise RuntimeError("temporal analysis worker labels are not initialized")
    if _WORKER_SEQUENCE_ROOT is None or _WORKER_METRIC_ROOT is None:
        raise RuntimeError("temporal analysis worker output roots are not initialized")
    feature_path = Path(feature_path_raw)
    date = feature_path.parent.name.removeprefix("date=")
    year = date[:4]
    sequence_path = _WORKER_SEQUENCE_ROOT / f"year={year}" / f"date={date}" / "sequence.npz"
    metric_path = _WORKER_METRIC_ROOT / f"year={year}" / f"date={date}.parquet"
    if sequence_path.exists():
        with np.load(sequence_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    else:
        features = pd.read_parquet(
            feature_path,
            columns=[
                "symbol",
                "decision_target_timestamp",
                *FEATURE_COLUMNS,
            ],
        )
        labels = _WORKER_LABELS.loc[
            _WORKER_LABELS["date"].astype(str).eq(date),
            ["symbol", TARGET_COLUMN],
        ]
        arrays = assemble_day_sequence(
            features,
            labels,
            clocks=_WORKER_CLOCKS,
            pool_symbols=_pool_symbols(date),
        )
        write_sequence_npz(sequence_path, arrays)

    if metric_path.exists():
        metrics = pd.read_parquet(metric_path)
    else:
        metrics = analyze_day_sequence(
            arrays,
            date=date,
            top_n=_WORKER_TOP_N,
            tail_fraction=_WORKER_TAIL_FRACTION,
        )
        write_frame_atomic(metrics, metric_path)
    return {
        "date": date,
        "symbols": int(len(arrays["symbols"])),
        "target_valid": int(np.isfinite(arrays["target"]).sum()),
        "pool_members": int(arrays["pool_member"].sum()),
        "metric_rows": int(len(metrics)),
        "sequence_path": str(sequence_path),
        "metric_path": str(metric_path),
    }


def _plot_curves(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    panels = [
        ("all_a", "mean_rank_ic", "All-A daily Rank IC"),
        ("pool_l", "mean_rank_ic", "Pool-L daily Rank IC"),
        ("all_a", "mean_top_n_excess", "All-A top-100 excess"),
        ("pool_l", "mean_top_n_excess", "Pool-L top-100 excess"),
    ]
    colors = {"1m": "#277da1", "10m": "#f8961e", "60m": "#43aa8b"}
    for axis, (universe, metric, title) in zip(axes.flat, panels, strict=True):
        subset = summary.loc[summary["universe"].eq(universe)]
        for horizon, group in subset.groupby("horizon", observed=True, sort=False):
            values = group.sort_values("clock_seconds")
            y = values[metric].to_numpy(dtype=float)
            if "return" in metric or "excess" in metric or "spread" in metric:
                y = y * 10_000.0
            axis.plot(
                values["clock_seconds"],
                y,
                label=str(horizon),
                color=colors.get(str(horizon)),
                linewidth=1.5,
            )
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    tick_seconds = [
        9 * 3600 + 30 * 60,
        10 * 3600 + 30 * 60,
        11 * 3600 + 30 * 60,
        13 * 3600,
        14 * 3600,
        15 * 3600,
    ]
    tick_labels = ["09:30", "10:30", "11:30", "13:00", "14:00", "15:00"]
    for axis in axes.flat:
        axis.set_xticks(tick_seconds, tick_labels)
        axis.legend(frameon=False, ncol=3)
    axes[1, 0].set_ylabel("bps")
    axes[1, 1].set_ylabel("bps")
    figure.suptitle("Intraday realized-return paths vs adjusted D→D+1 close return")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_year_labels(label_root: Path, year: int) -> pd.DataFrame:
    path = label_root / f"year={year}" / "labels.parquet"
    if not path.exists():
        raise SystemExit(f"missing daily label shard: {path}")
    labels = pd.read_parquet(path, columns=["date", "symbol", TARGET_COLUMN])
    labels["date"] = labels["date"].astype(str)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full-day return paths without fitting a model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_toml(args.config)
    feature_root = Path(config_str(config, "temporal_analysis", "feature_root", ""))
    label_root = Path(config_str(config, "temporal_analysis", "label_root", ""))
    sequence_root = Path(config_str(config, "temporal_analysis", "sequence_root", ""))
    if not feature_root or not label_root or not sequence_root:
        raise SystemExit("[temporal_analysis] requires feature_root, label_root, and sequence_root")
    start_date = config_str(config, "temporal_analysis", "start_date", "2019-01-02")
    end_date = config_str(config, "temporal_analysis", "end_date", "2025-12-31")
    clocks = config_clock_list(
        config,
        "temporal_analysis",
        "decision_ranges",
        ["09:30:00..11:29:00", "13:00:00..14:59:00"],
    )
    workers = config_int(
        config,
        "temporal_analysis",
        "workers",
        max(1, min(8, os.cpu_count() or 1)),
    )
    top_n = config_int(config, "temporal_analysis", "top_n", 100)
    tail_fraction = config_float(config, "temporal_analysis", "tail_fraction", 0.1)
    pool_path = config_str(
        config,
        "stock_pool",
        "path",
        "lml.bzw@ssd/data/pool_L.parquet",
    )
    output_dir = Path(args.output_dir)
    metric_root = output_dir / "daily_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    pool = load_stock_pool(pool_path)
    pool = pool.loc[(pool.index >= start_date) & (pool.index <= end_date)]
    date_summaries: list[dict[str, object]] = []
    for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
        paths = sorted((feature_root / f"year={year}").glob("date=*/labels.parquet"))
        paths = [
            path
            for path in paths
            if start_date <= path.parent.name.removeprefix("date=") <= end_date
        ]
        if not paths:
            continue
        labels = _load_year_labels(label_root, year)
        global _WORKER_LABELS, _WORKER_POOL, _WORKER_CLOCKS
        global _WORKER_SEQUENCE_ROOT, _WORKER_METRIC_ROOT
        global _WORKER_TOP_N, _WORKER_TAIL_FRACTION
        _WORKER_LABELS = labels
        _WORKER_POOL = pool
        _WORKER_CLOCKS = clocks
        _WORKER_SEQUENCE_ROOT = sequence_root
        _WORKER_METRIC_ROOT = metric_root
        _WORKER_TOP_N = top_n
        _WORKER_TAIL_FRACTION = tail_fraction
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_process_day, str(path)): path for path in paths}
            for completed, future in enumerate(as_completed(futures), start=1):
                summary = future.result()
                date_summaries.append(summary)
                if completed == 1 or completed % 10 == 0 or completed == len(futures):
                    print(
                        f"year={year} completed={completed}/{len(futures)} date={summary['date']}",
                        flush=True,
                    )

    metric_paths = sorted(metric_root.glob("year=*/date=*.parquet"))
    if not metric_paths:
        raise SystemExit("temporal analysis produced no daily metric shards")
    daily = pd.concat((pd.read_parquet(path) for path in metric_paths), ignore_index=True)
    overall, annual = summarize_temporal_metrics(daily)
    write_frame_atomic(daily, output_dir / "daily_clock_metrics.parquet")
    overall.to_csv(output_dir / "clock_summary.csv", index=False)
    annual.to_csv(output_dir / "clock_summary_by_year.csv", index=False)
    _plot_curves(overall, output_dir / "clock_curves.png")

    sequence_manifest = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "run_id": run_id(config, args.config),
        "feature_root": str(feature_root),
        "label_root": str(label_root),
        "start_date": start_date,
        "end_date": end_date,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "clocks": clocks,
        "days": len(date_summaries),
        "sequence_files": len(list(sequence_root.glob("year=*/date=*/sequence.npz"))),
        "storage": "uncompressed_npz_float16_raw_and_cross_section_rank_nan_mask",
    }
    write_json(sequence_root / "manifest.json", sequence_manifest, atomic=True)
    (sequence_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    trace = {
        **sequence_manifest,
        "workers": workers,
        "top_n": top_n,
        "tail_fraction": tail_fraction,
        "metric_rows": int(len(daily)),
        "summary_rows": int(len(overall)),
        "annual_summary_rows": int(len(annual)),
        "date_summaries": sorted(date_summaries, key=lambda item: str(item["date"])),
    }
    write_json(output_dir / "temporal_analysis_trace.json", trace, atomic=True)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    print_mapping(
        "temporal_analysis",
        {
            "days": len(date_summaries),
            "daily_metric_rows": len(daily),
            "summary_rows": len(overall),
            "sequence_root": str(sequence_root),
            "output_dir": str(output_dir),
        },
    )
