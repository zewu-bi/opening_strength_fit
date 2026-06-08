from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL, write_json
from opening_strength_fit.io import read_frame
from opening_strength_fit.pool_internal_plots import (
    slug_label,
    write_universe_sml_pool_internal_plots,
)
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

DEFAULT_POOLS = ("universe", "S", "M", "L")
GROUP_COLS = ("date", "decision_target_timestamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TopN pool-internal short/next-close excess for synced prediction shards."
        )
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help=(
            "Prediction parquet/csv file or directory. May be repeated. If a "
            "directory contains raw/predictions_*.parquet, those shards are used."
        ),
    )
    parser.add_argument(
        "--next-close-label-input",
        required=True,
        help="Next-close label parquet file or directory containing yearly label shards.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--short-label-col", default="label")
    parser.add_argument("--next-label-col", default=NEXT_CLOSE_LABEL_COL)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--report-dir",
        default="",
        help=(
            "Optional report directory. When set, writes the universe/S/M/L "
            "pool-internal excess and Rank IC SVG panels plus plot data."
        ),
    )
    parser.add_argument(
        "--plot-prefix",
        default="",
        help=(
            "Filename/directory prefix for generated report plots. Defaults to "
            "--variant when present, otherwise --run-id."
        ),
    )
    parser.add_argument(
        "--plot-variant-label",
        default="",
        help="Display label used in generated report plot titles. Defaults to --variant.",
    )
    parser.add_argument(
        "--pool",
        action="append",
        choices=["universe", "S", "M", "L"],
        help="Pools to evaluate. Defaults to universe, S, M, and L.",
    )
    parser.add_argument("--pool-date-lag-sessions", type=int, default=0)
    return parser.parse_args()


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"prediction input does not exist: {path}")
    raw_dir = path / "raw"
    if raw_dir.exists():
        files = sorted(raw_dir.glob("predictions_*.parquet"))
        if files:
            return files
    combined = path / "predictions_all.parquet"
    if combined.exists():
        return [combined]
    files = sorted(path.glob("predictions_*.parquet"))
    if files:
        return files
    raise SystemExit(f"no prediction parquet files found under: {path}")


def next_close_files(path: Path, years: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"next-close label input does not exist: {path}")
    files = sorted(path.glob("*.parquet"))
    if years:
        matched = [file for file in files if any(year in file.name for year in sorted(years))]
        if matched:
            return matched
    if files:
        return files
    raise SystemExit(f"no next-close parquet files found under: {path}")


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    return out


def read_predictions(paths: list[str], *, score_col: str, short_label_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col, short_label_col]
    files = [file for raw in paths for file in prediction_files(Path(raw))]
    frames = [read_frame(file, columns=required) for file in files]
    if not frames:
        raise SystemExit("no prediction files supplied")
    return normalize_keys(pd.concat(frames, ignore_index=True))


def read_next_close_labels(path: str, *, years: set[str], next_label_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, next_label_col]
    files = next_close_files(Path(path), years)
    frames = [read_frame(file, columns=required) for file in files]
    if not frames:
        raise SystemExit("no next-close label files supplied")
    labels = normalize_keys(pd.concat(frames, ignore_index=True))
    labels = labels.dropna(subset=list(KEY_COLUMNS) + [next_label_col])
    return labels.drop_duplicates(list(KEY_COLUMNS), keep="last")


def finite_corr(left: pd.Series, right: pd.Series, *, method: str = "spearman") -> float:
    values = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if len(values) < 2:
        return float("nan")
    if values["left"].nunique(dropna=True) < 2 or values["right"].nunique(dropna=True) < 2:
        return float("nan")
    return float(values["left"].corr(values["right"], method=method))


def clock_label(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    return timestamps.dt.strftime("%H:%M")


def evaluate_pool(
    frame: pd.DataFrame,
    *,
    pool_name: str,
    score_col: str,
    short_label_col: str,
    next_label_col: str,
    top_n: int,
) -> pd.DataFrame:
    work = frame.dropna(subset=[score_col, short_label_col, next_label_col]).copy()
    if work.empty:
        return pd.DataFrame()
    work["_score_rank"] = work.groupby(list(GROUP_COLS), sort=False)[score_col].rank(
        ascending=False,
        method="first",
    )
    work["_selected"] = work["_score_rank"].le(top_n)

    group = work.groupby(list(GROUP_COLS), sort=False)
    base = group.agg(
        candidate_rows=(score_col, "size"),
        pool_short_mean=(short_label_col, "mean"),
        pool_next_mean=(next_label_col, "mean"),
    )
    selected = (
        work.loc[work["_selected"]]
        .groupby(list(GROUP_COLS), sort=False)
        .agg(
            selected_rows=(score_col, "size"),
            selected_short_mean=(short_label_col, "mean"),
            selected_next_mean=(next_label_col, "mean"),
        )
    )
    rank_ic = group.apply(
        lambda item: pd.Series(
            {
                "short_rank_ic": finite_corr(item[score_col], item[short_label_col]),
                "next_rank_ic": finite_corr(item[score_col], item[next_label_col]),
            }
        )
    )
    metrics = base.join(selected, how="left").join(rank_ic, how="left").reset_index()
    metrics["pool"] = pool_name
    metrics["test_month"] = pd.to_datetime(metrics["date"]).dt.to_period("M").astype(str)
    metrics["clock"] = clock_label(metrics["decision_target_timestamp"])
    metrics["short_internal_excess_bps"] = (
        metrics["selected_short_mean"] - metrics["pool_short_mean"]
    ) * 10_000.0
    metrics["next_internal_excess_bps"] = (
        metrics["selected_next_mean"] - metrics["pool_next_mean"]
    ) * 10_000.0
    for column in (
        "pool_short_mean",
        "selected_short_mean",
        "pool_next_mean",
        "selected_next_mean",
    ):
        metrics[f"{column}_bps"] = metrics[column] * 10_000.0
    return metrics[
        [
            "pool",
            "test_month",
            "date",
            "decision_target_timestamp",
            "clock",
            "candidate_rows",
            "selected_rows",
            "pool_short_mean_bps",
            "selected_short_mean_bps",
            "short_internal_excess_bps",
            "pool_next_mean_bps",
            "selected_next_mean_bps",
            "next_internal_excess_bps",
            "short_rank_ic",
            "next_rank_ic",
        ]
    ]


def summarize_groups(group_metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if group_metrics.empty:
        return pd.DataFrame()
    out = (
        group_metrics.groupby(by, sort=False)
        .agg(
            groups=("short_internal_excess_bps", "size"),
            months=("test_month", "nunique"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            pool_short_mean_bps=("pool_short_mean_bps", "mean"),
            selected_short_mean_bps=("selected_short_mean_bps", "mean"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            pool_next_mean_bps=("pool_next_mean_bps", "mean"),
            selected_next_mean_bps=("selected_next_mean_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
        )
        .reset_index()
    )
    return out


def positive_month_summary(month_summary: pd.DataFrame) -> pd.DataFrame:
    if month_summary.empty:
        return pd.DataFrame()
    return (
        month_summary.groupby("pool", sort=False)
        .agg(
            short_positive_months=(
                "short_internal_excess_bps",
                lambda value: int((value > 0).sum()),
            ),
            next_positive_months=("next_internal_excess_bps", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )


def positive_clock_summary(clock_summary: pd.DataFrame) -> pd.DataFrame:
    if clock_summary.empty:
        return pd.DataFrame()
    return (
        clock_summary.groupby("pool", sort=False)
        .agg(
            short_positive_clocks=(
                "short_internal_excess_bps",
                lambda value: int((value > 0).sum()),
            ),
            next_positive_clocks=("next_internal_excess_bps", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    pools = tuple(args.pool or DEFAULT_POOLS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(
        args.predictions,
        score_col=args.score_col,
        short_label_col=args.short_label_col,
    )
    years = set(pd.to_datetime(predictions["date"], errors="coerce").dt.strftime("%Y").dropna())
    labels = read_next_close_labels(
        args.next_close_label_input,
        years=years,
        next_label_col=args.next_label_col,
    )
    frame = predictions.merge(labels, on=list(KEY_COLUMNS), how="left")
    missing_next = int(frame[args.next_label_col].isna().sum())
    if missing_next:
        print(f"warning: missing next-close labels for {missing_next} prediction rows")

    group_frames = []
    for pool in pools:
        if pool == "universe":
            pool_frame = frame
            pool_name = "universe"
        else:
            pool_path = DEFAULT_STOCK_POOL_PATHS[pool]
            pool_frame = frame.loc[
                stock_pool_membership_mask(
                    frame,
                    load_stock_pool(pool_path),
                    date_lag_sessions=args.pool_date_lag_sessions,
                )
            ].copy()
            pool_name = f"pool_{pool}"
        group_frames.append(
            evaluate_pool(
                pool_frame,
                pool_name=pool_name,
                score_col=args.score_col,
                short_label_col=args.short_label_col,
                next_label_col=args.next_label_col,
                top_n=args.top_n,
            )
        )

    group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    month_summary = summarize_groups(group_metrics, ["pool", "test_month"])
    clock_summary = summarize_groups(group_metrics, ["pool", "clock"])
    summary = summarize_groups(group_metrics, ["pool"])
    if not summary.empty:
        summary = summary.merge(positive_month_summary(month_summary), on="pool", how="left")
        summary = summary.merge(positive_clock_summary(clock_summary), on="pool", how="left")
        if args.run_id:
            summary.insert(1, "run_id", args.run_id)
        if args.variant:
            summary.insert(1, "variant", args.variant)

    group_metrics.to_csv(output_dir / "pool_internal_group_metrics.csv", index=False)
    month_summary_path = output_dir / "pool_internal_month_summary.csv"
    month_summary.to_csv(month_summary_path, index=False)
    clock_summary.to_csv(output_dir / "pool_internal_clock_summary.csv", index=False)
    summary.to_csv(output_dir / "pool_internal_summary.csv", index=False)
    report_plots = {}
    if args.report_dir:
        plot_prefix = args.plot_prefix or slug_label(args.variant or args.run_id)
        plot_variant_label = args.plot_variant_label or args.variant or args.run_id or plot_prefix
        plot_pools = tuple("universe" if pool == "universe" else f"pool_{pool}" for pool in pools)
        report_plots = write_universe_sml_pool_internal_plots(
            month_summary,
            Path(args.report_dir),
            input_path=month_summary_path,
            output_prefix=plot_prefix,
            variant_label=plot_variant_label,
            pools=plot_pools,
        )
    write_json(
        output_dir / "pool_internal_trace.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": args.run_id,
            "variant": args.variant,
            "predictions": args.predictions,
            "next_close_label_input": args.next_close_label_input,
            "rows": int(len(frame)),
            "missing_next_close_rows": missing_next,
            "pools": list(pools),
            "top_n": args.top_n,
            "pool_date_lag_sessions": args.pool_date_lag_sessions,
            "report_plots": report_plots,
        },
        ensure_ascii=True,
    )

    print("pool_internal_summary:")
    print(summary.to_string(index=False))
    if report_plots:
        print("\npool_internal_report_plots:")
        for label, path in report_plots.items():
            print(f"  {label}: {path}")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
