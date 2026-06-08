from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL, write_json
from opening_strength_fit.commands.weekly_pool_internal_plots import (
    build_weekly_pool_internal_summaries,
)
from opening_strength_fit.io import read_frame
from opening_strength_fit.pool_internal_plots import (
    slug_label,
    write_universe_sml_pool_internal_plots,
    write_weekly_pool_internal_rolling_plot,
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
    parser.add_argument(
        "--weekly-report-dir",
        default="",
        help="Optional directory for weekly trading-day-equal diagnostics derived from group metrics.",
    )
    parser.add_argument("--weekly-output-prefix", default="")
    parser.add_argument("--weekly-rolling-weeks", type=int, default=4)
    parser.add_argument(
        "--records-dir",
        default="",
        help="Optional experiments/results style directory for lightweight archived outputs.",
    )
    parser.add_argument(
        "--record-prefix",
        default="",
        help="Filename prefix used under --records-dir/backtests. Defaults to --run-id.",
    )
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


def halfyear_summary(month_summary: pd.DataFrame) -> pd.DataFrame:
    if month_summary.empty:
        return pd.DataFrame()
    frame = month_summary.copy()
    frame["year"] = frame["test_month"].astype(str).str.slice(0, 4).astype(int)
    month_num = frame["test_month"].astype(str).str.slice(5, 7).astype(int)
    frame["half"] = month_num.map(lambda value: "H1" if value <= 6 else "H2")
    return (
        frame.groupby(["pool", "year", "half"], sort=False)
        .agg(
            months=("test_month", "nunique"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
            short_positive_months=(
                "short_internal_excess_bps",
                lambda value: int((value > 0).sum()),
            ),
            next_positive_months=("next_internal_excess_bps", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )


def year_summary(month_summary: pd.DataFrame) -> pd.DataFrame:
    if month_summary.empty:
        return pd.DataFrame()
    frame = month_summary.copy()
    frame["year"] = frame["test_month"].astype(str).str.slice(0, 4).astype(int)
    return (
        frame.groupby(["pool", "year"], sort=False)
        .agg(
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
            short_positive_months=(
                "short_internal_excess_bps",
                lambda value: int((value > 0).sum()),
            ),
            next_positive_months=("next_internal_excess_bps", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out


def write_weekly_outputs(
    group_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    output_prefix: str,
    variant_label: str,
    pools: tuple[str, ...],
    rolling_weeks: int,
) -> dict[str, str]:
    daily, weekly, overall, worst = build_weekly_pool_internal_summaries(
        group_metrics,
        pools=pools,
        rolling_weeks=rolling_weeks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily_pool_internal_summary.csv"
    weekly_path = output_dir / "weekly_pool_internal_summary.csv"
    overall_path = output_dir / "weekly_pool_internal_overall_summary.csv"
    worst_path = output_dir / "weekly_worst_windows.csv"
    _csv_ready(daily).to_csv(daily_path, index=False, float_format="%.6f")
    _csv_ready(weekly).to_csv(weekly_path, index=False, float_format="%.6f")
    overall.to_csv(overall_path, index=False, float_format="%.6f")
    worst.to_csv(worst_path, index=False, float_format="%.6f")
    plot_paths = write_weekly_pool_internal_rolling_plot(
        weekly,
        output_dir,
        input_path=weekly_path,
        output_prefix=output_prefix,
        variant_label=variant_label,
        pools=pools,
        rolling_weeks=rolling_weeks,
    )
    trace_path = output_dir / "weekly_pool_internal_trace.json"
    write_json(
        trace_path,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "daily_summary": str(daily_path),
            "weekly_summary": str(weekly_path),
            "overall_summary": str(overall_path),
            "worst_windows": str(worst_path),
            "plot_paths": plot_paths,
            "pools": list(pools),
            "rolling_weeks": rolling_weeks,
            "weighting": (
                "date x pool is first averaged across decision clocks; weekly summaries and "
                "rolling windows are equal weighted by trading day"
            ),
        },
        ensure_ascii=True,
    )
    return {
        "daily": str(daily_path),
        "weekly": str(weekly_path),
        "overall": str(overall_path),
        "worst": str(worst_path),
        "trace": str(trace_path),
        **plot_paths,
    }


def record_pool_internal_outputs(
    *,
    output_dir: Path,
    records_dir: Path,
    record_prefix: str,
    report_plots: dict[str, str],
) -> list[Path]:
    backtests_dir = records_dir / "backtests"
    records = [
        (output_dir / "pool_internal_summary.csv", f"{record_prefix}_pool_internal_summary.csv"),
        (
            output_dir / "pool_internal_month_summary.csv",
            f"{record_prefix}_pool_internal_month_summary.csv",
        ),
        (
            output_dir / "pool_internal_clock_summary.csv",
            f"{record_prefix}_pool_internal_clock_summary.csv",
        ),
        (
            output_dir / "pool_internal_group_metrics.csv",
            f"{record_prefix}_pool_internal_group_metrics.csv",
        ),
        (
            output_dir / "pool_internal_halfyear_summary.csv",
            f"{record_prefix}_pool_internal_halfyear_summary.csv",
        ),
        (
            output_dir / "pool_internal_year_summary.csv",
            f"{record_prefix}_pool_internal_year_summary.csv",
        ),
        (output_dir / "pool_internal_trace.json", f"{record_prefix}_pool_internal_trace.json"),
    ]
    plot_records = {
        "pool_internal_plot_data": f"{record_prefix}_pool_internal_plot_data.csv",
        "pool_internal_figure": f"{record_prefix}_pool_internal_with_mean.svg",
        "rank_ic_plot_data": f"{record_prefix}_rank_ic_plot_data.csv",
        "rank_ic_figure": f"{record_prefix}_rank_ic_with_mean.svg",
        "short_excess_rank_ic_plot_data": f"{record_prefix}_short_excess_rank_ic_plot_data.csv",
        "short_excess_rank_ic_figure": f"{record_prefix}_short_excess_rank_ic_with_mean.svg",
        "next_excess_rank_ic_plot_data": f"{record_prefix}_next_excess_rank_ic_plot_data.csv",
        "next_excess_rank_ic_figure": f"{record_prefix}_next_excess_rank_ic_with_mean.svg",
    }
    for key, name in plot_records.items():
        if key in report_plots:
            records.append((Path(report_plots[key]), name))
    for key, name in plot_records.items():
        if not key.endswith("_plot_data") or key not in report_plots:
            continue
        trace_path = _plot_trace_path(Path(report_plots[key]))
        if trace_path.exists():
            trace_name = name.replace("_plot_data.csv", "_trace.json")
            if trace_name.endswith("_pool_internal_trace.json"):
                trace_name = f"{record_prefix}_pool_internal_with_mean_trace.json"
            records.append((trace_path, trace_name))

    copied: list[Path] = []
    for source, name in records:
        if not source.exists():
            continue
        destination = backtests_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _plot_trace_path(plot_data_path: Path) -> Path:
    name = plot_data_path.name
    if name.endswith("_plot_data.csv"):
        return plot_data_path.with_name(name.replace("_plot_data.csv", "_trace.json"))
    return plot_data_path.with_suffix(".json")


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
    halfyear = halfyear_summary(month_summary)
    yearly = year_summary(month_summary)
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
    halfyear.to_csv(output_dir / "pool_internal_halfyear_summary.csv", index=False)
    yearly.to_csv(output_dir / "pool_internal_year_summary.csv", index=False)
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
    weekly_outputs = {}
    if args.weekly_report_dir:
        weekly_outputs = write_weekly_outputs(
            group_metrics,
            Path(args.weekly_report_dir),
            output_prefix=args.weekly_output_prefix or args.plot_prefix or args.variant or args.run_id,
            variant_label=args.plot_variant_label or args.variant or args.run_id,
            pools=tuple("universe" if pool == "universe" else f"pool_{pool}" for pool in pools),
            rolling_weeks=args.weekly_rolling_weeks,
        )
    trace_path = output_dir / "pool_internal_trace.json"
    trace_payload = {
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
        "weekly_outputs": weekly_outputs,
        "record_paths": [],
    }
    write_json(trace_path, trace_payload, ensure_ascii=True)
    record_paths: list[Path] = []
    record_prefix = args.record_prefix or args.run_id
    if args.records_dir:
        if not record_prefix:
            raise SystemExit("--records-dir requires --record-prefix or --run-id")
        record_paths = record_pool_internal_outputs(
            output_dir=output_dir,
            records_dir=Path(args.records_dir),
            record_prefix=record_prefix,
            report_plots=report_plots,
        )
        trace_payload["record_paths"] = [str(path) for path in record_paths]
        write_json(trace_path, trace_payload, ensure_ascii=True)
        destination = (
            Path(args.records_dir)
            / "backtests"
            / f"{record_prefix}_pool_internal_trace.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, destination)

    print("pool_internal_summary:")
    print(summary.to_string(index=False))
    if report_plots:
        print("\npool_internal_report_plots:")
        for label, path in report_plots.items():
            print(f"  {label}: {path}")
    if weekly_outputs:
        print("\nweekly_pool_internal_outputs:")
        for label, path in weekly_outputs.items():
            print(f"  {label}: {path}")
    if record_paths:
        print("\nrecorded_pool_internal_outputs:")
        for path in record_paths:
            print(f"  {path}")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
