from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL, write_json
from opening_strength_fit.io import read_frame
from opening_strength_fit.pool_internal_company import (
    COMPANY_API_TIMES,
    COMPANY_SCORE_AGGS,
    COMPANY_SCORE_TRANSFORMS,
    build_company_neutral_score_matrix,  # noqa: F401
    build_company_score_matrix,  # noqa: F401
    company_backtest_neutral_comparison_plot_data,  # noqa: F401
    company_backtest_relative_plot_data,  # noqa: F401
    create_company_backtest_payload,  # noqa: F401
    decode_company_backtest_result,  # noqa: F401
    filter_company_backtest_scores,  # noqa: F401
    normalize_clock,  # noqa: F401
    run_company_backtest_analysis,
    write_company_api_outputs,  # noqa: F401
)
from opening_strength_fit.pool_internal_eval import (
    evaluate_pool,
    halfyear_summary,
    positive_clock_summary,
    positive_month_summary,
    quarter_summary,
    summarize_groups,
    year_summary,
)
from opening_strength_fit.pool_internal_plots import (
    slug_label,
    write_daily_pool_internal_cumulative_plot,
    write_universe_sml_pool_internal_plots,
    write_weekly_pool_internal_rolling_plot,
)
from opening_strength_fit.pool_internal_weekly import (
    build_daily_summary,
    build_weekly_pool_internal_summaries,
)
from opening_strength_fit.prediction_frames import (
    next_close_files,
    normalize_keys,
    prediction_files,
)
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

DEFAULT_POOLS = ("universe", "S", "M", "L")
BACKTEST_MODES = ("self", "company-api", "both")


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
        default="",
        help="Next-close label parquet file or directory containing yearly label shards.",
    )
    parser.add_argument(
        "--backtest-mode",
        default="self",
        choices=BACKTEST_MODES,
        help=(
            "self keeps the existing pool-internal label-based analysis; company-api "
            "sends scores to the company backtest API; both does both."
        ),
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
        "--plot-period",
        choices=["month", "quarter"],
        default="month",
        help="Period aggregation used for report bar plots. Defaults to month.",
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
    parser.add_argument(
        "--company-clock",
        action="append",
        help=(
            "Decision clock to keep for company API input, for example 09:40 or 0940. "
            "May be repeated. If omitted, all clocks are eligible before range filters."
        ),
    )
    parser.add_argument("--company-start-clock", default="")
    parser.add_argument("--company-end-clock", default="")
    parser.add_argument("--company-score-agg", default="mean", choices=COMPANY_SCORE_AGGS)
    parser.add_argument(
        "--company-score-transform",
        default="identity",
        choices=COMPANY_SCORE_TRANSFORMS,
        help=(
            "Transform scores before sending them to the company API. The default "
            "identity preserves the local higher-prediction-is-better convention; "
            "use negate only when reproducing historical runs or when an endpoint "
            "is known to prefer lower scores."
        ),
    )
    parser.add_argument(
        "--company-neutral-baseline",
        action="store_true",
        help=(
            "Also run a neutral stock-pool baseline through the company API by "
            "replacing every finite model score with --company-neutral-score. Use "
            "with --company-top-n 0 for a full pool baseline."
        ),
    )
    parser.add_argument("--company-neutral-score", type=float, default=0.0)
    parser.add_argument(
        "--company-top-n",
        type=int,
        default=None,
        help="Top N names per date sent to company API. Defaults to --top-n.",
    )
    parser.add_argument(
        "--company-pool",
        default="L",
        choices=["", "S", "M", "L"],
        help="Optional stock-pool membership filter before company API input construction.",
    )
    parser.add_argument(
        "--company-pool-path",
        default="",
        help="Override company API stock-pool parquet path. Defaults to --company-pool path.",
    )
    parser.add_argument("--company-api-time", default="950", choices=COMPANY_API_TIMES)
    parser.add_argument(
        "--company-daily",
        action="store_true",
        help="Send a daily score matrix instead of an intraday {time: matrix} payload.",
    )
    parser.add_argument("--company-tar", default="I500", choices=["I500", "large", "small"])
    parser.add_argument("--company-cap", type=float, default=None)
    parser.add_argument("--company-trgain", type=float, default=None)
    parser.add_argument("--company-vol-limit", type=float, default=None)
    parser.add_argument(
        "--company-fee",
        dest="company_fee",
        action="store_true",
        default=None,
        help="Explicitly enable API fees. Omit to use API default.",
    )
    parser.add_argument("--company-no-fee", dest="company_fee", action="store_false")
    parser.add_argument("--company-return-eod", action="store_true")
    parser.add_argument(
        "--company-skip-api",
        action="store_true",
        help="Only write company API score inputs and trace; do not POST to the API.",
    )
    parser.add_argument("--company-endpoint", action="append")
    parser.add_argument("--company-timeout", type=float, default=600.0)
    parser.add_argument(
        "--company-output-dir",
        default="",
        help="Directory for company API score inputs and raw outputs. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--company-plot-dir",
        default="",
        help="Directory for company API cumulative plot. Defaults to --report-dir or --output-dir.",
    )
    parser.add_argument("--company-series-key", default="")
    parser.add_argument("--company-series-label", default="")
    parser.add_argument("--company-series-color", default="")
    return parser.parse_args()


def read_predictions(paths: list[str], *, score_col: str, short_label_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col, short_label_col]
    files = [file for raw in paths for file in prediction_files(Path(raw))]
    print(f"reading_predictions: files={len(files)}")
    frames = []
    for file in files:
        frame = read_frame(file, columns=required)
        print(f"  {file}: rows={len(frame)}")
        frames.append(frame)
    if not frames:
        raise SystemExit("no prediction files supplied")
    return normalize_keys(pd.concat(frames, ignore_index=True))


def read_prediction_scores(paths: list[str], *, score_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col]
    files = [file for raw in paths for file in prediction_files(Path(raw))]
    print(f"reading_prediction_scores: files={len(files)}")
    frames = []
    for file in files:
        frame = read_frame(file, columns=required)
        print(f"  {file}: rows={len(frame)}")
        frames.append(frame)
    if not frames:
        raise SystemExit("no prediction files supplied")
    return normalize_keys(pd.concat(frames, ignore_index=True))


def read_next_close_labels(path: str, *, years: set[str], next_label_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, next_label_col]
    files = next_close_files(Path(path), years)
    print(f"reading_next_close_labels: files={len(files)} years={','.join(sorted(years))}")
    frames = []
    for file in files:
        frame = read_frame(file, columns=required)
        print(f"  {file}: rows={len(frame)}")
        frames.append(frame)
    if not frames:
        raise SystemExit("no next-close label files supplied")
    labels = normalize_keys(pd.concat(frames, ignore_index=True))
    labels = labels.dropna(subset=list(KEY_COLUMNS) + [next_label_col])
    return labels.drop_duplicates(list(KEY_COLUMNS), keep="last")


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
    record_subdir: str = "",
) -> list[Path]:
    backtests_dir = records_dir / "backtests" / record_subdir

    def record_name(suffix: str) -> str:
        return suffix if record_subdir else f"{record_prefix}_{suffix}"

    records = [
        (output_dir / "pool_internal_summary.csv", record_name("pool_internal_summary.csv")),
        (
            output_dir / "pool_internal_quarter_summary.csv",
            record_name("pool_internal_quarter_summary.csv"),
        ),
        (
            output_dir / "daily_pool_internal_summary.csv",
            record_name("daily_pool_internal_summary.csv"),
        ),
        (
            output_dir / "pool_internal_month_summary.csv",
            record_name("pool_internal_month_summary.csv"),
        ),
        (
            output_dir / "pool_internal_clock_summary.csv",
            record_name("pool_internal_clock_summary.csv"),
        ),
        (
            output_dir / "pool_internal_group_metrics.csv",
            record_name("pool_internal_group_metrics.csv"),
        ),
        (
            output_dir / "pool_internal_halfyear_summary.csv",
            record_name("pool_internal_halfyear_summary.csv"),
        ),
        (
            output_dir / "pool_internal_year_summary.csv",
            record_name("pool_internal_year_summary.csv"),
        ),
        (output_dir / "pool_internal_trace.json", record_name("pool_internal_trace.json")),
    ]
    plot_records = {
        "pool_internal_plot_data": record_name("pool_internal_plot_data.csv"),
        "pool_internal_figure": record_name("pool_internal_with_mean.svg"),
        "rank_ic_plot_data": record_name("rank_ic_plot_data.csv"),
        "rank_ic_figure": record_name("rank_ic_with_mean.svg"),
        "short_excess_rank_ic_plot_data": record_name("short_excess_rank_ic_plot_data.csv"),
        "short_excess_rank_ic_figure": record_name("short_excess_rank_ic_with_mean.svg"),
        "next_excess_rank_ic_plot_data": record_name("next_excess_rank_ic_plot_data.csv"),
        "next_excess_rank_ic_figure": record_name("next_excess_rank_ic_with_mean.svg"),
        "daily_cumulative_plot_data": record_name("daily_cumulative_plot_data.csv"),
        "daily_cumulative_figure": record_name("daily_cumulative.svg"),
        "company_backtest_plot_data": record_name("company_backtest_plot_data.csv"),
        "company_backtest_figure": record_name("company_backtest.svg"),
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
            if trace_name == "pool_internal_trace.json" or trace_name.endswith(
                "_pool_internal_trace.json"
            ):
                trace_name = record_name("pool_internal_with_mean_trace.json")
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

    run_self = args.backtest_mode in {"self", "both"}
    run_company = args.backtest_mode in {"company-api", "both"}
    if run_self and not args.next_close_label_input:
        raise SystemExit("--next-close-label-input is required when --backtest-mode uses self")

    if run_self:
        predictions = read_predictions(
            args.predictions,
            score_col=args.score_col,
            short_label_col=args.short_label_col,
        )
    else:
        predictions = read_prediction_scores(args.predictions, score_col=args.score_col)

    frame = pd.DataFrame()
    missing_next = 0
    group_metrics = pd.DataFrame()
    month_summary = pd.DataFrame()
    quarterly = pd.DataFrame()
    clock_summary = pd.DataFrame()
    halfyear = pd.DataFrame()
    yearly = pd.DataFrame()
    summary = pd.DataFrame()
    plot_pools = tuple("universe" if pool == "universe" else f"pool_{pool}" for pool in pools)
    report_plots = {}
    weekly_outputs = {}
    company_trace: dict[str, Any] = {}

    if run_self:
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
        print(
            f"analysis_frame: prediction_rows={len(predictions)} "
            f"label_rows={len(labels)} joined_rows={len(frame)}"
        )

        group_frames = []
        for pool in pools:
            if pool == "universe":
                pool_frame = frame
                pool_name = "universe"
            else:
                pool_path = DEFAULT_STOCK_POOL_PATHS[pool]
                print(f"loading_stock_pool: pool={pool} path={pool_path}")
                pool_frame = frame.loc[
                    stock_pool_membership_mask(
                        frame,
                        load_stock_pool(pool_path),
                        date_lag_sessions=args.pool_date_lag_sessions,
                    )
                ].copy()
                pool_name = f"pool_{pool}"
            print(f"evaluating_pool: pool={pool_name} rows={len(pool_frame)}")
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

        group_metrics = (
            pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
        )
        month_summary = summarize_groups(group_metrics, ["pool", "test_month"])
        quarterly = quarter_summary(group_metrics)
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
        quarter_summary_path = output_dir / "pool_internal_quarter_summary.csv"
        quarterly.to_csv(quarter_summary_path, index=False)
        clock_summary.to_csv(output_dir / "pool_internal_clock_summary.csv", index=False)
        halfyear.to_csv(output_dir / "pool_internal_halfyear_summary.csv", index=False)
        yearly.to_csv(output_dir / "pool_internal_year_summary.csv", index=False)
        summary.to_csv(output_dir / "pool_internal_summary.csv", index=False)
        daily = pd.DataFrame()
        if not group_metrics.empty:
            daily = build_daily_summary(group_metrics, pools=plot_pools)
        daily_summary_path = output_dir / "daily_pool_internal_summary.csv"
        _csv_ready(daily).to_csv(daily_summary_path, index=False)

        if args.report_dir:
            plot_prefix = args.plot_prefix or slug_label(args.variant or args.run_id)
            plot_variant_label = args.plot_variant_label or args.variant or args.run_id or plot_prefix
            plot_summary = quarterly if args.plot_period == "quarter" else month_summary
            plot_summary_path = (
                quarter_summary_path if args.plot_period == "quarter" else month_summary_path
            )
            report_plots = write_universe_sml_pool_internal_plots(
                plot_summary,
                Path(args.report_dir),
                input_path=plot_summary_path,
                output_prefix=plot_prefix,
                variant_label=plot_variant_label,
                pools=plot_pools,
            )
            if not daily.empty:
                report_plots.update(
                    write_daily_pool_internal_cumulative_plot(
                        daily,
                        Path(args.report_dir) / "cumulative",
                        input_path=daily_summary_path,
                        output_prefix=plot_prefix,
                        output_name=f"{plot_prefix}_{'_'.join(plot_pools)}_daily_cumulative",
                        variant_label=plot_variant_label,
                        pools=plot_pools,
                        x_label_mode="years_only",
                    )
                )
        if args.weekly_report_dir:
            weekly_outputs = write_weekly_outputs(
                group_metrics,
                Path(args.weekly_report_dir),
                output_prefix=args.weekly_output_prefix
                or args.plot_prefix
                or args.variant
                or args.run_id,
                variant_label=args.plot_variant_label or args.variant or args.run_id,
                pools=plot_pools,
                rolling_weeks=args.weekly_rolling_weeks,
            )

    if run_company:
        company_output_dir = Path(args.company_output_dir) if args.company_output_dir else (
            output_dir / "company_backtest_api"
        )
        default_plot_dir = Path(args.report_dir) if args.report_dir else output_dir
        company_plot_dir = Path(args.company_plot_dir) if args.company_plot_dir else (
            default_plot_dir / "company_backtest_api"
        )
        company_trace = run_company_backtest_analysis(
            predictions,
            company_output_dir,
            company_plot_dir,
            args=args,
        )
        report_plots.update(company_trace.get("plot_paths", {}))
    trace_path = output_dir / "pool_internal_trace.json"
    trace_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "variant": args.variant,
        "backtest_mode": args.backtest_mode,
        "predictions": args.predictions,
        "next_close_label_input": args.next_close_label_input,
        "rows": int(len(frame) if run_self else len(predictions)),
        "missing_next_close_rows": missing_next,
        "pools": list(pools),
        "top_n": args.top_n,
        "pool_date_lag_sessions": args.pool_date_lag_sessions,
        "plot_period": args.plot_period,
        "report_plots": report_plots,
        "weekly_outputs": weekly_outputs,
        "company_backtest": company_trace,
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
            Path(args.records_dir) / "backtests" / f"{record_prefix}_pool_internal_trace.json"
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
