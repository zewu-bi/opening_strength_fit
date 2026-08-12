from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL, write_json
from opening_strength_fit.artifact_catalog import print_recorded_artifacts
from opening_strength_fit.commands.arguments import add_arguments, add_options
from opening_strength_fit.io import read_frame
from opening_strength_fit.io.frames import csv_ready
from opening_strength_fit.pool_internal_artifacts import record_pool_internal_outputs
from opening_strength_fit.pool_internal_company import (  # noqa: F401
    COMPANY_API_TIMES,
    COMPANY_SCORE_AGGS,
    COMPANY_SCORE_TRANSFORMS,
    build_company_neutral_score_matrix,
    build_company_score_matrix,
    company_backtest_neutral_comparison_plot_data,
    company_backtest_relative_plot_data,
    create_company_backtest_payload,
    decode_company_backtest_result,
    filter_company_backtest_scores,
    normalize_clock,
    run_company_backtest_analysis,
    write_company_api_outputs,
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
)
from opening_strength_fit.pool_internal_weekly import (
    build_daily_summary,
    write_weekly_pool_internal_outputs,
)
from opening_strength_fit.prediction_frames import (
    next_close_files,
    normalize_keys,
    prediction_files,
)
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.stock_pool import filter_named_stock_pool

DEFAULT_POOLS = ("universe", "S", "M", "L")
BACKTEST_MODES = ("self", "company-api", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TopN pool-internal short/next-close excess for synced prediction shards."
    )
    add_options(
        parser,
        predictions={
            "action": "append",
            "required": True,
            "help": (
                "Prediction parquet/csv file or directory. May be repeated. If a "
                "directory contains raw/predictions_*.parquet, those shards are used."
            ),
        },
        next_close_label_input={
            "default": "",
            "help": "Next-close label parquet file or directory containing yearly label shards.",
        },
        backtest_mode={
            "default": "self",
            "choices": BACKTEST_MODES,
            "help": (
                "self keeps the existing pool-internal label-based analysis; company-api "
                "sends scores to the company backtest API; both does both."
            ),
        },
        output_dir={"required": True},
    )
    add_arguments(parser, "run-id variant", default="")
    add_options(
        parser,
        score_col={"default": "prediction"},
        short_label_col={"default": "label"},
        next_label_col={"default": NEXT_CLOSE_LABEL_COL},
        top_n={"type": int, "default": 100},
        report_dir={
            "default": "",
            "help": (
                "Optional report directory. When set, writes the universe/S/M/L "
                "pool-internal excess and Rank IC SVG panels plus plot data."
            ),
        },
        plot_prefix={
            "default": "",
            "help": (
                "Filename/directory prefix for generated report plots. Defaults to "
                "--variant when present, otherwise --run-id."
            ),
        },
        plot_variant_label={
            "default": "",
            "help": "Display label used in generated report plot titles. Defaults to --variant.",
        },
        plot_period={
            "choices": ["month", "quarter"],
            "default": "month",
            "help": "Period aggregation used for report bar plots. Defaults to month.",
        },
        pool={
            "action": "append",
            "choices": ["universe", "S", "M", "L"],
            "help": "Pools to evaluate. Defaults to universe, S, M, and L.",
        },
        pool_date_lag_sessions={"type": int, "default": 0},
        weekly_report_dir={
            "default": "",
            "help": "Optional directory for weekly trading-day-equal diagnostics derived from group metrics.",
        },
        weekly_output_prefix={"default": ""},
        weekly_rolling_weeks={"type": int, "default": 4},
        records_dir={
            "default": "",
            "help": "Optional experiments/results style directory for lightweight archived outputs.",
        },
        record_prefix={
            "default": "",
            "help": "Filename prefix used under --records-dir/backtests. Defaults to --run-id.",
        },
        company_clock={
            "action": "append",
            "help": (
                "Decision clock to keep for company API input, for example 09:40 or 0940. "
                "May be repeated. If omitted, all clocks are eligible before range filters."
            ),
        },
    )
    add_arguments(parser, "company-start-clock company-end-clock", default="")
    add_options(
        parser,
        company_score_agg={"default": "mean", "choices": COMPANY_SCORE_AGGS},
        company_score_transform={
            "default": "identity",
            "choices": COMPANY_SCORE_TRANSFORMS,
            "help": (
                "Transform scores before sending them to the company API. The default "
                "identity preserves the local higher-prediction-is-better convention; "
                "use negate only when reproducing historical runs or when an endpoint "
                "is known to prefer lower scores."
            ),
        },
        company_neutral_baseline={
            "action": "store_true",
            "help": (
                "Also run a neutral stock-pool baseline through the company API by "
                "replacing every finite model score with --company-neutral-score. Use "
                "with --company-top-n 0 for a full pool baseline."
            ),
        },
        company_neutral_score={"type": float, "default": 0.0},
        company_top_n={
            "type": int,
            "default": None,
            "help": "Top N names per date sent to company API. Defaults to --top-n.",
        },
        company_pool={
            "default": "L",
            "choices": ["", "S", "M", "L"],
            "help": "Optional stock-pool membership filter before company API input construction.",
        },
        company_pool_path={
            "default": "",
            "help": "Override company API stock-pool parquet path. Defaults to --company-pool path.",
        },
        company_api_time={"default": "950", "choices": COMPANY_API_TIMES},
        company_daily={
            "action": "store_true",
            "help": "Send a daily score matrix instead of an intraday {time: matrix} payload.",
        },
        company_tar={"default": "I500", "choices": ["I500", "large", "small"]},
    )
    add_arguments(parser, "company-cap company-trgain company-vol-limit", type=float, default=None)
    add_options(
        parser,
        company_fee={
            "dest": "company_fee",
            "action": "store_true",
            "default": None,
            "help": "Explicitly enable API fees. Omit to use API default.",
        },
        company_no_fee={"dest": "company_fee", "action": "store_false"},
        company_return_eod={"action": "store_true"},
        company_skip_api={
            "action": "store_true",
            "help": "Only write company API score inputs and trace; do not POST to the API.",
        },
        company_endpoint={"action": "append"},
        company_timeout={"type": float, "default": 600.0},
        company_output_dir={
            "default": "",
            "help": "Directory for company API score inputs and raw outputs. Defaults under --output-dir.",
        },
        company_plot_dir={
            "default": "",
            "help": "Directory for company API cumulative plot. Defaults to --report-dir or --output-dir.",
        },
    )
    add_arguments(
        parser,
        "company-series-key company-series-label company-series-color",
        default="",
    )
    return parser.parse_args()


def read_predictions(
    paths: list[str], *, score_col: str, short_label_col: str = ""
) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col, *([short_label_col] if short_label_col else [])]
    files = [file for raw in paths for file in prediction_files(Path(raw))]
    context = "reading_predictions" if short_label_col else "reading_prediction_scores"
    print(f"{context}: files={len(files)}")
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


@dataclass
class _SelfAnalysis:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_next: int = 0
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    report_plots: dict[str, str] = field(default_factory=dict)
    weekly_outputs: dict[str, str] = field(default_factory=dict)


def _run_self_analysis(
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    pools: tuple[str, ...],
    output_dir: Path,
    plot_pools: tuple[str, ...],
) -> _SelfAnalysis:
    report_plots: dict[str, str] = {}
    weekly_outputs: dict[str, str] = {}
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
        pool_name, pool_frame = filter_named_stock_pool(
            frame,
            pool,
            date_lag_sessions=args.pool_date_lag_sessions,
        )
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

    group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
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
    csv_ready(daily).to_csv(daily_summary_path, index=False)

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
        weekly_outputs, _ = write_weekly_pool_internal_outputs(
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

    return _SelfAnalysis(
        frame=frame,
        missing_next=missing_next,
        summary=summary,
        report_plots=report_plots,
        weekly_outputs=weekly_outputs,
    )


def main() -> None:
    args = parse_args()
    pools = tuple(args.pool or DEFAULT_POOLS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_self = args.backtest_mode in {"self", "both"}
    run_company = args.backtest_mode in {"company-api", "both"}
    if run_self and not args.next_close_label_input:
        raise SystemExit("--next-close-label-input is required when --backtest-mode uses self")

    predictions = read_predictions(
        args.predictions,
        score_col=args.score_col,
        short_label_col=args.short_label_col if run_self else "",
    )

    plot_pools = tuple("universe" if pool == "universe" else f"pool_{pool}" for pool in pools)
    analysis = (
        _run_self_analysis(args, predictions, pools, output_dir, plot_pools)
        if run_self
        else _SelfAnalysis()
    )
    report_plots = analysis.report_plots
    weekly_outputs = analysis.weekly_outputs
    company_trace: dict[str, Any] = {}

    if run_company:
        company_output_dir = Path(args.company_output_dir or output_dir / "company_backtest_api")
        default_plot_dir = Path(args.report_dir) if args.report_dir else output_dir
        company_plot_dir = Path(args.company_plot_dir or default_plot_dir / "company_backtest_api")
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
        "rows": int(len(analysis.frame) if run_self else len(predictions)),
        "missing_next_close_rows": analysis.missing_next,
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
    print(analysis.summary.to_string(index=False))
    if report_plots:
        print_mapping("pool_internal_report_plots", report_plots)
    if weekly_outputs:
        print_mapping("weekly_pool_internal_outputs", weekly_outputs)
    print_recorded_artifacts(record_paths, "pool_internal")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
