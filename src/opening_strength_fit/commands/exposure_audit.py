from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.artifact_catalog import (
    EXPOSURE_AUDIT_ARTIFACTS,
    print_recorded_artifacts,
    record_requested_artifacts,
)
from opening_strength_fit.commands import arguments as cmd
from opening_strength_fit.config import config_str
from opening_strength_fit.exposure_audit import (
    active_default_exposures,
    add_derived_exposure_columns,
    category_summary,
    daily_concentration,
    derivable_exposure_columns,
    derived_exposure_source_columns,
    exposure_group_metrics,
    exposure_specs,
    industry_group_metrics,
    normalize_audit_frame,
    summarize_concentration,
    summarize_exposure_groups,
    summarize_industry_groups,
)
from opening_strength_fit.io import (
    available_frame_columns,
    frame_columns,
    merge_frame_support,
    read_frame,
    read_frame_files,
    select_available_columns,
)
from opening_strength_fit.io import (
    frame_files_many as _generic_files,
)
from opening_strength_fit.prediction_frames import prediction_files_many as _prediction_files
from opening_strength_fit.schema import normalize_decision_keys
from opening_strength_fit.stock_pool import (
    filter_named_stock_pool,
)

DEFAULT_POOLS = ("L",)


def parse_args() -> argparse.Namespace:
    parser = cmd.command_parser(
        description=("Audit TopN or supplied-portfolio exposure versus its candidate pool.")
    )
    parser.add_argument("--predictions", action="append")
    parser.add_argument(
        "--exposure-input",
        action="append",
        help=(
            "Optional parquet/csv file or directory with exposure columns keyed by "
            "date,symbol or date,symbol,decision_target_timestamp. May be repeated."
        ),
    )
    cmd.add_arguments(parser, "output-dir run-id variant", default="")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--top-n", type=int, default=None)
    cmd.add_arguments(parser, "selection-col weight-col industry-col", default="")
    parser.add_argument(
        "--skip-score-exposure-corr",
        action="store_true",
        help="Skip score-vs-exposure Spearman approximation for faster large-run audits.",
    )
    parser.add_argument(
        "--exposure-col",
        action="append",
        help="Exposure column to audit. Defaults to known price/liquidity/momentum columns.",
    )
    parser.add_argument(
        "--pool",
        action="append",
        choices=["universe", "S", "M", "L"],
        help="Pool to audit. Defaults to L.",
    )
    parser.add_argument("--pool-date-lag-sessions", type=int, default=None)
    cmd.add_arguments(parser, "records-dir record-prefix", default="")
    return parser.parse_args()


def _merge_exposure_files(
    frame: pd.DataFrame,
    *,
    files: list[Path],
    exposure_columns: list[str],
    industry_col: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    merge_trace: list[dict[str, object]] = []
    for file in files:
        available = frame_columns(file)
        keys = ["date", "symbol"]
        if "decision_target_timestamp" in available:
            keys.append("decision_target_timestamp")
        missing_keys = sorted(set(keys) - available)
        if missing_keys:
            raise SystemExit(f"{file}: missing exposure key columns: {missing_keys}")
        columns = select_available_columns(keys, [*exposure_columns, industry_col], available)
        if len(columns) == len(keys):
            continue
        support = normalize_decision_keys(
            read_frame(file, columns=columns), key_columns=tuple(keys)
        )
        frame = merge_frame_support(frame, support, keys=keys)
        merge_trace.append(
            {
                "file": str(file),
                "keys": keys,
                "rows": int(len(support)),
                "columns": [column for column in columns if column not in keys],
            }
        )
        print(f"  {file}: rows={len(support)} keys={','.join(keys)}")
    return frame, merge_trace


def _load_audit_frame(
    *,
    prediction_paths: list[str],
    exposure_paths: list[str],
    score_col: str,
    selection_col: str,
    weight_col: str,
    top_n: int,
    exposure_columns: list[str],
    industry_col: str,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    prediction_files_list = _prediction_files(prediction_paths)
    print(f"reading_predictions: files={len(prediction_files_list)}")
    prediction_available = available_frame_columns(prediction_files_list)

    exposure_files_list = _generic_files(exposure_paths) if exposure_paths else []
    exposure_available = (
        available_frame_columns(exposure_files_list) if exposure_files_list else set()
    )
    available = prediction_available | exposure_available

    if exposure_columns:
        available_or_derivable = available | derivable_exposure_columns(available)
        missing_exposures = sorted(set(exposure_columns) - available_or_derivable)
        if missing_exposures:
            raise SystemExit(f"requested exposure columns are missing: {missing_exposures}")
        active_exposure_columns = exposure_columns
    else:
        active_exposure_columns = [spec.column for spec in active_default_exposures(available)]
    if not active_exposure_columns:
        raise SystemExit(
            "no exposure columns found; pass --exposure-col or include default exposure columns"
        )
    if industry_col and industry_col not in available:
        raise SystemExit(f"industry column is missing: {industry_col}")

    prediction_required = [*KEY_COLUMNS]
    if not selection_col and not weight_col and score_col not in prediction_available:
        raise SystemExit(f"score column is missing from predictions: {score_col}")
    if selection_col and selection_col not in prediction_available:
        raise SystemExit(f"selection column is missing from predictions: {selection_col}")
    if weight_col and weight_col not in prediction_available:
        raise SystemExit(f"weight column is missing from predictions: {weight_col}")
    prediction_columns = select_available_columns(
        KEY_COLUMNS, (score_col, selection_col, weight_col), prediction_available
    )

    external_exposure_columns = [
        column for column in active_exposure_columns if column in exposure_available
    ]
    derived_source_columns = derived_exposure_source_columns(active_exposure_columns, available)
    external_derived_source_columns = [
        column for column in derived_source_columns if column in exposure_available
    ]
    prediction_exposure_columns = [
        column
        for column in active_exposure_columns
        if column in prediction_available and column not in external_exposure_columns
    ]
    prediction_derived_source_columns = [
        column for column in derived_source_columns if column in prediction_available
    ]
    prediction_columns = select_available_columns(
        prediction_columns,
        [*prediction_exposure_columns, *prediction_derived_source_columns],
        prediction_available,
    )
    if (
        industry_col
        and industry_col in prediction_available
        and industry_col not in exposure_available
    ):
        prediction_columns.append(industry_col)

    predictions = normalize_audit_frame(
        read_frame_files(
            prediction_files_list,
            columns=prediction_columns,
            required=prediction_required,
        )
    )
    frame = predictions

    if exposure_files_list:
        print(f"reading_exposures: files={len(exposure_files_list)}")
        frame, exposure_merge_trace = _merge_exposure_files(
            frame,
            files=exposure_files_list,
            exposure_columns=[
                *external_exposure_columns,
                *external_derived_source_columns,
            ],
            industry_col=industry_col,
        )
    else:
        exposure_merge_trace = []
    frame, derived_sources = add_derived_exposure_columns(frame, active_exposure_columns)

    missing_after_join = {
        column: int(frame[column].isna().sum()) if column in frame.columns else len(frame)
        for column in active_exposure_columns
    }
    trace = {
        "prediction_files": [str(path) for path in prediction_files_list],
        "exposure_files": [str(path) for path in exposure_files_list],
        "exposure_file_merges": exposure_merge_trace,
        "prediction_rows": int(len(predictions)),
        "joined_rows": int(len(frame)),
        "active_exposure_columns": active_exposure_columns,
        "derived_exposure_sources": derived_sources,
        "missing_exposure_values_after_join": missing_after_join,
        "selection_mode": (
            "selection_col" if selection_col else "weight_col" if weight_col else "top_n"
        ),
        "top_n": top_n,
    }
    return frame, active_exposure_columns, trace


def main() -> None:
    args = parse_args()
    config, arguments, run_name = cmd.command_context(args, "exposure_audit")
    prediction_paths = arguments.list("predictions")
    if not prediction_paths:
        raise SystemExit("pass --predictions or set [exposure_audit].predictions")
    exposure_paths = arguments.list("exposure_input")
    pools = tuple(arguments.list("pool", DEFAULT_POOLS) or DEFAULT_POOLS)
    score_col = arguments.string("score_col", "prediction") or "prediction"
    selection_col = arguments.string("selection_col")
    weight_col = arguments.string("weight_col")
    industry_col = arguments.string("industry_col")
    top_n = arguments.integer("top_n", 100)
    pool_date_lag_sessions = arguments.integer("pool_date_lag_sessions", 0)
    requested_exposures = arguments.list("exposure_col", config_name="exposure_cols")
    skip_score_exposure_corr = arguments.flag("skip_score_exposure_corr")
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )

    frame, active_exposure_columns, load_trace = _load_audit_frame(
        prediction_paths=prediction_paths,
        exposure_paths=exposure_paths,
        score_col=score_col,
        selection_col=selection_col,
        weight_col=weight_col,
        top_n=top_n,
        exposure_columns=requested_exposures,
        industry_col=industry_col,
    )
    specs = exposure_specs(active_exposure_columns)
    group_frames = []
    daily_frames = []
    industry_frames = []
    for pool in pools:
        pool_name, pool_data = filter_named_stock_pool(
            frame,
            pool,
            date_lag_sessions=pool_date_lag_sessions,
        )
        print(f"auditing_pool: pool={pool_name} rows={len(pool_data)}")
        selection_options = {
            "pool": pool_name,
            "score_col": score_col,
            "top_n": top_n,
            "selection_col": selection_col,
            "weight_col": weight_col,
        }
        group_frames.append(
            exposure_group_metrics(
                pool_data,
                specs,
                **selection_options,
                compute_score_corr=not skip_score_exposure_corr,
            )
        )
        daily_frames.append(
            daily_concentration(
                pool_data,
                **selection_options,
                industry_col=industry_col,
            )
        )
        industry_frames.append(
            industry_group_metrics(
                pool_data,
                **selection_options,
                industry_col=industry_col,
            )
        )

    group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    month_summary = summarize_exposure_groups(
        group_metrics,
        ["pool", "test_month", "category", "exposure"],
    )
    summary = summarize_exposure_groups(group_metrics, ["pool", "category", "exposure"])
    if not summary.empty:
        summary = summary.sort_values(
            ["pool", "abs_selected_mean_z", "abs_rank_deviation"],
            ascending=[True, False, False],
        )
    categories = category_summary(summary)
    industry_metrics = (
        pd.concat(industry_frames, ignore_index=True) if industry_frames else pd.DataFrame()
    )
    industry_month_summary = summarize_industry_groups(
        industry_metrics,
        ["pool", "test_month", "industry_col", "industry"],
    )
    industry_summary = summarize_industry_groups(
        industry_metrics,
        ["pool", "industry_col", "industry"],
    )
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    concentration = summarize_concentration(daily)
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_name,
        "variant": args.variant or config_str(config, "run", "description", ""),
        "score_col": score_col,
        "selection_col": selection_col,
        "weight_col": weight_col,
        "industry_col": industry_col,
        "score_exposure_corr": not skip_score_exposure_corr,
        "pools": list(pools),
        "pool_date_lag_sessions": pool_date_lag_sessions,
        **load_trace,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in (
        ("group_metrics", group_metrics),
        ("month_summary", month_summary),
        ("summary", summary),
        ("category_summary", categories),
        ("industry_group_metrics", industry_metrics),
        ("industry_month_summary", industry_month_summary),
        ("industry_summary", industry_summary),
        ("daily_concentration", daily),
        ("concentration_summary", concentration),
    ):
        output.to_csv(output_dir / f"exposure_audit_{name}.csv", index=False)
    write_json(output_dir / "exposure_audit_trace.json", trace, ensure_ascii=True)

    records_dir = arguments.string("records_dir")
    record_paths = record_requested_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix=arguments.string("record_prefix") or run_name,
        names=EXPOSURE_AUDIT_ARTIFACTS,
    )

    print("\nexposure_audit_summary:")
    display_cols = "pool category exposure selected_mean_rank selected_mean_z selected_top_decile_share score_exposure_spearman".split()
    print(summary[display_cols].head(40).to_string(index=False) if not summary.empty else "empty")
    if not industry_summary.empty:
        print("\nexposure_audit_industry_summary:")
        industry_display_cols = "pool industry_col industry candidate_share selected_share active_share abs_active_share".split()
        print(industry_summary[industry_display_cols].head(30).to_string(index=False))
    print_recorded_artifacts(record_paths, "exposure_audit")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
