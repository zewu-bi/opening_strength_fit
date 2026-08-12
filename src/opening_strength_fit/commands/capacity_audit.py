from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.artifact_catalog import (
    CAPACITY_AUDIT_ARTIFACTS,
    print_recorded_artifacts,
    record_requested_artifacts,
)
from opening_strength_fit.capacity_acceptance import normalize_key_columns
from opening_strength_fit.capacity_audit import (
    CapacityConstraints,
    ask_depth_pairs,
    build_capacity_portfolios,
    summarize_capacity_daily,
    summarize_capacity_groups,
    summarize_capacity_months,
)
from opening_strength_fit.commands import arguments as cmd
from opening_strength_fit.config import config_str
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
        description=(
            "Build a score-ranked capacity-constrained portfolio per decision group "
            "and audit fill, participation, concentration, and capacity depth."
        )
    )
    parser.add_argument("--predictions", action="append")
    parser.add_argument(
        "--label-input",
        action="append",
        help="Deprecated and ignored; capacity audit no longer computes realized returns.",
    )
    parser.add_argument(
        "--capacity-input",
        action="append",
        help="Optional keyed parquet/csv file or directory containing capacity columns.",
    )
    cmd.add_arguments(parser, "output-dir run-id variant", default="")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--label-col", default="", help=argparse.SUPPRESS)
    parser.add_argument("--target-notional", type=float, default=None)
    cmd.add_arguments(parser, "capacity-notional-col capacity-volume-col", default="")
    parser.add_argument("--capacity-price-col", default="")
    cmd.add_arguments(parser, "max-participation-rate max-symbol-weight", type=float, default=None)
    parser.add_argument("--min-trade-notional", type=float, default=None)
    cmd.add_arguments(parser, "max-names ask-depth-levels", type=int, default=None)
    parser.add_argument("--ask-depth-participation-rate", type=float, default=None)
    parser.add_argument("--allow-decision-depth-fallback", action="store_true")
    parser.add_argument("--industry-col", default="")
    parser.add_argument("--max-industry-weight", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--slippage-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pool",
        action="append",
        choices=["universe", "S", "M", "L"],
        help="Pool to audit. Defaults to L.",
    )
    parser.add_argument("--pool-date-lag-sessions", type=int, default=None)
    cmd.add_arguments(parser, "records-dir record-prefix", default="")
    parser.add_argument(
        "--selected-output-limit",
        type=int,
        default=None,
        help="Maximum selected rows to write. 0 writes all selected rows.",
    )
    return parser.parse_args()


def _capacity_columns(constraints: CapacityConstraints) -> list[str]:
    columns = list(
        dict.fromkeys(
            filter(
                None,
                (
                    constraints.capacity_notional_col,
                    constraints.capacity_volume_col,
                    constraints.capacity_price_col,
                    constraints.industry_col,
                ),
            )
        )
    )
    if constraints.ask_depth_levels > 0:
        columns.extend(
            f"{prefix}_{level}"
            for level in range(1, int(constraints.ask_depth_levels) + 1)
            for prefix in ("entry_ask_price", "entry_ask_volume", "ask_price", "ask_volume")
        )
    return list(dict.fromkeys(columns))


def _validate_inputs(
    *,
    prediction_available: set[str],
    support_available: set[str],
    constraints: CapacityConstraints,
) -> None:
    available = prediction_available | support_available
    if constraints.score_col not in prediction_available:
        raise SystemExit(f"score column is missing from predictions: {constraints.score_col}")
    if constraints.max_participation_rate > 0:
        has_notional = (
            constraints.capacity_notional_col and constraints.capacity_notional_col in available
        )
        has_volume_price = (
            constraints.capacity_volume_col
            and constraints.capacity_volume_col in available
            and constraints.capacity_price_col in available
        )
        if not has_notional and not has_volume_price:
            raise SystemExit(
                "capacity participation requested but capacity columns are missing; "
                "set capacity_notional_col or capacity_volume_col/capacity_price_col"
            )
    if constraints.ask_depth_levels > 0:
        dummy = pd.DataFrame(columns=sorted(available))
        pairs = ask_depth_pairs(
            dummy,
            levels=constraints.ask_depth_levels,
            allow_decision_depth_fallback=constraints.allow_decision_depth_fallback,
        )
        if len(pairs) < constraints.ask_depth_levels:
            raise SystemExit(
                f"ask-depth audit requested {constraints.ask_depth_levels} levels but "
                f"only found {len(pairs)} usable level(s)"
            )
    if (
        constraints.industry_col
        and constraints.max_industry_weight > 0
        and constraints.industry_col not in available
    ):
        raise SystemExit(f"industry column is missing: {constraints.industry_col}")


def _load_audit_frame(
    *,
    prediction_paths: list[str],
    capacity_paths: list[str],
    constraints: CapacityConstraints,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_files_list = _prediction_files(prediction_paths)
    support_files = _generic_files(capacity_paths) if capacity_paths else []
    print(f"reading_predictions: files={len(prediction_files_list)}")
    prediction_available = available_frame_columns(prediction_files_list)
    support_available = available_frame_columns(support_files) if support_files else set()
    _validate_inputs(
        prediction_available=prediction_available,
        support_available=support_available,
        constraints=constraints,
    )

    support_columns = _capacity_columns(constraints)
    prediction_columns = select_available_columns(
        [*KEY_COLUMNS, constraints.score_col], support_columns, prediction_available
    )
    predictions = read_frame_files(
        prediction_files_list,
        columns=prediction_columns,
        required=KEY_COLUMNS,
    )
    frame = normalize_decision_keys(predictions, key_columns=KEY_COLUMNS)

    if support_files:
        print(f"reading_capacity_support: files={len(support_files)}")
        support_read_columns = select_available_columns(
            KEY_COLUMNS, support_columns, support_available
        )
        support = read_frame_files(
            support_files,
            columns=support_read_columns,
            required=KEY_COLUMNS,
        )
        support = normalize_decision_keys(support, key_columns=KEY_COLUMNS)
        frame = merge_frame_support(frame, support, keys=KEY_COLUMNS)

    trace = {
        "prediction_files": [str(path) for path in prediction_files_list],
        "capacity_files": [str(path) for path in support_files],
        "prediction_rows": int(len(predictions)),
        "joined_rows": int(len(frame)),
    }
    return frame, trace


def _read_prediction_file(
    file: Path,
    *,
    constraints: CapacityConstraints,
) -> pd.DataFrame:
    available = frame_columns(file)
    support_columns = _capacity_columns(constraints)
    columns = select_available_columns(
        [*KEY_COLUMNS, constraints.score_col], support_columns, available
    )
    frame = read_frame(file, columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return normalize_key_columns(frame[columns])


def _run_capacity_audit_streaming(
    *,
    prediction_paths: list[str],
    constraints: CapacityConstraints,
    pools: tuple[str, ...],
    pool_date_lag_sessions: int,
    selected_output_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    files = _prediction_files(prediction_paths)
    prediction_available = available_frame_columns(files)
    _validate_inputs(
        prediction_available=prediction_available,
        support_available=set(),
        constraints=constraints,
    )
    selected_frames = []
    group_frames = []
    stock_pools: dict[str, pd.DataFrame] = {}
    rows_read = 0
    selected_rows_full = 0
    selected_rows_kept = 0
    rows_by_file: dict[str, int] = {}
    print(f"reading_predictions_streaming: files={len(files)}")
    for file in files:
        frame = _read_prediction_file(file, constraints=constraints)
        rows_read += len(frame)
        rows_by_file[str(file)] = int(len(frame))
        print(f"  {file}: rows={len(frame)}")
        for pool in pools:
            pool_name, pool_data = filter_named_stock_pool(
                frame,
                pool,
                date_lag_sessions=pool_date_lag_sessions,
                cache=stock_pools,
            )
            print(f"auditing_capacity_pool: pool={pool_name} rows={len(pool_data)} file={file}")
            selected, metrics = build_capacity_portfolios(
                pool_data,
                constraints,
                pool=pool_name,
            )
            selected_rows_full += len(selected)
            if selected_output_limit <= 0:
                selected_frames.append(selected)
                selected_rows_kept += len(selected)
            elif selected_rows_kept < selected_output_limit:
                keep = min(selected_output_limit - selected_rows_kept, len(selected))
                if keep > 0:
                    selected_frames.append(selected.head(keep).copy())
                    selected_rows_kept += keep
            group_frames.append(metrics)

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    trace = {
        "prediction_files": [str(path) for path in files],
        "prediction_rows": rows_read,
        "joined_rows": rows_read,
        "rows_by_prediction_file": rows_by_file,
        "selected_rows_full": selected_rows_full,
        "selected_rows_kept": selected_rows_kept,
        "streaming": True,
    }
    return selected, group_metrics, trace


def main() -> None:
    args = parse_args()
    config, arguments, run_name = cmd.command_context(args, "capacity_audit")
    prediction_paths = arguments.list("predictions")
    if not prediction_paths:
        raise SystemExit("pass --predictions or set [capacity_audit].predictions")
    pools = tuple(arguments.list("pool", DEFAULT_POOLS) or DEFAULT_POOLS)
    constraints = arguments.resolve_dataclass(CapacityConstraints(target_notional=1_000_000_000.0))
    if not constraints.score_col:
        constraints = replace(constraints, score_col="prediction")
    label_paths = arguments.list("label_input")
    capacity_paths = arguments.list("capacity_input")
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    pool_date_lag_sessions = arguments.integer("pool_date_lag_sessions", 0)
    selected_output_limit = arguments.integer("selected_output_limit", 0)

    if not capacity_paths:
        selected, group_metrics, load_trace = _run_capacity_audit_streaming(
            prediction_paths=prediction_paths,
            constraints=constraints,
            pools=pools,
            pool_date_lag_sessions=pool_date_lag_sessions,
            selected_output_limit=selected_output_limit,
        )
    else:
        frame, load_trace = _load_audit_frame(
            prediction_paths=prediction_paths,
            capacity_paths=capacity_paths,
            constraints=constraints,
        )

        selected_frames = []
        group_frames = []
        for pool in pools:
            pool_name, pool_data = filter_named_stock_pool(
                frame,
                pool,
                date_lag_sessions=pool_date_lag_sessions,
            )
            print(f"auditing_capacity_pool: pool={pool_name} rows={len(pool_data)}")
            selected_part, metrics = build_capacity_portfolios(
                pool_data,
                constraints,
                pool=pool_name,
            )
            selected_frames.append(selected_part)
            group_frames.append(metrics)

        selected = (
            pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
        )
        group_metrics = (
            pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
        )
    daily_summary = summarize_capacity_daily(group_metrics)
    month_summary = summarize_capacity_months(group_metrics)
    summary = summarize_capacity_groups(group_metrics)
    selected_rows_full = int(load_trace.get("selected_rows_full", len(selected)))
    selected_output_rows = int(load_trace.get("selected_rows_kept", len(selected)))
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_name,
        "variant": args.variant or config_str(config, "run", "description", ""),
        "pools": list(pools),
        "pool_date_lag_sessions": pool_date_lag_sessions,
        "selected_rows_full": selected_rows_full,
        "selected_output_limit": int(selected_output_limit),
        "selected_output_rows": selected_output_rows,
        "constraints": constraints.__dict__,
        "ignored_label_inputs": label_paths,
        **load_trace,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in (
        ("group_metrics", group_metrics),
        ("daily_summary", daily_summary),
        ("month_summary", month_summary),
        ("summary", summary),
    ):
        output.to_csv(output_dir / f"capacity_audit_{name}.csv", index=False)
    write_json(output_dir / "capacity_audit_trace.json", trace, ensure_ascii=True)
    selected_to_write = (
        selected.head(selected_output_limit).copy()
        if 0 < selected_output_limit < len(selected)
        else selected
    )
    selected_to_write.to_csv(output_dir / "capacity_audit_selected.csv", index=False)

    records_dir = arguments.string("records_dir")
    record_paths = record_requested_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix=arguments.string("record_prefix") or run_name,
        names=CAPACITY_AUDIT_ARTIFACTS,
    )

    print("\ncapacity_audit_summary:")
    display_cols = (
        "pool groups fill_ratio fill_success_rate mean_top_depth_to_target "
        "p95_top_depth_to_target max_top_depth_to_target selected_rows filled_groups "
        "unfilled_groups max_symbol_weight"
    ).split()
    print(summary[display_cols].to_string(index=False) if not summary.empty else "empty")
    print_recorded_artifacts(record_paths, "capacity_audit")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
