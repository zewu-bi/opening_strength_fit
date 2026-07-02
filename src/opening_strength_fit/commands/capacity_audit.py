from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.capacity_audit import (
    CapacityConstraints,
    ask_depth_pairs,
    build_capacity_portfolios,
    summarize_capacity_daily,
    summarize_capacity_groups,
    summarize_capacity_months,
)
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import frame_columns, read_frame
from opening_strength_fit.prediction_frames import prediction_files
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

DEFAULT_POOLS = ("L",)
OUTPUT_FILES = (
    "capacity_audit_selected.csv",
    "capacity_audit_group_metrics.csv",
    "capacity_audit_daily_summary.csv",
    "capacity_audit_month_summary.csv",
    "capacity_audit_summary.csv",
    "capacity_audit_trace.json",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a score-ranked capacity-constrained portfolio per decision group "
            "and audit fill, participation, concentration, and capacity depth."
        )
    )
    parser.add_argument("--config", default="")
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
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--label-col", default="", help=argparse.SUPPRESS)
    parser.add_argument("--target-notional", type=float, default=None)
    parser.add_argument("--capacity-notional-col", default="")
    parser.add_argument("--capacity-volume-col", default="")
    parser.add_argument("--capacity-price-col", default="")
    parser.add_argument("--max-participation-rate", type=float, default=None)
    parser.add_argument("--max-symbol-weight", type=float, default=None)
    parser.add_argument("--min-trade-notional", type=float, default=None)
    parser.add_argument("--max-names", type=int, default=None)
    parser.add_argument("--ask-depth-levels", type=int, default=None)
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
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--record-prefix", default="")
    parser.add_argument(
        "--selected-output-limit",
        type=int,
        default=None,
        help="Maximum selected rows to write. 0 writes all selected rows.",
    )
    return parser.parse_args()


def _arg_list(
    args: argparse.Namespace, config: dict, name: str, default: Iterable[str]
) -> list[str]:
    value = getattr(args, name)
    if value:
        return list(value)
    return config_list(config, "capacity_audit", name, tuple(default))


def _arg_str(args: argparse.Namespace, config: dict, name: str, default: str) -> str:
    value = getattr(args, name)
    return (
        str(value)
        if value not in (None, "")
        else config_str(
            config,
            "capacity_audit",
            name,
            default,
        )
    )


def _arg_float(args: argparse.Namespace, config: dict, name: str, default: float) -> float:
    value = getattr(args, name)
    return (
        float(value)
        if value is not None
        else config_float(
            config,
            "capacity_audit",
            name,
            default,
        )
    )


def _arg_int(args: argparse.Namespace, config: dict, name: str, default: int) -> int:
    value = getattr(args, name)
    return (
        int(value)
        if value is not None
        else config_int(
            config,
            "capacity_audit",
            name,
            default,
        )
    )


def _frame_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    files = sorted(path.rglob("*.parquet"))
    if not files:
        files = sorted(path.rglob("*.csv")) + sorted(path.rglob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no parquet/csv files found under: {path}")
    return files


def _prediction_files(paths: list[str]) -> list[Path]:
    return [file for raw in paths for file in prediction_files(Path(raw))]


def _generic_files(paths: list[str]) -> list[Path]:
    return [file for raw in paths for file in _frame_files(Path(raw))]


def _available_columns(files: list[Path]) -> set[str]:
    columns: set[str] = set()
    for file in files:
        columns |= frame_columns(file)
    return columns


def _read_files(
    files: list[Path],
    *,
    columns: list[str],
    required: Iterable[str],
) -> pd.DataFrame:
    required_set = set(required)
    frames: list[pd.DataFrame] = []
    for file in files:
        available = frame_columns(file)
        missing = sorted(required_set - available)
        if missing:
            raise SystemExit(f"{file}: missing required columns: {missing}")
        read_columns = [column for column in columns if column in available]
        frame = read_frame(file, columns=read_columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        print(f"  {file}: rows={len(frame)}")
        frames.append(frame[columns])
    if not frames:
        raise SystemExit("no input files supplied")
    return pd.concat(frames, ignore_index=True)


def _merge_keyed(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty:
        return left
    keyed = right.drop_duplicates(list(KEY_COLUMNS), keep="last")
    overlap = [
        column for column in keyed.columns if column not in KEY_COLUMNS and column in left.columns
    ]
    merged = left.merge(
        keyed,
        on=list(KEY_COLUMNS),
        how="left",
        suffixes=("", "_support"),
        validate="many_to_one",
    )
    for column in overlap:
        support_col = f"{column}_support"
        if support_col in merged.columns:
            merged[column] = merged[column].combine_first(merged[support_col])
            merged = merged.drop(columns=[support_col])
    return merged


def _capacity_columns(constraints: CapacityConstraints) -> list[str]:
    columns = []
    for column in (
        constraints.capacity_notional_col,
        constraints.capacity_volume_col,
        constraints.capacity_price_col,
        constraints.industry_col,
    ):
        if column and column not in columns:
            columns.append(column)
    if constraints.ask_depth_levels > 0:
        for level in range(1, int(constraints.ask_depth_levels) + 1):
            for column in (
                f"entry_ask_price_{level}",
                f"entry_ask_volume_{level}",
                f"ask_price_{level}",
                f"ask_volume_{level}",
            ):
                if column not in columns:
                    columns.append(column)
    return columns


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
    prediction_available = _available_columns(prediction_files_list)
    support_available = _available_columns(support_files) if support_files else set()
    _validate_inputs(
        prediction_available=prediction_available,
        support_available=support_available,
        constraints=constraints,
    )

    support_columns = _capacity_columns(constraints)
    prediction_columns = [*KEY_COLUMNS, constraints.score_col]
    for column in support_columns:
        if column in prediction_available and column not in prediction_columns:
            prediction_columns.append(column)
    predictions = _read_files(
        prediction_files_list,
        columns=prediction_columns,
        required=KEY_COLUMNS,
    )
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    predictions["symbol"] = predictions["symbol"].astype(str)
    predictions["decision_target_timestamp"] = pd.to_datetime(
        predictions["decision_target_timestamp"],
        errors="coerce",
    )
    frame = predictions.dropna(subset=list(KEY_COLUMNS)).copy()

    if support_files:
        print(f"reading_capacity_support: files={len(support_files)}")
        support_read_columns = [*KEY_COLUMNS]
        for column in support_columns:
            if column in support_available and column not in support_read_columns:
                support_read_columns.append(column)
        support = _read_files(
            support_files,
            columns=support_read_columns,
            required=KEY_COLUMNS,
        )
        support["date"] = pd.to_datetime(support["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        support["symbol"] = support["symbol"].astype(str)
        support["decision_target_timestamp"] = pd.to_datetime(
            support["decision_target_timestamp"],
            errors="coerce",
        )
        support = support.dropna(subset=list(KEY_COLUMNS)).copy()
        frame = _merge_keyed(frame, support)

    trace = {
        "prediction_files": [str(path) for path in prediction_files_list],
        "capacity_files": [str(path) for path in _generic_files(capacity_paths)]
        if capacity_paths
        else [],
        "prediction_rows": int(len(predictions)),
        "joined_rows": int(len(frame)),
    }
    return frame, trace


def _pool_frame(
    frame: pd.DataFrame,
    *,
    pool: str,
    pool_date_lag_sessions: int,
    stock_pools: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, pd.DataFrame]:
    if pool == "universe":
        return "universe", frame
    pool_path = DEFAULT_STOCK_POOL_PATHS[pool]
    print(f"loading_stock_pool: pool={pool} path={pool_path}")
    if stock_pools is None:
        stock_pool = load_stock_pool(pool_path)
    else:
        stock_pool = stock_pools.get(pool)
        if stock_pool is None:
            stock_pool = load_stock_pool(pool_path)
            stock_pools[pool] = stock_pool
    mask = stock_pool_membership_mask(
        frame,
        stock_pool,
        date_lag_sessions=pool_date_lag_sessions,
    )
    return f"pool_{pool}", frame.loc[mask].copy()


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    return out.dropna(subset=list(KEY_COLUMNS)).copy()


def _read_prediction_file(
    file: Path,
    *,
    constraints: CapacityConstraints,
) -> pd.DataFrame:
    available = frame_columns(file)
    support_columns = _capacity_columns(constraints)
    columns = [*KEY_COLUMNS, constraints.score_col]
    for column in support_columns:
        if column in available and column not in columns:
            columns.append(column)
    frame = read_frame(file, columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return _normalize_keys(frame[columns])


def _run_capacity_audit_streaming(
    *,
    prediction_paths: list[str],
    constraints: CapacityConstraints,
    pools: tuple[str, ...],
    pool_date_lag_sessions: int,
    selected_output_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    files = _prediction_files(prediction_paths)
    prediction_available = _available_columns(files)
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
            pool_name, pool_data = _pool_frame(
                frame,
                pool=pool,
                pool_date_lag_sessions=pool_date_lag_sessions,
                stock_pools=stock_pools,
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


def _write_outputs(
    *,
    output_dir: Path,
    selected: pd.DataFrame,
    group_metrics: pd.DataFrame,
    daily_summary: pd.DataFrame,
    month_summary: pd.DataFrame,
    summary: pd.DataFrame,
    trace: dict[str, object],
    selected_output_limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_metrics.to_csv(output_dir / "capacity_audit_group_metrics.csv", index=False)
    daily_summary.to_csv(output_dir / "capacity_audit_daily_summary.csv", index=False)
    month_summary.to_csv(output_dir / "capacity_audit_month_summary.csv", index=False)
    summary.to_csv(output_dir / "capacity_audit_summary.csv", index=False)
    write_json(output_dir / "capacity_audit_trace.json", trace, ensure_ascii=True)
    selected_to_write = selected
    if selected_output_limit > 0 and len(selected) > selected_output_limit:
        selected_to_write = selected.head(selected_output_limit).copy()
    selected_to_write.to_csv(output_dir / "capacity_audit_selected.csv", index=False)


def record_capacity_audit_outputs(
    *,
    output_dir: Path,
    records_dir: Path,
    record_prefix: str,
) -> list[Path]:
    archive_dir = records_dir / "backtests" / record_prefix
    copied: list[Path] = []
    for name in OUTPUT_FILES:
        source = output_dir / name
        if not source.exists():
            continue
        destination = archive_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = args.run_id or (run_id(config, args.config) if args.config else "capacity_audit")
    prediction_paths = _arg_list(args, config, "predictions", ())
    if not prediction_paths:
        raise SystemExit("pass --predictions or set [capacity_audit].predictions")
    pools = tuple(_arg_list(args, config, "pool", DEFAULT_POOLS) or DEFAULT_POOLS)
    constraints = CapacityConstraints(
        target_notional=_arg_float(args, config, "target_notional", 1_000_000_000.0),
        score_col=_arg_str(args, config, "score_col", "prediction") or "prediction",
        capacity_notional_col=_arg_str(
            args,
            config,
            "capacity_notional_col",
            "turnover_diff_30t",
        ),
        capacity_volume_col=_arg_str(args, config, "capacity_volume_col", ""),
        capacity_price_col=_arg_str(args, config, "capacity_price_col", "ask_price_1"),
        max_participation_rate=_arg_float(args, config, "max_participation_rate", 0.10),
        max_symbol_weight=_arg_float(args, config, "max_symbol_weight", 0.01),
        min_trade_notional=_arg_float(args, config, "min_trade_notional", 0.0),
        max_names=_arg_int(args, config, "max_names", 0),
        ask_depth_levels=_arg_int(args, config, "ask_depth_levels", 0),
        ask_depth_participation_rate=_arg_float(
            args,
            config,
            "ask_depth_participation_rate",
            0.25,
        ),
        allow_decision_depth_fallback=bool(args.allow_decision_depth_fallback)
        or config_bool(config, "capacity_audit", "allow_decision_depth_fallback", False),
        industry_col=_arg_str(args, config, "industry_col", ""),
        max_industry_weight=_arg_float(args, config, "max_industry_weight", 0.0),
    )
    label_paths = _arg_list(args, config, "label_input", ())
    capacity_paths = _arg_list(args, config, "capacity_input", ())
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    pool_date_lag_sessions = (
        args.pool_date_lag_sessions
        if args.pool_date_lag_sessions is not None
        else config_int(config, "capacity_audit", "pool_date_lag_sessions", 0)
    )
    selected_output_limit = (
        args.selected_output_limit
        if args.selected_output_limit is not None
        else config_int(config, "capacity_audit", "selected_output_limit", 0)
    )

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
            pool_name, pool_data = _pool_frame(
                frame,
                pool=pool,
                pool_date_lag_sessions=pool_date_lag_sessions,
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
    _write_outputs(
        output_dir=output_dir,
        selected=selected,
        group_metrics=group_metrics,
        daily_summary=daily_summary,
        month_summary=month_summary,
        summary=summary,
        trace=trace,
        selected_output_limit=selected_output_limit,
    )

    record_paths: list[Path] = []
    records_dir = args.records_dir or config_str(config, "capacity_audit", "records_dir", "")
    if records_dir:
        record_prefix = (
            args.record_prefix
            or config_str(config, "capacity_audit", "record_prefix", "")
            or run_name
        )
        record_paths = record_capacity_audit_outputs(
            output_dir=output_dir,
            records_dir=Path(records_dir),
            record_prefix=record_prefix,
        )

    print("\ncapacity_audit_summary:")
    display_cols = [
        "pool",
        "groups",
        "fill_ratio",
        "fill_success_rate",
        "mean_top_depth_to_target",
        "p95_top_depth_to_target",
        "max_top_depth_to_target",
        "selected_rows",
        "filled_groups",
        "unfilled_groups",
        "max_symbol_weight",
    ]
    print(summary[display_cols].to_string(index=False) if not summary.empty else "empty")
    if record_paths:
        print("\nrecorded_capacity_audit_outputs:")
        for path in record_paths:
            print(f"  {path}")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
