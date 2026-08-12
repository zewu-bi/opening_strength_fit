from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.artifact_catalog import STRATEGY_ACCEPTANCE_ARTIFACTS
from opening_strength_fit.capacity_acceptance import load_label_frame
from opening_strength_fit.capacity_audit import (
    CapacityConstraints,
    build_capacity_portfolios,
    summarize_capacity_groups,
)
from opening_strength_fit.commands.arguments import CommandArguments, command_context
from opening_strength_fit.config import (
    config_float_tuple,
    config_int,
    config_str,
)
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic
from opening_strength_fit.prediction_frames import prediction_files
from opening_strength_fit.realistic_acceptance import (
    RealisticExecutionConstraints,
    apply_realistic_execution_constraints,
    merge_realistic_execution_context,
    summarize_realistic_acceptance,
    summarize_realistic_acceptance_overall,
)
from opening_strength_fit.schema import normalize_decision_keys
from opening_strength_fit.stock_pool import load_stock_pool, stock_pool_membership_mask
from opening_strength_fit.strategy_acceptance import (
    CAPACITY_ONLY,
    POLICIES,
    REALISTIC_NO_REFILL,
    VISIBLE_PRETRADE_REFILL,
    TailSettings,
    add_execution_context_columns,
    build_visible_pretrade_refill,
    group_targets_from_metrics,
    leave_one_period_out,
    monthly_block_bootstrap,
    summarize_overlap,
    summarize_selected_groups,
    summarize_tail_robustness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capacity, realistic no-refill, visible pre-trade refill, overlap, "
        "and tail-robustness acceptance from one prediction lineage."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _configured_constraints[T](config: dict, section: str, defaults: T) -> T:
    return CommandArguments(argparse.Namespace(), config, section).resolve_dataclass(defaults)


def capacity_constraints_from_config(config: dict) -> CapacityConstraints:
    return _configured_constraints(
        config,
        "capacity",
        CapacityConstraints(
            target_notional=50_000_000.0,
            capacity_notional_col="turnover_diff_10t",
            max_participation_rate=0.20,
        ),
    )


def execution_constraints_from_config(config: dict) -> RealisticExecutionConstraints:
    return _configured_constraints(
        config,
        "execution",
        RealisticExecutionConstraints(
            min_child_notional=10_000.0,
            round_lot_shares=100,
            price_col="capacity_price",
            status_col="status",
            tradable_statuses=("T0", "TRADE"),
            max_spread_bps=50.0,
            max_ask_depth_participation_rate=0.25,
        ),
    )


def _required_prediction_columns(
    available: set[str],
    capacity: CapacityConstraints,
    execution: RealisticExecutionConstraints,
) -> list[str]:
    required = {*KEY_COLUMNS, capacity.score_col}
    if capacity.max_participation_rate > 0:
        if capacity.capacity_notional_col:
            required.add(capacity.capacity_notional_col)
        else:
            required |= {capacity.capacity_volume_col, capacity.capacity_price_col}
    if capacity.capacity_price_col:
        required.add(capacity.capacity_price_col)
    missing = sorted(column for column in required if column and column not in available)
    if missing:
        raise SystemExit(f"prediction input missing required columns: {missing}")

    desired = set(required) | set(
        "status ask_price_1 bid_price_1 mid_price spread_bps ask1_to_limit_up_bps "
        "ask_depth_notional ask_depth_10 industry".split()
    )
    desired.update(
        filter(
            None,
            (
                execution.price_col,
                execution.status_col,
                execution.spread_bps_col,
                execution.limit_up_room_bps_col,
                execution.ask_depth_notional_col,
                execution.industry_col,
                capacity.industry_col,
            ),
        )
    )
    if capacity.ask_depth_levels > 0:
        desired.update(
            f"{prefix}_{level}"
            for level in range(1, capacity.ask_depth_levels + 1)
            for prefix in ("entry_ask_price", "entry_ask_volume", "ask_price", "ask_volume")
        )
    return sorted(column for column in desired if column in available)


def _validate_execution_context(
    frame: pd.DataFrame,
    constraints: RealisticExecutionConstraints,
) -> None:
    requirements = (
        (
            constraints.status_col and constraints.tradable_statuses,
            constraints.status_col,
            "status",
        ),
        (constraints.max_spread_bps > 0, constraints.spread_bps_col, "spread"),
        (constraints.min_limit_up_room_bps > 0, constraints.limit_up_room_bps_col, "limit-up room"),
        (
            constraints.max_ask_depth_participation_rate > 0,
            constraints.ask_depth_notional_col,
            "ask-depth",
        ),
    )
    for required, column, label in requirements:
        if required and column not in frame.columns:
            raise SystemExit(f"execution {label} column is missing: {column}")


def _context_frame(frame: pd.DataFrame, execution: RealisticExecutionConstraints) -> pd.DataFrame:
    columns = [*KEY_COLUMNS]
    for column in (
        execution.price_col,
        execution.status_col,
        execution.spread_bps_col,
        execution.limit_up_room_bps_col,
        execution.ask_depth_notional_col,
        execution.industry_col if execution.max_daily_industry_weight > 0 else "",
    ):
        if column and column in frame.columns and column not in columns:
            columns.append(column)
    return frame[columns].drop_duplicates(list(KEY_COLUMNS), keep="last")


def _policy_daily_and_summary(
    selected: pd.DataFrame,
    targets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    policy: str,
    constraints: RealisticExecutionConstraints,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_selected = selected.loc[selected["policy"].eq(policy)].drop(columns="policy")
    policy_targets = targets.loc[targets["policy"].eq(policy)].drop(columns="policy")
    daily = summarize_realistic_acceptance(
        policy_selected,
        policy_targets,
        labels,
        constraints=constraints,
        label_col=label_col,
    )
    daily.insert(0, "policy", policy)
    summary = summarize_realistic_acceptance_overall(daily.drop(columns="policy"))
    summary.insert(0, "policy", policy)
    return daily, summary


def _tail_summary_wide(tail: pd.DataFrame) -> pd.DataFrame:
    value_columns = "raw_net_bps_vs_target winsor_net_bps_vs_target trim_net_bps_vs_target tail_notional_share".split()
    parts = []
    for threshold, item in tail.groupby("threshold", observed=True):
        renamed = item[["policy", "pool", *value_columns]].rename(
            columns={column: f"{threshold}_{column}" for column in value_columns}
        )
        parts.append(renamed)
    if not parts:
        return pd.DataFrame(columns=["policy", "pool"])
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["policy", "pool"], how="outer")
    return out


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config, arguments, run_name = command_context(args, "strategy_acceptance")
    variant = arguments.string("variant", run_name)
    prediction_roots = arguments.list("predictions")
    label_inputs = arguments.list("label_input")
    label_col = arguments.string("label_col", "alpha_return_next_close")
    pool_code = arguments.string("pool", "L").upper()
    pool_path = arguments.string("pool_path", f"lml.bzw@ssd/data/pool_{pool_code}.parquet")
    pool_lag = arguments.integer("pool_date_lag_sessions", 0)
    policies = arguments.tuple("policies", POLICIES)
    unknown_policies = sorted(set(policies) - set(POLICIES))
    if unknown_policies:
        raise SystemExit(f"unsupported strategy policies: {unknown_policies}")
    if not prediction_roots:
        raise SystemExit("[strategy_acceptance].predictions is required")
    if not label_inputs:
        raise SystemExit("[strategy_acceptance].label_input is required")
    if pool_code not in {"L", "M", "S", "UNIVERSE"}:
        raise SystemExit("[strategy_acceptance].pool must be L, M, S, or universe")

    output_dir = Path(
        args.output_dir or config_str(config, "output", "local_dir", f"output/artifacts/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    capacity = capacity_constraints_from_config(config)
    execution = execution_constraints_from_config(config)
    tail_settings = TailSettings(
        quantiles=config_float_tuple(config, "tail", "quantiles", (0.95, 0.99)),
        bootstrap_samples=config_int(config, "tail", "bootstrap_samples", 2_000),
        bootstrap_seed=config_int(config, "tail", "bootstrap_seed", 20260722),
    )

    files = [file for root in prediction_roots for file in prediction_files(Path(root))]
    stock_pool = None if pool_code == "UNIVERSE" else load_stock_pool(pool_path)
    pool_name = "universe" if pool_code == "UNIVERSE" else f"pool_{pool_code}"
    selected_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    capacity_metric_parts: list[pd.DataFrame] = []
    refill_metric_parts: list[pd.DataFrame] = []
    seen_dates: set[str] = set()
    input_trace: list[dict[str, object]] = []

    for file in files:
        available = frame_columns(file)
        columns = _required_prediction_columns(available, capacity, execution)
        frame = read_frame(file, columns=columns)
        frame = normalize_decision_keys(frame, drop_missing=True)
        file_dates = set(frame["date"].astype(str).unique())
        overlap_dates = sorted(seen_dates & file_dates)
        if overlap_dates:
            raise SystemExit(f"prediction shards overlap dates: {overlap_dates[:5]}")
        seen_dates |= file_dates
        input_rows = len(frame)
        if stock_pool is not None:
            mask = stock_pool_membership_mask(
                frame,
                stock_pool,
                date_lag_sessions=pool_lag,
            )
            frame = frame.loc[mask].copy()
        frame = add_execution_context_columns(frame)
        _validate_execution_context(frame, execution)
        context = _context_frame(frame, execution)
        print(
            f"strategy_acceptance_input: file={file} rows={input_rows} pool_rows={len(frame)}",
            flush=True,
        )

        capacity_selected, capacity_metrics = build_capacity_portfolios(
            frame,
            capacity,
            pool=pool_name,
        )
        capacity_metric_parts.append(capacity_metrics)
        capacity_targets = group_targets_from_metrics(capacity_metrics, policy=CAPACITY_ONLY)
        capacity_selected = merge_realistic_execution_context(capacity_selected, context)
        capacity_selected["original_allocated_notional"] = capacity_selected["allocated_notional"]
        capacity_selected["execution_fill_rate"] = 1.0
        if CAPACITY_ONLY in policies:
            part = capacity_selected.copy()
            part.insert(0, "policy", CAPACITY_ONLY)
            selected_parts.append(part)
            target_parts.append(capacity_targets)

        if REALISTIC_NO_REFILL in policies:
            no_refill, _ = apply_realistic_execution_constraints(capacity_selected, execution)
            no_refill.insert(0, "policy", REALISTIC_NO_REFILL)
            selected_parts.append(no_refill)
            no_refill_targets = capacity_targets.copy()
            no_refill_targets["policy"] = REALISTIC_NO_REFILL
            target_parts.append(no_refill_targets)

        if VISIBLE_PRETRADE_REFILL in policies:
            refill, refill_metrics = build_visible_pretrade_refill(
                frame,
                pool=pool_name,
                capacity_constraints=capacity,
                execution_constraints=execution,
            )
            refill.insert(0, "policy", VISIBLE_PRETRADE_REFILL)
            selected_parts.append(refill)
            target_parts.append(
                group_targets_from_metrics(
                    refill_metrics,
                    policy=VISIBLE_PRETRADE_REFILL,
                )
            )
            refill_metric_parts.append(refill_metrics)

        input_trace.append(
            {
                "path": str(file),
                "input_rows": int(input_rows),
                "pool_rows": int(len(frame)),
                "dates": int(len(file_dates)),
                "capacity_selected_rows": int(len(capacity_selected)),
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    targets = pd.concat(target_parts, ignore_index=True)
    capacity_metrics = pd.concat(capacity_metric_parts, ignore_index=True)
    refill_metrics = (
        pd.concat(refill_metric_parts, ignore_index=True) if refill_metric_parts else pd.DataFrame()
    )
    labels = load_label_frame(
        tuple(label_inputs),
        label_col=label_col,
        dates=set(targets["date"].astype(str)),
    )

    daily_parts = []
    summary_parts = []
    group_parts = []
    for policy in policies:
        daily, summary = _policy_daily_and_summary(
            selected,
            targets,
            labels,
            policy=policy,
            constraints=execution,
            label_col=label_col,
        )
        daily_parts.append(daily)
        summary_parts.append(summary)
        group_parts.append(summarize_selected_groups(selected, targets, policy=policy))
        policy_path = output_dir / f"strategy_acceptance_selected_{policy}.parquet"
        write_frame_atomic(
            selected.loc[selected["policy"].eq(policy)].drop(columns="policy"),
            policy_path,
        )

    daily = pd.concat(daily_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    group_metrics = pd.concat(group_parts, ignore_index=True)
    positions, overlap_daily, adjacent, overlap_summary = summarize_overlap(
        selected,
        targets,
        capacity_total_notional=execution.capacity_total_notional,
    )
    tail, tail_monthly, tail_concentration = summarize_tail_robustness(
        selected,
        targets,
        labels,
        label_col=label_col,
        fee_bps=execution.fee_bps,
        settings=tail_settings,
    )
    bootstrap = monthly_block_bootstrap(daily, settings=tail_settings)
    leave_one_out = leave_one_period_out(daily)
    summary = summary.merge(overlap_summary, on=["policy", "pool"], how="left")
    summary = summary.merge(_tail_summary_wide(tail), on=["policy", "pool"], how="left")
    summary = summary.merge(bootstrap, on=["policy", "pool"], how="left")
    capacity_summary = summarize_capacity_groups(capacity_metrics)

    artifacts = (
        summary,
        daily,
        group_metrics,
        capacity_summary,
        positions,
        overlap_summary,
        overlap_daily,
        adjacent,
        tail,
        tail_monthly,
        tail_concentration,
        bootstrap,
        leave_one_out,
    )
    artifact_names = (
        "strategy_acceptance_summary.csv strategy_acceptance_daily.csv "
        "strategy_acceptance_group_metrics.csv strategy_acceptance_capacity_summary.csv "
        "strategy_acceptance_daily_positions.parquet strategy_acceptance_overlap_summary.csv "
        "strategy_acceptance_overlap_daily.csv strategy_acceptance_overlap_adjacent.csv "
        "strategy_acceptance_tail_summary.csv strategy_acceptance_tail_monthly.csv "
        "strategy_acceptance_tail_concentration.csv strategy_acceptance_bootstrap.csv "
        "strategy_acceptance_leave_one_out.csv"
    ).split()
    for artifact, name in zip(artifacts, artifact_names, strict=True):
        write_frame_atomic(artifact, output_dir / name)
    if not refill_metrics.empty:
        write_frame_atomic(
            refill_metrics, output_dir / "strategy_acceptance_refill_group_metrics.csv"
        )

    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_name,
        "variant": variant,
        "config": str(config_path),
        "source_revision": os.environ.get("OPENING_STRENGTH_SOURCE_REVISION", ""),
        "prediction_roots": prediction_roots,
        "prediction_files": input_trace,
        "label_inputs": label_inputs,
        "label_col": label_col,
        "pool": pool_name,
        "pool_path": pool_path if stock_pool is not None else "",
        "pool_date_lag_sessions": pool_lag,
        "policies": list(policies),
        "capacity_constraints": asdict(capacity),
        "execution_constraints": asdict(execution),
        "tail_settings": asdict(tail_settings),
        "selected_rows_by_policy": {
            policy: int(selected["policy"].eq(policy).sum()) for policy in policies
        },
        "group_targets_by_policy": {
            policy: int(targets["policy"].eq(policy).sum()) for policy in policies
        },
        "modeling_notes": {
            CAPACITY_ONLY: "Score-ranked per-decision capacity allocation before execution filters.",
            REALISTIC_NO_REFILL: (
                "Selected-order replay; rejected or clipped orders remain cash and lower-ranked "
                "candidates are not revisited."
            ),
            VISIBLE_PRETRADE_REFILL: (
                "Full candidate ranking is filtered and allocated using only decision-time "
                "status, spread, depth, capacity, lot, and daily concentration state. It is not "
                "an instantaneous retry after observing a realized fill failure."
            ),
            "holding": (
                "Overlap treats all opening slices as simultaneously funded until next close; "
                "this is an aggregate audit, not a general intraday entry/exit ledger."
            ),
        },
        "compact_artifacts": list(STRATEGY_ACCEPTANCE_ARTIFACTS),
    }
    write_json(output_dir / "strategy_acceptance_trace.json", trace, ensure_ascii=True)
    write_json(
        output_dir / "_SUCCESS",
        {
            "run_id": run_name,
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "format_version": 1,
        },
        ensure_ascii=True,
    )
    print("strategy_acceptance_summary:")
    print(summary.to_string(index=False))
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
