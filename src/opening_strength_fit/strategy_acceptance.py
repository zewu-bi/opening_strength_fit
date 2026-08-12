from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.capacity_acceptance import RETURN_BPS_DENOMINATOR
from opening_strength_fit.capacity_audit import (
    CapacityConstraints,
    add_capacity_limits,
    capacity_selection_record,
    finite_numeric,
    normalize_capacity_frame,
)
from opening_strength_fit.realistic_acceptance import (
    RealisticExecutionConstraints,
    round_notional_to_lot,
    row_depth_limit,
    tradable_mask,
    validate_execution_constraints,
)

CAPACITY_ONLY = "capacity_only"
REALISTIC_NO_REFILL = "realistic_no_refill"
VISIBLE_PRETRADE_REFILL = "visible_pretrade_refill"
POLICIES = (CAPACITY_ONLY, REALISTIC_NO_REFILL, VISIBLE_PRETRADE_REFILL)
GROUP_COLUMNS = ("pool", "date", "decision_target_timestamp")


@dataclass(frozen=True)
class TailSettings:
    quantiles: tuple[float, ...] = (0.95, 0.99)
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20260722


def add_execution_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the execution fields shared by replay and causal refill."""

    out = frame.copy()
    if "capacity_price" not in out.columns and "ask_price_1" in out.columns:
        out["capacity_price"] = finite_numeric(out["ask_price_1"])
    if "spread_bps" not in out.columns and {"ask_price_1", "bid_price_1"} <= set(out):
        ask = finite_numeric(out["ask_price_1"])
        bid = finite_numeric(out["bid_price_1"])
        mid = (ask + bid) / 2.0
        out["spread_bps"] = np.where(
            mid.gt(0),
            (ask - bid) / mid * RETURN_BPS_DENOMINATOR,
            np.nan,
        )
    if "ask_depth_notional" not in out.columns:
        if {"ask_depth_10", "ask_price_1"} <= set(out):
            out["ask_depth_notional"] = finite_numeric(out["ask_depth_10"]) * finite_numeric(
                out["ask_price_1"]
            )
        else:
            depth: pd.Series | None = None
            for level in range(1, 11):
                price_col = f"ask_price_{level}"
                volume_col = f"ask_volume_{level}"
                if price_col not in out.columns or volume_col not in out.columns:
                    continue
                part = finite_numeric(out[price_col]) * finite_numeric(out[volume_col])
                depth = part if depth is None else depth.add(part, fill_value=0.0)
            if depth is not None:
                out["ask_depth_notional"] = depth
    return out


def _selected_refill_record(
    row: pd.Series,
    *,
    pool: str,
    constraints: CapacityConstraints,
    allocated: float,
) -> dict[str, object]:
    record = capacity_selection_record(
        row,
        pool=pool,
        constraints=constraints,
        rank=int(row["candidate_rank"]),
        symbol=str(row["symbol"]),
        allocated=allocated,
        extra_allocations={"original_allocated_notional": allocated},
        audit_depth=False,
    )
    record.update(
        ask_depth_notional=float(row.get("ask_depth_notional", np.nan)),
        execution_fill_rate=1.0,
    )
    for column in (
        "status",
        "spread_bps",
        "ask1_to_limit_up_bps",
        "industry",
        "ask_price_1",
        "bid_price_1",
        "mid_price",
    ):
        if column in row.index:
            record[column] = row[column]
    return record


def build_visible_pretrade_refill(
    candidates: pd.DataFrame,
    *,
    pool: str,
    capacity_constraints: CapacityConstraints,
    execution_constraints: RealisticExecutionConstraints,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate from the full visible candidate ranking after execution eligibility checks.

    This is a decision-time policy, not an instantaneous retry after an observed fill failure.
    All eligibility and limits must therefore be present in ``candidates`` at the decision time.
    """

    validate_execution_constraints(execution_constraints)
    work = add_execution_context_columns(normalize_capacity_frame(candidates))
    work = add_capacity_limits(work, capacity_constraints)
    work[capacity_constraints.score_col] = finite_numeric(work[capacity_constraints.score_col])
    work = work.dropna(subset=[capacity_constraints.score_col]).copy()
    work = work.sort_values(
        ["date", "decision_target_timestamp", capacity_constraints.score_col, "symbol"],
        ascending=[True, True, False, True],
    )
    work["candidate_rank"] = (
        work.groupby(["date", "decision_target_timestamp"], sort=False).cumcount() + 1
    )
    work["_execution_eligible"] = tradable_mask(work, execution_constraints)

    target = float(capacity_constraints.target_notional)
    daily_symbol_cap = (
        float(execution_constraints.capacity_total_notional)
        * float(execution_constraints.max_daily_symbol_weight)
        if execution_constraints.max_daily_symbol_weight > 0
        else np.inf
    )
    daily_industry_cap = (
        float(execution_constraints.capacity_total_notional)
        * float(execution_constraints.max_daily_industry_weight)
        if execution_constraints.max_daily_industry_weight > 0
        else np.inf
    )
    selected_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for date, day in work.groupby("date", sort=False):
        symbol_remaining: dict[str, float] = {}
        industry_remaining: dict[str, float] = {}
        symbol_decisions: dict[str, set[pd.Timestamp]] = {}
        for decision_time, group in day.groupby("decision_target_timestamp", sort=True):
            remaining = target
            group_selected: list[dict[str, object]] = []
            eligible = group.loc[group["_execution_eligible"] & group["_row_limit_notional"].gt(0)]
            for _, row in eligible.iterrows():
                if capacity_constraints.max_names > 0 and len(group_selected) >= int(
                    capacity_constraints.max_names
                ):
                    break
                symbol = str(row["symbol"])
                used_decisions = symbol_decisions.setdefault(symbol, set())
                if (
                    execution_constraints.max_symbol_decision_count > 0
                    and decision_time not in used_decisions
                    and len(used_decisions) >= int(execution_constraints.max_symbol_decision_count)
                ):
                    continue
                symbol_room = symbol_remaining.setdefault(symbol, float(daily_symbol_cap))
                allowed = min(
                    float(row["_row_limit_notional"]),
                    remaining,
                    max(0.0, symbol_room),
                    row_depth_limit(row, execution_constraints),
                )
                industry_value = ""
                if (
                    execution_constraints.max_daily_industry_weight > 0
                    and execution_constraints.industry_col
                    and execution_constraints.industry_col in row.index
                ):
                    industry_value = str(row[execution_constraints.industry_col])
                    industry_room = industry_remaining.setdefault(
                        industry_value,
                        float(daily_industry_cap),
                    )
                    allowed = min(allowed, max(0.0, industry_room))
                allowed *= float(execution_constraints.execution_fill_rate)
                allowed = round_notional_to_lot(allowed, row, execution_constraints)
                if (
                    execution_constraints.min_child_notional > 0
                    and allowed < execution_constraints.min_child_notional
                ):
                    continue
                if allowed <= 0:
                    continue
                record = _selected_refill_record(
                    row,
                    pool=pool,
                    constraints=capacity_constraints,
                    allocated=allowed,
                )
                record["execution_fill_rate"] = float(execution_constraints.execution_fill_rate)
                group_selected.append(record)
                selected_rows.append(record)
                remaining -= allowed
                symbol_remaining[symbol] = max(0.0, symbol_room - allowed)
                used_decisions.add(pd.Timestamp(decision_time))
                if industry_value:
                    industry_remaining[industry_value] = max(
                        0.0,
                        industry_remaining[industry_value] - allowed,
                    )
                if remaining <= 1e-9:
                    break

            allocated = target - max(0.0, remaining)
            ranks = [int(item["rank"]) for item in group_selected]
            metric_rows.append(
                {
                    "pool": pool,
                    "date": str(date),
                    "decision_target_timestamp": pd.Timestamp(decision_time),
                    "clock": pd.Timestamp(decision_time).strftime("%H:%M"),
                    "candidate_rows": int(len(group)),
                    "eligible_rows": int(len(eligible)),
                    "selected_rows": int(len(group_selected)),
                    "target_notional": target,
                    "allocated_notional": allocated,
                    "cash_notional": max(0.0, target - allocated),
                    "fill_ratio": allocated / target,
                    "filled": allocated >= target * (1.0 - 1e-9),
                    "max_candidate_rank": max(ranks) if ranks else float("nan"),
                    "mean_candidate_rank": float(np.mean(ranks)) if ranks else float("nan"),
                }
            )

    return pd.DataFrame(selected_rows), pd.DataFrame(metric_rows)


def group_targets_from_metrics(metrics: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    targets = metrics[["pool", "date", "decision_target_timestamp", "target_notional"]].copy()
    targets = targets.rename(columns={"target_notional": "group_target_notional"})
    targets.insert(0, "policy", policy)
    return targets


def summarize_selected_groups(
    selected: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    keys = list(GROUP_COLUMNS)
    target = targets.loc[targets["policy"].eq(policy), keys + ["group_target_notional"]].copy()
    chosen = selected.loc[selected["policy"].eq(policy)].copy()
    if chosen.empty:
        allocated = pd.DataFrame(columns=[*keys, "selected_rows", "allocated_notional", "max_rank"])
    else:
        allocated = (
            chosen.groupby(keys, sort=False)
            .agg(
                selected_rows=("allocated_notional", "size"),
                allocated_notional=("allocated_notional", "sum"),
                max_rank=("rank", "max"),
            )
            .reset_index()
        )
    out = target.merge(allocated, on=keys, how="left")
    out[["selected_rows", "allocated_notional"]] = out[
        ["selected_rows", "allocated_notional"]
    ].fillna(0.0)
    out["fill_ratio"] = out["allocated_notional"] / out["group_target_notional"]
    out["cash_notional"] = (out["group_target_notional"] - out["allocated_notional"]).clip(
        lower=0.0
    )
    out.insert(0, "policy", policy)
    return out


def summarize_overlap(
    selected: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    capacity_total_notional: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = selected.copy()
    work["decision_target_timestamp"] = pd.to_datetime(
        work["decision_target_timestamp"], errors="coerce"
    )
    positions = (
        work.groupby(["policy", "pool", "date", "symbol"], observed=True, sort=False)
        .agg(
            allocated_notional=("allocated_notional", "sum"),
            decision_count=("decision_target_timestamp", "nunique"),
            first_decision=("decision_target_timestamp", "min"),
            last_decision=("decision_target_timestamp", "max"),
        )
        .reset_index()
    )
    target_daily = (
        targets.groupby(["policy", "pool", "date"], observed=True, sort=False)[
            "group_target_notional"
        ]
        .sum()
        .rename("target_notional")
        .reset_index()
    )
    daily_rows: list[dict[str, object]] = []
    for keys, item in positions.groupby(["policy", "pool", "date"], observed=True, sort=False):
        policy, pool, date = keys
        allocated = float(item["allocated_notional"].sum())
        shares = (
            item["allocated_notional"] / allocated if allocated > 0 else item["allocated_notional"]
        )
        hhi = float((shares**2).sum()) if allocated > 0 else float("nan")
        repeated = item["decision_count"].gt(1)
        daily_rows.append(
            {
                "policy": policy,
                "pool": pool,
                "date": date,
                "allocated_notional": allocated,
                "unique_symbols": int(len(item)),
                "repeated_symbols": int(repeated.sum()),
                "repeated_symbol_notional_share": (
                    float(item.loc[repeated, "allocated_notional"].sum() / allocated)
                    if allocated > 0
                    else float("nan")
                ),
                "max_daily_symbol_weight": (
                    float(item["allocated_notional"].max() / capacity_total_notional)
                    if allocated > 0
                    else float("nan")
                ),
                "symbol_hhi": hhi,
                "effective_symbols": float(1.0 / hhi) if hhi > 0 else float("nan"),
            }
        )
    daily = pd.DataFrame(daily_rows).merge(
        target_daily,
        on=["policy", "pool", "date"],
        how="right",
    )
    daily["allocated_notional"] = daily["allocated_notional"].fillna(0.0)
    daily["capital_fraction"] = daily["allocated_notional"] / float(capacity_total_notional)
    daily["fill_ratio"] = daily["allocated_notional"] / daily["target_notional"]

    per_decision = (
        work.groupby(
            ["policy", "pool", "date", "decision_target_timestamp", "symbol"],
            observed=True,
            sort=False,
        )["allocated_notional"]
        .sum()
        .reset_index()
    )
    adjacent_rows: list[dict[str, object]] = []
    for keys, item in per_decision.groupby(["policy", "pool", "date"], observed=True):
        policy, pool, date = keys
        times = sorted(item["decision_target_timestamp"].unique())
        by_time = {
            time: part.set_index("symbol")["allocated_notional"]
            for time, part in item.groupby("decision_target_timestamp", observed=True)
        }
        for left_time, right_time in zip(times, times[1:], strict=False):
            left = by_time[left_time]
            right = by_time[right_time]
            common = left.index.intersection(right.index)
            union = left.index.union(right.index)
            weighted = float(np.minimum(left.reindex(common), right.reindex(common)).sum())
            denominator = min(float(left.sum()), float(right.sum()))
            adjacent_rows.append(
                {
                    "policy": policy,
                    "pool": pool,
                    "date": date,
                    "left_decision": pd.Timestamp(left_time),
                    "right_decision": pd.Timestamp(right_time),
                    "left_symbols": int(len(left)),
                    "right_symbols": int(len(right)),
                    "common_symbols": int(len(common)),
                    "name_jaccard": float(len(common) / len(union)) if len(union) else np.nan,
                    "name_overlap_min": (
                        float(len(common) / min(len(left), len(right)))
                        if min(len(left), len(right)) > 0
                        else np.nan
                    ),
                    "weighted_overlap_min": weighted / denominator if denominator > 0 else np.nan,
                }
            )
    adjacent = pd.DataFrame(adjacent_rows)
    adjacent_summary = (
        adjacent.groupby(["policy", "pool"], observed=True)
        .agg(
            adjacent_pairs=("name_jaccard", "size"),
            mean_name_jaccard=("name_jaccard", "mean"),
            mean_name_overlap_min=("name_overlap_min", "mean"),
            mean_weighted_overlap_min=("weighted_overlap_min", "mean"),
        )
        .reset_index()
        if not adjacent.empty
        else pd.DataFrame(columns=["policy", "pool"])
    )
    summary = (
        daily.groupby(["policy", "pool"], observed=True)
        .agg(
            overlap_days=("date", "nunique"),
            overlap_mean_fill_ratio=("fill_ratio", "mean"),
            mean_capital_fraction=("capital_fraction", "mean"),
            mean_unique_symbols=("unique_symbols", "mean"),
            mean_repeated_symbols=("repeated_symbols", "mean"),
            mean_repeated_symbol_notional_share=("repeated_symbol_notional_share", "mean"),
            max_daily_symbol_weight=("max_daily_symbol_weight", "max"),
            mean_effective_symbols=("effective_symbols", "mean"),
        )
        .reset_index()
        .merge(adjacent_summary, on=["policy", "pool"], how="left")
    )
    return positions, daily, adjacent, summary


def _tail_record(
    item: pd.DataFrame,
    *,
    target_notional: float,
    label_col: str,
    fee_bps: float,
    quantile: float,
    threshold: float | None = None,
) -> dict[str, float]:
    labels = finite_numeric(item[label_col])
    notionals = finite_numeric(item["allocated_notional"]).fillna(0.0).clip(lower=0.0)
    valid = labels.notna() & notionals.gt(0)
    labels = labels.loc[valid]
    notionals = notionals.loc[valid]
    if labels.empty or target_notional <= 0:
        return {}
    threshold_value = float(labels.quantile(quantile)) if threshold is None else float(threshold)
    tail = labels.gt(threshold_value)
    gross_pnl = float((notionals * labels).sum())
    winsor_pnl = float((notionals * labels.clip(upper=threshold_value)).sum())
    trim_gross_pnl = float((notionals.loc[~tail] * labels.loc[~tail]).sum())
    fee_pnl = float(notionals.sum() * fee_bps / RETURN_BPS_DENOMINATOR)
    trim_fee_pnl = float(notionals.loc[~tail].sum() * fee_bps / RETURN_BPS_DENOMINATOR)
    scale = RETURN_BPS_DENOMINATOR / target_notional
    tail_occurrence = float(notionals.loc[tail].sum() * threshold_value * scale)
    severity = float((notionals.loc[tail] * (labels.loc[tail] - threshold_value)).sum() * scale)
    return {
        "threshold_label_bps": threshold_value * RETURN_BPS_DENOMINATOR,
        "rows": int(len(labels)),
        "selected_notional": float(notionals.sum()),
        "target_notional_denominator": target_notional,
        "raw_gross_bps_vs_target": gross_pnl * scale,
        "raw_net_bps_vs_target": (gross_pnl - fee_pnl) * scale,
        "winsor_gross_bps_vs_target": winsor_pnl * scale,
        "winsor_net_bps_vs_target": (winsor_pnl - fee_pnl) * scale,
        "trim_gross_bps_vs_target": trim_gross_pnl * scale,
        "trim_net_bps_vs_target": (trim_gross_pnl - trim_fee_pnl) * scale,
        "winsor_removed_bps": (gross_pnl - winsor_pnl) * scale,
        "non_tail_background_bps": trim_gross_pnl * scale,
        "tail_occurrence_at_threshold_bps": tail_occurrence,
        "tail_severity_above_threshold_bps": severity,
        "tail_total_contribution_bps": tail_occurrence + severity,
        "tail_row_share": float(tail.mean()),
        "tail_notional_share": float(notionals.loc[tail].sum() / notionals.sum()),
        "tail_mean_label_bps": float(labels.loc[tail].mean() * RETURN_BPS_DENOMINATOR),
        "non_tail_mean_label_bps": float(labels.loc[~tail].mean() * RETURN_BPS_DENOMINATOR),
    }


def summarize_tail_robustness(
    selected: pd.DataFrame,
    targets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_col: str,
    fee_bps: float,
    settings: TailSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = selected.merge(
        labels[list(KEY_COLUMNS) + [label_col]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    missing = int(merged[label_col].isna().sum())
    if missing:
        raise ValueError(f"strategy selected rows missing {label_col}: {missing}")
    target_by_policy = targets.groupby(["policy", "pool"], observed=True)[
        "group_target_notional"
    ].sum()
    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    merged["month"] = pd.to_datetime(merged["date"]).dt.to_period("M").astype(str)
    target_monthly = targets.copy()
    target_monthly["month"] = pd.to_datetime(target_monthly["date"]).dt.to_period("M").astype(str)
    target_monthly = target_monthly.groupby(["policy", "pool", "month"], observed=True)[
        "group_target_notional"
    ].sum()

    for (policy, pool), item in merged.groupby(["policy", "pool"], observed=True):
        total_target = float(target_by_policy.loc[(policy, pool)])
        for quantile in settings.quantiles:
            record = _tail_record(
                item,
                target_notional=total_target,
                label_col=label_col,
                fee_bps=fee_bps,
                quantile=quantile,
            )
            threshold = float(record["threshold_label_bps"]) / RETURN_BPS_DENOMINATOR
            summary_rows.append(
                {
                    "policy": policy,
                    "pool": pool,
                    "threshold": f"p{int(round(quantile * 100))}",
                    **record,
                }
            )
            for month, month_item in item.groupby("month", observed=True):
                month_target = float(target_monthly.loc[(policy, pool, month)])
                month_record = _tail_record(
                    month_item,
                    target_notional=month_target,
                    label_col=label_col,
                    fee_bps=fee_bps,
                    quantile=quantile,
                    threshold=threshold,
                )
                monthly_rows.append(
                    {
                        "policy": policy,
                        "pool": pool,
                        "month": month,
                        "threshold": f"p{int(round(quantile * 100))}",
                        **month_record,
                    }
                )

    contribution_rows: list[dict[str, object]] = []
    merged["gross_pnl"] = merged["allocated_notional"] * merged[label_col]
    merged["fee_pnl"] = merged["allocated_notional"] * fee_bps / RETURN_BPS_DENOMINATOR
    for (policy, pool), item in merged.groupby(["policy", "pool"], observed=True):
        total_target = float(target_by_policy.loc[(policy, pool)])
        total_gross = float(item["gross_pnl"].sum())
        for unit, columns in (
            ("date", ["date"]),
            ("symbol", ["symbol"]),
            ("symbol_date", ["date", "symbol"]),
        ):
            grouped = item.groupby(columns, observed=True)[["gross_pnl", "fee_pnl"]].sum()
            grouped = grouped.sort_values("gross_pnl", ascending=False)
            for top_n in (1, 5, 10):
                removed = grouped.head(top_n)
                remaining = grouped.iloc[top_n:]
                contribution_rows.append(
                    {
                        "policy": policy,
                        "pool": pool,
                        "unit": unit,
                        "top_n": top_n,
                        "groups": int(len(grouped)),
                        "top_gross_pnl": float(removed["gross_pnl"].sum()),
                        "top_gross_share": (
                            float(removed["gross_pnl"].sum() / total_gross)
                            if total_gross != 0
                            else np.nan
                        ),
                        "remaining_gross_bps_vs_target": float(
                            remaining["gross_pnl"].sum() / total_target * RETURN_BPS_DENOMINATOR
                        ),
                        "remaining_net_bps_vs_target": float(
                            (remaining["gross_pnl"].sum() - remaining["fee_pnl"].sum())
                            / total_target
                            * RETURN_BPS_DENOMINATOR
                        ),
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(contribution_rows),
    )


def monthly_block_bootstrap(
    daily: pd.DataFrame,
    *,
    settings: TailSettings,
) -> pd.DataFrame:
    rng = np.random.default_rng(settings.bootstrap_seed)
    rows: list[dict[str, object]] = []
    work = daily.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.to_period("M").astype(str)
    for (policy, pool), item in work.groupby(["policy", "pool"], observed=True):
        blocks = item.groupby("month", observed=True)["next_capital_net_return_bps"].sum()
        values = blocks.to_numpy(dtype="float64")
        if not len(values):
            continue
        samples = rng.choice(
            values, size=(settings.bootstrap_samples, len(values)), replace=True
        ).sum(axis=1)
        rows.append(
            {
                "policy": policy,
                "pool": pool,
                "block_unit": "month",
                "blocks": int(len(values)),
                "samples": int(settings.bootstrap_samples),
                "observed_cumulative_net_bps": float(values.sum()),
                "bootstrap_p05_cumulative_net_bps": float(np.quantile(samples, 0.05)),
                "bootstrap_median_cumulative_net_bps": float(np.quantile(samples, 0.50)),
                "bootstrap_p95_cumulative_net_bps": float(np.quantile(samples, 0.95)),
                "bootstrap_positive_probability": float(np.mean(samples > 0)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_period_out(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    date = pd.to_datetime(work["date"])
    periods = {
        "month": date.dt.to_period("M").astype(str),
        "quarter": date.dt.to_period("Q").astype(str),
    }
    rows: list[dict[str, object]] = []
    for unit, values in periods.items():
        work["period"] = values
        for (policy, pool), item in work.groupby(["policy", "pool"], observed=True):
            period_pnl = item.groupby("period", observed=True)["next_capital_net_return_bps"].sum()
            total = float(period_pnl.sum())
            for period, value in period_pnl.items():
                rows.append(
                    {
                        "policy": policy,
                        "pool": pool,
                        "unit": unit,
                        "omitted_period": period,
                        "omitted_period_net_bps": float(value),
                        "remaining_cumulative_net_bps": total - float(value),
                    }
                )
    return pd.DataFrame(rows)
