from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL
from opening_strength_fit.capacity_acceptance import (
    DEFAULT_CAPACITY_TOTAL_NOTIONAL,
    RETURN_BPS_DENOMINATOR,
    normalize_key_columns,
)
from opening_strength_fit.io import frame_columns, read_frame

REALISTIC_DAILY_SUMMARY = "realistic_acceptance_daily_summary.csv"
REALISTIC_SUMMARY = "realistic_acceptance_summary.csv"
REALISTIC_SELECTED = "realistic_acceptance_selected.csv"
REALISTIC_TRACE = "realistic_acceptance_trace.json"
DEFAULT_REALISTIC_LABEL_COL = NEXT_CLOSE_LABEL_COL

SELECTED_REQUIRED_COLUMNS = (
    "pool",
    "date",
    "symbol",
    "decision_target_timestamp",
    "allocated_notional",
    "target_notional",
)
SELECTED_OPTIONAL_COLUMNS = (
    "test_month",
    "clock",
    "rank",
    "score",
    "capacity_price",
    "execution_price",
    "ask_price_1",
    "bid_price_1",
    "mid_price",
    "spread_bps",
    "ask1_to_limit_up_bps",
    "status",
    "row_limit_notional",
    "capacity_notional",
    "capacity_limit_notional",
    "capacity_participation_rate",
    "ask_depth_notional",
    "ask_depth_limit_notional",
    "ask_depth_participation_rate",
    "industry",
)


@dataclass(frozen=True)
class RealisticExecutionConstraints:
    capacity_total_notional: float = DEFAULT_CAPACITY_TOTAL_NOTIONAL
    fee_bps: float = 8.0
    max_daily_symbol_weight: float = 0.005
    # Deprecated compatibility fields. Per-decision capacity is enforced when
    # building the selected input, not by a second daily turnover budget here.
    max_daily_symbol_participation_rate: float = 0.10
    daily_capacity_method: str = "max"
    execution_fill_rate: float = 1.0
    min_child_notional: float = 0.0
    max_symbol_decision_count: int = 0
    round_lot_shares: int = 0
    price_col: str = ""
    status_col: str = ""
    tradable_statuses: tuple[str, ...] = ()
    spread_bps_col: str = "spread_bps"
    max_spread_bps: float = 0.0
    limit_up_room_bps_col: str = "ask1_to_limit_up_bps"
    min_limit_up_room_bps: float = 0.0
    ask_depth_notional_col: str = "ask_depth_notional"
    max_ask_depth_participation_rate: float = 0.0
    industry_col: str = "industry"
    max_daily_industry_weight: float = 0.0


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def validate_execution_constraints(constraints: RealisticExecutionConstraints) -> None:
    if constraints.capacity_total_notional <= 0:
        raise ValueError("capacity_total_notional must be positive")
    if constraints.fee_bps < 0:
        raise ValueError("fee_bps must be >= 0")
    if constraints.max_daily_symbol_weight < 0:
        raise ValueError("max_daily_symbol_weight must be >= 0")
    if constraints.max_daily_symbol_participation_rate < 0:
        raise ValueError("max_daily_symbol_participation_rate must be >= 0")
    if not 0 <= constraints.execution_fill_rate <= 1:
        raise ValueError("execution_fill_rate must be between 0 and 1")
    if constraints.min_child_notional < 0:
        raise ValueError("min_child_notional must be >= 0")
    if constraints.max_symbol_decision_count < 0:
        raise ValueError("max_symbol_decision_count must be >= 0")
    if constraints.round_lot_shares < 0:
        raise ValueError("round_lot_shares must be >= 0")
    if constraints.max_spread_bps < 0:
        raise ValueError("max_spread_bps must be >= 0")
    if constraints.min_limit_up_room_bps < 0:
        raise ValueError("min_limit_up_room_bps must be >= 0")
    if constraints.max_ask_depth_participation_rate < 0:
        raise ValueError("max_ask_depth_participation_rate must be >= 0")
    if constraints.max_daily_industry_weight < 0:
        raise ValueError("max_daily_industry_weight must be >= 0")


def load_realistic_selected(
    paths: tuple[str | Path, ...],
    *,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    frames = []
    required_set = set(SELECTED_REQUIRED_COLUMNS)
    requested_optional = tuple(dict.fromkeys((*SELECTED_OPTIONAL_COLUMNS, *extra_columns)))
    for raw in paths:
        path = Path(raw)
        available = frame_columns(path)
        missing = sorted(required_set - available)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        columns = [
            column
            for column in (*SELECTED_REQUIRED_COLUMNS, *requested_optional)
            if column in available
        ]
        frames.append(read_frame(path, columns=columns))
    if not frames:
        raise ValueError("no selected input files supplied")
    selected = pd.concat(frames, ignore_index=True)
    required = (*SELECTED_REQUIRED_COLUMNS, *requested_optional)
    for column in requested_optional:
        if column not in selected.columns:
            selected[column] = pd.NA
    selected = selected[list(required)]
    selected = normalize_key_columns(selected)
    for column in (
        "allocated_notional",
        "target_notional",
        "rank",
        "score",
        "capacity_price",
        "execution_price",
        "ask_price_1",
        "bid_price_1",
        "mid_price",
        "spread_bps",
        "ask1_to_limit_up_bps",
        "row_limit_notional",
        "capacity_notional",
        "capacity_limit_notional",
        "capacity_participation_rate",
        "ask_depth_notional",
        "ask_depth_limit_notional",
        "ask_depth_participation_rate",
    ):
        if column in selected.columns:
            selected[column] = _finite_numeric(selected[column])
    selected = selected.dropna(subset=["allocated_notional", "target_notional"]).copy()
    selected = selected.loc[selected["allocated_notional"].gt(0)].copy()
    if selected.empty:
        raise ValueError("selected inputs have no positive allocated_notional rows")
    return selected


def realistic_context_columns(constraints: RealisticExecutionConstraints) -> tuple[str, ...]:
    columns: list[str] = []
    for column in (
        constraints.price_col,
        constraints.status_col,
        constraints.spread_bps_col if constraints.max_spread_bps > 0 else "",
        constraints.limit_up_room_bps_col if constraints.min_limit_up_room_bps > 0 else "",
        constraints.ask_depth_notional_col
        if constraints.max_ask_depth_participation_rate > 0
        else "",
        constraints.industry_col if constraints.max_daily_industry_weight > 0 else "",
    ):
        if column:
            columns.append(str(column))
    return tuple(dict.fromkeys(columns))


def load_realistic_execution_context(
    paths: tuple[str | Path, ...],
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    context_columns = tuple(column for column in columns if column not in KEY_COLUMNS)
    if not paths or not context_columns:
        return pd.DataFrame()
    frames = []
    required_set = set(KEY_COLUMNS)
    for raw in paths:
        path = Path(raw)
        available = frame_columns(path)
        missing_keys = sorted(required_set - available)
        if missing_keys:
            raise ValueError(f"{path} missing context key columns: {missing_keys}")
        read_columns = [
            *KEY_COLUMNS,
            *(column for column in context_columns if column in available),
        ]
        frames.append(read_frame(path, columns=read_columns))
    context = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if context.empty:
        return context
    context = normalize_key_columns(context)
    return context.drop_duplicates(list(KEY_COLUMNS), keep="last")


def merge_realistic_execution_context(
    selected: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        return selected
    selected = normalize_key_columns(selected)
    context = normalize_key_columns(context)
    context_value_columns = [column for column in context.columns if column not in KEY_COLUMNS]
    if not context_value_columns:
        return selected
    merged = selected.merge(
        context[list(KEY_COLUMNS) + context_value_columns],
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
        suffixes=("", "_context"),
    )
    for column in context_value_columns:
        context_column = f"{column}_context"
        if context_column not in merged.columns:
            continue
        if column in selected.columns:
            merged[column] = merged[column].combine_first(merged[context_column])
        else:
            merged[column] = merged[context_column]
        merged = merged.drop(columns=[context_column])
    return merged


def round_notional_to_lot(
    notional: float,
    row: pd.Series,
    constraints: RealisticExecutionConstraints,
) -> float:
    if constraints.round_lot_shares <= 0 or not constraints.price_col:
        return notional
    if constraints.price_col not in row or pd.isna(row[constraints.price_col]):
        return notional
    price = float(row[constraints.price_col])
    if price <= 0:
        return 0.0
    shares = np.floor(notional / price / float(constraints.round_lot_shares))
    return float(shares * float(constraints.round_lot_shares) * price)


def tradable_mask(frame: pd.DataFrame, constraints: RealisticExecutionConstraints) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if constraints.status_col and constraints.tradable_statuses:
        if constraints.status_col in frame.columns:
            allowed = {str(status).upper() for status in constraints.tradable_statuses}
            mask &= frame[constraints.status_col].astype(str).str.upper().isin(allowed)
    if constraints.max_spread_bps > 0 and constraints.spread_bps_col in frame.columns:
        spread = _finite_numeric(frame[constraints.spread_bps_col])
        mask &= spread.notna() & spread.le(float(constraints.max_spread_bps))
    if constraints.min_limit_up_room_bps > 0 and constraints.limit_up_room_bps_col in frame.columns:
        room = _finite_numeric(frame[constraints.limit_up_room_bps_col])
        mask &= room.notna() & room.ge(float(constraints.min_limit_up_room_bps))
    return mask


def row_depth_limit(row: pd.Series, constraints: RealisticExecutionConstraints) -> float:
    if constraints.max_ask_depth_participation_rate <= 0:
        return np.inf
    if not constraints.ask_depth_notional_col:
        return np.inf
    if constraints.ask_depth_notional_col not in row or pd.isna(
        row[constraints.ask_depth_notional_col]
    ):
        return np.inf
    depth = float(row[constraints.ask_depth_notional_col])
    if depth <= 0:
        return 0.0
    return depth * float(constraints.max_ask_depth_participation_rate)


def apply_realistic_execution_constraints(
    selected: pd.DataFrame,
    constraints: RealisticExecutionConstraints,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_execution_constraints(constraints)
    work = selected.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["decision_target_timestamp"] = pd.to_datetime(
        work["decision_target_timestamp"],
        errors="coerce",
    )
    work["rank"] = _finite_numeric(work.get("rank", pd.Series(index=work.index))).fillna(1e12)
    work["score"] = _finite_numeric(work.get("score", pd.Series(index=work.index))).fillna(-np.inf)
    work = work.dropna(subset=["date", "symbol", "decision_target_timestamp"]).copy()
    work = work.sort_values(
        ["pool", "date", "decision_target_timestamp", "rank", "score", "symbol"],
        ascending=[True, True, True, True, False, True],
    )

    group_targets = (
        work.groupby(["pool", "date", "decision_target_timestamp"], sort=False)
        .agg(group_target_notional=("target_notional", "first"))
        .reset_index()
    )

    adjusted_rows: list[dict[str, object]] = []
    daily_symbol_cap = (
        constraints.capacity_total_notional * float(constraints.max_daily_symbol_weight)
        if constraints.max_daily_symbol_weight > 0
        else np.inf
    )

    for (_pool, _date), day in work.groupby(["pool", "date"], sort=False):
        day = day.copy()
        day = day.loc[tradable_mask(day, constraints)].copy()
        if day.empty:
            continue
        symbol_remaining = {
            str(symbol): float(daily_symbol_cap)
            for symbol in day["symbol"].dropna().astype(str).unique()
        }
        daily_industry_cap = (
            constraints.capacity_total_notional * float(constraints.max_daily_industry_weight)
            if constraints.max_daily_industry_weight > 0
            else np.inf
        )
        if (
            constraints.max_daily_industry_weight > 0
            and constraints.industry_col
            and constraints.industry_col in day.columns
        ):
            industry_remaining = {
                str(industry): float(daily_industry_cap)
                for industry in day[constraints.industry_col].dropna().astype(str).unique()
            }
        else:
            industry_remaining = {}
        group_remaining = (
            day.groupby("decision_target_timestamp", sort=False)["target_notional"]
            .first()
            .astype("float64")
            .to_dict()
        )
        symbol_decisions: dict[str, set[pd.Timestamp]] = {}
        for _, row in day.iterrows():
            symbol = str(row["symbol"])
            decision_time = row["decision_target_timestamp"]
            original = float(row["allocated_notional"])
            if original <= 0:
                continue
            if constraints.max_symbol_decision_count > 0:
                used_decisions = symbol_decisions.setdefault(symbol, set())
                if decision_time not in used_decisions and len(used_decisions) >= int(
                    constraints.max_symbol_decision_count
                ):
                    continue
            allowed = min(
                original,
                max(0.0, float(symbol_remaining.get(symbol, 0.0))),
                max(0.0, float(group_remaining.get(decision_time, 0.0))),
                row_depth_limit(row, constraints),
            )
            industry_value = ""
            if industry_remaining and constraints.industry_col in row:
                industry_value = str(row[constraints.industry_col])
                allowed = min(allowed, max(0.0, float(industry_remaining.get(industry_value, 0.0))))
            allowed *= float(constraints.execution_fill_rate)
            allowed = round_notional_to_lot(allowed, row, constraints)
            if constraints.min_child_notional > 0 and allowed < constraints.min_child_notional:
                continue
            if allowed <= 0:
                continue
            symbol_remaining[symbol] = max(0.0, float(symbol_remaining.get(symbol, 0.0)) - allowed)
            group_remaining[decision_time] = max(
                0.0,
                float(group_remaining.get(decision_time, 0.0)) - allowed,
            )
            if constraints.max_symbol_decision_count > 0:
                symbol_decisions.setdefault(symbol, set()).add(decision_time)
            if industry_value:
                industry_remaining[industry_value] = max(
                    0.0,
                    float(industry_remaining.get(industry_value, 0.0)) - allowed,
                )
            out = row.to_dict()
            out["original_allocated_notional"] = original
            out["allocated_notional"] = allowed
            out["execution_fill_rate"] = constraints.execution_fill_rate
            adjusted_rows.append(out)

    adjusted = pd.DataFrame(adjusted_rows)
    if adjusted.empty:
        adjusted = work.head(0).copy()
        adjusted["original_allocated_notional"] = pd.Series(dtype="float64")
        adjusted["execution_fill_rate"] = pd.Series(dtype="float64")
    return adjusted.reset_index(drop=True), group_targets


def summarize_realistic_acceptance(
    adjusted: pd.DataFrame,
    group_targets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    constraints: RealisticExecutionConstraints,
    label_col: str = DEFAULT_REALISTIC_LABEL_COL,
) -> pd.DataFrame:
    validate_execution_constraints(constraints)
    if adjusted.empty:
        merged = adjusted.copy()
        merged["_gross_pnl"] = pd.Series(dtype="float64")
        merged["_fee_pnl"] = pd.Series(dtype="float64")
        merged["_net_pnl"] = pd.Series(dtype="float64")
    else:
        merged = adjusted.merge(
            labels[list(KEY_COLUMNS) + [label_col]],
            on=list(KEY_COLUMNS),
            how="left",
            validate="many_to_one",
        )
        missing_label_rows = int(merged[label_col].isna().sum())
        if missing_label_rows:
            raise ValueError(f"realistic selected rows missing {label_col}: {missing_label_rows}")
        merged["_gross_pnl"] = merged["allocated_notional"] * merged[label_col]
        merged["_fee_pnl"] = (
            merged["allocated_notional"] * float(constraints.fee_bps) / RETURN_BPS_DENOMINATOR
        )
        merged["_net_pnl"] = merged["_gross_pnl"] - merged["_fee_pnl"]

    daily_targets = group_targets.groupby(["pool", "date"], sort=False).agg(
        capacity_decision_groups=("decision_target_timestamp", "size"),
        target_notional=("group_target_notional", "sum"),
    )
    daily = merged.groupby(["pool", "date"], sort=False).agg(
        gross_pnl=("_gross_pnl", "sum"),
        fee_pnl=("_fee_pnl", "sum"),
        net_pnl=("_net_pnl", "sum"),
        selected_allocated_notional=("allocated_notional", "sum"),
        original_selected_allocated_notional=("original_allocated_notional", "sum"),
        selected_rows=("allocated_notional", "size"),
    )
    daily = daily_targets.join(daily, how="left").fillna(
        {
            "gross_pnl": 0.0,
            "fee_pnl": 0.0,
            "net_pnl": 0.0,
            "selected_allocated_notional": 0.0,
            "original_selected_allocated_notional": 0.0,
            "selected_rows": 0,
        }
    )
    daily = daily.reset_index()
    daily["week_start"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily["capacity_total_notional"] = float(constraints.capacity_total_notional)
    daily["capacity_daily_capital_fraction"] = daily["target_notional"] / float(
        constraints.capacity_total_notional
    )
    daily["fill_ratio"] = daily["selected_allocated_notional"] / daily["target_notional"]
    daily["cash_notional"] = (daily["target_notional"] - daily["selected_allocated_notional"]).clip(
        lower=0.0
    )
    daily["gross_next_return_bps"] = (
        daily["gross_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["fee_bps"] = daily["fee_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    daily["next_net_return_bps"] = (
        daily["net_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["next_capital_net_return_bps"] = (
        daily["net_pnl"] / float(constraints.capacity_total_notional) * RETURN_BPS_DENOMINATOR
    )
    daily["next_net_pnl"] = daily["net_pnl"]
    return daily[
        [
            "pool",
            "date",
            "week_start",
            "capacity_decision_groups",
            "selected_rows",
            "selected_allocated_notional",
            "original_selected_allocated_notional",
            "target_notional",
            "capacity_total_notional",
            "capacity_daily_capital_fraction",
            "fill_ratio",
            "cash_notional",
            "gross_next_return_bps",
            "fee_bps",
            "next_net_return_bps",
            "next_capital_net_return_bps",
            "next_net_pnl",
            "gross_pnl",
            "fee_pnl",
        ]
    ]


def summarize_realistic_acceptance_overall(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    out = daily.groupby("pool", sort=False).agg(
        days=("date", "nunique"),
        capacity_decision_groups=("capacity_decision_groups", "sum"),
        mean_selected_rows=("selected_rows", "mean"),
        mean_target_notional=("target_notional", "mean"),
        mean_selected_allocated_notional=("selected_allocated_notional", "mean"),
        mean_fill_ratio=("fill_ratio", "mean"),
        min_fill_ratio=("fill_ratio", "min"),
        mean_capacity_daily_capital_fraction=("capacity_daily_capital_fraction", "mean"),
        mean_next_net_return_bps=("next_net_return_bps", "mean"),
        final_next_cumulative_net_return_bps=("next_capital_net_return_bps", "sum"),
        total_net_pnl=("next_net_pnl", "sum"),
    )
    return out.reset_index()


def constraints_trace(constraints: RealisticExecutionConstraints) -> dict[str, object]:
    return asdict(constraints)
