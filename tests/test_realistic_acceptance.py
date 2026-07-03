from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.realistic_acceptance import (
    RealisticExecutionConstraints,
    apply_realistic_execution_constraints,
    merge_realistic_execution_context,
    summarize_realistic_acceptance,
)


def test_realistic_replay_keeps_per_decision_capacity_allocations() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "decision_target_timestamp": [
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
            ],
            "rank": [1, 1],
            "score": [2.0, 2.0],
            "target_notional": [100.0, 100.0],
            "allocated_notional": [100.0, 100.0],
            "capacity_notional": [1_200.0, 1_200.0],
        }
    )
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        max_daily_symbol_participation_rate=0.10,
        daily_capacity_method="max",
    )

    constrained, group_targets = apply_realistic_execution_constraints(selected, constraints)

    assert constrained["allocated_notional"].tolist() == pytest.approx([100.0, 100.0])
    assert group_targets["group_target_notional"].tolist() == pytest.approx([100.0, 100.0])


def test_realistic_replay_applies_daily_symbol_weight_without_turnover_budget() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "decision_target_timestamp": [
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
            ],
            "rank": [1, 1],
            "score": [2.0, 2.0],
            "target_notional": [100.0, 100.0],
            "allocated_notional": [100.0, 100.0],
            "capacity_notional": [1_200.0, 1_200.0],
        }
    )
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=0.12,
        max_daily_symbol_participation_rate=0.10,
        daily_capacity_method="max",
    )

    constrained, _group_targets = apply_realistic_execution_constraints(selected, constraints)

    assert constrained["allocated_notional"].tolist() == pytest.approx([100.0, 20.0])


def test_realistic_replay_filters_execution_status_and_spread_from_context() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L", "pool_L"],
            "date": ["2024-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": ["2024-01-02 09:31:00"] * 3,
            "rank": [1, 2, 3],
            "score": [3.0, 2.0, 1.0],
            "target_notional": [300.0] * 3,
            "allocated_notional": [100.0, 100.0, 100.0],
        }
    )
    context = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": pd.to_datetime(["2024-01-02 09:31:00"] * 3),
            "status": ["TRADE", "HALT", "TRADE"],
            "spread_bps": [5.0, 5.0, 80.0],
        }
    )
    selected = merge_realistic_execution_context(selected, context)
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        status_col="status",
        tradable_statuses=("TRADE",),
        max_spread_bps=10.0,
    )

    constrained, _group_targets = apply_realistic_execution_constraints(selected, constraints)

    assert constrained["symbol"].tolist() == ["000001.SZ"]
    assert constrained["allocated_notional"].tolist() == pytest.approx([100.0])


def test_realistic_replay_rounds_lots_and_drops_small_children() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_target_timestamp": ["2024-01-02 09:31:00"] * 2,
            "rank": [1, 2],
            "score": [2.0, 1.0],
            "target_notional": [2000.0, 2000.0],
            "allocated_notional": [1234.0, 800.0],
            "capacity_price": [10.0, 10.0],
        }
    )
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=10_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        round_lot_shares=100,
        price_col="capacity_price",
        min_child_notional=1_000.0,
    )

    constrained, _group_targets = apply_realistic_execution_constraints(selected, constraints)

    assert constrained["symbol"].tolist() == ["000001.SZ"]
    assert constrained["allocated_notional"].tolist() == pytest.approx([1000.0])


def test_realistic_replay_applies_depth_and_industry_caps() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L", "pool_L"],
            "date": ["2024-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "industry": ["bank", "bank", "tech"],
            "decision_target_timestamp": ["2024-01-02 09:31:00"] * 3,
            "rank": [1, 2, 3],
            "score": [3.0, 2.0, 1.0],
            "target_notional": [300.0] * 3,
            "allocated_notional": [100.0, 100.0, 100.0],
            "ask_depth_notional": [1_000.0, 1_000.0, 150.0],
        }
    )
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        max_daily_industry_weight=0.15,
        max_ask_depth_participation_rate=0.5,
    )

    constrained, _group_targets = apply_realistic_execution_constraints(selected, constraints)

    assert constrained["symbol"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert constrained["allocated_notional"].tolist() == pytest.approx([100.0, 50.0, 75.0])


def test_realistic_summary_keeps_unfilled_cash_in_daily_target() -> None:
    selected = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "decision_target_timestamp": [
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
            ],
            "rank": [1, 1],
            "score": [2.0, 2.0],
            "target_notional": [100.0, 100.0],
            "allocated_notional": [100.0, 100.0],
            "capacity_notional": [1_200.0, 1_200.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "decision_target_timestamp": [
                pd.Timestamp("2024-01-02 09:31:00"),
                pd.Timestamp("2024-01-02 09:32:00"),
            ],
            "alpha_return_next_close": [0.01, 0.01],
        }
    )
    constraints = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        max_daily_symbol_participation_rate=0.10,
        daily_capacity_method="max",
    )
    constrained, group_targets = apply_realistic_execution_constraints(selected, constraints)

    daily = summarize_realistic_acceptance(
        constrained,
        group_targets,
        labels,
        constraints=constraints,
    )

    row = daily.iloc[0]
    assert row["target_notional"] == pytest.approx(200.0)
    assert row["selected_allocated_notional"] == pytest.approx(200.0)
    assert row["cash_notional"] == pytest.approx(0.0)
    assert row["fill_ratio"] == pytest.approx(1.0)
    assert row["gross_next_return_bps"] == pytest.approx(100.0)
    assert row["next_capital_net_return_bps"] == pytest.approx(20.0)
