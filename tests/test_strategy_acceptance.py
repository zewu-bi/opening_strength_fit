from __future__ import annotations

import json

import pandas as pd
import pytest

from opening_strength_fit.capacity_audit import CapacityConstraints, build_capacity_portfolios
from opening_strength_fit.commands.strategy_acceptance import main
from opening_strength_fit.realistic_acceptance import (
    RealisticExecutionConstraints,
    apply_realistic_execution_constraints,
    merge_realistic_execution_context,
)
from opening_strength_fit.strategy_acceptance import (
    CAPACITY_ONLY,
    REALISTIC_NO_REFILL,
    VISIBLE_PRETRADE_REFILL,
    TailSettings,
    build_visible_pretrade_refill,
    group_targets_from_metrics,
    monthly_block_bootstrap,
    summarize_overlap,
    summarize_tail_robustness,
)


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": ["2024-01-02 09:31:00"] * 3,
            "prediction": [3.0, 2.0, 1.0],
            "turnover_diff_10t": [1_000.0] * 3,
            "ask_price_1": [10.0] * 3,
            "bid_price_1": [9.99] * 3,
            "ask_depth_10": [10_000.0] * 3,
            "status": ["HALT", "TRADE", "TRADE"],
        }
    )


def test_visible_pretrade_refill_revisits_full_ranking() -> None:
    candidates = _candidate_frame()
    capacity = CapacityConstraints(
        target_notional=100.0,
        capacity_notional_col="turnover_diff_10t",
        capacity_price_col="ask_price_1",
        max_participation_rate=0.10,
        max_symbol_weight=1.0,
    )
    execution = RealisticExecutionConstraints(
        capacity_total_notional=1_000.0,
        fee_bps=0.0,
        max_daily_symbol_weight=1.0,
        status_col="status",
        tradable_statuses=("TRADE",),
        max_spread_bps=0.0,
        max_ask_depth_participation_rate=0.0,
    )

    capacity_selected, _ = build_capacity_portfolios(candidates, capacity, pool="pool_L")
    context = candidates[["date", "symbol", "decision_target_timestamp", "status"]]
    capacity_selected = merge_realistic_execution_context(capacity_selected, context)
    no_refill, _ = apply_realistic_execution_constraints(capacity_selected, execution)
    refill, metrics = build_visible_pretrade_refill(
        candidates,
        pool="pool_L",
        capacity_constraints=capacity,
        execution_constraints=execution,
    )

    assert no_refill.empty
    assert refill["symbol"].tolist() == ["000002.SZ"]
    assert refill["rank"].tolist() == [2]
    assert refill["allocated_notional"].tolist() == pytest.approx([100.0])
    assert metrics.loc[0, "fill_ratio"] == pytest.approx(1.0)
    assert metrics.loc[0, "max_candidate_rank"] == pytest.approx(2.0)


def test_overlap_and_tail_summaries_share_policy_contract() -> None:
    times = pd.to_datetime(["2024-01-02 09:31:00", "2024-01-02 09:31:00", "2024-01-02 09:32:00"])
    selected = pd.DataFrame(
        {
            "policy": [VISIBLE_PRETRADE_REFILL] * 3,
            "pool": ["pool_L"] * 3,
            "date": ["2024-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "decision_target_timestamp": times,
            "allocated_notional": [50.0, 50.0, 100.0],
        }
    )
    metrics = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-02"],
            "decision_target_timestamp": times[[0, 2]],
            "target_notional": [100.0, 100.0],
        }
    )
    targets = group_targets_from_metrics(metrics, policy=VISIBLE_PRETRADE_REFILL)
    labels = selected[["date", "symbol", "decision_target_timestamp"]].copy()
    labels["alpha_return_next_close"] = [0.01, -0.01, 0.20]

    positions, daily, adjacent, overlap = summarize_overlap(
        selected,
        targets,
        capacity_total_notional=1_000.0,
    )
    tail, monthly, concentration = summarize_tail_robustness(
        selected,
        targets,
        labels,
        label_col="alpha_return_next_close",
        fee_bps=0.0,
        settings=TailSettings(quantiles=(0.50,), bootstrap_samples=100, bootstrap_seed=7),
    )

    assert positions.loc[positions["symbol"].eq("000001.SZ"), "decision_count"].item() == 2
    assert daily.loc[0, "repeated_symbol_notional_share"] == pytest.approx(0.75)
    assert adjacent.loc[0, "common_symbols"] == 1
    assert overlap.loc[0, "mean_name_jaccard"] == pytest.approx(0.5)
    assert tail.loc[0, "winsor_gross_bps_vs_target"] < tail.loc[0, "raw_gross_bps_vs_target"]
    assert tail.loc[0, "tail_notional_share"] == pytest.approx(0.5)
    assert len(monthly) == 1
    assert set(concentration["unit"]) == {"date", "symbol", "symbol_date"}


def test_monthly_block_bootstrap_is_deterministic() -> None:
    daily = pd.DataFrame(
        {
            "policy": [CAPACITY_ONLY] * 4,
            "pool": ["pool_L"] * 4,
            "date": ["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"],
            "next_capital_net_return_bps": [1.0, 2.0, -1.0, 4.0],
        }
    )
    settings = TailSettings(bootstrap_samples=200, bootstrap_seed=11)

    first = monthly_block_bootstrap(daily, settings=settings)
    second = monthly_block_bootstrap(daily, settings=settings)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "observed_cumulative_net_bps"] == pytest.approx(6.0)


def test_strategy_acceptance_cli_writes_unified_artifacts(tmp_path, monkeypatch) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    label_path = tmp_path / "labels.parquet"
    pool_path = tmp_path / "pool_L.parquet"
    output_dir = tmp_path / "acceptance"
    config_path = tmp_path / "acceptance.toml"

    parts = []
    labels = []
    for clock in ("09:31:00", "09:32:00"):
        part = _candidate_frame().copy()
        part["decision_target_timestamp"] = f"2024-01-02 {clock}"
        parts.append(part)
        for index, symbol in enumerate(part["symbol"]):
            labels.append(
                {
                    "date": "2024-01-02",
                    "symbol": symbol,
                    "decision_target_timestamp": f"2024-01-02 {clock}",
                    "alpha_return_next_close": [0.02, 0.01, -0.01][index],
                }
            )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(prediction_path, index=False)
    pd.DataFrame(labels).to_parquet(label_path, index=False)
    pd.DataFrame(
        {symbol: [True] for symbol in predictions["symbol"].unique()},
        index=pd.Index(["2024-01-02"], name="date"),
    ).to_parquet(pool_path)
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                'id = "strategy_acceptance_test"',
                'kind = "strategy_acceptance"',
                'description = "test"',
                'status = "running"',
                "",
                "[strategy_acceptance]",
                f'predictions = ["{prediction_path}"]',
                f'label_input = ["{label_path}"]',
                'label_col = "alpha_return_next_close"',
                'pool = "L"',
                f'pool_path = "{pool_path}"',
                'policies = ["capacity_only", "realistic_no_refill", "visible_pretrade_refill"]',
                "",
                "[capacity]",
                "target_notional = 100",
                'capacity_notional_col = "turnover_diff_10t"',
                'capacity_price_col = "ask_price_1"',
                "max_participation_rate = 0.10",
                "max_symbol_weight = 1.0",
                "",
                "[execution]",
                "capacity_total_notional = 1000",
                "fee_bps = 0",
                "max_daily_symbol_weight = 1.0",
                'status_col = "status"',
                'tradable_statuses = ["TRADE"]',
                "max_spread_bps = 50",
                "min_child_notional = 0",
                "round_lot_shares = 0",
                "max_ask_depth_participation_rate = 0.25",
                "",
                "[tail]",
                "quantiles = [0.50]",
                "bootstrap_samples = 100",
                "bootstrap_seed = 7",
                "",
                "[output]",
                f'local_dir = "{output_dir}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["osf-audit-strategy-acceptance", "--config", str(config_path)],
    )

    main()

    summary = pd.read_csv(output_dir / "strategy_acceptance_summary.csv")
    trace = json.loads((output_dir / "strategy_acceptance_trace.json").read_text())
    assert set(summary["policy"]) == {
        CAPACITY_ONLY,
        REALISTIC_NO_REFILL,
        VISIBLE_PRETRADE_REFILL,
    }
    fill = summary.set_index("policy")["mean_fill_ratio"]
    assert fill.loc[REALISTIC_NO_REFILL] == pytest.approx(0.0)
    assert fill.loc[VISIBLE_PRETRADE_REFILL] == pytest.approx(1.0)
    assert trace["modeling_notes"][VISIBLE_PRETRADE_REFILL].startswith("Full candidate")
    assert (output_dir / "strategy_acceptance_tail_summary.csv").exists()
    assert (output_dir / "_SUCCESS").exists()
