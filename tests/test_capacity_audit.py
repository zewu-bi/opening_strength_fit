from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.capacity_audit import (
    CapacityConstraints,
    build_capacity_portfolios,
    summarize_capacity_groups,
)
from opening_strength_fit.commands.capacity_audit import main


def _capacity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2022-01-03",
                "symbol": "000001.SZ",
                "decision_target_timestamp": "2022-01-03 09:31:00",
                "prediction": 3.0,
                "turnover_diff_30t": 10_000.0,
            },
            {
                "date": "2022-01-03",
                "symbol": "000002.SZ",
                "decision_target_timestamp": "2022-01-03 09:31:00",
                "prediction": 2.0,
                "turnover_diff_30t": 4_000.0,
            },
            {
                "date": "2022-01-03",
                "symbol": "600000.SH",
                "decision_target_timestamp": "2022-01-03 09:31:00",
                "prediction": 1.0,
                "turnover_diff_30t": 2_000.0,
            },
        ]
    )


def test_capacity_portfolio_allocates_until_target_or_limits() -> None:
    constraints = CapacityConstraints(
        target_notional=1_000.0,
        max_participation_rate=0.10,
        max_symbol_weight=0.50,
    )

    selected, metrics = build_capacity_portfolios(
        _capacity_frame(),
        constraints,
        pool="pool_L",
    )
    summary = summarize_capacity_groups(metrics).set_index("pool")

    assert selected["allocated_notional"].tolist() == pytest.approx([500.0, 400.0, 100.0])
    assert selected["target_weight"].tolist() == pytest.approx([0.5, 0.4, 0.1])
    assert metrics.loc[0, "fill_ratio"] == pytest.approx(1.0)
    assert metrics.loc[0, "filled"]
    assert metrics.loc[0, "top_depth_to_target"] == pytest.approx(3.0)
    assert metrics.loc[0, "max_capacity_participation_rate"] == pytest.approx(0.10)
    assert summary.loc["pool_L", "fill_success_rate"] == pytest.approx(1.0)
    assert summary.loc["pool_L", "mean_top_depth_to_target"] == pytest.approx(3.0)
    assert "capital_net_return_bps" not in metrics.columns
    assert "capital_excess_bps" not in summary.columns


def test_capacity_audit_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "capacity"
    _capacity_frame().to_parquet(predictions, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "osf-audit-capacity",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--pool",
            "universe",
            "--target-notional",
            "1000",
            "--max-participation-rate",
            "0.10",
            "--max-symbol-weight",
            "0.50",
        ],
    )

    main()

    summary = pd.read_csv(output_dir / "capacity_audit_summary.csv")
    selected = pd.read_csv(output_dir / "capacity_audit_selected.csv")
    assert summary.loc[0, "pool"] == "universe"
    assert summary.loc[0, "fill_ratio"] == pytest.approx(1.0)
    assert summary.loc[0, "mean_top_depth_to_target"] == pytest.approx(3.0)
    assert selected["allocated_notional"].sum() == pytest.approx(1_000.0)
    assert (output_dir / "capacity_audit_trace.json").exists()
