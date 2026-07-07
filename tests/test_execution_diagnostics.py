from __future__ import annotations

import math

import pandas as pd

from opening_strength_fit.execution_diagnostics import (
    DiagnosticCase,
    run_ask_level_attribution_case,
    run_execution_context_case,
)


def test_ask_level_attribution_splits_allocated_notional_by_visible_levels(tmp_path) -> None:
    selected_path = tmp_path / "selected.csv"
    prediction_root = tmp_path / "predictions"
    output_dir = tmp_path / "out"
    prediction_root.mkdir()

    selected = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_target_timestamp": [
                "2025-01-02 09:31:00",
                "2025-01-02 09:31:00",
            ],
            "allocated_notional": [100.0, 50.0],
        }
    )
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_target_timestamp": [
                "2025-01-02 09:31:00",
                "2025-01-02 09:31:00",
            ],
            "ask_price_1": [10.0, 10.0],
            "ask_volume_1": [5.0, 10.0],
            "ask_price_2": [10.0, 10.0],
            "ask_volume_2": [10.0, 0.0],
        }
    ).to_parquet(prediction_root / "predictions.parquet", index=False)

    output_path = run_ask_level_attribution_case(
        DiagnosticCase(
            name="test",
            selected_path=selected_path,
            prediction_root=prediction_root,
            output_dir=output_dir,
        ),
        levels=(1, 2),
    )

    summary = pd.read_csv(output_path).set_index("bucket")
    assert summary.loc["ask1", "filled_notional"] == 100.0
    assert summary.loc["ask2", "filled_notional"] == 50.0
    assert summary.loc["ask1", "full_within_rows"] == 1.0
    assert summary.loc["ask2", "full_within_rows"] == 2.0
    assert summary.loc["beyond_ask2", "filled_notional"] == 0.0


def test_execution_context_derives_spread_limit_distance_and_depth(tmp_path) -> None:
    selected_path = tmp_path / "selected.csv"
    prediction_root = tmp_path / "predictions"
    output_dir = tmp_path / "out"
    prediction_root.mkdir()

    pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001.SZ"],
            "decision_target_timestamp": ["2025-01-02 09:31:00"],
        }
    ).to_csv(selected_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000002.SZ"],
            "decision_target_timestamp": [
                "2025-01-02 09:31:00",
                "2025-01-02 09:31:00",
            ],
            "ask_price_1": [10.0, 20.0],
            "bid_price_1": [9.8, 19.5],
            "limit_up_price": [11.0, 21.0],
            "ask_price_2": [10.1, 20.2],
            "ask_volume_1": [100.0, 50.0],
            "ask_volume_2": [50.0, 50.0],
            "status": ["active", "ignored"],
        }
    ).to_parquet(prediction_root / "predictions.parquet", index=False)

    output_path = run_execution_context_case(
        DiagnosticCase(
            name="test",
            selected_path=selected_path,
            prediction_root=prediction_root,
            output_dir=output_dir,
        )
    )

    context = pd.read_parquet(output_path)
    assert len(context) == 1
    row = context.iloc[0]
    assert row["symbol"] == "000001.SZ"
    assert row["capacity_price"] == 10.0
    assert math.isclose(row["spread_bps"], (10.0 - 9.8) / 9.9 * 10_000.0)
    assert math.isclose(row["ask1_to_limit_up_bps"], (11.0 - 10.0) / 10.0 * 10_000.0)
    assert row["ask_depth_notional"] == 10.0 * 100.0 + 10.1 * 50.0
