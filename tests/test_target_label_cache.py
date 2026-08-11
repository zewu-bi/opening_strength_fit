from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.commands.target_label_cache import _merge_short_label_input


def _base_frame() -> pd.DataFrame:
    decision_times = pd.to_datetime(["2025-01-02 09:30:00", "2025-01-02 09:31:00"])
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "timestamp": decision_times,
            "decision_target_timestamp": decision_times,
            "label": [0.01, 0.02],
            "gross_label": [0.011, 0.021],
            "valid_label": [True, True],
            "feature": [1.0, 2.0],
        }
    )


def _short_sidecar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001.SZ"],
            "decision_target_timestamp": pd.to_datetime(["2025-01-02 09:31:00"]),
            "label": [0.03],
            "gross_label": [0.031],
            "valid_label": [True],
            "sell_vwap": [10.3],
            "hold_seconds": [180],
        }
    )


def test_merge_short_label_input_replaces_outcome_and_invalidates_context(tmp_path) -> None:
    sidecar_path = tmp_path / "short.parquet"
    _short_sidecar().to_parquet(sidecar_path, index=False)

    merged, stats = _merge_short_label_input(
        _base_frame(),
        sidecar_path,
        label_col="label",
        source_label_col="label",
        source_valid_col="valid_label",
    )

    context = merged.loc[
        merged["decision_target_timestamp"].dt.strftime("%H:%M:%S") == "09:30:00"
    ].iloc[0]
    sample = merged.loc[
        merged["decision_target_timestamp"].dt.strftime("%H:%M:%S") == "09:31:00"
    ].iloc[0]
    assert pd.isna(context["label"])
    assert not bool(context["valid_label"])
    assert sample["label"] == pytest.approx(0.03)
    assert sample["gross_label"] == pytest.approx(0.031)
    assert sample["sell_vwap"] == pytest.approx(10.3)
    assert sample["hold_seconds"] == 180
    assert sample["feature"] == pytest.approx(2.0)
    assert stats == {
        "sidecar_rows": 1,
        "matched_rows": 1,
        "valid_rows": 1,
        "base_rows_without_sidecar": 1,
    }


def test_merge_short_label_input_rejects_duplicate_keys(tmp_path) -> None:
    sidecar_path = tmp_path / "short.parquet"
    pd.concat([_short_sidecar(), _short_sidecar()], ignore_index=True).to_parquet(
        sidecar_path,
        index=False,
    )

    with pytest.raises(SystemExit, match="keys are not unique"):
        _merge_short_label_input(
            _base_frame(),
            sidecar_path,
            label_col="label",
            source_label_col="label",
            source_valid_col="valid_label",
        )
