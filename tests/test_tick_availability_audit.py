from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.commands.tick_availability_audit import (
    normalize_receipt_epoch_us,
    summarize_availability,
)


def test_summarize_availability_excludes_missing_receipt_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "Symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "receipt_timestamp_unit": ["microseconds", "microseconds", "missing"],
            "receipt_timestamp_covered": [True, True, False],
            "receipt_after_decision_seconds": [0.5, -0.2, pd.NA],
            "exchange_state_age_seconds": [0.0, 1.0, 2.0],
        }
    )

    summary = summarize_availability(frame)

    assert summary["rows"] == 3
    assert summary["receipt_timestamp_rows"] == 2
    assert summary["receipt_timestamp_coverage"] == pytest.approx(2 / 3)
    assert summary["receipt_after_decision_rows"] == 1
    assert summary["receipt_after_decision_fraction_of_covered"] == pytest.approx(0.5)
    assert summary["receipt_after_decision_gt_2s_fraction_of_covered"] == 0.0


def test_normalize_receipt_epoch_us_handles_source_unit_change() -> None:
    normalized, units = normalize_receipt_epoch_us(
        pd.Series([0, 1_735_781_460_500_000, 1_743_485_106_193_573_500])
    )

    assert pd.isna(normalized.iloc[0])
    assert normalized.iloc[1] == 1_735_781_460_500_000
    assert normalized.iloc[2] == pytest.approx(1_743_485_106_193_573.5)
    assert units.tolist() == ["missing", "microseconds", "nanoseconds"]
