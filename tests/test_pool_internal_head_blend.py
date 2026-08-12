from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/scripts/run_pool_internal_head_blend.py",
    run_name="pool_internal_head_blend_compat",
)


def _ranked_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 4,
            "decision_target_timestamp": pd.to_datetime(["2025-01-02 09:31"] * 4),
            "base_prediction": [0.4, 0.1, 0.3, 0.2],
            "overlay_prediction": [0.1, 0.4, 0.2, np.nan],
        }
    )
    frame["overlay_available"] = frame["overlay_prediction"].notna()
    for name in ("base", "overlay"):
        frame[f"{name}_rank"], frame[f"{name}_rank_score"] = SCRIPT["normalized_descending_rank"](
            frame, f"{name}_prediction"
        )
    return frame


def test_head_blend_preserves_base_overlay_and_gated_boost_modes() -> None:
    frame = _ranked_frame()
    pd.testing.assert_series_equal(
        SCRIPT["build_variant_score"](frame, {"mode": "base"}),
        frame["base_rank_score"],
    )
    boosted = SCRIPT["build_variant_score"](
        frame,
        {"mode": "head_boost", "overlay_head_n": 2, "base_gate_n": 3, "weight": 0.4},
    )
    assert boosted.gt(frame["base_rank_score"]).tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="unsupported blend variant"):
        SCRIPT["build_variant_score"](frame, {"mode": "unknown"})


def test_head_blend_summaries_keep_period_and_tail_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "variant": ["blend"] * 4,
            "top_n": [100] * 4,
            "test_month": ["2025-01"] * 2 + ["2025-02"] * 2,
            "date": ["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"],
            "short_internal_excess_bps": [1.0, 2.0, -1.0, 4.0],
            "next_internal_excess_bps": [3.0, -2.0, 5.0, 6.0],
            "short_rank_ic": [0.1, 0.2, 0.3, 0.4],
            "next_rank_ic": [0.2, 0.1, 0.4, 0.3],
        }
    )
    summary, monthly, quarterly = SCRIPT["summarize_group_metrics"](metrics)
    assert summary.loc[0, ["groups", "months"]].tolist() == [4, 2]
    assert len(monthly) == 2
    assert len(quarterly) == 1

    selected = pd.DataFrame(
        {"variant": ["blend"] * 10, "top_n": [100] * 10, "next_excess_bps": range(10)}
    )
    tail = SCRIPT["tail_summary"](selected)
    assert tail.loc[0, "raw_mean_bps"] == pytest.approx(4.5)
    assert tail.loc[0, "p95_winsor_mean_bps"] < tail.loc[0, "raw_mean_bps"]
