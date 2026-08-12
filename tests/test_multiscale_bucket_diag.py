from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    add_ic_rows,
    load_ranked_pool_shard,
    spearman_rank_ic,
)

ROOT = Path(__file__).parents[1]
AUCTION_COMPAT = runpy.run_path(
    ROOT / "experiments/scripts/run_auction_fresh_rank_bucket_diagnostics.py",
    run_name="auction_fresh_rank_bucket_compat",
)
MECH_COMPAT = runpy.run_path(
    ROOT / "experiments/scripts/run_mech328_v2_multiscale_bucket_diagnostics.py",
    run_name="mech328_v2_multiscale_compat",
)


def test_spearman_rank_ic_matches_pandas_with_ties() -> None:
    scores = np.array([4.0, 4.0, 3.0, 2.0, 1.0])
    outcomes = np.array([2.0, 1.0, 2.0, 0.0, 0.0])

    expected = pd.Series(scores).corr(pd.Series(outcomes), method="spearman")

    assert spearman_rank_ic(scores, outcomes) == pytest.approx(expected)


def test_add_ic_rows_uses_original_prediction_values() -> None:
    top = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 4,
            "decision_target_timestamp": ["09:31:00"] * 4,
            "month": ["2025-01"] * 4,
            "prediction": [4.0, 4.0, 2.0, 1.0],
            "excess_bps": [2.0, 0.0, 1.0, 0.0],
        }
    )

    topk_rows, bucket_rows, window_rows = add_ic_rows(
        top,
        variant="test",
        top_k=[4],
        bucket_widths=[4],
        window_widths=[4],
        window_stride=1,
        top_n=4,
    )
    expected = pd.Series(top["prediction"]).corr(
        pd.Series(top["excess_bps"]),
        method="spearman",
    )

    assert topk_rows[0][-1] == pytest.approx(expected)
    assert bucket_rows[0][-1] == pytest.approx(expected)
    assert window_rows[0][-1] == pytest.approx(expected)


def test_load_ranked_pool_shard_is_shared_join_and_sort_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "decision_target_timestamp": ["2025-01-02 09:31:00"] * 3,
            "prediction": [0.2, 0.3, 0.1],
        }
    )
    labels = predictions.iloc[:, :3].copy()
    labels["alpha_return_next_close"] = [0.01, 0.03, -0.01]
    pool = pd.DataFrame(
        [[True, True, True]],
        index=["2025-01-02"],
        columns=predictions["symbol"],
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: predictions.copy())

    frame, trace = load_ranked_pool_shard(
        pred_path=Path("predictions.parquet"),
        labels=labels,
        pool=pool,
    )

    assert frame["symbol"].tolist() == ["000002.SZ", "000001.SZ", "000003.SZ"]
    assert frame["score_rank"].tolist() == [1, 2, 3]
    assert frame["excess_bps"].sum() == pytest.approx(0.0)
    assert trace == {
        "prediction_pool_rows": 3,
        "joined_rows": 3,
        "groups": 1,
        "duplicate_keys": 0,
        "missing_labels": 0,
    }


def test_historical_diagnostic_presets_delegate_to_shared_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    auction_globals = AUCTION_COMPAT["main"].__globals__
    monkeypatch.setitem(auction_globals, "run_rank_bucket_reaudit", calls.append)
    monkeypatch.setitem(auction_globals, "run_multiscale_bucket_diagnostics", calls.append)
    AUCTION_COMPAT["main"]()
    assert len(calls) == 2
    assert (
        calls[0].run_ids == calls[1].run_ids == {"auction_fresh_pruned": AUCTION_COMPAT["RUN_ID"]}
    )
    assert calls[0].months == calls[1].months == list(AUCTION_COMPAT["DEFAULT_DIAGNOSTIC_MONTHS"])

    calls.clear()
    monkeypatch.setitem(
        MECH_COMPAT["main"].__globals__, "run_multiscale_bucket_diagnostics", calls.append
    )
    MECH_COMPAT["main"]()
    assert len(calls) == 1
    assert calls[0].run_ids == {"mech328_v2": MECH_COMPAT["RUN_ID"]}
    assert calls[0].months == list(MECH_COMPAT["DEFAULT_DIAGNOSTIC_MONTHS"])
