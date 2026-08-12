from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
EVIDENCE = (
    ROOT / "experiments/evidence/backtests/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
)
HISTORICAL_EVIDENCE = (
    ROOT / "experiments/evidence/backtests/nn_ds350_label12_36m_grouped_gated_v2_mse_v1"
)


def test_ds350_max30_evidence_is_complete_and_self_consistent() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["cases"] == 15
    assert manifest["training_folds"] == 120
    assert manifest["four_figure_case"] == "w0931_0940_h1m"
    for name in (
        "01_signal_acceptance.svg",
        "02_top100_cumulative.svg",
        "03_top1000_bucket_curve.svg",
        "04_top1000_return_distribution.svg",
    ):
        assert name in manifest["files"]

    for name, metadata in manifest["files"].items():
        path = EVIDENCE / name
        assert path.stat().st_size == metadata["bytes"]
        assert path.stat().st_size < 1_000_000
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    matrix = pd.read_csv(EVIDENCE / "matrix_summary.csv")
    folds = pd.read_csv(EVIDENCE / "training_fold_summary.csv")
    assert matrix["case"].nunique() == 15
    assert len(folds) == 120
    assert folds.groupby("case", observed=True).size().eq(8).all()
    assert (folds["best_epoch"] > 10).sum() == 107
    assert (folds["best_epoch"] == 30).sum() == 36


def test_ds350_top1000_evidence_reused_embedded_next_label() -> None:
    for name in ("trace_top1000_bucket.json", "trace_top1000_distribution.json"):
        trace = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
        assert trace["prediction_next_label_col"] == "label_next_close"
        assert trace["next_label_root"] is None
        assert all(
            month["next_label_source"] == "prediction:label_next_close"
            for month in trace["months"].values()
        )


def test_ds350_max10_evidence_has_compact_complete_traces() -> None:
    manifest = json.loads((HISTORICAL_EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    trace_bundle = json.loads((HISTORICAL_EVIDENCE / "traces.json").read_text(encoding="utf-8"))
    expected_cases = set(pd.read_csv(HISTORICAL_EVIDENCE / "matrix_group_pooled.csv")["case"])

    assert manifest["status"] == "superseded"
    assert manifest["cases"] == trace_bundle["case_count"] == len(expected_cases) == 15
    assert set(trace_bundle["cases"]) == expected_cases
    assert "traces.json" in manifest["files"]
    assert not any(name.startswith("traces/") for name in manifest["files"])

    for case, trace in trace_bundle["cases"].items():
        assert trace["variant"] == case
        assert trace["run_id"] == f"nn_ds350_{case}_36m_grouped_gated_v2_mse_v1"
        assert trace["rows"] > 0
        assert 0 <= trace["missing_next_close_rows"] < trace["rows"]

    for name, metadata in manifest["files"].items():
        path = HISTORICAL_EVIDENCE / name
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
