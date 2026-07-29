from __future__ import annotations

import numpy as np
import pytest

from opening_strength_fit.temporal_analysis import write_sequence_npz
from opening_strength_fit.temporal_modeling import (
    evaluate_temporal_model,
    fit_temporal_model,
)

torch = pytest.importorskip("torch")


def _write_day(path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    symbols = np.array([f"{index:06d}.SZ" for index in range(16)])
    values = rng.normal(0.0, 0.01, size=(16, 3, 20)).astype(np.float32)
    target = (values[:, 0, :].mean(axis=1) + rng.normal(0, 0.001, 16)).astype(np.float32)
    arrays = {
        "symbols": symbols,
        "clock_seconds": np.arange(20, dtype=np.int32) * 60 + 34200,
        "values": values,
        "valid": np.isfinite(values),
        "target": target,
        "pool_member": np.arange(16) % 2 == 0,
    }
    write_sequence_npz(path, arrays)


def test_temporal_model_one_epoch_smoke(tmp_path) -> None:
    paths = []
    for index, date in enumerate(["2025-01-02", "2025-01-03", "2025-01-06"]):
        path = tmp_path / f"date={date}" / "sequence.npz"
        _write_day(path, seed=index)
        paths.append(path)
    common = {
        "device": "cpu",
        "batch_size": 8,
        "value_mode": "cross_section_rank",
        "latest_clocks": {"1m": "14:47", "10m": "14:38", "60m": "13:48"},
        "raw_scale": 0.02,
        "evaluation_universe": "pool_l",
        "top_n": 2,
    }
    model, history, trace = fit_temporal_model(
        paths[:2],
        paths[2:],
        architecture="tcn",
        train_universe="pool_l",
        hidden_width=8,
        dropout=0.0,
        epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        loss_name="huber",
        head_fraction=0.1,
        selection_metric="top_n_excess",
        patience=1,
        seed=7,
        **common,
    )
    metrics, predictions = evaluate_temporal_model(
        model,
        paths[2:],
        include_predictions=True,
        **common,
    )
    assert len(history) == 1
    assert trace["parameter_count"] > 0
    assert len(predictions) == 16
    assert np.isfinite(metrics["loss"])
