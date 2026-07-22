from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _tiny_labeled_frame() -> pd.DataFrame:
    rows = []
    dates = ("2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06")
    symbols = ("000001.SZ", "000002.SZ", "600000.SH")
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            feature_signal = float(symbol_index - 1) + date_index * 0.25
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "timestamp": pd.Timestamp(f"{date} 09:31:00"),
                    "decision_time": "09:31:00",
                    "decision_target_timestamp": pd.Timestamp(f"{date} 09:31:00"),
                    "decision_lag_seconds": 0.0,
                    "label": feature_signal * 0.01 + (symbol_index - 1) * 0.002,
                    "valid_label": True,
                    "feature_signal": feature_signal,
                    "feature_liquidity": float((date_index + 1) * (symbol_index + 2)),
                }
            )
    return pd.DataFrame(rows)


def test_training_cli_smoke_writes_core_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "tiny_labeled.parquet"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "tiny_training_smoke.toml"
    _tiny_labeled_frame().to_parquet(input_path, index=False)
    config_path.write_text(
        f"""
[run]
id = "tiny_training_smoke"

[data]
source = "labeled_pvc"
labeled_path = "{input_path.as_posix()}"

[window]
mode = "chronological"
test_start_date = "2022-01-05"
test_end_date = "2022-01-06"

[model]
name = "ridge"
alpha = 1.0

[features]
include_feature_prefixes = ["feature_"]

[evaluation]
top_n = 1
score_bins = 3
bucket_mode = "daily"
selection_mode = "cross_section"
ic_mode = "cross_section"

[universe]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opening_strength_fit.commands.experiment_run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (output_dir / "predictions_2022-01.parquet").exists()
    assert (output_dir / "predictions.parquet").exists()
    assert not (output_dir / "score_buckets_2022-01.csv").exists()
    assert (output_dir / "score_buckets.csv").exists()
    assert (output_dir / "metrics_by_year.csv").exists()
    assert (output_dir / "metrics.json").exists()

    success = json.loads((output_dir / "_SUCCESS").read_text(encoding="utf-8"))
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    metrics = pd.read_csv(output_dir / "metrics_by_year.csv")

    assert success == {
        "run_id": "tiny_training_smoke",
        "windows": 1,
        "status": "completed",
        "format_version": 1,
    }
    assert len(predictions) == 6
    assert predictions["prediction"].notna().all()
    assert metrics.loc[0, "run_id"] == "tiny_training_smoke"
    assert int(metrics.loc[0, "test_rows"]) == 6
    assert metrics_payload["reproducibility"]["config_path"] == str(config_path)
    assert (
        metrics_payload["reproducibility"]["config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert metrics_payload["train_stats_by_window"]["2022-01"]["feature_names"] == [
        "feature_liquidity",
        "feature_signal",
    ]
