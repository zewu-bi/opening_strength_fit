from pathlib import Path

from opening_strength_fit.config import load_toml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1.toml"
JOB = ROOT / "experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_sharded_job.yaml"


def test_dataset350_training_matrix_has_requested_twelve_cases() -> None:
    config = load_toml(CONFIG)
    cases = config["matrix"]["cases"]
    assert len(cases) == 12
    assert len({case["name"] for case in cases}) == 12
    horizons = [case["horizon"] for case in cases]
    assert horizons.count("1m") == 3
    assert horizons.count("3m") == 3
    assert horizons.count("10m") == 2
    assert horizons.count("1h") == 2
    assert horizons.count("close") == 2
    assert "5m" not in horizons

    for case in cases:
        window_slug = case["name"].split("_h", maxsplit=1)[0].removeprefix("w")
        assert f"opening_{window_slug}_features_350" in case["feature_path"]
        assert f"opening_{window_slug}_labels_h" in case["label_path"]

    assert config["data"]["expected_feature_count"] == 350
    assert config["features"]["feature_value_transform"] == "none"
    assert config["model"]["target_col"] == "target_label"
    assert config["model"]["random_state"] == 31


def test_dataset350_training_matrix_job_maps_cases_to_oos_shards() -> None:
    text = JOB.read_text()
    assert "completions: 96" in text
    assert "parallelism: 8" in text
    assert "CASE_INDEX=$((INDEX / 8))" in text
    assert "SPLIT_INDEX=$((INDEX % 8))" in text
    assert "--feature-input" in text
    assert "--label-input" in text
    assert "--run-id" in text
    assert "h5m" not in text
