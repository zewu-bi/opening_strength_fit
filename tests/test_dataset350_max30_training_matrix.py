from copy import deepcopy
from pathlib import Path

from opening_strength_fit.config import load_toml

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "experiments/runs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1.toml"
MAX30_CONFIG = ROOT / "experiments/runs/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1.toml"
MAX30_JOBS = (
    ROOT / "experiments/jobs/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1_sharded_job.yaml"
)
MAX30_QUEUE = ROOT / "experiments/scripts/run_ds350_label15_max30_training_queue.sh"

MAX30_ROOT = (
    "/mnt/output/opening_strength_fit/nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
)
BASELINE_ROOT = "/mnt/output/opening_strength_fit/nn/nn_ds350_label12_36m_grouped_gated_v2_mse_v1"


def test_max30_changes_only_the_intended_training_budget_and_run_identity() -> None:
    baseline = load_toml(BASELINE_CONFIG)
    max30 = load_toml(MAX30_CONFIG)

    assert max30["model"]["max_epochs"] == 30
    assert max30["model"]["early_stopping_patience"] == 3
    assert max30["matrix"] == baseline["matrix"]

    expected_model = deepcopy(baseline["model"])
    expected_model["max_epochs"] = 30
    expected_model["early_stopping_patience"] = 3
    assert max30["model"] == expected_model

    assert max30["data"] == baseline["data"]
    assert max30["universe"] == baseline["universe"]
    assert max30["features"] == baseline["features"]
    assert max30["window"] == baseline["window"]
    assert max30["evaluation"] == baseline["evaluation"]
    assert max30["output"]["k8s_dir"] == MAX30_ROOT
    assert max30["output"]["k8s_dir"] != baseline["output"]["k8s_dir"]
    assert "max30" in max30["run"]["id"]


def test_max30_manifest_has_fifteen_isolated_eight_fold_jobs() -> None:
    config = load_toml(MAX30_CONFIG)
    text = MAX30_JOBS.read_text()
    cases = [case["name"] for case in config["matrix"]["cases"]]

    assert "kind: List" in text
    assert text.count("name: os-nn-ds350-m30-") == 15
    assert all(f"osf-case: {case}" in text for case in cases)
    assert "completionMode: Indexed" in text
    assert "completions: 8" in text
    assert "parallelism: 8" in text
    assert text.count("suspend: true") == 1
    assert "suspend: false" not in text
    assert 'requests: {cpu: "8", memory: 256Gi' in text
    assert 'limits: {cpu: "16", memory: 384Gi' in text
    assert all(model in text for model in ("A100-80", "H100-80", "H20"))
    assert f'OUT="{MAX30_ROOT}/${{CASE}}/month_${{TEST_START}}"' in text
    assert BASELINE_ROOT not in text
    assert MAX30_CONFIG.name in text
    assert text.count(f"image: {config['k8s']['helper_image']}") == 1


def test_max30_queue_runs_all_fifteen_labels_with_at_most_eight_gpus() -> None:
    text = MAX30_QUEUE.read_text()

    assert text.count("  os-nn-ds350-m30-") == 15
    assert 'if [[ "${SUCCEEDED:-0}" -ge 8 ]]' in text
    assert "all 15 ds350 max-30 label jobs completed" in text
    assert "os-nn-ds350-w0931-h1m-v2" not in text
