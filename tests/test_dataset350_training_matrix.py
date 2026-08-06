from pathlib import Path

from opening_strength_fit.commands.k8s_rendering_common import avoid_nodes_affinity_yaml
from opening_strength_fit.config import load_toml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1.toml"
W0931_JOBS = ROOT / "experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w0931_jobs.yaml"
W1001_JOBS = ROOT / "experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w1001_jobs.yaml"
W1401_JOBS = ROOT / "experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w1401_jobs.yaml"
H5M_JOBS = ROOT / "experiments/jobs/nn_ds350_label15_36m_grouped_gated_v2_mse_v1_h5m_jobs.yaml"
LABEL15_QUEUE = ROOT / "experiments/scripts/run_ds350_label15_training_queue.sh"
LABEL15_ANALYSIS_JOB = (
    ROOT / "experiments/jobs/support/ds350_label15_pool_internal_analysis_job.yaml"
)
TRAINING_IMAGE = (
    "registry.corp.highfortfunds.com/bizewu/opening-strength-fit@"
    "sha256:cb7120cf0549ddb4a91533587b04aaf38847b5b78c13ddfbe586d10e22b106b4"
)
H5M_TRAINING_IMAGE = (
    "registry.corp.highfortfunds.com/bizewu/opening-strength-fit@"
    "sha256:1c35beb1275aca13e5a71fd03bc13ae703078b29b3e2a744c77e386a32c4917a"
)


def test_required_gpu_models_render_into_the_same_node_affinity_term() -> None:
    rendered = avoid_nodes_affinity_yaml(
        {
            "k8s": {
                "avoid_nodes": ["node22"],
                "required_node_label_values": {
                    "gpu_model": ["A100-80", "H100-80", "H20"]
                },
            }
        },
        indent=6,
    )
    assert rendered.count("nodeSelectorTerms:") == 1
    assert "- key: kubernetes.io/hostname" in rendered
    assert "operator: NotIn" in rendered
    assert "- key: gpu_model" in rendered
    assert "operator: In" in rendered
    assert all(model in rendered for model in ("A100-80", "H100-80", "H20"))


def test_dataset350_training_matrix_has_all_fifteen_cases() -> None:
    config = load_toml(CONFIG)
    cases = config["matrix"]["cases"]
    assert len(cases) == 15
    assert len({case["name"] for case in cases}) == 15
    horizons = [case["horizon"] for case in cases]
    assert horizons.count("1m") == 3
    assert horizons.count("3m") == 3
    assert horizons.count("10m") == 2
    assert horizons.count("1h") == 2
    assert horizons.count("close") == 2
    assert horizons.count("5m") == 3

    for case in cases:
        window_slug = case["name"].split("_h", maxsplit=1)[0].removeprefix("w")
        assert f"opening_{window_slug}_features_350" in case["feature_path"]
        assert f"opening_{window_slug}_labels_h" in case["label_path"]

    assert config["data"]["expected_feature_count"] == 350
    assert config["data"]["trusted_model_ready_split"] is True
    assert config["universe"]["enabled"] is False
    assert config["features"]["feature_value_transform"] == "none"
    assert config["model"]["target_col"] == "target_label"
    assert config["model"]["random_state"] == 31
    assert config["model"]["training_tensor_storage"] == "cuda_resident"
    assert config["model"]["cuda_resident_reserve_gib"] == 12.0
    assert config["k8s"]["resources"]["cpu_request"] == "8"
    assert config["k8s"]["resources"]["cpu_limit"] == "16"
    assert config["k8s"]["resources"]["memory_request"] == "256Gi"
    assert config["k8s"]["helper_image"] == H5M_TRAINING_IMAGE
    assert config["k8s"]["required_node_label_values"]["gpu_model"] == [
        "A100-80",
        "H100-80",
        "H20",
    ]


def test_dataset350_w0931_uses_one_indexed_job_per_label() -> None:
    text = W0931_JOBS.read_text()
    cases = [
        "w0931_0940_h1m",
        "w0931_0940_h3m",
        "w0931_0940_h10m",
        "w0931_0940_h1h",
        "w0931_0940_hclose",
    ]
    assert "kind: List" in text
    assert text.count("name: os-nn-ds350-w0931-") == 5
    assert all(f"osf-case: {case}" in text for case in cases)
    assert "completionMode: Indexed" in text
    assert "completions: 8" in text
    assert "parallelism: 8" in text
    assert text.count("suspend: true") == 5
    assert "suspend: false" not in text
    assert 'requests: {cpu: "8", memory: 256Gi' in text
    assert 'limits: {cpu: "16", memory: 384Gi' in text
    assert "- key: gpu_model" in text
    assert all(model in text for model in ("A100-80", "H100-80", "H20"))
    assert "--feature-input" in text
    assert "--label-input" in text
    assert "--run-id" in text
    assert "h5m" not in text
    assert text.count(f"image: {TRAINING_IMAGE}") == 1


def test_dataset350_w1001_queues_five_indexed_label_jobs() -> None:
    text = W1001_JOBS.read_text()
    cases = [
        "w1001_1010_h1m",
        "w1001_1010_h3m",
        "w1001_1010_h10m",
        "w1001_1010_h1h",
        "w1001_1010_hclose",
    ]
    assert "kind: List" in text
    assert text.count("name: os-nn-ds350-w1001-") == 5
    assert all(f"osf-case: {case}" in text for case in cases)
    assert text.count("suspend: true") == 5
    assert "suspend: false" not in text
    assert "completionMode: Indexed" in text
    assert "completions: 8" in text
    assert "parallelism: 8" in text
    assert "opening_1001_1010_features_350" in text
    assert 'requests: {cpu: "8", memory: 256Gi' in text
    assert 'limits: {cpu: "16", memory: 384Gi' in text
    assert "- key: gpu_model" in text
    assert "h5m" not in text
    assert text.count(f"image: {TRAINING_IMAGE}") == 1


def test_dataset350_w1401_queues_two_indexed_label_jobs() -> None:
    text = W1401_JOBS.read_text()
    assert "kind: List" in text
    assert text.count("name: os-nn-ds350-w1401-") == 2
    assert "osf-case: w1401_1410_h1m" in text
    assert "osf-case: w1401_1410_h3m" in text
    assert text.count("suspend: true") == 2
    assert "suspend: false" not in text
    assert "completionMode: Indexed" in text
    assert "completions: 8" in text
    assert "parallelism: 8" in text
    assert "opening_1401_1410_features_350" in text
    assert "opening_1401_1410_labels_h1m_v2" in text
    assert "opening_1401_1410_labels_h3m_v2" in text
    assert 'requests: {cpu: "8", memory: 256Gi' in text
    assert 'limits: {cpu: "16", memory: 384Gi' in text
    assert "- key: gpu_model" in text
    assert "h5m" not in text
    assert text.count(f"image: {TRAINING_IMAGE}") == 1


def test_dataset350_h5m_jobs_cover_all_three_windows() -> None:
    text = H5M_JOBS.read_text()
    cases = ["w0931_0940_h5m", "w1001_1010_h5m", "w1401_1410_h5m"]
    assert "kind: List" in text
    assert text.count("name: os-nn-ds350-w") == 3
    assert all(f"osf-case: {case}" in text for case in cases)
    assert text.count("suspend: true") == 3
    assert "suspend: false" not in text
    assert "completionMode: Indexed" in text
    assert "completions: 8" in text
    assert "parallelism: 8" in text
    assert "opening_${WINDOW}_features_350" in text
    assert "opening_${WINDOW}_labels_h5m_v2" in text
    assert 'requests: {cpu: "8", memory: 256Gi' in text
    assert 'limits: {cpu: "16", memory: 384Gi' in text
    assert "- key: gpu_model" in text
    assert text.count(f"image: {H5M_TRAINING_IMAGE}") == 1


def test_dataset350_label15_queue_appends_three_h5m_jobs() -> None:
    text = LABEL15_QUEUE.read_text()
    assert text.count("  os-nn-ds350-w") == 15
    assert text.index("os-nn-ds350-w1401-h3m-v2") < text.index(
        "os-nn-ds350-w0931-h5m-v2"
    )
    assert "os-nn-ds350-w1001-h5m-v2" in text
    assert "os-nn-ds350-w1401-h5m-v2" in text
    assert "all 15 ds350 label jobs completed" in text


def test_dataset350_label15_analysis_reuses_pool_internal_summary_without_plots() -> None:
    text = LABEL15_ANALYSIS_JOB.read_text()
    assert "completionMode: Indexed" in text
    assert "completions: 15" in text
    assert "parallelism: 5" in text
    assert text.count("                w0931_0940_h") == 6
    assert text.count("                w1001_1010_h") == 6
    assert text.count("                w1401_1410_h") == 3
    assert "osf-analyze-pool-internal-top100" in text
    assert "--pool universe" in text
    assert "--pool L" in text
    assert "--top-n 100" in text
    assert "--pool-date-lag-sessions 0" in text
    assert "pool_internal_quarter_summary.csv" in text
    assert "--report-dir" not in text
