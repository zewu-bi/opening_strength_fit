from pathlib import Path

from opening_strength_fit.commands.k8s_rendering_common import avoid_nodes_affinity_yaml
from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s_builder_rendering import render_matrix_training_jobs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1.toml"
LABEL15_QUEUE = ROOT / "experiments/scripts/run_ds350_label15_training_queue.sh"
LABEL15_ANALYSIS_JOB = (
    ROOT / "experiments/jobs/support/ds350_label15_pool_internal_analysis_job.yaml"
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
                "required_node_label_values": {"gpu_model": ["A100-80", "H100-80", "H20"]},
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


def test_dataset350_matrix_renderer_preserves_all_fifteen_indexed_jobs() -> None:
    config = load_toml(CONFIG)
    text = render_matrix_training_jobs(CONFIG.relative_to(ROOT), config, H5M_TRAINING_IMAGE)
    cases = [case["name"] for case in config["matrix"]["cases"]]

    assert text.count("kind: Job") == 15
    assert text.count("name: os-nn-ds350-w") == 15
    assert all(f"osf-case: {case}" in text for case in cases)
    assert text.count("completionMode: Indexed") == 15
    assert text.count("completions: 8") == 15
    assert text.count("parallelism: 8") == 15
    assert text.count("suspend: true") == 15
    assert "suspend: false" not in text
    assert text.count('cpu: "8"') == 15
    assert text.count("memory: 256Gi") == 15
    assert text.count('cpu: "16"') == 15
    assert text.count("memory: 384Gi") == 15
    assert text.count("- key: gpu_model") == 15
    assert all(model in text for model in ("A100-80", "H100-80", "H20"))
    assert text.count("--feature-input") == 15
    assert text.count("--label-input") == 15
    assert text.count("--run-id") == 15
    assert text.count(f"image: {H5M_TRAINING_IMAGE}") == 15


def test_dataset350_label15_queue_appends_three_h5m_jobs() -> None:
    text = LABEL15_QUEUE.read_text()
    assert text.count("  os-nn-ds350-w") == 15
    assert text.index("os-nn-ds350-w1401-h3m-v2") < text.index("os-nn-ds350-w0931-h5m-v2")
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
