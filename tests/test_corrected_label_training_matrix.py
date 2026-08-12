from __future__ import annotations

import runpy
import tomllib
from pathlib import Path

from opening_strength_fit.commands.k8s_rendering import render_sharded_training_job
from opening_strength_fit.k8s_builder_rendering import render_indexed_builder_job

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
CANONICAL = RUNS / (
    "nn_delay6_v6_decision_clock_state_36m_2022_2025_w0931_0940_"
    "auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
)
TRAINING_CONFIGS = tuple(
    sorted(RUNS.glob("nn_v6_*_corrected_nextclose_36m_grouped_gated_v2_mse.toml"))
)
TARGET_CONFIGS = tuple(sorted(RUNS.glob("build_target_v6_*_corrected_nextclose.toml")))
PREPARE_COMPAT = runpy.run_path(
    ROOT / "experiments/scripts/prepare_corrected_label_training_matrix.py",
    run_name="prepare_corrected_label_training_matrix_compat",
)
TOP1000_COMPAT = runpy.run_path(
    ROOT / "experiments/scripts/render_corrected_label_top1000_jobs.py",
    run_name="render_corrected_label_top1000_jobs_compat",
)


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_six_training_configs_preserve_latest_v6_training_contract() -> None:
    canonical = _load(CANONICAL)
    assert len(TRAINING_CONFIGS) == 6

    expected = {
        "nn_v6_w0931_0940_short1m": ("09:31:00", "09:40:00", 60, "608Gi", "960Gi"),
        "nn_v6_w0931_0940_short3m": ("09:31:00", "09:40:00", 180, "608Gi", "960Gi"),
        "nn_v6_w1001_1010_short1m": ("10:01:00", "10:10:00", 60, "950Gi", "1400Gi"),
        "nn_v6_w1001_1010_short3m": ("10:01:00", "10:10:00", 180, "950Gi", "1400Gi"),
        "nn_v6_w1401_1410_short1m": ("14:01:00", "14:10:00", 60, "608Gi", "960Gi"),
        "nn_v6_w1401_1410_short3m": ("14:01:00", "14:10:00", 180, "608Gi", "960Gi"),
    }
    for path in TRAINING_CONFIGS:
        config = _load(path)
        run_id = config["run"]["id"]
        prefix = next(prefix for prefix in expected if run_id.startswith(prefix))
        start, end, hold, memory_request, memory_limit = expected[prefix]
        assert config["run"]["status"] == "completed"
        assert config["sample"]["start_time"] == start
        assert config["sample"]["end_time"] == end
        assert config["labels"]["hold_seconds"] == hold
        assert config["data"]["labeled_path"].endswith("corrected_nextclose")
        assert "opening_2013_2025_next_close_labels_v1" not in str(config)
        assert len(config["k8s"]["wait_for_paths"]) == 14
        for section in ("universe", "features", "filters", "window", "model", "evaluation"):
            assert config[section] == canonical[section]
        for key in ("avoid_nodes", "shard_parallelism"):
            assert config["k8s"][key] == canonical["k8s"][key]
        resources = config["k8s"]["resources"]
        for key in ("cpu_request", "cpu_limit", "gpu_limit"):
            assert resources[key] == canonical["k8s"]["resources"][key]
        assert resources["memory_request"] == memory_request
        assert resources["memory_limit"] == memory_limit
        assert config["k8s"]["node_selector"] == canonical["k8s"]["node_selector"]
        manifest = render_sharded_training_job(path, config, config["k8s"]["helper_image"])
        assert manifest.count("kind: Job") == 1
        assert "completionMode: Indexed" in manifest
        assert "completions: 8" in manifest
        assert "parallelism: 8" in manifest
        assert "JOB_COMPLETION_INDEX" in manifest
        assert (
            "TEST_STARTS=(2022-01 2022-07 2023-01 2023-07 2024-01 2024-07 2025-01 2025-07)"
            in manifest
        )
        assert (
            "TEST_ENDS=(2022-06 2022-12 2023-06 2023-12 2024-06 2024-12 2025-06 2025-12)"
            in manifest
        )


def test_target_matrix_has_three_1m_and_three_3m_corrected_mixed_targets() -> None:
    assert len(TARGET_CONFIGS) == 6
    one_minute = 0
    three_minute = 0
    for path in TARGET_CONFIGS:
        config = _load(path)
        target = config["target_cache"]
        assert target["mode"] == "mixed"
        assert target["long_label_weight"] == 0.30
        assert "next_close_decision_clock_state_clock6" in target["long_label_input"]
        assert "opening_2013_2025_next_close_labels_v1" not in str(config)
        if "short_label_input" in target:
            three_minute += 1
            assert "h180_vwap60" in target["short_label_input"]
            assert target["short_label_col"] == "label"
            assert target["short_valid_col"] == "valid_label"
        else:
            one_minute += 1
            assert "short1m" in config["run"]["id"]
        manifest = render_indexed_builder_job(path, config, config["k8s"]["helper_image"])
        assert "completionMode: Indexed" in manifest
        assert "completions: 7" in manifest
        assert "--long-label-input" in manifest
        if "short_label_input" in target:
            assert "--short-label-input" in manifest
            success_path = config["target_cache_shards"]["short_label_success_template"]
            assert success_path.replace("{year}", "${YEAR}") in manifest
    assert (one_minute, three_minute) == (3, 3)


def test_historical_matrix_renderers_delegate_to_canonical_configs() -> None:
    assert PREPARE_COMPAT["TARGET_CONFIGS"] == TARGET_CONFIGS
    assert PREPARE_COMPAT["TRAINING_CONFIGS"] == TRAINING_CONFIGS
    for path in TARGET_CONFIGS:
        assert "completionMode: Indexed" in PREPARE_COMPAT["_render"](
            path, render_indexed_builder_job
        )
    for path in TRAINING_CONFIGS:
        assert "completionMode: Indexed" in PREPARE_COMPAT["_render"](
            path, render_sharded_training_job
        )


def test_historical_top1000_renderer_preserves_job_constraints() -> None:
    run_id = TOP1000_COMPAT["RUN_TEMPLATE"].format(window="0931_0940", horizon="short1m")
    for mode, node in (("rank", "node13"), ("hist", "node14")):
        config = TOP1000_COMPAT["_config"](run_id, "0931_0940", "short1m", mode, node)
        manifest = TOP1000_COMPAT["render_top1000_job"](
            Path("compat.toml"), config, TOP1000_COMPAT["IMAGE"]
        )
        assert f"name: os-top1000-corrected-0931-0940-1m-{mode}" in manifest
        assert f'kubernetes.io/hostname: "{node}"' in manifest
        assert "key: has_gpu" in manifest
        assert "nvidia.com/gpu" not in manifest
        assert run_id in manifest
        assert (
            f"touch /mnt/output/opening_strength_fit/nn/{run_id}/analysis/top1000_corrected_acceptance/_{mode.upper()}_SUCCESS"
            in manifest
        )
        assert ("--top1000-bucket-return-histogram-only" in manifest) == (mode == "hist")
