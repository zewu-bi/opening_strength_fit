from __future__ import annotations

import unittest
from pathlib import Path

from opening_strength_fit.commands.k8s_rendering import (
    _k8s_job_name,  # noqa: E402
    render_pool_internal_analysis_job,  # noqa: E402
    render_sharded_training_job,  # noqa: E402
    render_training_job,  # noqa: E402
    resolve_render_image,  # noqa: E402
    training_command,  # noqa: E402
)
from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT, temporary_pod_name


class K8sHelperTest(unittest.TestCase):
    def test_temporary_pod_name_stays_within_kubernetes_limit(self) -> None:
        name = temporary_pod_name(
            "opening-strength-sync-artifacts",
            "lgbm_delay2_postopen_0931_0940_baseline_v1",
        )

        self.assertLessEqual(len(name), KUBERNETES_NAME_LIMIT)
        self.assertTrue(name.startswith("opening-strength-sync-artifacts-"))

    def test_rendered_job_name_stays_within_kubernetes_limit(self) -> None:
        name = _k8s_job_name(
            "opening-strength",
            "rolling_alpha_conditioned_top100_validation_v1",
            "sharded",
        )

        self.assertLessEqual(len(name), KUBERNETES_NAME_LIMIT)
        self.assertTrue(name.startswith("opening-strength-"))
        self.assertIn("-sharded-", name)

    def test_render_image_requires_explicit_immutable_reference(self) -> None:
        with self.assertRaisesRegex(SystemExit, "missing container image"):
            resolve_render_image("", allow_mutable=False)
        with self.assertRaisesRegex(SystemExit, "refusing mutable image"):
            resolve_render_image("registry/opening-strength-fit:latest", allow_mutable=False)

        self.assertEqual(
            resolve_render_image("registry/opening-strength-fit:20260715-maintenance-v1"),
            "registry/opening-strength-fit:20260715-maintenance-v1",
        )

    def test_rendered_job_name_keeps_mixed_weight_token(self) -> None:
        name = _k8s_job_name(
            "opening-strength",
            "lgbm_delay2_18m_postopen_mixed_w030_rolling_v1",
            "sharded",
            max_length=KUBERNETES_NAME_LIMIT - 2,
        )

        self.assertLessEqual(len(name), KUBERNETES_NAME_LIMIT - 2)
        self.assertIn("mixed-w030", name)
        self.assertNotIn("postopen-mi", name)

    def test_rolling_monthly_sharded_job_uses_indexed_pods(self) -> None:
        config = {
            "run": {
                "id": "rolling_alpha_conditioned_top100_validation_v1",
                "kind": "alpha_conditioned_rolling_validation",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 12,
                "test_start_month": "2021-08",
                "test_end_month": "2021-10",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/run"},
            "k8s": {
                "backoff_limit_per_index": 0,
                "resources": {"memory_limit": "256Gi"},
            },
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/rolling.toml"),
            config,
            "image:tag",
        )

        self.assertIn("completionMode: Indexed", manifest)
        self.assertIn("completions: 3", manifest)
        self.assertIn("backoffLimitPerIndex: 0", manifest)
        self.assertNotIn("backoffLimit: 0", manifest)
        self.assertIn("JOB_COMPLETION_INDEX", manifest)
        self.assertIn("TEST_STARTS=(2021-08 2021-09 2021-10)", manifest)
        self.assertIn("TEST_ENDS=(2021-08 2021-09 2021-10)", manifest)
        self.assertNotIn("for MONTH in", manifest)

    def test_rolling_halfyear_sharded_job_uses_window_starts(self) -> None:
        config = {
            "run": {
                "id": "lgbm_delay2_36m_halfyear_rolling_v1",
                "kind": "exploration",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2018-01",
                "test_end_month": "2024-12",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/run"},
            "k8s": {
                "job_name": "os-lgbm-36m-2018-2024-w030-halfyear",
                "resources": {"memory_limit": "512Gi"},
            },
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/rolling.toml"),
            config,
            "image:tag",
        )

        self.assertIn("completions: 14", manifest)
        self.assertIn("TEST_STARTS=(2018-01 2018-07", manifest)
        self.assertIn("2024-01 2024-07)", manifest)
        self.assertIn("TEST_ENDS=(2018-06 2018-12", manifest)
        self.assertIn("2024-06 2024-12)", manifest)
        self.assertIn('--test-start-month "${TEST_START}"', manifest)
        self.assertIn('--test-end-month "${TEST_END}"', manifest)

    def test_rolling_monthly_can_render_independent_shard_jobs(self) -> None:
        config = {
            "run": {"id": "nn_independent_shards", "kind": "exploration"},
            "model": {"name": "torch_mlp"},
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-12",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/nn/run"},
            "k8s": {
                "job_name": "os-nn-independent",
                "shard_job_mode": "separate",
                "config_map_name": "os-nn-independent-config",
                "config_map_volume_name": "run-config",
                "config_map_mount_path": "/app/opening_strength_fit/experiments/runs/run.toml",
                "config_map_sub_path": "run.toml",
                "resources": {"memory_limit": "896Gi"},
            },
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/run.toml"),
            config,
            "image:tag",
        )

        self.assertEqual(manifest.count("kind: Job"), 2)
        self.assertIn("name: os-nn-independent-s0", manifest)
        self.assertIn("name: os-nn-independent-s1", manifest)
        self.assertNotIn("completionMode: Indexed", manifest)
        self.assertNotIn("JOB_COMPLETION_INDEX", manifest)
        self.assertIn("TEST_START=2022-01", manifest)
        self.assertIn("TEST_START=2022-07", manifest)
        self.assertIn('OUT="${ROOT}/month_2022-01"', manifest)
        self.assertIn('OUT="${ROOT}/month_2022-07"', manifest)
        self.assertEqual(manifest.count("name: os-nn-independent-config"), 2)
        self.assertEqual(manifest.count("subPath: run.toml"), 2)

    def test_v2_sharded_job_uses_run_and_fold_directories(self) -> None:
        config = {
            "run": {"id": "new_halfyear_run", "kind": "exploration"},
            "model": {"name": "lightgbm"},
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-12",
            },
            "output": {"layout": "v2"},
            "k8s": {"resources": {"memory_limit": "128Gi"}},
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/new_halfyear_run.toml"),
            config,
            "image:tag",
        )

        self.assertIn(
            "ROOT=/mnt/output/opening_strength_fit/runs/models/lightgbm/new_halfyear_run",
            manifest,
        )
        self.assertIn('OUT="${ROOT}/fold_${TEST_START}_${TEST_END}"', manifest)
        self.assertNotIn('OUT="${ROOT}/month_${TEST_START}"', manifest)

    def test_osf_train_shards_require_predictions_before_skipping(self) -> None:
        config = {
            "run": {
                "id": "lgbm_delay2_36m_halfyear_rolling_v1",
                "kind": "exploration",
            },
            "model": {"name": "lightgbm"},
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2025-01",
                "test_end_month": "2025-12",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/run"},
            "k8s": {"resources": {"memory_limit": "512Gi"}},
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/rolling.toml"),
            config,
            "image:tag",
        )

        self.assertIn('[ -f "${OUT}/_SUCCESS" ]', manifest)
        self.assertIn('[ -f "${OUT}/metrics_by_year.csv" ]', manifest)
        self.assertIn('[ -f "${OUT}/predictions.parquet" ]', manifest)
        self.assertIn("required outputs already exist", manifest)

    def test_sharded_job_respects_explicit_short_job_name(self) -> None:
        config = {
            "run": {
                "id": "lgbm_delay2_36m_2023_mixed_w030_soft_core_reg_light_rolling_v1",
                "kind": "exploration",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_start_month": "2023-01",
                "test_end_month": "2023-12",
            },
            "output": {
                "k8s_dir": "/mnt/output/opening_strength_fit/os-lgbm-36m-2023-w030-baseline"
            },
            "k8s": {
                "job_name": "os-lgbm-36m-2023-w030-baseline",
                "resources": {"memory_limit": "512Gi"},
            },
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/os-lgbm-36m-2023-w030-baseline.toml"),
            config,
            "image:tag",
        )

        self.assertIn("name: os-lgbm-36m-2023-w030-baseline", manifest)
        self.assertNotIn("sharded-", manifest)

    def test_next_close_label_cache_uses_cache_builder_script(self) -> None:
        config = {"run": {"kind": "next_close_label_cache"}}

        self.assertEqual(training_command(config), "osf-build-next-close-labels")

    def test_feature_hygiene_uses_hygiene_audit_script(self) -> None:
        config = {
            "run": {
                "id": "feature_hygiene_v1",
                "kind": "feature_hygiene",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-06",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/feature_hygiene_v1"},
            "k8s": {"resources": {"memory_limit": "128Gi"}},
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/feature_hygiene_v1.toml"),
            config,
            "image:tag",
        )

        self.assertEqual(training_command(config), "osf-audit-feature-hygiene")
        self.assertIn('[ -f "${OUT}/feature_hygiene_trace.json" ]', manifest)
        self.assertIn('[ -f "${OUT}/feature_hygiene.csv" ]', manifest)

    def test_training_job_waits_for_configured_paths(self) -> None:
        config = {
            "run": {
                "id": "feature_hygiene_v1",
                "kind": "feature_hygiene",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/feature_hygiene_v1"},
            "k8s": {
                "wait_for_paths": ["/mnt/output/source/feature_importance.csv"],
                "resources": {"memory_limit": "128Gi"},
            },
        }

        manifest = render_training_job(
            Path("experiments/runs/feature_hygiene_v1.toml"),
            config,
            "image:tag",
        )

        self.assertIn('WAIT_PATHS=("/mnt/output/source/feature_importance.csv")', manifest)
        self.assertIn("exec osf-audit-feature-hygiene", manifest)

    def test_ensemble_model_uses_standard_training_script(self) -> None:
        config = {"run": {"kind": "exploration"}, "model": {"name": "ensemble"}}

        self.assertEqual(training_command(config), "osf-train")

    def test_torch_mlp_model_uses_standard_training_script_and_gpu_resource(self) -> None:
        config = {
            "run": {"id": "nn_smoke", "kind": "exploration"},
            "model": {"name": "torch_mlp"},
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/nn/nn_smoke"},
            "k8s": {
                "resources": {"gpu_limit": 1, "memory_limit": "128Gi"},
                "node_selector": {"has_gpu": "true"},
            },
        }

        manifest = render_training_job(
            Path("experiments/runs/nn_smoke.toml"),
            config,
            "image:tag",
        )

        self.assertEqual(training_command(config), "osf-train")
        self.assertIn('nvidia.com/gpu: "1"', manifest)
        self.assertIn('has_gpu: "true"', manifest)
        self.assertIn("exec osf-train", manifest)

    def test_pool_internal_analysis_job_uses_cluster_artifacts(self) -> None:
        config = {
            "run": {
                "id": "lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1",
                "kind": "exploration",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-12",
            },
            "evaluation": {"top_n": 100},
            "output": {
                "k8s_dir": "/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1"
            },
            "k8s": {
                "namespace": "bizewu",
                "pvc": "bizewu-private-data",
                "clickhouse_secret": "opening-strength-clickhouse",
            },
            "analysis": {
                "pool_internal": {
                    "enabled": True,
                    "job_name": "os-analyze-36m-2225-regmid",
                    "variant": "reg_mid",
                    "env_secrets": ["xy-fit-ceph-credentials"],
                    "pools": ["universe", "L"],
                    "resources": {"memory_limit": "384Gi"},
                }
            },
        }

        manifest = render_pool_internal_analysis_job(
            Path("experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1.toml"),
            config,
            "image:tag",
        )

        self.assertIn("name: os-analyze-36m-2225-regmid", manifest)
        self.assertIn("name: opening-strength-clickhouse", manifest)
        self.assertIn("name: xy-fit-ceph-credentials", manifest)
        self.assertIn(
            "/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1/month_2022-01/metrics_by_year.csv",
            manifest,
        )
        self.assertIn(
            "/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1/month_2022-07/metrics_by_year.csv",
            manifest,
        )
        self.assertIn("--predictions \\", manifest)
        self.assertIn("--pool \\\n                universe", manifest)
        self.assertIn("--pool \\\n                L", manifest)
        self.assertIn("memory: 384Gi", manifest)

    def test_pool_internal_analysis_does_not_inherit_training_gpu_selector(self) -> None:
        config = {
            "run": {
                "id": "nn_mlp_base",
                "kind": "exploration",
            },
            "window": {
                "mode": "rolling_monthly",
                "train_months": 36,
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-06",
            },
            "output": {"k8s_dir": "/mnt/output/opening_strength_fit/nn/nn_mlp_base"},
            "k8s": {
                "namespace": "bizewu",
                "pvc": "bizewu-private-data",
                "clickhouse_secret": "opening-strength-clickhouse",
                "node_selector": {"has_gpu": "true", "mem_per_gpu_tier": "high"},
                "avoid_nodes": ["node20"],
            },
            "analysis": {
                "pool_internal": {
                    "enabled": True,
                    "job_name": "os-analyze-nn-base",
                    "variant": "nn_base",
                    "env_secrets": ["xy-fit-ceph-credentials"],
                    "pools": ["universe", "L"],
                    "resources": {"memory_limit": "384Gi"},
                }
            },
        }

        manifest = render_pool_internal_analysis_job(
            Path("experiments/runs/nn_mlp_base.toml"),
            config,
            "image:tag",
        )

        self.assertNotIn('has_gpu: "true"', manifest)
        self.assertNotIn('mem_per_gpu_tier: "high"', manifest)
        self.assertIn("- node20", manifest)

    def test_v2_pool_internal_analysis_waits_for_fold_outputs(self) -> None:
        config = {
            "run": {"id": "new_nn_run", "kind": "exploration"},
            "window": {
                "mode": "rolling_monthly",
                "test_months": 6,
                "test_stride_months": 6,
                "test_start_month": "2022-01",
                "test_end_month": "2022-12",
            },
            "output": {"layout": "v2"},
            "analysis": {"pool_internal": {"enabled": True}},
        }

        manifest = render_pool_internal_analysis_job(
            Path("experiments/runs/new_nn_run.toml"),
            config,
            "image:tag",
        )

        root = "/mnt/output/opening_strength_fit/runs/models/other/new_nn_run"
        self.assertIn(f"{root}/fold_2022-01_2022-06/metrics_by_year.csv", manifest)
        self.assertIn(f"{root}/fold_2022-07_2022-12/metrics_by_year.csv", manifest)
        self.assertIn(f"--predictions \\\n                {root}", manifest)


if __name__ == "__main__":
    unittest.main()
