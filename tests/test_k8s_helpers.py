from __future__ import annotations

import unittest
from pathlib import Path

from opening_strength_fit.commands.k8s_rendering import (
    _k8s_job_name,  # noqa: E402
    render_pool_internal_analysis_job,  # noqa: E402
    render_sharded_training_job,  # noqa: E402
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
            "k8s": {"resources": {"memory_limit": "256Gi"}},
        }

        manifest = render_sharded_training_job(
            Path("experiments/runs/rolling.toml"),
            config,
            "image:tag",
        )

        self.assertIn("completionMode: Indexed", manifest)
        self.assertIn("completions: 3", manifest)
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

    def test_ensemble_model_uses_standard_training_script(self) -> None:
        config = {"run": {"kind": "exploration"}, "model": {"name": "ensemble"}}

        self.assertEqual(training_command(config), "osf-train")

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


if __name__ == "__main__":
    unittest.main()
