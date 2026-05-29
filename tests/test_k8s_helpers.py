from __future__ import annotations

from pathlib import Path
import sys
import unittest

from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT, temporary_pod_name

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from render_k8s_job import _k8s_job_name  # noqa: E402
from render_k8s_job import render_sharded_training_job  # noqa: E402


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
        self.assertIn("MONTHS=(2021-08 2021-09 2021-10)", manifest)
        self.assertNotIn("for MONTH in", manifest)


if __name__ == "__main__":
    unittest.main()
