from __future__ import annotations

import unittest

from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT, temporary_pod_name


class K8sHelperTest(unittest.TestCase):
    def test_temporary_pod_name_stays_within_kubernetes_limit(self) -> None:
        name = temporary_pod_name(
            "opening-strength-sync-artifacts",
            "lgbm_delay2_postopen_0931_0940_baseline_v1",
        )

        self.assertLessEqual(len(name), KUBERNETES_NAME_LIMIT)
        self.assertTrue(name.startswith("opening-strength-sync-artifacts-"))


if __name__ == "__main__":
    unittest.main()
