from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_feature_dependence import _feature_group_name, _feature_groups  # noqa: E402


class FeatureAuditGroupTest(unittest.TestCase):
    def test_default_groups_separate_v1_and_v2_postopen_features(self) -> None:
        groups = _feature_groups({})

        self.assertEqual(
            _feature_group_name("postopen_v2_ask_depth_3", groups),
            "postopen_v2",
        )
        self.assertEqual(
            _feature_group_name("postopen_ask_volume_1_diff_1m", groups),
            "postopen_v1",
        )
        self.assertEqual(_feature_group_name("preopen_volume", groups), "preopen")
        self.assertEqual(
            _feature_group_name("volume_diff_1t", groups),
            "trade_flow",
        )
        self.assertEqual(
            _feature_group_name("ask_volume_1", groups),
            "orderbook_depth",
        )
        self.assertEqual(_feature_group_name("return_10t", groups), "momentum")

    def test_enabled_groups_filter_defaults(self) -> None:
        groups = _feature_groups(
            {
                "feature_audit": {
                    "enabled_groups": ["preopen", "postopen_v2"],
                }
            }
        )

        self.assertEqual([group.name for group in groups], ["preopen", "postopen_v2"])


if __name__ == "__main__":
    unittest.main()
