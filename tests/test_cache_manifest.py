from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opening_strength_fit.cache_manifest import (
    build_cache_manifest,
    write_cache_manifest,
)


class CacheManifestTest(unittest.TestCase):
    def test_build_cache_manifest_is_json_safe(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2022-01-04",
                    "symbol": "000001.SZ",
                    "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                    "decision_time": "09:30:00",
                    "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                    "decision_lag_seconds": 0.0,
                    "label": 0.01,
                    "valid_label": True,
                },
                {
                    "date": "2022-01-04",
                    "symbol": "000002.SZ",
                    "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                    "decision_time": "09:31:00",
                    "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                    "decision_lag_seconds": 1.0,
                    "label": None,
                    "valid_label": False,
                },
            ]
        )

        manifest = build_cache_manifest(
            frame,
            cache_path="/tmp/cache.parquet",
            config={"cache": {"schema_version": "test_v1"}},
            run_name="cache_test",
        )

        self.assertEqual(manifest["cache_schema_version"], "test_v1")
        self.assertEqual(manifest["summary"]["rows"], 2)
        self.assertEqual(manifest["decision_time_counts"]["09:30:00"], 1)
        self.assertEqual(manifest["label_summary"]["valid_labels"], 1)
        self.assertEqual(manifest["required_columns"]["missing"], [])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_cache_manifest(manifest, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_id"], "cache_test")


if __name__ == "__main__":
    unittest.main()
