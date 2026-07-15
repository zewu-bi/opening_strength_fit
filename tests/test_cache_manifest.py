from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opening_strength_fit.cache_manifest import (
    build_cache_manifest,
    cache_manifest_path,
    validate_cache_manifest,
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

    def test_manifest_fingerprint_ignores_model_and_output_sections(self) -> None:
        frame = self._labeled_frame()
        build_config = {
            "sample": {"decision_times": ["09:31:00"]},
            "model": {"name": "lightgbm"},
            "output": {"local_dir": "one"},
        }
        read_config = {
            "sample": {"decision_times": ["09:31:00"]},
            "model": {"name": "torch_mlp"},
            "output": {"local_dir": "two"},
        }

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.parquet"
            frame.to_parquet(cache_path, index=False)
            manifest = build_cache_manifest(
                frame,
                cache_path=cache_path,
                config=build_config,
                run_name="cache_test",
            )
            write_cache_manifest(manifest, cache_manifest_path(cache_path))

            validated = validate_cache_manifest(cache_path, read_config)

        self.assertEqual(validated["run_id"], "cache_test")

    def test_manifest_rejects_changed_cache_building_config(self) -> None:
        frame = self._labeled_frame()
        build_config = {"sample": {"decision_times": ["09:31:00"]}}
        read_config = {"sample": {"decision_times": ["09:32:00"]}}

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.parquet"
            frame.to_parquet(cache_path, index=False)
            manifest = build_cache_manifest(
                frame,
                cache_path=cache_path,
                config=build_config,
                run_name="cache_test",
            )
            write_cache_manifest(manifest, cache_manifest_path(cache_path))

            with self.assertRaisesRegex(SystemExit, "config fingerprint does not match"):
                validate_cache_manifest(cache_path, read_config)

    def test_manifest_rejects_cache_file_schema_mismatch(self) -> None:
        frame = self._labeled_frame()

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.parquet"
            frame.to_parquet(cache_path, index=False)
            manifest = build_cache_manifest(
                frame,
                cache_path=cache_path,
                config={},
                run_name="cache_test",
            )
            write_cache_manifest(manifest, cache_manifest_path(cache_path))
            frame.drop(columns=["label"]).to_parquet(cache_path, index=False)

            with self.assertRaisesRegex(SystemExit, "manifest schema columns"):
                validate_cache_manifest(cache_path, {})

    def test_manifest_can_be_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.parquet"
            cache_path.touch()

            with self.assertRaisesRegex(SystemExit, "manifest does not exist"):
                validate_cache_manifest(cache_path, {}, required=True)

    @staticmethod
    def _labeled_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2022-01-04"],
                "symbol": ["000001.SZ"],
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")],
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")],
                "label": [0.01],
                "valid_label": [True],
            }
        )


if __name__ == "__main__":
    unittest.main()
