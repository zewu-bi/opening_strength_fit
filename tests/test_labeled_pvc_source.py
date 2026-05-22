from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from opening_strength_fit.training import _load_labeled_pvc_frame
from opening_strength_fit.training import _resolved_data_source


class LabeledPvcSourceTest(unittest.TestCase):
    def test_labeled_pvc_source_reads_data_labeled_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeled.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2022-01-04",
                        "symbol": "000001.SZ",
                        "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                        "label": 0.01,
                        "valid_label": True,
                    }
                ]
            ).to_parquet(path, index=False)

            args = argparse.Namespace(labeled_input=None)
            config = {
                "data": {"source": "labeled_pvc", "labeled_path": str(path)},
                "universe": {"enabled": False},
            }

            labeled = _load_labeled_pvc_frame(args, config)

        self.assertEqual(len(labeled), 1)
        self.assertIn("label", labeled.columns)
        self.assertEqual(str(labeled.loc[0, "symbol"]), "000001.SZ")

    def test_labeled_pvc_is_explicit_data_source(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "labeled_pvc"}}

        self.assertEqual(_resolved_data_source(args, config, ""), "labeled_pvc")

    def test_auto_source_prefers_data_labeled_path(self) -> None:
        args = argparse.Namespace(input=None, labeled_input=None, data_source=None)
        config = {"data": {"source": "auto", "labeled_path": "/mnt/cache/labeled.parquet"}}

        self.assertEqual(_resolved_data_source(args, config, ""), "labeled_pvc")


if __name__ == "__main__":
    unittest.main()
