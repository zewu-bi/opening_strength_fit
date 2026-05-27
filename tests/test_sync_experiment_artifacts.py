from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sync_experiment_artifacts import combine_metric_frames  # noqa: E402


def _metric_row(month: str, rows: int, top_return: float) -> dict[str, object]:
    return {
        "run_id": "test_run",
        "test_year": int(month[:4]),
        "test_month": month,
        "train_start_date": "2021-01-01",
        "train_end_date": "2021-12-31",
        "test_start_date": f"{month}-01",
        "test_end_date": f"{month}-28",
        "test_rows": rows,
        "top_score_mean_return": top_return,
    }


class SyncExperimentArtifactsTest(unittest.TestCase):
    def test_yearly_shard_metrics_are_combined_locally(self) -> None:
        frames = [
            pd.DataFrame([_metric_row("2021-01", 10, 0.01)]),
            pd.DataFrame([_metric_row("2022-01", 20, 0.02)]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = combine_metric_frames(
                frames,
                monthly=False,
                run_id="test_run",
                output_dir=Path(directory),
            )
            combined = pd.read_csv(Path(directory) / "test_run_metrics_by_year.csv")

        self.assertEqual([path.name for path in paths], ["test_run_metrics_by_year.csv"])
        self.assertEqual(combined["test_year"].tolist(), [2021, 2022])

    def test_monthly_shard_metrics_write_month_and_year_outputs(self) -> None:
        frames = [
            pd.DataFrame([_metric_row("2022-01", 1, 0.01)]),
            pd.DataFrame([_metric_row("2022-02", 3, 0.05)]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = combine_metric_frames(
                frames,
                monthly=True,
                run_id="test_run",
                output_dir=Path(directory),
            )
            monthly = pd.read_csv(Path(directory) / "test_run_metrics_by_month.csv")
            yearly = pd.read_csv(Path(directory) / "test_run_metrics_by_year.csv")

        self.assertEqual(
            [path.name for path in paths],
            ["test_run_metrics_by_year.csv", "test_run_metrics_by_month.csv"],
        )
        self.assertEqual(monthly["test_month"].tolist(), ["2022-01", "2022-02"])
        self.assertEqual(int(yearly.loc[0, "test_rows"]), 4)
        self.assertAlmostEqual(float(yearly.loc[0, "top_score_mean_return"]), 0.04)


if __name__ == "__main__":
    unittest.main()
