from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from opening_strength_fit.commands.artifact_sync import (
    combine_metric_frames,  # noqa: E402
    combine_rolling_validation_shards,  # noqa: E402
    pull_gap_attribution_artifacts,  # noqa: E402
    pull_next_close_labels,  # noqa: E402
    pull_pool_internal_analysis_artifacts,  # noqa: E402
    pull_rolling_validation_shards,  # noqa: E402
    pull_score_risk_artifacts,  # noqa: E402
    record_lightweight_artifacts,  # noqa: E402
    record_metrics,  # noqa: E402
)
from opening_strength_fit.commands.artifact_sync_metrics import next_close_label_years  # noqa: E402
from opening_strength_fit.k8s import RunSpec  # noqa: E402


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

    def test_record_metrics_archives_year_and_month_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "metrics"
            metrics_dir.mkdir()
            (metrics_dir / "test_run_metrics_by_year.csv").write_text("year\n", encoding="utf-8")
            (metrics_dir / "test_run_metrics_by_month.csv").write_text("month\n", encoding="utf-8")

            paths = record_metrics("test_run", metrics_dir, root / "results")

        self.assertEqual(
            [path.name for path in paths],
            ["test_run_metrics_by_year.csv", "test_run_metrics_by_month.csv"],
        )

    def test_record_metrics_accepts_results_metrics_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "results" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "test_run_metrics_by_year.csv").write_text("year\n", encoding="utf-8")

            paths = record_metrics("test_run", metrics_dir, root / "results")

        self.assertEqual(paths, [metrics_dir / "test_run_metrics_by_year.csv"])

    def test_next_close_label_years_use_halfyear_window_range(self) -> None:
        spec = RunSpec(
            run_id="halfyear",
            pvc_dir="/mnt/output/opening_strength_fit/halfyear",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            test_start_month="2024-07",
            test_end_month="2025-12",
        )

        self.assertEqual(next_close_label_years(spec), [2024, 2025])

    def test_pull_next_close_labels_uses_standard_local_directory(self) -> None:
        spec = RunSpec(
            run_id="halfyear",
            pvc_dir="/mnt/output/opening_strength_fit/halfyear",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            test_start_month="2025-01",
            test_end_month="2025-12",
        )

        fetched_remote_paths = []

        def fake_fetch(_hfcli, _spec, _pod, remote_path, local_path):
            fetched_remote_paths.append(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("label\n", encoding="utf-8")
            return True

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "opening_strength_fit.commands.artifact_sync_metrics.fetch_remote_file_if_exists",
                side_effect=fake_fetch,
            ):
                paths = pull_next_close_labels(
                    "hfcli",
                    spec,
                    "helper-pod",
                    Path(directory),
                    label_pvc_dir="/labels",
                )

        self.assertEqual(
            [path.as_posix() for path in paths],
            [f"{directory}/next_close_labels_2025/opening_2025_next_close_labels_v1.parquet"],
        )
        self.assertEqual(
            fetched_remote_paths,
            ["/labels/opening_2025_next_close_labels_v1.parquet"],
        )

    def test_score_risk_artifacts_are_pulled_to_local_run_dir(self) -> None:
        spec = RunSpec(
            run_id="score_learned_risk_sweep_v1",
            pvc_dir="/mnt/output/opening_strength_fit/score_learned_risk_sweep_v1",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            kind="score_risk_sweep",
        )

        fetched_remote_paths = []

        def fake_fetch(_hfcli, _spec, _pod, remote_path, local_path):
            fetched_remote_paths.append(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("artifact\n", encoding="utf-8")
            return True

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "opening_strength_fit.commands.artifact_sync_artifacts.fetch_remote_file_if_exists",
                side_effect=fake_fetch,
            ):
                paths = pull_score_risk_artifacts(
                    "hfcli",
                    spec,
                    "helper-pod",
                    Path(directory),
                )
            output_dir = Path(directory) / spec.run_id

            self.assertIn(output_dir / "score_risk_summary.csv", paths)
            self.assertTrue((output_dir / "score_risk_trace.json").exists())
            self.assertTrue((output_dir / "artifact_fetch_trace.json").exists())
            self.assertFalse((output_dir / "clickhouse_next_close_labels.parquet").exists())
            self.assertFalse(
                any("clickhouse_next_close_labels.parquet" in path for path in fetched_remote_paths)
            )

    def test_lightweight_artifacts_record_summaries_only(self) -> None:
        spec = RunSpec(
            run_id="rolling_alpha_conditioned_top100_validation_v1",
            pvc_dir="/mnt/output/opening_strength_fit/rolling_alpha_conditioned_top100_validation_v1",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=2021,
            test_end_year=2021,
            test_start_month="2021-08",
            test_end_month="2021-09",
            kind="alpha_conditioned_rolling_validation",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "artifacts" / spec.run_id
            output_dir.mkdir(parents=True)
            (output_dir / "rolling_summary.csv").write_text(
                "summary\n",
                encoding="utf-8",
            )
            (output_dir / "rolling_month_summary.csv").write_text(
                "month\n",
                encoding="utf-8",
            )
            (output_dir / "rolling_trace.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "rolling_group_metrics.csv").write_text(
                "group\n",
                encoding="utf-8",
            )

            paths = record_lightweight_artifacts(
                spec,
                root / "artifacts",
                root / "results",
            )

        self.assertEqual(
            [path.relative_to(Path(directory) / "results").as_posix() for path in paths],
            [
                "backtests/rolling_alpha_conditioned_top100_validation_v1/summary.csv",
                "backtests/rolling_alpha_conditioned_top100_validation_v1/month_summary.csv",
                "backtests/rolling_alpha_conditioned_top100_validation_v1/trace.json",
            ],
        )

    def test_pool_internal_analysis_artifacts_pull_and_record(self) -> None:
        spec = RunSpec(
            run_id="lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1",
            pvc_dir="/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_reg_mid_v1",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            test_start_month="2022-01",
            test_end_month="2025-12",
            kind="exploration",
            pool_internal_analysis_enabled=True,
            pool_internal_analysis_dir="/mnt/output/opening_strength_fit/run/analysis/pool_internal_top100",
            pool_internal_record_prefix="reg_mid_record",
        )

        def fake_fetch(_hfcli, _spec, _pod, _remote_dir, local_dir):
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "pool_internal_summary.csv").write_text(
                "pool,short_internal_excess_bps\npool_L,1.0\n",
                encoding="utf-8",
            )
            (local_dir / "pool_internal_month_summary.csv").write_text(
                "pool,test_month\npool_L,2022-01\n",
                encoding="utf-8",
            )
            (local_dir / "pool_internal_clock_summary.csv").write_text(
                "pool,clock\npool_L,09:31\n",
                encoding="utf-8",
            )
            (local_dir / "pool_internal_group_metrics.csv").write_text(
                "pool,date\npool_L,2022-01-04\n",
                encoding="utf-8",
            )
            (local_dir / "pool_internal_halfyear_summary.csv").write_text(
                "pool,year,half\npool_L,2022,H1\n",
                encoding="utf-8",
            )
            (local_dir / "pool_internal_year_summary.csv").write_text(
                "pool,year\npool_L,2022\n",
                encoding="utf-8",
            )
            report = (
                local_dir
                / "reports"
                / "reg_mid_universe_sml_pool_internal_with_mean"
                / "reg_mid_universe_sml_top100_pool_internal_with_mean.svg"
            )
            report.parent.mkdir(parents=True)
            report.write_text("<svg />\n", encoding="utf-8")
            plot_data = report.with_name(
                "reg_mid_universe_sml_pool_internal_with_mean_plot_data.csv"
            )
            plot_data.write_text("test_month,pool\nMean,pool_L\n", encoding="utf-8")
            trace = {
                "report_plots": {
                    "pool_internal_figure": (
                        "/mnt/output/opening_strength_fit/run/analysis/pool_internal_top100/"
                        "reports/reg_mid_universe_sml_pool_internal_with_mean/"
                        "reg_mid_universe_sml_top100_pool_internal_with_mean.svg"
                    ),
                    "pool_internal_plot_data": (
                        "/mnt/output/opening_strength_fit/run/analysis/pool_internal_top100/"
                        "reports/reg_mid_universe_sml_pool_internal_with_mean/"
                        "reg_mid_universe_sml_pool_internal_with_mean_plot_data.csv"
                    ),
                }
            }
            (local_dir / "pool_internal_trace.json").write_text(
                json.dumps(trace),
                encoding="utf-8",
            )
            return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "opening_strength_fit.commands.artifact_sync_artifacts.fetch_remote_directory_if_exists",
                side_effect=fake_fetch,
            ):
                pulled = pull_pool_internal_analysis_artifacts(
                    "hfcli",
                    spec,
                    "helper-pod",
                    root / "artifacts",
                )
            recorded = record_lightweight_artifacts(
                spec,
                root / "artifacts",
                root / "results",
            )

        self.assertIn(root / "artifacts" / spec.run_id / "pool_internal_summary.csv", pulled)
        recorded_rel = [path.relative_to(root / "results").as_posix() for path in recorded]
        self.assertIn(
            "backtests/reg_mid_record/pool_internal_summary.csv",
            recorded_rel,
        )
        self.assertIn(
            "backtests/reg_mid_record/pool_internal_with_mean.svg",
            recorded_rel,
        )

    def test_presentation_core_archive_profile_prunes_reports_from_config(self) -> None:
        spec = RunSpec(
            run_id="presentation_run",
            pvc_dir="/mnt/output/opening_strength_fit/presentation_run",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            kind="pool_internal_analysis",
            pool_internal_analysis_enabled=True,
            pool_internal_analysis_dir="/mnt/output/opening_strength_fit/presentation_run/analysis",
            pool_internal_archive_profile="presentation_core",
        )

        def fake_fetch(_hfcli, _spec, _pod, _remote_dir, local_dir):
            local_dir.mkdir(parents=True)
            keep = local_dir / "reports" / "variant_universe_pool_l_short_excess_rank_ic_with_mean"
            stale = local_dir / "reports" / "variant_universe_sml_short_excess_rank_ic_with_mean"
            drop = local_dir / "reports" / "variant_rank_ic_with_mean"
            keep.mkdir(parents=True)
            stale.mkdir(parents=True)
            drop.mkdir(parents=True)
            (keep / "keep.svg").write_text("<svg />\n", encoding="utf-8")
            (stale / "stale.svg").write_text("<svg />\n", encoding="utf-8")
            (drop / "drop.svg").write_text("<svg />\n", encoding="utf-8")
            (local_dir / "pool_internal_trace.json").write_text(
                json.dumps(
                    {
                        "report_plots": {
                            "short_excess_rank_ic_figure": (
                                "/mnt/output/opening_strength_fit/presentation_run/analysis/"
                                "reports/variant_universe_pool_l_short_excess_rank_ic_with_mean/"
                                "keep.svg"
                            ),
                            "rank_ic_figure": (
                                "/mnt/output/opening_strength_fit/presentation_run/analysis/"
                                "reports/variant_rank_ic_with_mean/drop.svg"
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "opening_strength_fit.commands.artifact_sync_artifacts.fetch_remote_directory_if_exists",
                side_effect=fake_fetch,
            ):
                pull_pool_internal_analysis_artifacts(
                    "hfcli",
                    spec,
                    "helper-pod",
                    root,
                )
            output_dir = root / spec.run_id
            trace = json.loads((output_dir / "pool_internal_trace.json").read_text())

            self.assertTrue(
                (
                    output_dir
                    / "reports"
                    / "variant_universe_pool_l_short_excess_rank_ic_with_mean"
                ).exists()
            )
            self.assertFalse(
                (
                    output_dir / "reports" / "variant_universe_sml_short_excess_rank_ic_with_mean"
                ).exists()
            )
            self.assertFalse((output_dir / "reports" / "variant_rank_ic_with_mean").exists())
            self.assertEqual(
                set(trace["report_plots"]),
                {"short_excess_rank_ic_figure"},
            )

    def test_rolling_validation_shards_are_combined_locally(self) -> None:
        rows = [
            {
                "test_month": "2021-08",
                "variant": "alpha_rank",
                "risk_model": "",
                "penalty": 0.0,
                "candidate_alpha_rank_min": 0.0,
                "date": "2021-08-02",
                "clock": "09:31",
                "candidate_rows": 100,
                "selected_rows": 100,
                "short_top_mean_bps": 1.0,
                "short_top_excess_bps": 2.0,
                "next_top_mean_bps": -1.0,
                "next_top_excess_bps": -2.0,
                "selected_gap_risk_rank": 0.1,
                "selected_binary_risk_rank": 0.2,
            },
            {
                "test_month": "2021-09",
                "variant": "alpha_rank",
                "risk_model": "",
                "penalty": 0.0,
                "candidate_alpha_rank_min": 0.0,
                "date": "2021-09-01",
                "clock": "09:31",
                "candidate_rows": 100,
                "selected_rows": 100,
                "short_top_mean_bps": 3.0,
                "short_top_excess_bps": 4.0,
                "next_top_mean_bps": 5.0,
                "next_top_excess_bps": 6.0,
                "selected_gap_risk_rank": 0.3,
                "selected_binary_risk_rank": 0.4,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in rows:
                shard_dir = root / f"month_{row['test_month']}"
                shard_dir.mkdir(parents=True)
                pd.DataFrame([row]).to_csv(
                    shard_dir / "rolling_group_metrics.csv",
                    index=False,
                )

            paths = combine_rolling_validation_shards(
                root,
                months=["2021-08", "2021-09"],
                missing_months=[],
            )
            summary = pd.read_csv(root / "rolling_summary.csv")

        self.assertIn(root / "rolling_summary.csv", paths)
        self.assertEqual(int(summary.loc[0, "months"]), 2)
        self.assertAlmostEqual(float(summary.loc[0, "short_top_excess_bps"]), 3.0)
        self.assertAlmostEqual(float(summary.loc[0, "next_top_excess_bps"]), 2.0)

    def test_rolling_validation_artifacts_can_be_pulled_from_month_shards(self) -> None:
        spec = RunSpec(
            run_id="rolling_alpha_conditioned_top100_validation_v1",
            pvc_dir="/mnt/output/opening_strength_fit/rolling_alpha_conditioned_top100_validation_v1",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=2021,
            test_end_year=2021,
            test_start_month="2021-08",
            test_end_month="2021-08",
            kind="alpha_conditioned_rolling_validation",
        )
        row = {
            "test_month": "2021-08",
            "variant": "alpha_rank",
            "risk_model": "",
            "penalty": 0.0,
            "candidate_alpha_rank_min": 0.0,
            "date": "2021-08-02",
            "clock": "09:31",
            "candidate_rows": 100,
            "selected_rows": 100,
            "short_top_mean_bps": 1.0,
            "short_top_excess_bps": 2.0,
            "next_top_mean_bps": -1.0,
            "next_top_excess_bps": -2.0,
            "selected_gap_risk_rank": 0.1,
            "selected_binary_risk_rank": 0.2,
        }

        def fake_fetch(_hfcli, _spec, _pod, remote_path, local_path):
            if not remote_path.endswith("rolling_group_metrics.csv"):
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(local_path, index=False)
            return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "opening_strength_fit.commands.artifact_sync_artifacts.fetch_remote_file_if_exists",
                side_effect=fake_fetch,
            ):
                paths = pull_rolling_validation_shards(
                    "hfcli",
                    spec,
                    "helper-pod",
                    root,
                )

            self.assertTrue((root / "rolling_summary.csv").exists())
            self.assertIn(root / "month_2021-08" / "rolling_group_metrics.csv", paths)

    def test_gap_attribution_artifacts_are_pulled_without_group_metrics(self) -> None:
        spec = RunSpec(
            run_id="gap_risk_penalized_attribution_v1",
            pvc_dir="/mnt/output/opening_strength_fit/gap_risk_penalized_attribution_v1",
            namespace="bizewu",
            pvc="bizewu-private-data",
            mount_path="/mnt/output",
            pull_secret="highfort",
            image="image",
            test_start_year=0,
            test_end_year=0,
            kind="gap_risk_attribution",
        )
        fetched_remote_paths = []

        def fake_fetch(_hfcli, _spec, _pod, remote_path, local_path):
            fetched_remote_paths.append(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("artifact\n", encoding="utf-8")
            return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "opening_strength_fit.commands.artifact_sync_artifacts.fetch_remote_file_if_exists",
                side_effect=fake_fetch,
            ):
                paths = pull_gap_attribution_artifacts(
                    "hfcli",
                    spec,
                    "helper-pod",
                    root,
                )

            self.assertIn(
                root / spec.run_id / "gap_attribution_outcomes_overall.csv",
                paths,
            )
        self.assertFalse(
            any("gap_attribution_group_metrics.csv" in path for path in fetched_remote_paths)
        )


if __name__ == "__main__":
    unittest.main()
