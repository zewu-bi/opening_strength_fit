from __future__ import annotations

from pathlib import Path

from opening_strength_fit.commands.gap_risk_attribution import prediction_path
from opening_strength_fit.k8s import load_run_spec
from opening_strength_fit.prediction_frames import prediction_files
from opening_strength_fit.pvc_layout import (
    LEGACY_LAYOUT,
    PVC_LAYOUT_V2,
    output_layout,
    rolling_shard_dir_candidates,
    rolling_shard_dir_name,
    run_output_dir,
    run_storage_group,
    yearly_shard_dir_name,
)


def test_explicit_k8s_dir_preserves_legacy_layout() -> None:
    config = {
        "run": {"id": "legacy_run"},
        "output": {"k8s_dir": "/mnt/output/opening_strength_fit/legacy_run"},
    }

    assert output_layout(config) == LEGACY_LAYOUT
    assert run_output_dir(config, "legacy_run") == "/mnt/output/opening_strength_fit/legacy_run"
    assert rolling_shard_dir_name("2022-01", "2022-06", output_layout(config)) == ("month_2022-01")


def test_new_runs_default_to_v2_layout() -> None:
    config = {
        "run": {"id": "new_run", "kind": "exploration"},
        "model": {"name": "lightgbm"},
        "k8s": {"mount_path": "/data"},
    }

    assert output_layout(config) == PVC_LAYOUT_V2
    assert run_storage_group(config) == "models/lightgbm"
    assert run_output_dir(config, "new_run") == (
        "/data/opening_strength_fit/runs/models/lightgbm/new_run"
    )
    assert rolling_shard_dir_name("2022-01", "2022-06", output_layout(config)) == (
        "fold_2022-01_2022-06"
    )
    assert yearly_shard_dir_name(2022, output_layout(config)) == "fold_2022-01_2022-12"


def test_layout_read_candidates_include_v2_and_legacy() -> None:
    assert rolling_shard_dir_candidates(
        "2022-01",
        "2022-06",
        preferred_layout=PVC_LAYOUT_V2,
    ) == ("fold_2022-01_2022-06", "month_2022-01")


def test_load_run_spec_records_v2_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "new_run.toml"
    config_path.write_text(
        """\
[run]
id = "new_run"
kind = "exploration"

[model]
name = "torch_mlp"

[window]
test_start_month = "2022-01"
test_end_month = "2022-06"
""",
        encoding="utf-8",
    )

    spec = load_run_spec(config_path)

    assert spec.output_layout == PVC_LAYOUT_V2
    assert spec.pvc_dir == "/mnt/output/opening_strength_fit/runs/models/nn/new_run"


def test_run_storage_groups_data_and_analysis_runs() -> None:
    assert run_storage_group({"run": {"kind": "cache_transform"}}) == "data/cache-transform"
    assert run_storage_group({"run": {"kind": "capacity_audit"}}) == ("analyses/capacity-audit")
    assert run_storage_group({}) == "legacy/untracked"


def test_prediction_files_read_v2_fold_directories(tmp_path: Path) -> None:
    first = tmp_path / "fold_2022-01_2022-06"
    second = tmp_path / "fold_2022-07_2022-12"
    first.mkdir()
    second.mkdir()
    (first / "predictions.parquet").touch()
    (second / "predictions.parquet").touch()

    assert prediction_files(tmp_path) == [
        first / "predictions.parquet",
        second / "predictions.parquet",
    ]


def test_gap_attribution_reads_month_from_v2_fold(tmp_path: Path) -> None:
    expected = tmp_path / "fold_2021-08_2022-01" / "predictions.parquet"
    expected.parent.mkdir()
    expected.touch()

    assert prediction_path(tmp_path, "2021-11") == expected
