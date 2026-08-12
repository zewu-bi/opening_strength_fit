from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from opening_strength_fit.commands.feature_hygiene_audit import (
    _file_overlaps_date_range,
    _sample_labeled_pvc_frame,
    build_prune_report,
    feature_correlation_pairs,
    load_feature_importance,
    main,
    summarize_feature_hygiene,
)


def test_feature_hygiene_flags_constant_and_near_constant_features() -> None:
    frame = pd.DataFrame(
        {
            "active": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "near_constant": [0.0, 0.0, 0.0, 1.0],
        }
    )
    features = ["active", "constant", "near_constant"]

    hygiene = summarize_feature_hygiene(
        frame,
        features,
        group_by_feature={feature: "g" for feature in features},
        near_constant_top_ratio=0.75,
    ).set_index("feature")

    assert bool(hygiene.loc["active", "constant"]) is False
    assert bool(hygiene.loc["constant", "constant"]) is True
    assert bool(hygiene.loc["near_constant", "near_constant"]) is True
    assert hygiene.loc["near_constant", "top_frequency_ratio"] == 0.75


def test_correlation_pairs_default_to_same_group_only() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "cross": [1.0, 2.0, 3.0, 4.0],
        }
    )
    group_by_feature = {"a": "same", "b": "same", "cross": "other"}

    pairs, _ = feature_correlation_pairs(
        frame,
        ["a", "b", "cross"],
        group_by_feature=group_by_feature,
        method="spearman",
        threshold=0.99,
        same_group_only=True,
        min_periods=2,
    )

    assert pairs[["feature_a", "feature_b"]].to_dict("records") == [
        {"feature_a": "a", "feature_b": "b"}
    ]


def test_prune_report_keeps_highest_importance_representative() -> None:
    frame = pd.DataFrame(
        {
            "weak_dup": [1.0, 2.0, 3.0, 4.0],
            "strong_dup": [2.0, 4.0, 6.0, 8.0],
            "independent": [4.0, 1.0, 3.0, 2.0],
        }
    )
    features = ["weak_dup", "strong_dup", "independent"]
    group_by_feature = {feature: "postopen_v2" for feature in features}
    importance = pd.DataFrame(
        [
            {"feature": "weak_dup", "importance_gain_mean": 1.0, "importance_split_mean": 1.0},
            {
                "feature": "strong_dup",
                "importance_gain_mean": 10.0,
                "importance_split_mean": 3.0,
            },
        ]
    )
    hygiene = summarize_feature_hygiene(
        frame,
        features,
        group_by_feature=group_by_feature,
        importance=importance,
    )
    pairs, _ = feature_correlation_pairs(
        frame,
        features,
        group_by_feature=group_by_feature,
        method="spearman",
        threshold=0.99,
        min_periods=2,
    )

    candidates, clusters, keep_list, drop_list = build_prune_report(
        features,
        hygiene,
        pairs,
        candidate_threshold=0.99,
        near_duplicate_threshold=0.995,
    )

    assert clusters.loc[0, "representative"] == "strong_dup"
    assert "weak_dup" in drop_list
    assert "strong_dup" in keep_list
    weak_candidate = candidates.set_index("feature").loc["weak_dup"]
    assert weak_candidate["action"] == "drop"
    assert weak_candidate["keep_feature"] == "strong_dup"


def test_explicit_missing_importance_path_is_required(tmp_path) -> None:
    missing_path = tmp_path / "missing_feature_importance.csv"

    with pytest.raises(SystemExit, match="feature importance file not found"):
        load_feature_importance(missing_path, required=True)


def test_labeled_cache_file_overlap_uses_year_from_name() -> None:
    path = Path("opening_2025_delay2_mixed_w030_labeled_v1.parquet")

    assert _file_overlaps_date_range(path, "2024-12-01", "2025-01-02")
    assert _file_overlaps_date_range(path, "2025-02-21", "2025-07-01")
    assert not _file_overlaps_date_range(path, "2024-01-01", "2024-07-01")


def test_labeled_pvc_historical_context_samples_each_target_day(tmp_path) -> None:
    input_dir = tmp_path / "cache"
    input_dir.mkdir()
    rows = []
    for date, value in [
        ("2021-12-29", 10.0),
        ("2021-12-30", 12.0),
        ("2022-01-04", 14.0),
        ("2022-01-27", 20.0),
        ("2022-01-28", 22.0),
        ("2022-02-01", 24.0),
    ]:
        rows.append(
            {
                "date": date,
                "symbol": "000001.SZ",
                "timestamp": f"{date} 09:31:00",
                "decision_time": "09:31:00",
                "decision_target_timestamp": f"{date} 09:31:00",
                "decision_lag_seconds": 0,
                "label": 0.1,
                "valid_label": True,
                "base_feature": value,
            }
        )
    pd.DataFrame(rows).to_csv(input_dir / "opening_2022_labeled.csv", index=False)

    args = SimpleNamespace(
        labeled_input="",
        sample_months=None,
        test_start_month=None,
        test_end_month=None,
        sample_month_stride=None,
        days_per_month=None,
        sample_rows=None,
        random_state=None,
    )
    config = {
        "data": {"labeled_path": str(input_dir)},
        "universe": {"enabled": False},
        "sample": {
            "mode": "decision_points",
            "decision_times": ["09:31:00"],
            "decision_max_lag_seconds": 5,
        },
        "features": {
            "include_historical_same_minute_surprise": True,
            "historical_surprise_columns": ["base_feature"],
            "historical_surprise_windows": [2],
            "historical_surprise_min_periods": 2,
            "historical_surprise_modes": ["ratio"],
        },
        "feature_hygiene": {
            "sample_months": ["2022-01", "2022-02"],
            "days_per_month": 1,
            "sample_rows": 2,
            "random_state": 7,
            "historical_context_calendar_days": 10,
        },
    }

    sampled = _sample_labeled_pvc_frame(args, config)

    assert sampled["date"].tolist() == ["2022-01-04", "2022-02-01"]
    surprise_col = "hist_surprise_base_feature_2d_ratio"
    assert surprise_col in sampled.columns
    assert sampled[surprise_col].notna().all()


def test_feature_hygiene_main_writes_reports_for_labeled_input(tmp_path) -> None:
    input_path = tmp_path / "labeled.csv"
    output_dir = tmp_path / "audit"
    pd.DataFrame(
        {
            "date": ["2022-01-04"] * 4,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "timestamp": pd.date_range("2022-01-04 09:31:00", periods=4, freq="s"),
            "label": [0.1, 0.2, 0.3, 0.4],
            "valid_label": [True, True, True, True],
            "weak_dup": [1.0, 2.0, 3.0, 4.0],
            "strong_dup": [2.0, 4.0, 6.0, 8.0],
            "constant_feature": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(input_path, index=False)

    main(
        [
            "--input",
            str(input_path),
            "--input-kind",
            "labeled",
            "--output-dir",
            str(output_dir),
            "--corr-threshold",
            "0.99",
            "--near-duplicate-threshold",
            "0.995",
            "--min-corr-periods",
            "2",
        ]
    )

    expected = {
        "feature_hygiene.csv",
        "feature_correlation_pairs.csv",
        "feature_correlation_clusters.csv",
        "feature_prune_candidates.csv",
        "feature_keep_list.txt",
        "feature_drop_list.txt",
        "feature_hygiene_trace.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    drops = (output_dir / "feature_drop_list.txt").read_text(encoding="utf-8").splitlines()
    assert "constant_feature" in drops
