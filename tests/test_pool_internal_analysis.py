from __future__ import annotations

import pickle

import pandas as pd
import pytest

from opening_strength_fit.commands.pool_internal_analysis import (
    build_company_neutral_score_matrix,
    build_company_score_matrix,
    company_backtest_neutral_comparison_plot_data,
    company_backtest_relative_plot_data,
    create_company_backtest_payload,
    decode_company_backtest_result,
    filter_company_backtest_scores,
    normalize_clock,
    parse_args,
    write_company_api_outputs,
)
from opening_strength_fit.pool_internal_eval import (
    halfyear_summary,
    year_summary,
)
from opening_strength_fit.prediction_frames import prediction_files


def _month_row(month: str, short_bps: float, next_bps: float) -> dict[str, object]:
    return {
        "pool": "pool_S",
        "test_month": month,
        "candidate_rows": 1000.0,
        "selected_rows": 100.0,
        "pool_short_mean_bps": 0.0,
        "selected_short_mean_bps": short_bps,
        "short_internal_excess_bps": short_bps,
        "pool_next_mean_bps": 0.0,
        "selected_next_mean_bps": next_bps,
        "next_internal_excess_bps": next_bps,
        "short_rank_ic": 0.10,
        "next_rank_ic": 0.01,
    }


def test_pool_internal_halfyear_and_year_summaries() -> None:
    month_summary = pd.DataFrame(
        [
            _month_row("2025-01", 4.0, -1.0),
            _month_row("2025-02", 8.0, 3.0),
            _month_row("2025-07", 2.0, 5.0),
        ]
    )

    halfyear = halfyear_summary(month_summary)
    yearly = year_summary(month_summary)

    h1 = halfyear.loc[halfyear["half"].eq("H1")].iloc[0]
    assert h1["months"] == 2
    assert h1["short_internal_excess_bps"] == pytest.approx(6.0)
    assert h1["next_positive_months"] == 1

    year = yearly.iloc[0]
    assert year["year"] == 2025
    assert year["months"] == 3
    assert year["short_positive_months"] == 3
    assert year["next_internal_excess_bps"] == pytest.approx(7.0 / 3.0)


def test_prediction_files_can_read_k8s_shard_layout(tmp_path) -> None:
    first = tmp_path / "month_2022-01" / "predictions.parquet"
    second = tmp_path / "month_2022-07" / "predictions_2022.parquet"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    paths = prediction_files(tmp_path)

    assert paths == [first, second]


def test_company_score_transform_default_preserves_local_direction(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_pool_internal_top100",
            "--predictions",
            "predictions.parquet",
            "--output-dir",
            "out",
        ],
    )

    args = parse_args()

    assert args.company_score_transform == "identity"


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2022-01-03",
                "symbol": "000001.SZ",
                "decision_target_timestamp": "2022-01-03 09:31:00",
                "prediction": 1.0,
            },
            {
                "date": "2022-01-03",
                "symbol": "000001.SZ",
                "decision_target_timestamp": "2022-01-03 09:40:00",
                "prediction": 3.0,
            },
            {
                "date": "2022-01-03",
                "symbol": "000002.SZ",
                "decision_target_timestamp": "2022-01-03 09:40:00",
                "prediction": 5.0,
            },
            {
                "date": "2022-01-03",
                "symbol": "600000.SH",
                "decision_target_timestamp": "2022-01-03 09:40:00",
                "prediction": 4.0,
            },
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "decision_target_timestamp": "2022-01-04 09:40:00",
                "prediction": 6.0,
            },
            {
                "date": "2022-01-04",
                "symbol": "600000.SH",
                "decision_target_timestamp": "2022-01-04 09:40:00",
                "prediction": 2.0,
            },
        ]
    )


def test_company_backtest_clock_pool_filter_and_score_matrix(tmp_path) -> None:
    pool_path = tmp_path / "pool.parquet"
    pool = pd.DataFrame(
        {
            "000001.SZ": [True, True],
            "000002.SZ": [False, False],
            "600000.SH": [True, True],
        },
        index=pd.Index(["2022-01-03", "2022-01-04"], name="date"),
    )
    pool.to_parquet(pool_path)

    filtered, filter_stats = filter_company_backtest_scores(
        _prediction_frame(),
        score_col="prediction",
        clocks=None,
        start_clock="09:31",
        end_clock="09:40",
        pool="",
        pool_path=str(pool_path),
        pool_date_lag_sessions=0,
    )
    score, score_long, stats = build_company_score_matrix(
        filtered,
        score_col="prediction",
        score_agg="mean",
        score_transform="identity",
        top_n=1,
    )

    assert normalize_clock("940") == "09:40"
    assert filter_stats["rows_after_filters"] == 5
    assert filter_stats["clocks"] == ["09:31", "09:40"]
    assert stats["score_dates"] == 2
    assert stats["top_n"] == 1
    assert score.loc[pd.Timestamp("2022-01-03"), "600000.SH"] == pytest.approx(4.0)
    assert score.loc[pd.Timestamp("2022-01-04"), "000001.SZ"] == pytest.approx(6.0)
    assert score_long.groupby("date").size().tolist() == [1, 1]


def test_company_backtest_score_transform_keeps_top_selection_direction() -> None:
    score, score_long, stats = build_company_score_matrix(
        _prediction_frame(),
        score_col="prediction",
        score_agg="mean",
        score_transform="negate",
        top_n=1,
    )

    assert stats["score_transform"] == "negate"
    assert score.loc[pd.Timestamp("2022-01-03"), "000002.SZ"] == pytest.approx(-5.0)
    assert score.loc[pd.Timestamp("2022-01-04"), "000001.SZ"] == pytest.approx(-6.0)
    assert score_long["prediction"].tolist() == pytest.approx([-5.0, -6.0])


def test_company_neutral_score_matrix_preserves_finite_mask() -> None:
    score = pd.DataFrame(
        {
            "000001.SZ": [1.0, None],
            "000002.SZ": [2.0, 3.0],
        },
        index=pd.DatetimeIndex(["2022-01-03", "2022-01-04"], name="date"),
    )
    neutral, neutral_long, stats = build_company_neutral_score_matrix(
        score,
        score_col="prediction",
        neutral_score=0.0,
        base_stats={"top_n": 0},
    )

    assert neutral.loc[pd.Timestamp("2022-01-03"), "000001.SZ"] == pytest.approx(0.0)
    assert pd.isna(neutral.loc[pd.Timestamp("2022-01-04"), "000001.SZ"])
    assert neutral_long["prediction"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert stats["score_transform"] == "neutral_constant"
    assert stats["score_rows_long"] == 3


def test_company_backtest_relative_plot_data_aligns_dates() -> None:
    left = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "profit": [0.03, 0.01],
            "alpha": [0.02, 0.00],
        }
    )
    right = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2022-01-04", "2022-01-05"]),
            "profit": [0.005, 0.02],
            "alpha": [-0.005, 0.01],
        }
    )

    relative = company_backtest_relative_plot_data(
        left,
        right,
        series_key="model_minus_neutral",
        series_label="model - neutral",
    )

    assert relative["week_start"].tolist() == [pd.Timestamp("2022-01-04")]
    assert relative["profit_bps"].iloc[0] == pytest.approx(50.0)
    assert relative["alpha_bps"].iloc[0] == pytest.approx(50.0)
    assert relative["profit_cumulative_bps"].iloc[0] == pytest.approx(50.0)


def test_company_backtest_neutral_comparison_plot_data_splits_panels() -> None:
    model = pd.DataFrame(
        {
            "pool": ["model", "model"],
            "pool_label": ["model", "model"],
            "week_start": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "profit": [0.01, 0.02],
            "alpha": [0.00, 0.01],
            "profit_bps": [100.0, 200.0],
            "alpha_bps": [0.0, 100.0],
            "profit_cumulative_bps": [100.0, 300.0],
            "alpha_cumulative_bps": [0.0, 100.0],
        }
    )
    neutral = pd.DataFrame(
        {
            "pool": ["neutral_pool", "neutral_pool"],
            "pool_label": ["neutral_pool", "neutral_pool"],
            "week_start": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "profit": [0.005, 0.01],
            "alpha": [-0.005, 0.00],
            "profit_bps": [50.0, 100.0],
            "alpha_bps": [-50.0, 0.0],
            "profit_cumulative_bps": [50.0, 150.0],
            "alpha_cumulative_bps": [-50.0, -50.0],
        }
    )

    out = company_backtest_neutral_comparison_plot_data(
        model,
        neutral,
        model_key="model",
        model_label="model",
        neutral_key="neutral_pool",
        neutral_label="neutral_pool",
        delta_key="model",
        delta_label="model",
    )

    model_rows = out.loc[out["pool"].eq("model")]
    delta = model_rows.dropna(subset=["incremental_cumulative_bps"])
    assert model_rows["profit_cumulative_bps"].notna().sum() == 2
    assert out.loc[out["pool"].eq("neutral_pool"), "incremental_cumulative_bps"].isna().all()
    assert delta["profit_cumulative_bps"].isna().all()
    assert delta["incremental_cumulative_bps"].tolist() == pytest.approx([50.0, 150.0])


def test_company_backtest_payload_decode_and_write_outputs(tmp_path) -> None:
    score = pd.DataFrame(
        {"000001.SZ": [1.5]},
        index=pd.DatetimeIndex(["2022-01-03"], name="date"),
    )
    payload = create_company_backtest_payload(
        score,
        api_time="950",
        daily=False,
        tar="I500",
        cap=1_000_000.0,
        trgain=0.4,
        fee=False,
        vol_limit=0.2,
        return_eod=True,
    )
    request = pickle.loads(payload)

    assert set(request["score"]) == {"950"}
    assert request["score"]["950"]["index"] == ["2022-01-03"]
    assert request["fee"] is False
    assert request["return_eod"] is True

    raw = pickle.dumps(
        {
            "alpha": {"2022-01-03": 0.01},
            "profit": {"2022-01-03": 0.02},
            "overday": {"2022-01-03": 0.015},
            "inday": {"2022-01-03": 0.005},
            "turnover": {"2022-01-03": 0.4},
            "rent": {"2022-01-03": 0.0},
            "count": {"2022-01-03": 100},
            "solve_rate": {"data": [0.99], "index": ["2022-01-03"]},
        }
    )
    result = decode_company_backtest_result(raw)
    summary = write_company_api_outputs(tmp_path, result)

    assert float(result["alpha"].iloc[0]) == pytest.approx(0.01)
    assert summary["series"]["turnover"]["mean"] == pytest.approx(0.4)
    assert summary["solve_rate_mean"] == pytest.approx(0.99)
    assert (tmp_path / "alpha.csv").exists()
    assert (tmp_path / "backtest_result.pkl").exists()
