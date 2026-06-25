from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.commands.exposure_audit import main
from opening_strength_fit.commands.exposure_input_build import build_exposure_input
from opening_strength_fit.exposure_audit import (
    daily_concentration,
    exposure_group_metrics,
    exposure_specs,
    industry_group_metrics,
    summarize_concentration,
    summarize_exposure_groups,
    summarize_industry_groups,
)


def _audit_frame() -> pd.DataFrame:
    rows = []
    for clock in ["09:31:00", "09:32:00"]:
        for symbol, score, price, turnover, industry in [
            ("000001.SZ", 3.0, 30.0, 300.0, "bank"),
            ("000002.SZ", 2.0, 20.0, 200.0, "tech"),
            ("600000.SH", 1.0, 10.0, 100.0, "tech"),
        ]:
            rows.append(
                {
                    "date": "2022-01-03",
                    "symbol": symbol,
                    "decision_target_timestamp": f"2022-01-03 {clock}",
                    "prediction": score,
                    "buy_price": price,
                    "turnover_diff_10t": turnover,
                    "industry": industry,
                }
            )
    return pd.DataFrame(rows)


def test_exposure_group_metrics_quantify_selected_bias() -> None:
    metrics = exposure_group_metrics(
        _audit_frame(),
        exposure_specs(["buy_price", "turnover_diff_10t"]),
        pool="pool_L",
        score_col="prediction",
        top_n=1,
    )
    summary = summarize_exposure_groups(metrics, ["pool", "category", "exposure"]).set_index(
        "exposure"
    )

    assert summary.loc["buy_price", "selected_mean"] == pytest.approx(30.0)
    assert summary.loc["buy_price", "candidate_mean"] == pytest.approx(20.0)
    assert summary.loc["buy_price", "selected_mean_rank"] == pytest.approx(1.0)
    assert summary.loc["buy_price", "selected_mean_z"] == pytest.approx(1.224744871, rel=1e-6)
    assert summary.loc["turnover_diff_10t", "score_exposure_spearman"] == pytest.approx(1.0)


def test_daily_concentration_counts_repeated_intraday_symbols() -> None:
    daily = daily_concentration(
        _audit_frame(),
        pool="pool_L",
        score_col="prediction",
        top_n=1,
        industry_col="industry",
    )
    overall = summarize_concentration(daily).set_index("pool")
    row = daily.iloc[0]

    assert row["selected_rows"] == 2
    assert row["selected_symbols"] == 1
    assert row["selected_repeat_rate"] == pytest.approx(0.5)
    assert row["selected_symbol_max_share"] == pytest.approx(1.0)
    assert row["selected_industries"] == 1
    assert overall.loc["pool_L", "selected_effective_symbols"] == pytest.approx(1.0)


def test_industry_group_metrics_measure_active_share() -> None:
    metrics = industry_group_metrics(
        _audit_frame(),
        industry_col="industry",
        pool="pool_L",
        score_col="prediction",
        top_n=1,
    )
    summary = summarize_industry_groups(
        metrics,
        ["pool", "industry_col", "industry"],
    ).set_index("industry")

    assert summary.loc["bank", "candidate_share"] == pytest.approx(1.0 / 3.0)
    assert summary.loc["bank", "selected_share"] == pytest.approx(1.0)
    assert summary.loc["bank", "active_share"] == pytest.approx(2.0 / 3.0)
    assert summary.loc["tech", "active_share"] == pytest.approx(-2.0 / 3.0)


def test_exposure_audit_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "audit"
    _audit_frame().to_parquet(predictions, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "osf-audit-exposure",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--pool",
            "universe",
            "--top-n",
            "1",
            "--exposure-col",
            "buy_price",
            "--industry-col",
            "industry",
        ],
    )

    main()

    summary = pd.read_csv(output_dir / "exposure_audit_summary.csv")
    concentration = pd.read_csv(output_dir / "exposure_audit_concentration_summary.csv")
    assert summary["exposure"].tolist() == ["buy_price"]
    assert concentration.loc[0, "pool"] == "universe"
    assert (output_dir / "exposure_audit_industry_summary.csv").exists()
    assert (output_dir / "exposure_audit_trace.json").exists()


def test_exposure_audit_cli_joins_daily_exposure_input(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.parquet"
    exposure_input = tmp_path / "daily_exposure.parquet"
    output_dir = tmp_path / "audit"
    _audit_frame().drop(columns=["buy_price", "industry"]).to_parquet(predictions, index=False)
    pd.DataFrame(
        [
            {
                "date": "2022-01-03",
                "symbol": "000001.SZ",
                "market_cap": 300.0,
                "industry_sw1": "bank",
            },
            {
                "date": "2022-01-03",
                "symbol": "000002.SZ",
                "market_cap": 200.0,
                "industry_sw1": "tech",
            },
            {
                "date": "2022-01-03",
                "symbol": "600000.SH",
                "market_cap": 100.0,
                "industry_sw1": "tech",
            },
        ]
    ).to_parquet(exposure_input, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "osf-audit-exposure",
            "--predictions",
            str(predictions),
            "--exposure-input",
            str(exposure_input),
            "--output-dir",
            str(output_dir),
            "--pool",
            "universe",
            "--top-n",
            "1",
            "--exposure-col",
            "market_cap",
            "--industry-col",
            "industry_sw1",
        ],
    )

    main()

    summary = pd.read_csv(output_dir / "exposure_audit_summary.csv")
    industries = pd.read_csv(output_dir / "exposure_audit_industry_summary.csv")
    assert summary.loc[0, "exposure"] == "market_cap"
    assert summary.loc[0, "selected_mean_rank"] == pytest.approx(1.0)
    assert set(industries["industry"]) == {"bank", "tech"}


class _FakeExposureClient:
    def query_df(self, query: str, parameters: dict) -> pd.DataFrame:
        rows = []
        for date in parameters["dates"]:
            for symbol in parameters["symbols"]:
                if "TotalMarketValue" in query:
                    rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "market_cap": 100.0,
                            "float_market_cap": 80.0,
                            "daily_amount": 10.0,
                            "daily_turnover_rate": 1.2,
                            "free_turnover_rate": 0.9,
                            "close_price": 20.0,
                        }
                    )
                else:
                    rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "industry_sw1": "bank" if symbol == "000001.SZ" else "tech",
                            "industry_sw2": "",
                            "industry_sw3": "",
                            "industry_code_sw1": "",
                            "industry_code_sw2": "",
                            "industry_code_sw3": "",
                        }
                    )
        return pd.DataFrame(rows)


def test_build_exposure_input_from_daily_bar_and_industry() -> None:
    keys = _audit_frame()[["date", "symbol", "decision_target_timestamp"]]
    exposures = build_exposure_input(
        keys=keys,
        client=_FakeExposureClient(),
        daily_bar_table="stock.daily_bar_jy_local",
        industry_table="stock.industry_local",
        date_chunk_size=30,
    )

    assert len(exposures) == 3
    assert "log_market_cap" in exposures.columns
    assert exposures.loc[exposures["symbol"] == "000001.SZ", "industry"].iloc[0] == "bank"
