from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import get_tick_client
from opening_strength_fit.feature_utils import finite_numeric as finite

SELECTED_PATH = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/"
    "opening_h1m_maxpath_deep_audit_v1/current_selected_members.parquet"
)
OUTPUT_DIR = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/opening_h1m_top1000_pool_profile_v1"
)
SYMBOL_REGEX = r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"
STYLE_FACTORS = (
    "pe_positive",
    "pb_positive",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d_ann",
    "position_52w",
)


def query_year(client: object, year: int) -> pd.DataFrame:
    start = f"{year - 1}-07-01"
    end = f"{year}-12-31"
    query = """
select
    toString(TradingDay) as date,
    Symbol as symbol,
    ClosePrice as close_price,
    AdjFactor as adj_factor,
    PE as pe,
    PB as pb,
    High52Weeks as high_52w,
    Low52Weeks as low_52w,
    STStatus as st_status
from stock.daily_bar_jy
where TradingDay between {start_date:Date} and {end_date:Date}
  and match(Symbol, {symbol_regex:String})
"""
    frame = client.query_df(
        query,
        parameters={
            "start_date": start,
            "end_date": end,
            "symbol_regex": SYMBOL_REGEX,
        },
    )
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    for column in (
        "close_price",
        "adj_factor",
        "pe",
        "pb",
        "high_52w",
        "low_52w",
        "st_status",
    ):
        frame[column] = finite(frame[column])
    frame = (
        frame.sort_values(["symbol", "date"], kind="mergesort")
        .drop_duplicates(["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )

    grouped = frame.groupby("symbol", sort=False)
    frame["adj_close"] = frame["close_price"].mul(frame["adj_factor"])
    frame["stock_return"] = grouped["adj_close"].pct_change(fill_method=None)
    frame["pe_positive"] = grouped["pe"].shift(1).where(lambda x: x.gt(0))
    frame["pb_positive"] = grouped["pb"].shift(1).where(lambda x: x.gt(0))
    previous_close = grouped["adj_close"].shift(1)
    frame["momentum_20d"] = previous_close.div(grouped["adj_close"].shift(21)).sub(1)
    frame["momentum_60d"] = previous_close.div(grouped["adj_close"].shift(61)).sub(1)
    frame["volatility_20d_ann"] = (
        grouped["stock_return"]
        .rolling(20, min_periods=15)
        .std()
        .reset_index(level=0, drop=True)
        .groupby(frame["symbol"], sort=False)
        .shift(1)
        .mul(np.sqrt(252.0))
    )
    previous_raw_close = grouped["close_price"].shift(1)
    previous_high = grouped["high_52w"].shift(1)
    previous_low = grouped["low_52w"].shift(1)
    frame["position_52w"] = previous_raw_close.sub(previous_low).div(
        previous_high.sub(previous_low).where(previous_high.gt(previous_low))
    )
    frame["position_52w"] = frame["position_52w"].where(frame["position_52w"].between(-0.05, 1.05))
    frame = frame.loc[frame["date"].str.startswith(str(year))].copy()
    for factor in STYLE_FACTORS:
        frame[f"{factor}_pct"] = frame.groupby("date", sort=False)[factor].rank(
            method="average", pct=True
        )
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_keys = pd.read_parquet(SELECTED_PATH, columns=["date", "symbol"])
    selected_keys["date"] = selected_keys["date"].astype(str)
    selected_keys["symbol"] = selected_keys["symbol"].astype(str)
    client = get_tick_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
    )

    selected_parts: list[pd.DataFrame] = []
    market_values: dict[str, list[np.ndarray]] = {factor: [] for factor in STYLE_FACTORS}
    market_counts: list[dict[str, float | int]] = []
    for year in range(2022, 2026):
        print(f"querying and computing {year}", flush=True)
        market = query_year(client, year)
        keys = selected_keys.loc[selected_keys["date"].str.startswith(str(year))]
        keep = [
            "date",
            "symbol",
            "st_status",
            *STYLE_FACTORS,
            *(f"{factor}_pct" for factor in STYLE_FACTORS),
        ]
        chosen = keys.merge(market[keep], on=["date", "symbol"], how="left", validate="one_to_one")
        selected_parts.append(chosen)
        for factor in STYLE_FACTORS:
            values = market[factor].dropna().to_numpy(dtype=np.float64, copy=True)
            market_values[factor].append(values)
        market_counts.append(
            {
                "year": year,
                "rows": len(market),
                "st_rows": int(market["st_status"].fillna(0).ne(0).sum()),
                "st_share_pct": market["st_status"].fillna(0).ne(0).mean() * 100,
            }
        )
        del market, chosen

    selected = pd.concat(selected_parts, ignore_index=True)
    selected.to_parquet(OUTPUT_DIR / "selected_style_factors.parquet", index=False)
    pd.DataFrame(market_counts).to_csv(OUTPUT_DIR / "market_st_by_year.csv", index=False)

    rows: list[dict[str, float | int | str]] = []
    for factor in STYLE_FACTORS:
        market = np.concatenate(market_values[factor])
        chosen = finite(selected[factor]).dropna().to_numpy(dtype=np.float64)
        chosen_pct = finite(selected[f"{factor}_pct"])
        rows.append(
            {
                "factor": factor,
                "market_valid_rows": len(market),
                "selected_valid_rows": len(chosen),
                "selected_coverage_pct": len(chosen) / len(selected) * 100,
                "market_mean": float(np.mean(market)),
                "market_p10": float(np.quantile(market, 0.10)),
                "market_p50": float(np.quantile(market, 0.50)),
                "market_p90": float(np.quantile(market, 0.90)),
                "selected_mean": float(np.mean(chosen)),
                "selected_p10": float(np.quantile(chosen, 0.10)),
                "selected_p50": float(np.quantile(chosen, 0.50)),
                "selected_p90": float(np.quantile(chosen, 0.90)),
                "selected_mean_market_pct": float(chosen_pct.mean() * 100),
                "selected_bottom_decile_pct": float(chosen_pct.le(0.10).mean() * 100),
                "selected_top_decile_pct": float(chosen_pct.ge(0.90).mean() * 100),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "style_factor_exposure.csv", index=False)

    selected_st = selected["st_status"].fillna(0).ne(0)
    market_rows = sum(row["rows"] for row in market_counts)
    market_st_rows = sum(row["st_rows"] for row in market_counts)
    pd.DataFrame(
        [
            {
                "market_rows": market_rows,
                "market_st_rows": market_st_rows,
                "market_st_share_pct": market_st_rows / market_rows * 100,
                "selected_rows": len(selected),
                "selected_st_rows": int(selected_st.sum()),
                "selected_st_share_pct": selected_st.mean() * 100,
                "st_representation_ratio": selected_st.mean() / (market_st_rows / market_rows),
            }
        ]
    ).to_csv(OUTPUT_DIR / "st_exposure.csv", index=False)
    (OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
