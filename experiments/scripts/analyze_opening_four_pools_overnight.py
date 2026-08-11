from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.feature_utils import finite_numeric as finite
from opening_strength_fit.schema import normalize_date_series as normalize_date
from opening_strength_fit.schema import normalize_text_series as decode

ROOT = Path("/mnt/output/opening_strength_fit")
DAILY_ROOT = ROOT / "cache/opening_0931_0940_raw_source"
MAX_PATH = ROOT / "runs/analyses/opening_h1m_maxpath_deep_audit_v1/current_selected_members.parquet"
MIN_PATH = ROOT / "runs/analyses/opening_h1m_minpath_top1000_v1/selected_members.parquet"
ABS_PATH = ROOT / "runs/analyses/opening_h1m_meanabs_top1000_v1/selected_members.parquet"
OUTPUT_DIR = ROOT / "runs/analyses/opening_h1m_four_pools_overnight_v1"
KEYS = ["date", "symbol"]


def load_pool(path: Path, score: str, *, ascending: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*KEYS, score])
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(
        ["date", score, "symbol"],
        ascending=[True, ascending, True],
        kind="mergesort",
    )


def read_market() -> pd.DataFrame:
    parts = []
    for year in range(2022, 2026):
        path = DAILY_ROOT / f"year={year}" / "daily_reference.parquet"
        frame = pd.read_parquet(path, columns=["TradingDay", "Symbol", "ClosePrice"]).rename(
            columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "close"}
        )
        frame["date"] = normalize_date(frame["date"])
        frame["symbol"] = decode(frame["symbol"])
        frame["close"] = finite(frame["close"])
        parts.append(frame.drop_duplicates(KEYS, keep="last"))
    market = pd.concat(parts, ignore_index=True).drop_duplicates(KEYS, keep="last")
    dates = sorted(market["date"].dropna().unique())
    next_dates = pd.DataFrame({"date": dates[:-1], "next_date": dates[1:]})
    market = market.merge(next_dates, on="date", how="left", validate="many_to_one")
    next_close = market[["date", "symbol", "close"]].rename(
        columns={"date": "next_date", "close": "next_close"}
    )
    market = market.merge(next_close, on=["next_date", "symbol"], how="left", validate="one_to_one")
    market["overnight_return_bps"] = (
        market["next_close"].div(market["close"].where(market["close"].gt(0))).sub(1) * 10_000.0
    )
    return market


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    maximum = load_pool(MAX_PATH, "max_z", ascending=False)
    minimum = load_pool(MIN_PATH, "min_z", ascending=True)
    absolute = load_pool(ABS_PATH, "mean_abs_z", ascending=False)
    maximum["rank"] = maximum.groupby("date", sort=False).cumcount() + 1
    maximum["side"] = 0
    minimum["rank"] = minimum.groupby("date", sort=False).cumcount() + 1
    minimum["side"] = 1
    combined = (
        pd.concat([maximum, minimum], ignore_index=True)
        .sort_values(["date", "rank", "side", "symbol"], kind="mergesort")
        .drop_duplicates(KEYS, keep="first")
        .sort_values(["date", "rank", "side", "symbol"], kind="mergesort")
        .groupby("date", sort=False)
        .head(1_000)
    )
    pools = {
        "最大1000": maximum,
        "最小1000": minimum,
        "500并补1000": combined,
        "绝对值1000": absolute,
    }
    market = read_market()
    market_daily = (
        market.groupby("date", sort=False)["overnight_return_bps"]
        .mean()
        .rename("market_return_bps")
        .reset_index()
    )
    rows = []
    daily_parts = []
    for name, pool in pools.items():
        joined = pool[KEYS].merge(
            market[KEYS + ["overnight_return_bps"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        daily = (
            joined.groupby("date", sort=False)["overnight_return_bps"]
            .agg(valid_names="count", pool_return_bps="mean")
            .reset_index()
            .merge(market_daily, on="date", validate="one_to_one")
        )
        daily["active_return_bps"] = daily["pool_return_bps"].sub(daily["market_return_bps"])
        daily["方案"] = name
        daily_parts.append(daily)
        rows.append(
            {
                "方案": name,
                "days": daily["pool_return_bps"].notna().sum(),
                "pool_return_bps": daily["pool_return_bps"].mean(),
                "market_return_bps": daily["market_return_bps"].mean(),
                "active_return_bps": daily["active_return_bps"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(OUTPUT_DIR / "daily_returns.csv", index=False)
    (OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
