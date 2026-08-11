from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_date_series as _date
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows as _normalize_keys
from opening_strength_fit.schema import normalize_text_series as _text

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
EXCLUDED_DATES = {"2026-03-18", "2026-04-22", "2026-05-06"}


root = Path("/mnt/output/opening_strength_fit")
prediction_path = (
    root / "nn/holdout/nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1/"
    "predictions_unfiltered.parquet"
)
predictions = _normalize_keys(
    pd.read_parquet(
        prediction_path,
        columns=[*KEYS, "prediction", "label_short"],
    )
)
predictions = predictions.loc[~predictions["date"].isin(EXCLUDED_DATES)]
ordered = predictions.sort_values(
    [*GROUPS, "prediction", "symbol"],
    ascending=[True, True, False, True],
    kind="mergesort",
)
top = ordered.groupby(GROUPS, sort=False).head(100).copy()

feature_path = root / "datasets/opening_0931_0940_features_350/year=2026/features.parquet"
features = _normalize_keys(
    pd.read_parquet(
        feature_path,
        columns=[*KEYS, "mid_price", "spread_bps", "ask_volume_1"],
    )
)
top = top.merge(features, on=KEYS, how="left", validate="one_to_one")

daily_path = root / "cache/opening_0931_0940_raw_source/year=2026/daily_reference.parquet"
daily = pd.read_parquet(
    daily_path,
    columns=[
        "TradingDay",
        "Symbol",
        "ClosePrice",
        "PreClosePrice",
        "STStatus",
        "TradeStatus",
        "UpdownLimitStatus",
    ],
).rename(
    columns={
        "TradingDay": "date",
        "Symbol": "symbol",
        "ClosePrice": "daily_close",
        "PreClosePrice": "prev_close",
        "STStatus": "st_status",
        "TradeStatus": "trade_status",
        "UpdownLimitStatus": "updown_limit_status",
    }
)
daily["date"] = _date(daily["date"])
daily["symbol"] = _text(daily["symbol"])
for column in ("daily_close", "prev_close", "st_status", "updown_limit_status"):
    daily[column] = pd.to_numeric(daily[column], errors="coerce")
daily["trade_status"] = _text(daily["trade_status"]).str.upper()
daily["final_up_limit"] = daily["updown_limit_status"].eq(1)
symbol = daily["symbol"].astype(str)
ratio = np.where(
    daily["st_status"].fillna(0).ne(0),
    0.05,
    np.where(symbol.str.startswith("30") | symbol.str.startswith("68"), 0.20, 0.10),
)
standard_limit = (
    np.floor(daily["prev_close"].to_numpy() * (1.0 + ratio) * 100.0 + 0.50000001) / 100.0
)
daily["upper_limit_price"] = pd.Series(standard_limit, index=daily.index).where(
    ~daily["final_up_limit"], daily["daily_close"]
)
daily.loc[daily["trade_status"].isin({"NEW", "新股", "N"}), "upper_limit_price"] = np.nan
daily = daily[
    ["date", "symbol", "daily_close", "upper_limit_price", "final_up_limit"]
].drop_duplicates(["date", "symbol"], keep="last")
top = top.merge(daily, on=["date", "symbol"], how="left", validate="many_to_one")

mid = pd.to_numeric(top["mid_price"], errors="coerce")
spread = pd.to_numeric(top["spread_bps"], errors="coerce")
decision_ask = mid * (1.0 + spread / 20_000.0)
decision_bid = mid * (1.0 - spread / 20_000.0)
ask_available = decision_ask.gt(0) & pd.to_numeric(top["ask_volume_1"], errors="coerce").gt(0)
decision_state = decision_ask.where(ask_available, decision_bid.where(decision_bid.gt(0)))
upper = pd.to_numeric(top["upper_limit_price"], errors="coerce")
entry_price = pd.to_numeric(top["daily_close"], errors="coerce") / (
    1.0 + pd.to_numeric(top["label_short"], errors="coerce")
)
entry_price = entry_price.where(entry_price.gt(0) & np.isfinite(entry_price))
top["decision_ask_available"] = ask_available
top["decision_room_bps"] = (upper - decision_ask) / decision_ask * 10_000.0
top["entry_room_bps"] = (upper - entry_price) / entry_price * 10_000.0
top["decision_sealed_at_limit"] = (
    upper.notna()
    & decision_state.notna()
    & decision_state.sub(upper).abs().le(0.0051)
    & ~ask_available
)
top["decision_within_100bps"] = ask_available & top["decision_room_bps"].between(-1.0, 100.0)
top["entry_at_limit"] = (
    upper.notna() & entry_price.notna() & entry_price.sub(upper).abs().le(0.0051)
)
top["entry_within_100bps"] = top["entry_room_bps"].between(-1.0, 100.0)
top["final_up_limit"] = top["final_up_limit"].fillna(False).astype(bool)


def summarize(part: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(part)),
        "decision_ask_available_pct": float(part["decision_ask_available"].mean() * 100.0),
        "decision_sealed_at_limit_pct": float(part["decision_sealed_at_limit"].mean() * 100.0),
        "decision_within_100bps_pct": float(part["decision_within_100bps"].mean() * 100.0),
        "entry_label_available_pct": float(entry_price.loc[part.index].notna().mean() * 100.0),
        "entry_at_limit_pct": float(part["entry_at_limit"].mean() * 100.0),
        "entry_within_100bps_pct": float(part["entry_within_100bps"].mean() * 100.0),
        "decision_room_bps_p10": float(part["decision_room_bps"].quantile(0.10)),
        "decision_room_bps_median": float(part["decision_room_bps"].median()),
        "entry_room_bps_p10": float(part["entry_room_bps"].quantile(0.10)),
        "entry_room_bps_median": float(part["entry_room_bps"].median()),
    }


result = {
    "status": "ok",
    "scope": "strict 2026H1 all-A close-model Top100",
    "selected": summarize(top),
    "selected_final_limit": summarize(top.loc[top["final_up_limit"]]),
    "selected_final_limit_share_pct": float(top["final_up_limit"].mean() * 100.0),
}
output_dir = root / "audits/ds350_2026h1_entry_limit_state_v1"
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(output_dir / "_SUCCESS").touch()
print(json.dumps(result, ensure_ascii=False, indent=2))
