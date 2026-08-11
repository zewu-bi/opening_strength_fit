from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from opening_strength_fit import ds350_holdout_analysis as common

TARGET_NOTIONAL = 50_000_000.0
MAX_SYMBOL_NOTIONAL = TARGET_NOTIONAL * 0.01
TURNOVER_PARTICIPATION = 0.20
DEPTH_PARTICIPATION = 0.25
ASK_LEVELS = tuple(range(1, 11))


def _predictions(model: str) -> pd.DataFrame:
    path = (
        common.ROOT
        / "nn/holdout"
        / f"nn_ds350_w0931_h{model}_train2023_2025_test2026h1_purge1_v1"
        / "predictions_unfiltered.parquet"
    )
    frame = common._normalize(
        pd.read_parquet(
            path,
            columns=[
                *common.KEYS,
                "prediction",
                "label_short",
            ],
        )
    )
    frame = frame.loc[~frame["date"].isin(common.EXCLUDED_DATES)].copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["own_label"] = pd.to_numeric(frame.pop("label_short"), errors="coerce")
    return frame.dropna(subset=[*common.KEYS, "prediction"])


def _raw_depth(keys: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Status",
        *[name for level in ASK_LEVELS for name in (f"AskPrice{level}", f"AskVolume{level}")],
    ]
    out = common.sample_tick_states(
        keys,
        columns=columns,
        delay_seconds=6,
        max_workers=4,
        progress_label="raw",
    ).rename(
        columns={
            "Status": "raw_entry_status",
            "raw_state_age_seconds": "raw_entry_state_age_seconds",
            **{
                f"Ask{kind}{level}": f"entry_ask_{kind.lower()}_{level}"
                for level in ASK_LEVELS
                for kind in ("Price", "Volume")
            },
        }
    )
    out["raw_entry_status"] = common._text(out["raw_entry_status"]).str.upper()
    for level in ASK_LEVELS:
        for kind in ("price", "volume"):
            column = f"entry_ask_{kind}_{level}"
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _raw_turnover(keys: pd.DataFrame) -> pd.DataFrame:
    raw_root = common.ROOT / "cache/opening_0931_0940_raw_source/year=2026/ticks"
    items = [(date, wanted.copy()) for date, wanted in keys.groupby("date", sort=True)]

    def read_day(date: str, wanted: pd.DataFrame) -> pd.DataFrame:
        path = raw_root / f"date={date}.parquet"
        if not path.exists():
            return pd.DataFrame()
        symbols = sorted(set(wanted["symbol"]))
        ticks = pd.read_parquet(
            path,
            columns=["Symbol", "ExchTimeOffsetUs", "Turnover"],
            filters=[("Symbol", "in", symbols)],
        )
        ticks["Symbol"] = common._text(ticks["Symbol"])
        ticks["ExchTimeOffsetUs"] = pd.to_numeric(ticks["ExchTimeOffsetUs"], errors="coerce")
        ticks["Turnover"] = pd.to_numeric(ticks["Turnover"], errors="coerce")
        ticks = ticks.dropna(subset=["ExchTimeOffsetUs"]).sort_values(
            ["Symbol", "ExchTimeOffsetUs"], kind="mergesort"
        )
        rows: list[pd.DataFrame] = []
        for symbol, part in wanted.groupby("symbol", sort=False):
            state = ticks.loc[ticks["Symbol"].eq(symbol)]
            if state.empty:
                continue
            offsets = state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
            turnover = state["Turnover"].to_numpy(dtype="float64")
            targets = (
                (
                    part["decision_target_timestamp"]
                    - part["decision_target_timestamp"].dt.normalize()
                )
                / pd.Timedelta(microseconds=1)
            ).to_numpy(dtype="int64")
            positions = np.searchsorted(offsets, targets, side="right") - 1
            prior = positions - 10
            valid = prior >= 0
            if not valid.any():
                continue
            out = part.loc[valid, common.KEYS].copy()
            diff = turnover[positions[valid]] - turnover[prior[valid]]
            out["raw_turnover_diff_10t"] = np.where(diff >= 0.0, diff, np.nan)
            rows.append(out)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(read_day, date, wanted): date for date, wanted in items}
        for index, future in enumerate(as_completed(futures), start=1):
            part = future.result()
            if not part.empty:
                rows.append(part)
            print(f"turnover progress {index}/{len(items)} {futures[future]}", flush=True)
    return pd.concat(rows, ignore_index=True).drop_duplicates(common.KEYS, keep="last")


def _add_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    valid_status = out["raw_entry_status"].isin(common.TRADABLE_STATUSES)
    depth_parts: list[pd.Series] = []
    for level in ASK_LEVELS:
        price = pd.to_numeric(out[f"entry_ask_price_{level}"], errors="coerce")
        volume = pd.to_numeric(out[f"entry_ask_volume_{level}"], errors="coerce")
        depth_parts.append((price * volume).where(price.gt(0) & volume.gt(0), 0.0))
    out["entry_ask1_notional"] = depth_parts[0].where(valid_status, 0.0)
    out["entry_ask10_notional"] = sum(depth_parts).where(valid_status, 0.0)
    out["turnover_cap"] = (
        pd.to_numeric(out["raw_turnover_diff_10t"], errors="coerce").clip(lower=0.0)
        * TURNOVER_PARTICIPATION
    ).fillna(0.0)
    out["cap_turnover_only"] = np.minimum(MAX_SYMBOL_NOTIONAL, out["turnover_cap"])
    out["cap_depth_ask1_only"] = np.minimum(
        MAX_SYMBOL_NOTIONAL, out["entry_ask1_notional"] * DEPTH_PARTICIPATION
    )
    out["cap_depth_ask10_only"] = np.minimum(
        MAX_SYMBOL_NOTIONAL, out["entry_ask10_notional"] * DEPTH_PARTICIPATION
    )
    out["cap_with_ask1"] = np.minimum(
        out["cap_turnover_only"], out["entry_ask1_notional"] * DEPTH_PARTICIPATION
    )
    out["cap_with_ask10"] = np.minimum(
        out["cap_turnover_only"], out["entry_ask10_notional"] * DEPTH_PARTICIPATION
    )
    return out


def _quantiles(values: pd.Series) -> dict[str, float]:
    valid = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "p10": float(valid.quantile(0.10)),
        "median": float(valid.quantile(0.50)),
        "p90": float(valid.quantile(0.90)),
        "mean": float(valid.mean()),
    }


def _summarize_segment(frame: pd.DataFrame, *, total_rows: int) -> dict[str, object]:
    status_valid = frame["raw_entry_status"].isin(common.TRADABLE_STATUSES)
    return {
        "rows": int(len(frame)),
        "row_share_pct": float(len(frame) / total_rows * 100.0),
        "raw_entry_status_valid_pct": float(status_valid.mean() * 100.0),
        "ask1_notional_cny": _quantiles(frame.loc[status_valid, "entry_ask1_notional"]),
        "ask10_notional_cny": _quantiles(frame.loc[status_valid, "entry_ask10_notional"]),
        "raw_turnover_diff_10t_cny": _quantiles(frame["raw_turnover_diff_10t"]),
        "cap_depth_ask1_only_cny": _quantiles(frame["cap_depth_ask1_only"]),
        "cap_depth_ask10_only_cny": _quantiles(frame["cap_depth_ask10_only"]),
        "cap_with_ask1_cny": _quantiles(frame["cap_with_ask1"]),
        "cap_with_ask10_cny": _quantiles(frame["cap_with_ask10"]),
    }


def _group_fill(frame: pd.DataFrame, cap: str) -> dict[str, float]:
    fill = (
        frame.groupby(common.GROUPS, sort=False)[cap].sum().clip(upper=TARGET_NOTIONAL)
        / TARGET_NOTIONAL
    )
    return {
        "mean_pct": float(fill.mean() * 100.0),
        "p10_pct": float(fill.quantile(0.10) * 100.0),
        "median_pct": float(fill.median() * 100.0),
        "min_pct": float(fill.min() * 100.0),
        "full_group_pct": float(fill.ge(1.0 - 1e-9).mean() * 100.0),
    }


def _model_summary(frame: pd.DataFrame) -> dict[str, object]:
    final = frame.loc[frame["final_up_limit"]].copy()
    other = frame.loc[~frame["final_up_limit"]].copy()
    ask10_total = float(frame["cap_with_ask10"].sum())
    ask1_total = float(frame["cap_with_ask1"].sum())
    return {
        "rows": int(len(frame)),
        "groups": int(frame.groupby(common.GROUPS, sort=False).ngroups),
        "top100_segments": {
            "all": _summarize_segment(frame, total_rows=len(frame)),
            "eventual_limit": _summarize_segment(final, total_rows=len(frame)),
            "nonlimit": _summarize_segment(other, total_rows=len(frame)),
        },
        "fixed_top100_fill_ratio": {
            "25pct_ask1_depth_only": _group_fill(frame, "cap_depth_ask1_only"),
            "25pct_ask10_depth_only": _group_fill(frame, "cap_depth_ask10_only"),
            "old_turnover_rule_without_depth": _group_fill(frame, "cap_turnover_only"),
            "plus_25pct_ask1": _group_fill(frame, "cap_with_ask1"),
            "plus_25pct_ask10": _group_fill(frame, "cap_with_ask10"),
        },
        "eventual_limit_share_of_fillable_notional_pct": {
            "ask1_basis": float(final["cap_with_ask1"].sum() / ask1_total * 100.0),
            "ask10_basis": float(final["cap_with_ask10"].sum() / ask10_total * 100.0),
        },
    }


def main() -> None:
    reference = common._market_reference()[["date", "symbol", "final_up_limit"]]
    tops: dict[str, pd.DataFrame] = {}
    for model in ("1m", "close"):
        frame = _predictions(model).merge(
            reference, on=["date", "symbol"], how="left", validate="many_to_one"
        )
        frame["final_up_limit"] = frame["final_up_limit"].fillna(False).astype(bool)
        tops[model] = common._top(frame)

    keys = pd.concat([frame[common.KEYS] for frame in tops.values()], ignore_index=True)
    keys = keys.drop_duplicates(common.KEYS, keep="last")
    output = common.ROOT / "audits/ds350_2026h1_top100_capacity_v2"
    depth_v1 = common.ROOT / "audits/ds350_2026h1_top100_capacity_v1/raw_top100_entry_depth.parquet"
    raw_path = output / "raw_top100_entry_depth.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    elif depth_v1.exists():
        raw = pd.read_parquet(depth_v1)
    else:
        raw = _raw_depth(keys)
    turnover_path = output / "raw_top100_turnover_10t.parquet"
    turnover = pd.read_parquet(turnover_path) if turnover_path.exists() else _raw_turnover(keys)
    raw = raw.merge(turnover, on=common.KEYS, how="left", validate="one_to_one")

    result: dict[str, object] = {
        "status": "ok",
        "sample": "strict 2026H1 all-A Top100; train 2023-2025; purge one session",
        "entry": "last raw state known at decision_target_timestamp + 6 seconds",
        "capacity_convention": {
            "strategy_capital_cny": 1_000_000_000.0,
            "slices": 20,
            "target_per_decision_cny": TARGET_NOTIONAL,
            "max_symbol_weight_per_decision": 0.01,
            "max_symbol_notional_cny": MAX_SYMBOL_NOTIONAL,
            "turnover_diff_10t_participation": TURNOVER_PARTICIPATION,
            "displayed_ask_depth_participation": DEPTH_PARTICIPATION,
            "important": "Fixed Top100 only: no refill from rank 101 onward.",
        },
        "models": {},
    }
    for model, top in tops.items():
        enriched = _add_capacity(top.merge(raw, on=common.KEYS, how="left", validate="one_to_one"))
        result["models"][model] = _model_summary(enriched)

    output.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    turnover.to_parquet(turnover_path, index=False)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
