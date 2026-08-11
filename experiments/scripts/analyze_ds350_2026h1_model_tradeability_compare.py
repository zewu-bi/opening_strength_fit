from __future__ import annotations

import json

import pandas as pd

from opening_strength_fit import ds350_holdout_analysis as common
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)


def main() -> None:
    reference = common._market_reference()
    close_labels = common._close_labels()
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    frames: dict[str, pd.DataFrame] = {}
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for model in ("1m", "close"):
        frame = common.attach_market_outcomes(
            common._prediction(model),
            close_labels,
            reference,
        )
        frames[model] = frame
        selected[(model, "all_a")] = common._top(frame)
        in_pool = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
        if bool(in_pool.any()):
            selected[(model, "pool_L")] = common._top(frame.loc[in_pool].copy())

    final_keys = pd.concat(
        [part.loc[part["final_up_limit"], common.KEYS] for part in selected.values()],
        ignore_index=True,
    ).drop_duplicates(common.KEYS, keep="last")
    output = common.ROOT / "audits/ds350_2026h1_model_tradeability_compare_v1"
    raw_cache = output / "raw_final_limit_entries.parquet"
    raw_entries = (
        pd.read_parquet(raw_cache) if raw_cache.exists() else common._raw_entries(final_keys)
    )

    result: dict[str, object] = {
        "status": "ok",
        "sample": "strict 2026H1; train 2023-2025; purge one session",
        "selection": "Top100 selected causally from unfiltered predictions before outcome availability",
        "models": {},
        "top100_overlap_pct": {},
    }
    for scope in ("all_a", "pool_L"):
        if ("1m", scope) not in selected or ("close", scope) not in selected:
            result["top100_overlap_pct"][scope] = None
            result["models"][f"scope_{scope}_status"] = (
                "unavailable: stock-pool membership has no rows for strict 2026H1"
            )
            continue
        left = selected[("1m", scope)][common.GROUPS + ["symbol"]]
        right = selected[("close", scope)][common.GROUPS + ["symbol"]]
        overlap = (
            left.merge(right, on=common.GROUPS + ["symbol"], how="inner")
            .groupby(common.GROUPS)
            .size()
        )
        result["top100_overlap_pct"][scope] = float(overlap.mean())
        for model in ("1m", "close"):
            frame = frames[model]
            if scope == "pool_L":
                frame = frame.loc[
                    stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
                ].copy()
            top = selected[(model, scope)].merge(
                raw_entries,
                on=common.KEYS,
                how="left",
                validate="one_to_one",
            )
            result["models"][f"{model}_{scope}"] = {
                "performance": common._metrics(frame, top),
                "selected_final_limit_tradeability": common._tradeability(top),
            }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw_entries.to_parquet(output / "raw_final_limit_entries.parquet", index=False)
    (output / "_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
