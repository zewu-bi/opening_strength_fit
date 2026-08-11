from __future__ import annotations

import json

import numpy as np
import pandas as pd

from opening_strength_fit import ds350_holdout_analysis as common

RUNS = {
    "mse": "nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1",
    "huber": "nn_ds350_w0931_hclose_huber_train2023_2025_test2026h1_purge1_v1",
    "huber80_mse20": ("nn_ds350_w0931_hclose_huber80_mse20_train2023_2025_test2026h1_purge1_v1"),
}


def _prediction(run_id: str) -> pd.DataFrame:
    return common.load_prediction(run_id)


def _daily_robustness(frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float]:
    candidate = (
        frame.dropna(subset=["return_close"])
        .groupby(common.GROUPS, sort=False)["return_close"]
        .mean()
    )
    picked = (
        selected.dropna(subset=["return_close"])
        .groupby(common.GROUPS, sort=False)["return_close"]
        .mean()
    )
    group_excess = (picked - candidate).dropna() * 10_000.0
    daily = group_excess.groupby(level="date").mean()
    standard_error = daily.std(ddof=1) / np.sqrt(len(daily))
    return {
        "dates": int(len(daily)),
        "daily_mean_excess_bps": float(daily.mean()),
        "daily_t_stat": float(daily.mean() / standard_error) if standard_error > 0 else np.nan,
        "positive_day_pct": float(daily.gt(0).mean() * 100.0),
    }


def main() -> None:
    reference = common._market_reference()
    close_labels = common._close_labels()
    frames: dict[str, pd.DataFrame] = {}
    selected: dict[str, pd.DataFrame] = {}
    for name, run_id in RUNS.items():
        frames[name] = common.attach_market_outcomes(
            _prediction(run_id),
            close_labels,
            reference,
        )
        selected[name] = common._top(frames[name])

    final_keys = pd.concat(
        [top.loc[top["final_up_limit"], common.KEYS] for top in selected.values()],
        ignore_index=True,
    ).drop_duplicates(common.KEYS, keep="last")
    output = common.ROOT / "audits/ds350_2026h1_close_loss_compare_v1"
    raw_cache = output / "raw_final_limit_entries.parquet"
    raw_entries = (
        pd.read_parquet(raw_cache) if raw_cache.exists() else common._raw_entries(final_keys)
    )

    result: dict[str, object] = {
        "status": "ok",
        "sample": "strict 2026H1 all-A; train 2023-2025; purge one session",
        "controlled_change": "loss function only; same features, labels, architecture, seed, and split",
        "runs": {},
        "top100_overlap_with_mse_pct": {},
    }
    mse_keys = selected["mse"][common.KEYS]
    for name, frame in frames.items():
        top = selected[name].merge(raw_entries, on=common.KEYS, how="left", validate="one_to_one")
        metrics = common._metrics(frame, top)
        total = float(metrics["same_day_close_excess_bps"])
        limit_part = float(metrics["same_day_close_limit_contribution_bps"])
        metrics["limit_contribution_over_total_pct"] = (
            limit_part / total * 100.0 if abs(total) > 1e-12 else np.nan
        )
        result["runs"][name] = {
            "run_id": RUNS[name],
            "performance": metrics,
            "daily_robustness": _daily_robustness(frame, top),
            "selected_final_limit_tradeability": common._tradeability(top),
        }
        overlap = (
            mse_keys.merge(top[common.KEYS], on=common.KEYS, how="inner")
            .groupby(common.GROUPS, sort=False)
            .size()
        )
        result["top100_overlap_with_mse_pct"][name] = float(overlap.mean())

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw_entries.to_parquet(raw_cache, index=False)
    (output / "_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
