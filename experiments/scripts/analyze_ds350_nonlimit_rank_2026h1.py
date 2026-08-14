from __future__ import annotations

import json

import analyze_ds350_2026h1_loss_compare as loss_compare
import analyze_ds350_2026h1_model_tradeability_compare as common
import numpy as np
import pandas as pd

from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

RUNS = {
    "baseline_mixed_mse": "nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1",
    "baseline_mixed_huber": "nn_ds350_w0931_hclose_huber_train2023_2025_test2026h1_purge1_v1",
    "baseline_mixed_huber80_mse20": "nn_ds350_w0931_hclose_huber80_mse20_train2023_2025_test2026h1_purge1_v1",
    "close_z_all_mse": "nn_ds350_w0931_hclose_closez_mse_train2023_2025_test2026h1_purge1_v1",
    "mixed_nonup_mse": "nn_ds350_w0931_hclose_mixed_nonup_mse_train2023_2025_test2026h1_purge1_v1",
    "mixed_ordinary_mse": "nn_ds350_w0931_hclose_mixed_ordinary_mse_train2023_2025_test2026h1_purge1_v1",
    "rank_mixed_all_mse": "nn_ds350_w0931_hclose_rankmixed_mse_train2023_2025_test2026h1_purge1_v1",
    "rank_close_all_mse": "nn_ds350_w0931_hclose_rankclose_mse_train2023_2025_test2026h1_purge1_v1",
    "rank_close_nonup_mse": "nn_ds350_w0931_hclose_rankclose_nonup_mse_train2023_2025_test2026h1_purge1_v1",
    "rank_close_ordinary_mse": "nn_ds350_w0931_hclose_rankclose_ordinary_mse_train2023_2025_test2026h1_purge1_v1",
}

OUTCOMES = {
    "same_day_close": "return_close",
    "entry_to_next_close": "entry_to_next_close",
    "close_to_next_open": "close_to_next_open",
    "next_open_to_next_close": "next_open_to_next_close",
    "close_to_next_close": "close_to_next_close",
}


def _prediction(run_id: str) -> pd.DataFrame:
    path = common.ROOT / "nn/holdout" / run_id / "predictions_unfiltered.parquet"
    if not path.exists():
        raise SystemExit(f"missing predictions: {path}")
    frame = common._normalize(
        pd.read_parquet(path, columns=[*common.KEYS, "prediction", "label_short"])
    )
    frame = frame.loc[~frame["date"].isin(common.EXCLUDED_DATES)].copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["own_label"] = pd.to_numeric(frame.pop("label_short"), errors="coerce")
    return frame.dropna(subset=[*common.KEYS, "prediction"])


def _next_close_labels() -> pd.DataFrame:
    path = common.ROOT / "datasets/opening_0931_0940_labels_hclose_v1/year=2026/labels.parquet"
    frame = common._normalize(pd.read_parquet(path, columns=[*common.KEYS, "label_next_close"]))
    frame["entry_to_next_close"] = pd.to_numeric(frame.pop("label_next_close"), errors="coerce")
    return frame.drop_duplicates(common.KEYS, keep="last")


def _limit_states() -> pd.DataFrame:
    path = common.ROOT / "cache/opening_0931_0940_raw_source/year=2026/daily_reference.parquet"
    frame = pd.read_parquet(path, columns=["TradingDay", "Symbol", "UpdownLimitStatus"]).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "UpdownLimitStatus": "limit_state",
        }
    )
    frame["date"] = common._date(frame["date"])
    frame["symbol"] = common._text(frame["symbol"])
    frame["limit_state"] = pd.to_numeric(frame["limit_state"], errors="coerce")
    return frame.drop_duplicates(["date", "symbol"], keep="last")


def _auc(frame: pd.DataFrame) -> float:
    work = frame.dropna(subset=["prediction", "limit_state"]).copy()
    positive = work["limit_state"].eq(1)
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if not positives or not negatives:
        return float("nan")
    rank_sum = float(work["prediction"].rank(method="average").loc[positive].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _daily_mean_t(values: pd.Series) -> dict[str, float]:
    daily = values.groupby(level="date").mean().dropna()
    standard_error = daily.std(ddof=1) / np.sqrt(len(daily))
    return {
        "dates": int(len(daily)),
        "mean_bps": float(daily.mean()),
        "t_stat": float(daily.mean() / standard_error) if standard_error > 0 else np.nan,
        "positive_day_pct": float(daily.gt(0).mean() * 100.0),
    }


def _outcome_metrics(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    outcome: str,
) -> dict[str, object]:
    candidate = frame.dropna(subset=[outcome])
    picked = selected.dropna(subset=[outcome])
    base_count = candidate.groupby(common.GROUPS, sort=False)[outcome].size()
    pick_count = picked.groupby(common.GROUPS, sort=False)[outcome].size()
    base_mean = candidate.groupby(common.GROUPS, sort=False)[outcome].mean()
    pick_mean = picked.groupby(common.GROUPS, sort=False)[outcome].mean()
    excess = (pick_mean - base_mean).dropna() * 10_000.0
    result: dict[str, object] = {
        "rank_ic": common._rank_ic(frame, outcome),
        "selected_raw_bps": float(pick_mean.mean() * 10_000.0),
        "universe_raw_bps": float(base_mean.mean() * 10_000.0),
        "excess_bps": float(excess.mean()),
        "daily_excess": _daily_mean_t(excess),
    }
    contribution_sum = 0.0
    for state, name in ((1, "up_limit"), (-1, "down_limit"), (0, "ordinary")):
        base_sum = (
            candidate.loc[candidate["limit_state"].eq(state)]
            .groupby(common.GROUPS, sort=False)[outcome]
            .sum()
        )
        pick_sum = (
            picked.loc[picked["limit_state"].eq(state)]
            .groupby(common.GROUPS, sort=False)[outcome]
            .sum()
        )
        contribution = (
            pick_sum.reindex(excess.index).fillna(0.0) / pick_count.reindex(excess.index)
            - base_sum.reindex(excess.index).fillna(0.0) / base_count.reindex(excess.index)
        ) * 10_000.0
        mean_contribution = float(contribution.mean())
        result[f"{name}_contribution_bps"] = mean_contribution
        contribution_sum += mean_contribution
    result["contribution_sum_bps"] = contribution_sum

    for state, name in ((0, "ordinary"), (-1, "down_limit")):
        subset = frame.loc[frame["limit_state"].eq(state)].copy()
        if subset.empty:
            continue
        subset_top = common._top(subset).dropna(subset=[outcome])
        subset_base = (
            subset.dropna(subset=[outcome]).groupby(common.GROUPS, sort=False)[outcome].mean()
        )
        subset_pick = subset_top.groupby(common.GROUPS, sort=False)[outcome].mean()
        result[f"{name}_rank_ic"] = common._rank_ic(subset, outcome)
        result[f"{name}_reselect_excess_bps"] = float(
            (subset_pick - subset_base).dropna().mean() * 10_000.0
        )
    return result


def _evaluate(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    selected = common._top(frame)
    result: dict[str, object] = {
        "groups": int(frame.groupby(common.GROUPS, sort=False).ngroups),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "candidate_up_limit_pct": float(frame["limit_state"].eq(1).mean() * 100.0),
        "selected_up_limit_pct": float(selected["limit_state"].eq(1).mean() * 100.0),
        "candidate_down_limit_pct": float(frame["limit_state"].eq(-1).mean() * 100.0),
        "selected_down_limit_pct": float(selected["limit_state"].eq(-1).mean() * 100.0),
        "final_up_limit_auc": _auc(frame),
        "outcomes": {},
    }
    base_up = float(frame["limit_state"].eq(1).mean())
    result["up_limit_enrichment_x"] = (
        float(selected["limit_state"].eq(1).mean()) / base_up if base_up else np.nan
    )
    for name, outcome in OUTCOMES.items():
        result["outcomes"][name] = _outcome_metrics(frame, selected, outcome)
    return result, selected


def _overlap(left: pd.DataFrame, right: pd.DataFrame) -> float:
    by_group = (
        left[common.KEYS]
        .merge(right[common.KEYS], on=common.KEYS, how="inner")
        .groupby(common.GROUPS, sort=False)
        .size()
    )
    return float(by_group.mean())


def main() -> None:
    reference = common._market_reference()
    close_labels = common._close_labels()
    next_close_labels = _next_close_labels()
    limit_states = _limit_states()
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])

    output: dict[str, object] = {
        "status": "ok",
        "sample": "strict 2026H1; train 2023-2025; purge one session; excluded known bad dates",
        "selection": "Top100 per decision cross-section",
        "runs": {},
        "top100_overlap_with_baseline": {"all_a": {}, "pool_l": {}},
    }
    selected_by_universe: dict[str, dict[str, pd.DataFrame]] = {
        "all_a": {},
        "pool_l": {},
    }
    for name, run_id in RUNS.items():
        frame = loss_compare._attach_outcomes(_prediction(run_id), close_labels, reference)
        frame = frame.merge(next_close_labels, on=common.KEYS, how="left", validate="one_to_one")
        frame = frame.merge(limit_states, on=["date", "symbol"], how="left", validate="many_to_one")
        frame = frame.dropna(subset=["limit_state"])
        universes = {
            "all_a": frame,
            "pool_l": frame.loc[
                stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
            ].copy(),
        }
        output["runs"][name] = {"run_id": run_id, "universes": {}}
        for universe, universe_frame in universes.items():
            metrics, selected = _evaluate(universe_frame)
            output["runs"][name]["universes"][universe] = metrics
            selected_by_universe[universe][name] = selected
        print(json.dumps({"completed_run": name, "run_id": run_id}))

    for universe in ("all_a", "pool_l"):
        baseline = selected_by_universe[universe]["baseline_mixed_mse"]
        for name, selected in selected_by_universe[universe].items():
            output["top100_overlap_with_baseline"][universe][name] = _overlap(baseline, selected)

    destination = common.ROOT / "audits/ds350_2026h1_nonlimit_rank_training_v1"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "_SUCCESS").touch()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
