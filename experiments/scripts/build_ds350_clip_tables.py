from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.io import read_frame
from opening_strength_fit.prediction_frames import normalize_keys, prediction_files
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

GROUP_COLUMNS = ["date", "decision_target_timestamp"]
OUTCOMES = ("own_label", "same_day_close", "next_close")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build comparable Pool-L Top100 tables for DS350 experiments."
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--clip-1m-root", type=Path, required=True)
    parser.add_argument("--clip-close-root", type=Path, required=True)
    parser.add_argument("--ordinary-1m-root", type=Path)
    parser.add_argument("--ordinary-close-root", type=Path)
    parser.add_argument("--pure-short-1m-root", type=Path)
    parser.add_argument("--pure-short-close-root", type=Path)
    parser.add_argument("--close-label-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def prediction_frame(root: Path) -> pd.DataFrame:
    frames = []
    for path in prediction_files(root):
        frame = read_frame(path, columns=[*KEY_COLUMNS, "prediction", "label"])
        frames.append(frame)
        print(f"read predictions path={path} rows={len(frame)}", flush=True)
    out = normalize_keys(pd.concat(frames, ignore_index=True))
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["own_label"] = pd.to_numeric(out.pop("label"), errors="coerce")
    return out


def outcome_labels(root: Path, years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = root / f"year={year}" / "labels.parquet"
        frame = read_frame(
            path,
            columns=[*KEY_COLUMNS, "label_short", "label_next_close"],
        )
        frames.append(frame)
        print(f"read outcome labels path={path} rows={len(frame)}", flush=True)
    out = normalize_keys(pd.concat(frames, ignore_index=True))
    out = out.rename(
        columns={
            "label_short": "same_day_close",
            "label_next_close": "next_close",
        }
    )
    for column in ("same_day_close", "next_close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.drop_duplicates(list(KEY_COLUMNS), keep="last")


def limit_states(root: Path, years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = root / f"year={year}" / "daily_reference.parquet"
        frame = read_frame(
            path,
            columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
        ).rename(
            columns={
                "TradingDay": "date",
                "Symbol": "symbol",
                "UpdownLimitStatus": "limit_state",
            }
        )
        frames.append(frame)
        print(f"read limit states path={path} rows={len(frame)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = normalize_clickhouse_date(out["date"])
    out["symbol"] = decode_clickhouse_text(out["symbol"])
    out["limit_state"] = pd.to_numeric(out["limit_state"], errors="coerce")
    return out.drop_duplicates(["date", "symbol"], keep="last")


def finite_rank_ic(frame: pd.DataFrame, outcome: str) -> float:
    values = frame[["prediction", outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return np.nan
    if values["prediction"].nunique() < 2 or values[outcome].nunique() < 2:
        return np.nan
    return float(values["prediction"].corr(values[outcome], method="spearman"))


def outcome_metrics(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    outcome: str,
    *,
    compute_rank_ic: bool = True,
) -> dict[str, float]:
    candidate = frame.dropna(subset=[outcome])
    picked = selected.dropna(subset=[outcome])
    candidate_group = candidate.groupby(GROUP_COLUMNS, sort=False)
    picked_group = picked.groupby(GROUP_COLUMNS, sort=False)
    candidate_count = candidate_group[outcome].size()
    picked_count = picked_group[outcome].size()
    candidate_mean = candidate_group[outcome].mean()
    picked_mean = picked_group[outcome].mean()
    excess = (picked_mean - candidate_mean).dropna() * 10_000.0

    result = {
        "rank_ic": None,
        "excess_bps": float(excess.mean()),
    }
    if compute_rank_ic:
        rank_ic = candidate_group.apply(lambda item: finite_rank_ic(item, outcome))
        result["rank_ic"] = float(rank_ic.mean())
    contribution_sum = 0.0
    for is_final_limit, name in ((True, "final_limit"), (False, "non_final_limit")):
        candidate_mask = candidate["limit_state"].eq(1).eq(is_final_limit)
        picked_mask = picked["limit_state"].eq(1).eq(is_final_limit)
        candidate_sum = (
            candidate.loc[candidate_mask].groupby(GROUP_COLUMNS, sort=False)[outcome].sum()
        )
        picked_sum = picked.loc[picked_mask].groupby(GROUP_COLUMNS, sort=False)[outcome].sum()
        contribution = (
            picked_sum.reindex(excess.index).fillna(0.0) / picked_count.reindex(excess.index)
            - candidate_sum.reindex(excess.index).fillna(0.0)
            / candidate_count.reindex(excess.index)
        ) * 10_000.0
        value = float(contribution.mean())
        result[f"{name}_contribution_bps"] = value
        contribution_sum += value
    result["contribution_sum_bps"] = contribution_sum
    result["contribution_residual_bps"] = result["excess_bps"] - contribution_sum
    return result


def evaluate_model(
    root: Path,
    *,
    labels: pd.DataFrame,
    daily: pd.DataFrame,
    pool: pd.DataFrame,
    top_n: int,
    rank_ic_outcomes: tuple[str, ...] = OUTCOMES,
) -> dict[str, object]:
    frame = prediction_frame(root)
    pool_mask = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
    frame = frame.loc[pool_mask].copy()
    frame = frame.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    frame = frame.merge(daily, on=["date", "symbol"], how="left", validate="many_to_one")

    # One eligibility set keeps the selected Top100 identical across the three horizons.
    frame = frame.dropna(subset=["prediction", *OUTCOMES]).copy()
    frame["_score_rank"] = frame.groupby(GROUP_COLUMNS, sort=False)["prediction"].rank(
        ascending=False,
        method="first",
    )
    selected = frame.loc[frame["_score_rank"].le(top_n)].copy()
    metrics = {}
    for outcome in OUTCOMES:
        metrics[outcome] = outcome_metrics(
            frame,
            selected,
            outcome,
            compute_rank_ic=outcome in rank_ic_outcomes,
        )
    return {
        "root": str(root),
        "groups": int(frame.groupby(GROUP_COLUMNS, sort=False).ngroups),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "candidate_rows_mean": float(frame.groupby(GROUP_COLUMNS, sort=False).size().mean()),
        "selected_rows_mean": float(selected.groupby(GROUP_COLUMNS, sort=False).size().mean()),
        "candidate_final_limit_pct": float(frame["limit_state"].eq(1).mean() * 100.0),
        "selected_final_limit_pct": float(selected["limit_state"].eq(1).mean() * 100.0),
        "outcomes": metrics,
    }


def table_frames(
    results: dict[str, dict[str, object]],
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_1 = []
    rows_2 = []
    for label, key in (("1m", f"{prefix}_1m"), ("close", f"{prefix}_close")):
        outcomes = results[key]["outcomes"]
        own = outcomes["own_label"]
        close = outcomes["same_day_close"]
        next_close = outcomes["next_close"]
        rows_1.append(
            {
                "Label": label,
                "IC": own["rank_ic"],
                "Label对应超额": own["excess_bps"],
                "持有到收盘超额": close["excess_bps"],
                "次日收盘超额": next_close["excess_bps"],
            }
        )
        rows_2.append(
            {
                "Label": label,
                "Label总超额": own["excess_bps"],
                "最终涨停贡献": own["final_limit_contribution_bps"],
                "非最终涨停贡献": own["non_final_limit_contribution_bps"],
                "持有到收盘总超额": close["excess_bps"],
                "收盘最终涨停贡献": close["final_limit_contribution_bps"],
                "收盘非最终涨停贡献": close["non_final_limit_contribution_bps"],
            }
        )
    return pd.DataFrame(rows_1), pd.DataFrame(rows_2)


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    years = range(2022, 2026)
    labels = outcome_labels(args.close_label_root, years)
    daily = limit_states(args.raw_root, years)
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])

    roots = {
        "clip3_1m": args.clip_1m_root,
        "clip3_close": args.clip_close_root,
    }
    optional_experiments = (
        ("ordinary_train", args.ordinary_1m_root, args.ordinary_close_root),
        ("pure_short", args.pure_short_1m_root, args.pure_short_close_root),
    )
    for prefix, root_1m, root_close in optional_experiments:
        if (root_1m is None) != (root_close is None):
            raise SystemExit(f"{prefix} requires both 1m and close roots")
        if root_1m is not None:
            roots[f"{prefix}_1m"] = root_1m
            roots[f"{prefix}_close"] = root_close
    if args.baseline_root:
        roots.update(
            {
                "baseline_1m": args.baseline_root / "w0931_0940_h1m",
                "baseline_close": args.baseline_root / "w0931_0940_hclose",
            }
        )
    results = {}
    for name, root in roots.items():
        print(f"evaluate model={name} root={root}", flush=True)
        results[name] = evaluate_model(
            root,
            labels=labels,
            daily=daily,
            pool=pool,
            top_n=args.top_n,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for prefix in ("baseline", "clip3", "ordinary_train", "pure_short"):
        if f"{prefix}_1m" not in results:
            continue
        table_1, table_2 = table_frames(results, prefix)
        table_1.to_csv(args.output_dir / f"{prefix}_table_1.csv", index=False)
        table_2.to_csv(args.output_dir / f"{prefix}_table_2.csv", index=False)
        print(f"{prefix.upper()}_TABLE_1", flush=True)
        print(table_1.to_csv(index=False).strip(), flush=True)
        print(f"{prefix.upper()}_TABLE_2", flush=True)
        print(table_2.to_csv(index=False).strip(), flush=True)
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
