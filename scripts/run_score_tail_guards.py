from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.labels import normalize_return_label_frame


DEFAULT_INPUT = (
    "output/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/"
    "predictions_all.parquet"
)
DEFAULT_NEXT_CLOSE_LABELS = (
    "output/reports/lgbm_delay2_postopen_0931_0940_baseline_v1_four_panel/"
    "clickhouse_next_close_labels.parquet"
)
DEFAULT_OUTPUT_DIR = "output/reports/lgbm_delay2_postopen_tail_guards_v1"
KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
BASE_COLUMNS = (*KEY_COLUMNS, "prediction", "label")
GUARD_COLUMNS = (
    "spread_bps",
    "turnover",
    "turnover_diff_10t",
    "turnover_diff_30t",
    "return_10t",
    "return_30t",
    "ask_depth_10",
    "depth_imbalance_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep visible-information TopN tail guards over an existing score file "
            "and evaluate short/next-close TopN outcomes."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--next-close-label-input", default=DEFAULT_NEXT_CLOSE_LABELS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--start-clock", default="09:31")
    parser.add_argument("--end-clock", default="09:40")
    return parser.parse_args()


def clock_range(start: str, end: str) -> list[str]:
    start_ts = pd.Timestamp(f"2000-01-01 {start}")
    end_ts = pd.Timestamp(f"2000-01-01 {end}")
    if end_ts < start_ts:
        raise SystemExit("--end-clock must be >= --start-clock")
    clocks = []
    current = start_ts
    while current <= end_ts:
        clocks.append(current.strftime("%H:%M"))
        current += pd.Timedelta(minutes=1)
    return clocks


def read_frame(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def existing_columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def load_predictions(path: Path, clocks: list[str], score_col: str) -> pd.DataFrame:
    available = existing_columns(path)
    requested = [column for column in (*BASE_COLUMNS, *GUARD_COLUMNS) if column in available]
    missing = [column for column in (*BASE_COLUMNS, score_col) if column not in available]
    if missing:
        raise SystemExit(f"prediction input missing columns: {missing}")
    if score_col not in requested:
        requested.append(score_col)

    frame = read_frame(path, columns=requested)
    frame = frame.dropna(subset=["date", "symbol", "decision_target_timestamp", score_col, "label"]).copy()
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["decision_target_timestamp"] = pd.to_datetime(
        frame["decision_target_timestamp"],
        errors="coerce",
    )
    frame["clock"] = frame["decision_target_timestamp"].dt.strftime("%H:%M")
    frame = frame.loc[frame["clock"].isin(clocks)].copy()
    if score_col != "prediction":
        frame["prediction"] = pd.to_numeric(frame[score_col], errors="coerce")
    return frame.dropna(subset=["decision_target_timestamp", "prediction"])


def load_next_close_labels(path: Path) -> pd.DataFrame:
    required = [*KEY_COLUMNS, "alpha_return_next_close"]
    labels = read_frame(path, columns=required)
    return normalize_return_label_frame(
        labels,
        key_columns=KEY_COLUMNS,
        label_col="alpha_return_next_close",
    )


def add_group_ranks(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    group_cols = ["date", "decision_target_timestamp"]
    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{column}_rp"] = values.groupby([out[col] for col in group_cols]).rank(
            pct=True,
            method="average",
        )
    return out


def between(column: str, low: float, high: float) -> Callable[[pd.DataFrame], pd.Series]:
    rank_col = f"{column}_rp"

    def _mask(frame: pd.DataFrame) -> pd.Series:
        return frame[rank_col].between(low, high, inclusive="both")

    return _mask


def ge(column: str, value: float) -> Callable[[pd.DataFrame], pd.Series]:
    rank_col = f"{column}_rp"

    def _mask(frame: pd.DataFrame) -> pd.Series:
        return frame[rank_col] >= value

    return _mask


def le(column: str, value: float) -> Callable[[pd.DataFrame], pd.Series]:
    rank_col = f"{column}_rp"

    def _mask(frame: pd.DataFrame) -> pd.Series:
        return frame[rank_col] <= value

    return _mask


def combine_masks(
    *parts: Callable[[pd.DataFrame], pd.Series],
) -> Callable[[pd.DataFrame], pd.Series]:
    def _mask(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for part in parts:
            mask &= part(frame).fillna(False)
        return mask

    return _mask


def guard_variants() -> dict[str, Callable[[pd.DataFrame], pd.Series] | None]:
    return {
        "baseline": None,
        "spread_le_p80": le("spread_bps", 0.80),
        "spread_le_p70": le("spread_bps", 0.70),
        "spread_le_p60": le("spread_bps", 0.60),
        "liquid_spread_p80_tdiff30_ge_p50": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover_diff_30t", 0.50),
        ),
        "liquid_spread_p70_tdiff30_ge_p50": combine_masks(
            le("spread_bps", 0.70),
            ge("turnover_diff_30t", 0.50),
        ),
        "liquid_spread_p80_turnover_ge_p50": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover", 0.50),
        ),
        "liquid_spread_no_chase": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover_diff_30t", 0.50),
            le("return_30t", 0.80),
        ),
        "liquid_spread_no_chase_10t": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover_diff_10t", 0.50),
            le("return_10t", 0.80),
        ),
        "liquid_spread_no_chase_strict": combine_masks(
            le("spread_bps", 0.70),
            ge("turnover_diff_30t", 0.50),
            le("return_30t", 0.70),
        ),
        "mid_heat": combine_masks(
            le("spread_bps", 0.80),
            between("turnover_diff_30t", 0.30, 0.90),
            between("return_30t", 0.20, 0.80),
        ),
        "mid_heat_10t": combine_masks(
            le("spread_bps", 0.80),
            between("turnover_diff_10t", 0.30, 0.90),
            between("return_10t", 0.20, 0.80),
        ),
        "next_flip_guard_10t": combine_masks(
            le("spread_bps", 0.80),
            between("turnover_diff_10t", 0.10, 0.80),
            between("return_10t", 0.20, 0.70),
            ge("ask_depth_10", 0.40),
            between("depth_imbalance_10", 0.20, 0.70),
        ),
        "next_flip_guard_10t_robust": combine_masks(
            le("spread_bps", 0.80),
            between("turnover_diff_10t", 0.30, 0.80),
            between("return_10t", 0.20, 0.70),
            ge("ask_depth_10", 0.40),
            between("depth_imbalance_10", 0.30, 0.70),
        ),
        "depth_balanced": combine_masks(
            le("spread_bps", 0.80),
            ge("ask_depth_10", 0.30),
            between("depth_imbalance_10", 0.30, 0.70),
        ),
        "all_guards_soft": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover_diff_30t", 0.40),
            le("return_30t", 0.80),
            ge("ask_depth_10", 0.30),
            between("depth_imbalance_10", 0.25, 0.75),
        ),
        "all_guards_soft_10t": combine_masks(
            le("spread_bps", 0.80),
            ge("turnover_diff_10t", 0.40),
            le("return_10t", 0.80),
            ge("ask_depth_10", 0.30),
            between("depth_imbalance_10", 0.25, 0.75),
        ),
        "all_guards_strict": combine_masks(
            le("spread_bps", 0.70),
            ge("turnover_diff_30t", 0.50),
            le("return_30t", 0.70),
            ge("ask_depth_10", 0.40),
            between("depth_imbalance_10", 0.30, 0.70),
        ),
    }


def variant_required_rank_columns(mask_fn: Callable[[pd.DataFrame], pd.Series] | None) -> set[str]:
    if mask_fn is None:
        return set()
    names = set(getattr(mask_fn, "__code__", ()).co_names)
    return {name for name in names if name.endswith("_rp")}


def summarize_variant(
    frame: pd.DataFrame,
    *,
    variant: str,
    mask_fn: Callable[[pd.DataFrame], pd.Series] | None,
    top_n: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for (date, timestamp), group in frame.groupby(
        ["date", "decision_target_timestamp"],
        sort=True,
    ):
        candidates = group if mask_fn is None else group.loc[mask_fn(group)]
        selected = candidates.sort_values("prediction", ascending=False).head(top_n)
        all_short = float(group["label"].mean())
        all_next = float(group["alpha_return_next_close"].mean())
        short_mean = float(selected["label"].mean()) if len(selected) else float("nan")
        next_mean = (
            float(selected["alpha_return_next_close"].mean())
            if len(selected)
            else float("nan")
        )
        rows.append(
            {
                "variant": variant,
                "date": str(date),
                "decision_target_timestamp": pd.Timestamp(timestamp),
                "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                "rows": int(len(group)),
                "candidate_rows": int(len(candidates)),
                "selected_rows": int(len(selected)),
                "short_all_mean_bps": all_short * 10_000.0,
                "short_top_mean_bps": short_mean * 10_000.0,
                "short_top_excess_bps": (short_mean - all_short) * 10_000.0,
                "short_top_win_rate": float((selected["label"] > 0).mean())
                if len(selected)
                else float("nan"),
                "next_all_mean_bps": all_next * 10_000.0,
                "next_top_mean_bps": next_mean * 10_000.0,
                "next_top_excess_bps": (next_mean - all_next) * 10_000.0,
                "next_top_win_rate": float(
                    (selected["alpha_return_next_close"] > 0).mean()
                )
                if len(selected)
                else float("nan"),
            }
        )

    group_metrics = pd.DataFrame(rows)
    minute = minute_summary(group_metrics)
    best = {
        "variant": variant,
        "groups": int(group_metrics["short_top_excess_bps"].notna().sum()),
        "top_n": int(top_n),
        "avg_candidate_rows": float(group_metrics["candidate_rows"].mean()),
        "avg_selected_rows": float(group_metrics["selected_rows"].mean()),
        "full_top_n_rate": float((group_metrics["selected_rows"] >= top_n).mean()),
        "short_top_mean_bps": float(group_metrics["short_top_mean_bps"].mean()),
        "short_top_excess_bps": float(group_metrics["short_top_excess_bps"].mean()),
        "short_top_win_rate": float(group_metrics["short_top_win_rate"].mean()),
        "next_top_mean_bps": float(group_metrics["next_top_mean_bps"].mean()),
        "next_top_excess_bps": float(group_metrics["next_top_excess_bps"].mean()),
        "next_top_win_rate": float(group_metrics["next_top_win_rate"].mean()),
        "next_excess_positive_rate": float(
            (group_metrics["next_top_excess_bps"] > 0).mean()
        ),
        "next_mean_positive_rate": float((group_metrics["next_top_mean_bps"] > 0).mean()),
        "next_positive_minute_count": int((minute["next_top_excess_bps"] > 0).sum()),
        "next_min_minute_excess_bps": float(minute["next_top_excess_bps"].min()),
        "all_minutes_next_excess_positive": bool(
            (minute["next_top_excess_bps"] > 0).all()
        ),
    }
    return rows, best


def minute_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        group_metrics.groupby(["variant", "clock"], as_index=False)
        .agg(
            groups=("date", "size"),
            rows=("rows", "sum"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            short_top_win_rate=("short_top_win_rate", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_top_win_rate=("next_top_win_rate", "mean"),
        )
        .sort_values(["variant", "clock"])
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clocks = clock_range(args.start_clock, args.end_clock)

    predictions = load_predictions(Path(args.input), clocks, args.score_col)
    next_labels = load_next_close_labels(Path(args.next_close_label_input))
    frame = predictions.merge(next_labels, on=list(KEY_COLUMNS), how="inner")
    frame = frame.dropna(subset=["label", "prediction", "alpha_return_next_close"]).copy()
    frame = add_group_ranks(frame, GUARD_COLUMNS)

    rank_columns = set(frame.columns)
    group_rows: list[dict[str, object]] = []
    summary_rows = []
    skipped = []
    for variant, mask_fn in guard_variants().items():
        try:
            rows, summary = summarize_variant(
                frame,
                variant=variant,
                mask_fn=mask_fn,
                top_n=int(args.top_n),
            )
        except KeyError as exc:
            skipped.append({"variant": variant, "missing": str(exc)})
            continue
        missing_rank_columns = sorted(variant_required_rank_columns(mask_fn) - rank_columns)
        if missing_rank_columns:
            skipped.append({"variant": variant, "missing": ",".join(missing_rank_columns)})
            continue
        group_rows.extend(rows)
        summary_rows.append(summary)

    group_metrics = pd.DataFrame(group_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["next_top_excess_bps", "short_top_excess_bps"],
        ascending=[False, False],
    )
    minutes = minute_summary(group_metrics)

    group_metrics.to_csv(output_dir / "tail_guard_group_metrics.csv", index=False)
    minutes.to_csv(output_dir / "tail_guard_minute_summary.csv", index=False)
    summary.to_csv(output_dir / "tail_guard_summary.csv", index=False)
    trace = {
        "input": args.input,
        "next_close_label_input": args.next_close_label_input,
        "output_dir": str(output_dir),
        "top_n": int(args.top_n),
        "clocks": clocks,
        "rows": int(len(frame)),
        "groups": int(
            frame[["date", "decision_target_timestamp"]].drop_duplicates().shape[0]
        ),
        "guard_columns": [column for column in GUARD_COLUMNS if column in frame.columns],
        "skipped": skipped,
        "best_by_next_excess": summary.head(5).to_dict(orient="records"),
    }
    (output_dir / "tail_guard_trace.json").write_text(
        json.dumps(json_safe(trace), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("tail_guard_summary")
    print(
        summary[
            [
                "variant",
                "avg_selected_rows",
                "short_top_excess_bps",
                "next_top_excess_bps",
                "next_positive_minute_count",
                "next_min_minute_excess_bps",
                "all_minutes_next_excess_positive",
            ]
        ]
        .head(20)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    if skipped:
        print("\nskipped")
        print(pd.DataFrame(skipped).to_string(index=False))
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
