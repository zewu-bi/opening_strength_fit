from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    clock_range,
    load_next_close_label_file,
    mean_aggregations,
    selection_group_metrics,
    write_json,
)
from opening_strength_fit.next_close_labels import (
    add_next_close_label_arguments,
    load_or_fetch_next_close_labels_from_args,
)
from opening_strength_fit.prediction_frames import read_clock_predictions
from opening_strength_fit.schema import DECISION_KEY_COLUMNS

DEFAULT_INPUT = (
    "output/legacy/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/predictions_all.parquet"
)
DEFAULT_OUTPUT_DIR = "output/legacy/reports/lgbm_delay2_postopen_tail_guards_v1"
KEY_COLUMNS = DECISION_KEY_COLUMNS
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--start-clock", default="09:31")
    parser.add_argument("--end-clock", default="09:40")
    add_next_close_label_arguments(parser, include_connection=True)
    return parser.parse_args()


def load_predictions(path: Path, clocks: list[str], score_col: str) -> pd.DataFrame:
    trailing = () if score_col in BASE_COLUMNS else (score_col,)
    frame = read_clock_predictions(
        path,
        required_columns=BASE_COLUMNS,
        optional_columns=("buy_price", *GUARD_COLUMNS),
        trailing_required_columns=trailing,
        dropna_columns=(*KEY_COLUMNS, score_col, "label"),
        clocks=clocks,
        context="prediction input",
    )
    if score_col != "prediction":
        frame["prediction"] = pd.to_numeric(frame[score_col], errors="coerce")
    return frame.dropna(subset=["decision_target_timestamp", "prediction"])


def load_next_close_labels(path: Path) -> pd.DataFrame:
    return load_next_close_label_file(path, key_columns=KEY_COLUMNS)


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


def guard_mask(
    bounds: tuple[tuple[str, float, float], ...],
) -> Callable[[pd.DataFrame], pd.Series]:
    def _mask(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for column, low, high in bounds:
            mask &= frame[f"{column}_rp"].between(low, high, inclusive="both").fillna(False)
        return mask

    return _mask


def guard_variants() -> dict[str, Callable[[pd.DataFrame], pd.Series] | None]:
    inf = float("inf")
    specs = {
        "spread_le_p80": "spread_bps::0.8",
        "spread_le_p70": "spread_bps::0.7",
        "spread_le_p60": "spread_bps::0.6",
        "liquid_spread_p80_tdiff30_ge_p50": "spread_bps::0.8 turnover_diff_30t:0.5:",
        "liquid_spread_p70_tdiff30_ge_p50": "spread_bps::0.7 turnover_diff_30t:0.5:",
        "liquid_spread_p80_turnover_ge_p50": "spread_bps::0.8 turnover:0.5:",
        "liquid_spread_no_chase": "spread_bps::0.8 turnover_diff_30t:0.5: return_30t::0.8",
        "liquid_spread_no_chase_10t": "spread_bps::0.8 turnover_diff_10t:0.5: return_10t::0.8",
        "liquid_spread_no_chase_strict": "spread_bps::0.7 turnover_diff_30t:0.5: return_30t::0.7",
        "mid_heat": "spread_bps::0.8 turnover_diff_30t:0.3:0.9 return_30t:0.2:0.8",
        "mid_heat_10t": "spread_bps::0.8 turnover_diff_10t:0.3:0.9 return_10t:0.2:0.8",
        "next_flip_guard_10t": "spread_bps::0.8 turnover_diff_10t:0.1:0.8 return_10t:0.2:0.7 ask_depth_10:0.4: depth_imbalance_10:0.2:0.7",
        "next_flip_guard_10t_robust": "spread_bps::0.8 turnover_diff_10t:0.3:0.8 return_10t:0.2:0.7 ask_depth_10:0.4: depth_imbalance_10:0.3:0.7",
        "depth_balanced": "spread_bps::0.8 ask_depth_10:0.3: depth_imbalance_10:0.3:0.7",
        "all_guards_soft": "spread_bps::0.8 turnover_diff_30t:0.4: return_30t::0.8 ask_depth_10:0.3: depth_imbalance_10:0.25:0.75",
        "all_guards_soft_10t": "spread_bps::0.8 turnover_diff_10t:0.4: return_10t::0.8 ask_depth_10:0.3: depth_imbalance_10:0.25:0.75",
        "all_guards_strict": "spread_bps::0.7 turnover_diff_30t:0.5: return_30t::0.7 ask_depth_10:0.4: depth_imbalance_10:0.3:0.7",
    }
    return {
        "baseline": None,
        **{
            name: guard_mask(
                tuple(
                    (column, float(low) if low else -inf, float(high) if high else inf)
                    for column, low, high in (bound.split(":") for bound in encoded.split())
                )
            )
            for name, encoded in specs.items()
        },
    }


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
        rows.append(
            {
                "variant": variant,
                **selection_group_metrics(
                    group,
                    selected,
                    date=date,
                    timestamp=timestamp,
                    candidate_counts={"candidate_rows": int(len(candidates))},
                ),
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
        "next_excess_positive_rate": float((group_metrics["next_top_excess_bps"] > 0).mean()),
        "next_mean_positive_rate": float((group_metrics["next_top_mean_bps"] > 0).mean()),
        "next_positive_minute_count": int((minute["next_top_excess_bps"] > 0).sum()),
        "next_min_minute_excess_bps": float(minute["next_top_excess_bps"].min()),
        "all_minutes_next_excess_positive": bool((minute["next_top_excess_bps"] > 0).all()),
    }
    return rows, best


def minute_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        group_metrics.groupby(["variant", "clock"], as_index=False)
        .agg(
            groups=("date", "size"),
            rows=("rows", "sum"),
            **mean_aggregations(
                *"candidate_rows selected_rows short_top_mean_bps short_top_excess_bps "
                "short_top_win_rate next_top_mean_bps next_top_excess_bps "
                "next_top_win_rate".split()
            ),
        )
        .sort_values(["variant", "clock"])
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clocks = clock_range(args.start_clock, args.end_clock)

    predictions = load_predictions(Path(args.input), clocks, args.score_col)
    next_labels = load_or_fetch_next_close_labels_from_args(
        predictions,
        args=args,
        output_dir=output_dir,
    )
    frame = predictions.merge(next_labels, on=list(KEY_COLUMNS), how="inner")
    frame = frame.dropna(subset=["label", "prediction", "alpha_return_next_close"]).copy()
    frame = add_group_ranks(frame, GUARD_COLUMNS)

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
        "groups": int(frame[["date", "decision_target_timestamp"]].drop_duplicates().shape[0]),
        "guard_columns": [column for column in GUARD_COLUMNS if column in frame.columns],
        "skipped": skipped,
        "best_by_next_excess": summary.head(5).to_dict(orient="records"),
    }
    write_json(output_dir / "tail_guard_trace.json", trace)

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
