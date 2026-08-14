from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.io import read_frame
from opening_strength_fit.prediction_frames import normalize_keys
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

GROUP_COLUMNS = ["date", "decision_target_timestamp"]
VALUE_COLUMNS = [
    "raw_label",
    "next_close",
    "mixed_target",
    "clip3_target",
    "pure_target",
]
STATE_NAMES = {1: "final_up", 0: "ordinary", -1: "final_down"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DS350 label tails by final limit state.")
    parser.add_argument("--h1m-root", type=Path, required=True)
    parser.add_argument("--h1m-clip3-root", type=Path, required=True)
    parser.add_argument("--h1m-pure-root", type=Path, required=True)
    parser.add_argument("--hclose-root", type=Path, required=True)
    parser.add_argument("--hclose-clip3-root", type=Path, required=True)
    parser.add_argument("--hclose-pure-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-start", type=int, default=2022)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--sample-per-state-year", type=int, default=100_000)
    return parser.parse_args()


def load_states(root: Path, year: int) -> pd.DataFrame:
    frame = read_frame(
        root / f"year={year}" / "daily_reference.parquet",
        columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "UpdownLimitStatus": "limit_state",
        }
    )
    frame["date"] = normalize_clickhouse_date(frame["date"])
    frame["symbol"] = decode_clickhouse_text(frame["symbol"])
    frame["limit_state"] = pd.to_numeric(frame["limit_state"], errors="coerce")
    return frame.drop_duplicates(["date", "symbol"], keep="last")


def load_horizon(
    root: Path,
    clip3_root: Path,
    pure_root: Path,
    states: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    year_name = f"year={year}"
    base = normalize_keys(
        read_frame(
            root / year_name / "labels.parquet",
            columns=[*KEY_COLUMNS, "label_short", "label_next_close", "target_label"],
        )
    )
    clip3 = read_frame(clip3_root / year_name / "labels.parquet", columns=["target_label"])
    pure = read_frame(pure_root / year_name / "labels.parquet", columns=["target_label"])
    if len(base) != len(clip3) or len(base) != len(pure):
        raise SystemExit(
            f"variant row mismatch year={year}: "
            f"base={len(base)} clip3={len(clip3)} pure={len(pure)}"
        )
    out = base.loc[:, [*KEY_COLUMNS]].copy()
    out["raw_label"] = pd.to_numeric(base["label_short"], errors="coerce")
    out["next_close"] = pd.to_numeric(base["label_next_close"], errors="coerce")
    out["mixed_target"] = pd.to_numeric(base["target_label"], errors="coerce")
    out["clip3_target"] = pd.to_numeric(clip3["target_label"], errors="coerce").to_numpy()
    out["pure_target"] = pd.to_numeric(pure["target_label"], errors="coerce").to_numpy()
    out = out.merge(states, on=["date", "symbol"], how="left", validate="many_to_one")
    return out


def state_name(values: pd.Series) -> pd.Series:
    return values.map(STATE_NAMES).fillna("unknown")


def basic_records(frame: pd.DataFrame, horizon: str, year: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    names = state_name(frame["limit_state"])
    for variable in VALUE_COLUMNS:
        values = pd.to_numeric(frame[variable], errors="coerce")
        for state in ("final_up", "ordinary", "final_down", "unknown"):
            selected = values.loc[names.eq(state)].replace([np.inf, -np.inf], np.nan).dropna()
            if selected.empty:
                continue
            array = selected.to_numpy(dtype="float64", copy=False)
            records.append(
                {
                    "horizon": horizon,
                    "year": year,
                    "variable": variable,
                    "state": state,
                    "count": len(array),
                    "sum": float(array.sum()),
                    "sum_sq": float(np.square(array).sum()),
                    "sum_abs": float(np.abs(array).sum()),
                    "positive": int((array > 0.0).sum()),
                    "zero": int((array == 0.0).sum()),
                    "minimum": float(array.min()),
                    "maximum": float(array.max()),
                }
            )
    return records


def sampled_records(
    frame: pd.DataFrame,
    horizon: str,
    year: int,
    sample_per_state: int,
) -> list[pd.DataFrame]:
    samples = []
    names = state_name(frame["limit_state"])
    for offset, state in enumerate(("final_up", "ordinary", "final_down", "unknown")):
        subset = frame.loc[names.eq(state), VALUE_COLUMNS]
        if subset.empty:
            continue
        count = min(int(sample_per_state), len(subset))
        if count < len(subset):
            subset = subset.sample(n=count, random_state=year * 10 + offset)
        subset = subset.copy()
        subset["horizon"] = horizon
        subset["year"] = year
        subset["state"] = state
        samples.append(subset)
    return samples


def tail_records(frame: pd.DataFrame, horizon: str, year: int) -> list[dict[str, object]]:
    records = []
    for variable in ("raw_label", "mixed_target", "clip3_target", "pure_target"):
        valid = frame.dropna(subset=[variable]).copy()
        groups = valid.groupby(GROUP_COLUMNS, sort=False)[variable]
        rank = groups.rank(ascending=False, method="first")
        count = groups.transform("size")
        selections = {
            "top_1pct": rank.le(np.ceil(count * 0.01)),
            "top_3pct": rank.le(np.ceil(count * 0.03)),
            "top100": rank.le(100),
        }
        for selection, mask in selections.items():
            selected = valid.loc[mask]
            state = pd.to_numeric(selected["limit_state"], errors="coerce")
            records.append(
                {
                    "horizon": horizon,
                    "year": year,
                    "variable": variable,
                    "selection": selection,
                    "rows": len(selected),
                    "final_up_rows": int(state.eq(1).sum()),
                    "ordinary_rows": int(state.eq(0).sum()),
                    "final_down_rows": int(state.eq(-1).sum()),
                }
            )
    return records


def clipping_records(frame: pd.DataFrame, horizon: str, year: int) -> list[dict[str, object]]:
    short = pd.to_numeric(frame["raw_label"], errors="coerce")
    long = pd.to_numeric(frame["next_close"], errors="coerce")
    usable = short.notna() & long.notna()
    keys = [frame[column] for column in GROUP_COLUMNS]
    records = []
    names = state_name(frame["limit_state"])
    for component, values in (("raw_label", short), ("next_close", long)):
        valid = values.where(usable)
        grouped = valid.groupby(keys, sort=False)
        mean = grouped.transform("mean")
        square_mean = valid.pow(2).groupby(keys, sort=False).transform("mean")
        std = np.sqrt((square_mean - mean.pow(2)).clip(lower=0.0))
        clipped = usable & (values.sub(mean).abs() > 3.0 * std)
        for state in ("final_up", "ordinary", "final_down", "unknown"):
            state_valid = usable & names.eq(state)
            count = int(state_valid.sum())
            if count == 0:
                continue
            records.append(
                {
                    "horizon": horizon,
                    "year": year,
                    "component": component,
                    "state": state,
                    "valid_rows": count,
                    "clipped_rows": int((clipped & names.eq(state)).sum()),
                }
            )
    return records


def aggregate_distribution(yearly: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon", "variable", "state"]
    grouped = yearly.groupby(keys, as_index=False).agg(
        count=("count", "sum"),
        total=("sum", "sum"),
        sum_sq=("sum_sq", "sum"),
        sum_abs=("sum_abs", "sum"),
        positive=("positive", "sum"),
        zero=("zero", "sum"),
        minimum=("minimum", "min"),
        maximum=("maximum", "max"),
    )
    grouped["mean"] = grouped["total"] / grouped["count"]
    variance = grouped["sum_sq"] / grouped["count"] - grouped["mean"].pow(2)
    grouped["std"] = np.sqrt(variance.clip(lower=0.0))
    grouped["abs_mean"] = grouped["sum_abs"] / grouped["count"]
    grouped["positive_pct"] = grouped["positive"] / grouped["count"] * 100.0
    grouped["zero_pct"] = grouped["zero"] / grouped["count"] * 100.0
    grouped["row_pct"] = (
        grouped["count"]
        / grouped.groupby(["horizon", "variable"])["count"].transform("sum")
        * 100.0
    )
    grouped["square_mass_pct"] = (
        grouped["sum_sq"]
        / grouped.groupby(["horizon", "variable"])["sum_sq"].transform("sum")
        * 100.0
    )
    grouped["absolute_mass_pct"] = (
        grouped["sum_abs"]
        / grouped.groupby(["horizon", "variable"])["sum_abs"].transform("sum")
        * 100.0
    )

    quantile_rows = []
    quantiles = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
    for (horizon, state), sample in samples.groupby(["horizon", "state"], sort=False):
        for variable in VALUE_COLUMNS:
            values = pd.to_numeric(sample[variable], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            result = np.quantile(values, quantiles)
            row: dict[str, object] = {
                "horizon": horizon,
                "variable": variable,
                "state": state,
                "sample_rows": len(values),
            }
            for quantile, value in zip(quantiles, result, strict=True):
                row[f"q{quantile:g}"] = float(value)
            quantile_rows.append(row)
    quantile_frame = pd.DataFrame(quantile_rows)
    return grouped.merge(quantile_frame, on=keys, how="left")


def aggregate_tails(yearly: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon", "variable", "selection"]
    out = yearly.groupby(keys, as_index=False).agg(
        rows=("rows", "sum"),
        final_up_rows=("final_up_rows", "sum"),
        ordinary_rows=("ordinary_rows", "sum"),
        final_down_rows=("final_down_rows", "sum"),
    )
    for state in ("final_up", "ordinary", "final_down"):
        out[f"{state}_pct"] = out[f"{state}_rows"] / out["rows"] * 100.0
    return out


def separation_summary(samples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for horizon, frame in samples.groupby("horizon", sort=False):
        ordinary = frame.loc[frame["state"].eq("ordinary")]
        final_up = frame.loc[frame["state"].eq("final_up")]
        for variable in VALUE_COLUMNS:
            ordinary_values = pd.to_numeric(ordinary[variable], errors="coerce").dropna()
            up_values = pd.to_numeric(final_up[variable], errors="coerce").dropna()
            if ordinary_values.empty or up_values.empty:
                continue
            q95 = float(ordinary_values.quantile(0.95))
            q99 = float(ordinary_values.quantile(0.99))
            records.append(
                {
                    "horizon": horizon,
                    "variable": variable,
                    "ordinary_q95": q95,
                    "ordinary_q99": q99,
                    "final_up_q25": float(up_values.quantile(0.25)),
                    "final_up_median": float(up_values.median()),
                    "final_up_q75": float(up_values.quantile(0.75)),
                    "final_up_above_ordinary_q95_pct": float(up_values.gt(q95).mean() * 100.0),
                    "final_up_above_ordinary_q99_pct": float(up_values.gt(q99).mean() * 100.0),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if args.year_end < args.year_start:
        raise SystemExit("year-end must not precede year-start")
    horizon_roots = {
        "1m": (args.h1m_root, args.h1m_clip3_root, args.h1m_pure_root),
        "close": (args.hclose_root, args.hclose_clip3_root, args.hclose_pure_root),
    }
    basic = []
    samples = []
    tails = []
    clipping = []
    for year in range(args.year_start, args.year_end + 1):
        states = load_states(args.raw_root, year)
        print(f"loaded states year={year} rows={len(states)}", flush=True)
        for horizon, roots in horizon_roots.items():
            frame = load_horizon(*roots, states, year)
            print(f"loaded labels horizon={horizon} year={year} rows={len(frame)}", flush=True)
            basic.extend(basic_records(frame, horizon, year))
            samples.extend(
                sampled_records(
                    frame,
                    horizon,
                    year,
                    sample_per_state=args.sample_per_state_year,
                )
            )
            tails.extend(tail_records(frame, horizon, year))
            clipping.extend(clipping_records(frame, horizon, year))
            del frame

    basic_frame = pd.DataFrame(basic)
    sample_frame = pd.concat(samples, ignore_index=True)
    tail_frame = pd.DataFrame(tails)
    clipping_frame = pd.DataFrame(clipping)
    distribution = aggregate_distribution(basic_frame, sample_frame)
    tail_summary = aggregate_tails(tail_frame)
    separation = separation_summary(sample_frame)
    clipping_summary = clipping_frame.groupby(
        ["horizon", "component", "state"], as_index=False
    ).agg(valid_rows=("valid_rows", "sum"), clipped_rows=("clipped_rows", "sum"))
    clipping_summary["clipped_pct"] = (
        clipping_summary["clipped_rows"] / clipping_summary["valid_rows"] * 100.0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basic_frame.to_csv(args.output_dir / "distribution_by_year.csv", index=False)
    distribution.to_csv(args.output_dir / "distribution_summary.csv", index=False)
    tail_frame.to_csv(args.output_dir / "tail_composition_by_year.csv", index=False)
    tail_summary.to_csv(args.output_dir / "tail_composition_summary.csv", index=False)
    separation.to_csv(args.output_dir / "state_separation_summary.csv", index=False)
    clipping_summary.to_csv(args.output_dir / "clip3_saturation_summary.csv", index=False)
    (args.output_dir / "_SUCCESS").touch()

    print("DISTRIBUTION_FOCUS", flush=True)
    focus = distribution.loc[
        distribution["state"].isin(["ordinary", "final_up"]),
        [
            "horizon",
            "variable",
            "state",
            "row_pct",
            "mean",
            "std",
            "q0.5",
            "q0.95",
            "q0.99",
            "square_mass_pct",
        ],
    ]
    print(focus.to_csv(index=False).strip(), flush=True)
    print("TAIL_COMPOSITION", flush=True)
    print(tail_summary.to_csv(index=False).strip(), flush=True)
    print("STATE_SEPARATION", flush=True)
    print(separation.to_csv(index=False).strip(), flush=True)
    print("CLIP3_SATURATION", flush=True)
    print(clipping_summary.to_csv(index=False).strip(), flush=True)


if __name__ == "__main__":
    main()
