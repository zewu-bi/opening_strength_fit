import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.reports import build_yearly_table, preferred_metric_column


def fmt(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


def print_opening_summary(df: pd.DataFrame, source: Path) -> None:
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "ridge"
    print("opening_summary:")
    print(f"  years: {int(df['test_year'].min())} -> {int(df['test_year'].max())}")
    print(f"  runs: {len(df)}")
    print(f"  features: {int(df['features'].iloc[0]):,}")
    if model_name == "ridge" and "alpha" in df.columns:
        print(f"  model: Ridge(alpha={df['alpha'].iloc[0]})")
    else:
        print(f"  model: {model_name}")
    if "selection_mode" in df.columns:
        print(f"  selection_mode: {df['selection_mode'].iloc[0]}")
    print(f"  source: {source}")


def print_yearly_results(df: pd.DataFrame) -> None:
    print("\nyearly_results:")
    table = build_yearly_table(df)
    print(table.to_string(index=False, float_format="{:.6f}".format))


def print_opening_assessment(df: pd.DataFrame) -> None:
    rank_col = preferred_metric_column(df, "group_rank_ic_mean", "daily_rank_ic_mean")
    rank_ir_col = preferred_metric_column(df, "group_rank_ic_ir", "daily_rank_ic_ir")
    ic_col = preferred_metric_column(df, "group_ic_mean", "daily_ic_mean")
    rank = df[rank_col]
    rank_ir = df[rank_ir_col]
    ic = df[ic_col]
    r2 = df["model_test_r2"]
    valid_rank = rank.dropna()

    print("\nopening_assessment:")
    print(f"  rank_ic_metric: {rank_col}")
    if rank_col == "daily_rank_ic_mean" and "group_rank_ic_mean" in df.columns:
        print("  rank_ic_note: grouped rank IC unavailable; using daily sanity metric")
    print(f"  rank_ic_mean_avg: {fmt(rank.mean())}")
    print(f"  rank_ic_ir_avg: {fmt(rank_ir.mean())}")
    print(f"  ic_mean_avg: {fmt(ic.mean())}")
    print(f"  model_r2_avg: {fmt(r2.mean())}")
    print(f"  positive_rank_ic_years: {(rank > 0).sum()}/{len(df)}")
    if valid_rank.empty:
        print("  strongest_year: n/a (rank IC unavailable)")
        print("  weakest_year: n/a (rank IC unavailable)")
        print("  verdict: unavailable: rank IC is all nan for this output")
        return

    strongest = df.loc[valid_rank.idxmax()]
    weakest = df.loc[valid_rank.idxmin()]
    print(f"  strongest_year: {int(strongest.test_year)} (rank_ic_mean={fmt(strongest[rank_col])})")
    print(f"  weakest_year: {int(weakest.test_year)} (rank_ic_mean={fmt(weakest[rank_col])})")

    if (rank > 0).all():
        verdict = f"pass: {rank_col} is positive in every tested year"
    elif (rank > 0).mean() >= 0.8:
        verdict = f"watch: {rank_col} is mostly positive, but not every year"
    else:
        verdict = f"needs_attention: {rank_col} is not consistently positive"
    print(f"  verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="",
        help="Result output directory containing metrics_by_year.csv.",
    )
    parser.add_argument(
        "--metrics-csv",
        default="",
        help="Optional direct path to a pulled metrics_by_year.csv file.",
    )
    args = parser.parse_args()

    if bool(args.input_dir) == bool(args.metrics_csv):
        raise SystemExit("Use exactly one of --input-dir or --metrics-csv.")

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        if not metrics_path.exists():
            raise SystemExit(f"metrics csv not found: {metrics_path}")
        source = metrics_path
    else:
        input_dir = Path(args.input_dir)
        metrics_path = input_dir / "metrics_by_year.csv"
        if not metrics_path.exists():
            print("metrics_by_year.csv not found")
            print("\navailable files:")
            for path in sorted(input_dir.glob("*")):
                print(f"  {path.name}")
            raise SystemExit(0)
        source = input_dir

    df = pd.read_csv(metrics_path).sort_values("test_year")

    print_opening_summary(df, source)
    print_yearly_results(df)
    print_opening_assessment(df)


if __name__ == "__main__":
    main()
