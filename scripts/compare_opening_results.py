import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.reports import build_yearly_table
from opening_strength_fit.reports import preferred_metric_column


RESULT_COLUMNS = {
    "run",
    "run_id",
    "test_year",
    "train_start_date",
    "train_end_date",
    "test_start_date",
    "test_end_date",
    "train_rows",
    "train_dates",
    "train_symbols",
    "test_rows",
    "test_dates",
    "test_symbols",
    "features",
    "model_test_r2",
    "rows",
    "dates",
    "symbols",
    "sample_grain",
    "ic_grouping",
    "overall_ic",
    "overall_rank_ic",
    "group_ic_mean",
    "group_ic_std",
    "group_ic_ir",
    "group_rank_ic_mean",
    "group_rank_ic_std",
    "group_rank_ic_ir",
    "daily_ic_mean",
    "daily_ic_std",
    "daily_ic_ir",
    "daily_rank_ic_mean",
    "daily_rank_ic_std",
    "daily_rank_ic_ir",
    "mean_label",
    "win_rate",
    "selection_mode",
    "top_n",
}

DEFAULT_RUNS = (
    (
        "ridge",
        "experiments/results/metrics/ridge_opening_full_metrics_by_year.csv",
    ),
)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path)


def read_run(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"metrics file not found for {label}: {path}")
    df = pd.read_csv(path).sort_values("test_year").copy()
    df.insert(0, "run", label)
    return df


def first_value(df: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in df.columns:
        return default
    value = df[column].dropna()
    if value.empty:
        return default
    return value.iloc[0]


def display_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def model_spec(df: pd.DataFrame) -> str:
    model_name = first_value(df, "model_name", "ridge")
    return f"Ridge(alpha={display_value(first_value(df, 'alpha'))})" if model_name == "ridge" else str(model_name)


def _metric_series(df: pd.DataFrame, preferred: str, fallback: str) -> tuple[str, pd.Series]:
    column = preferred_metric_column(df, preferred, fallback)
    return column, df[column]


def _extreme_year(df: pd.DataFrame, series: pd.Series, method: str) -> object:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    index = valid.idxmax() if method == "max" else valid.idxmin()
    return int(df.loc[index, "test_year"])


def _extreme_value(series: pd.Series, method: str) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.max() if method == "max" else valid.min())


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in runs.groupby("run", sort=False):
        rank_col, rank = _metric_series(
            df,
            "group_rank_ic_mean",
            "daily_rank_ic_mean",
        )
        rank_ir_col, rank_ir = _metric_series(
            df,
            "group_rank_ic_ir",
            "daily_rank_ic_ir",
        )
        ic_col, ic = _metric_series(df, "group_ic_mean", "daily_ic_mean")
        r2 = df["model_test_r2"]
        rows.append(
            {
                "run": label,
                "model": model_spec(df),
                "years": f"{int(df['test_year'].min())}-{int(df['test_year'].max())}",
                "features": int(df["features"].iloc[0]),
                "selection": first_value(df, "selection_mode", ""),
                "rank_ic_metric": rank_col,
                "rank_ic_mean_avg": rank.mean(),
                "rank_ic_ir_metric": rank_ir_col,
                "rank_ic_ir_avg": rank_ir.mean(),
                "ic_metric": ic_col,
                "ic_mean_avg": ic.mean(),
                "model_r2_avg": r2.mean(),
                "positive_rank_ic_years": f"{int((rank > 0).sum())}/{len(df)}",
                "best_year": _extreme_year(df, rank, "max"),
                "best_rank_ic_mean": _extreme_value(rank, "max"),
                "worst_year": _extreme_year(df, rank, "min"),
                "worst_rank_ic_mean": _extreme_value(rank, "min"),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_mean_avg", ascending=False)


def build_parameters(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in runs.groupby("run", sort=False):
        param_columns = [
            column for column in df.columns if column not in RESULT_COLUMNS
        ]
        for column in param_columns:
            value = first_value(df, column)
            if value != "":
                rows.append(
                    {"run": label, "parameter": column, "value": display_value(value)}
                )
    return pd.DataFrame(rows)


def build_diagnostics(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in runs.groupby("run", sort=False):
        df = df.sort_values("test_year")
        rank_col, rank = _metric_series(
            df,
            "group_rank_ic_mean",
            "daily_rank_ic_mean",
        )
        rank_ir_col, rank_ir = _metric_series(
            df,
            "group_rank_ic_ir",
            "daily_rank_ic_ir",
        )
        r2 = df["model_test_r2"]
        valid_rank = rank.dropna()
        rank_first = float(valid_rank.iloc[0]) if not valid_rank.empty else float("nan")
        rank_last = float(valid_rank.iloc[-1]) if not valid_rank.empty else float("nan")
        rank_change = rank_last - rank_first
        rows.append(
            {
                "run": label,
                "rank_ic_metric": rank_col,
                "rank_ic_first_year": rank_first,
                "rank_ic_last_year": rank_last,
                "rank_ic_change": rank_change,
                "rank_ic_change_pct": rank_change / rank_first if rank_first else float("nan"),
                "rank_ic_ir_metric": rank_ir_col,
                "rank_ic_ir_min": float(rank_ir.min()),
                "model_r2_min": float(r2.min()),
                "negative_r2_years": int((r2 < 0).sum()),
                "worst_rank_ic_year": _extreme_year(df, rank, "min"),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_change")


def build_yearly(runs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, df in runs.groupby("run", sort=False):
        table = build_yearly_table(df)
        table.insert(0, "run", label)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def write_plot(summary: pd.DataFrame, yearly: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for label, df in yearly.groupby("run", sort=False):
        axes[0, 0].plot(df["year"], df["rank_ic_mean"], marker="o", label=label)
        axes[0, 1].plot(df["year"], df["ic_mean"], marker="o", label=label)
        axes[1, 0].plot(df["year"], df["rank_ic_ir"], marker="o", label=label)
        axes[1, 1].plot(df["year"], df["model_r2"], marker="o", label=label)

    axes[0, 0].set_title("Rank IC Mean")
    axes[0, 1].set_title("IC Mean")
    axes[1, 0].set_title("Rank IC IR")
    axes[1, 1].set_title("Model R2")

    for ax in axes.ravel():
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Test year")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(summary)))
    fig.suptitle("Opening-Strength Model Comparison", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return format(value, floatfmt)
        return str(value)

    rows = [[cell(value) for value in row] for row in df.to_numpy()]
    headers = [str(column) for column in df.columns]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    out = [fmt_row(headers)]
    out.append("| " + " | ".join("-" * width for width in widths) + " |")
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Run metrics to compare, formatted as label=path. Defaults to current k8s metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/reports/opening_model_comparison",
        help="Directory for comparison CSV, Markdown, and PNG output.",
    )
    args = parser.parse_args()

    run_specs = args.run or [(label, Path(path)) for label, path in DEFAULT_RUNS]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.concat([read_run(label, path) for label, path in run_specs], ignore_index=True)
    summary = build_summary(runs)
    parameters = build_parameters(runs)
    diagnostics = build_diagnostics(runs)
    yearly = build_yearly(runs)

    summary_path = output_dir / "opening_model_comparison_summary.csv"
    parameters_path = output_dir / "opening_model_comparison_parameters.csv"
    diagnostics_path = output_dir / "opening_model_comparison_diagnostics.csv"
    yearly_path = output_dir / "opening_model_comparison_yearly.csv"
    report_path = output_dir / "opening_model_comparison.md"
    plot_path = output_dir / "opening_model_comparison.png"

    summary.to_csv(summary_path, index=False)
    parameters.to_csv(parameters_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    write_plot(summary, yearly, plot_path)

    report = [
        "# Opening-Strength Model Comparison",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Diagnostics",
        "",
        markdown_table(diagnostics),
        "",
        "## Model Parameters",
        "",
        markdown_table(parameters),
        "",
        "## Yearly Metrics",
        "",
        markdown_table(
            yearly[
                [
                    "run",
                    "year",
                    "rank_ic_mean",
                    "rank_ic_ir",
                    "ic_mean",
                    "ic_ir",
                    "model_r2",
                    "pooled_rank_ic",
                ]
            ]
        ),
        "",
        f"![Opening model comparison]({plot_path.name})",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("comparison_outputs:")
    print(f"  summary: {summary_path}")
    print(f"  parameters: {parameters_path}")
    print(f"  diagnostics: {diagnostics_path}")
    print(f"  yearly: {yearly_path}")
    print(f"  report: {report_path}")
    print(f"  plot: {plot_path}")
    print()
    print(summary.to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
