import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from opening_strength_fit.backtest import load_backtest_series, summarize_daily_series


DEFAULT_RUNS = (
    ("gbm", Path("output/backtest/gbm_opening_1y_next_month")),
)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be label=/path/to/backtest_dir")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cumulative alpha/profit curves across backtest runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Formatted as label=/path/to/backtest_output_dir. Defaults to current full-window runs.",
    )
    parser.add_argument(
        "--output",
        default="output/reports/backtest_model_comparison/cumulative_comparison.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--summary",
        default="output/reports/backtest_model_comparison/comparison_summary.json",
        help="JSON summary path.",
    )
    parser.add_argument(
        "--alpha-only-output",
        default=(
            "output/reports/backtest_model_comparison/"
            "cumulative_alpha_comparison.png"
        ),
        help="PNG output path for the alpha-only comparison chart.",
    )
    args = parser.parse_args()

    alpha_series = {}
    profit_series = {}
    summary = {}

    for label, run_dir in args.run or DEFAULT_RUNS:
        alpha = load_backtest_series(run_dir / "alpha.csv")
        profit = load_backtest_series(run_dir / "profit.csv")
        alpha_series[label] = alpha.cumsum()
        profit_series[label] = profit.cumsum()
        summary[label] = {
            "alpha": summarize_daily_series(alpha),
            "profit": summarize_daily_series(profit),
            "run_dir": str(run_dir),
        }

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for label, series in alpha_series.items():
        axes[0].plot(series.index, series.values, linewidth=1.8, label=label)
    axes[0].set_title("Backtest Comparison: Cumulative Alpha")
    axes[0].set_ylabel("Cumulative Alpha")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for label, series in profit_series.items():
        axes[1].plot(series.index, series.values, linewidth=1.8, label=label)
    axes[1].set_title("Backtest Comparison: Cumulative Profit")
    axes[1].set_ylabel("Cumulative Profit")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    alpha_only_fig, alpha_only_ax = plt.subplots(figsize=(14, 6))
    for label, series in alpha_series.items():
        alpha_only_ax.plot(series.index, series.values, linewidth=1.8, label=label)
    alpha_only_ax.set_title("Backtest Comparison: Cumulative Alpha")
    alpha_only_ax.set_ylabel("Cumulative Alpha")
    alpha_only_ax.set_xlabel("Date")
    alpha_only_ax.grid(True, alpha=0.25)
    alpha_only_ax.legend()
    alpha_only_fig.tight_layout()

    alpha_only_output_path = Path(args.alpha_only_output)
    alpha_only_output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_only_fig.savefig(alpha_only_output_path, dpi=180, bbox_inches="tight")
    plt.close(alpha_only_fig)

    summary["plot"] = str(output_path)
    summary["alpha_only_plot"] = str(alpha_only_output_path)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("compare_backtest_runs_complete:")
    print(f"  plot: {output_path}")
    print(f"  alpha_only_plot: {alpha_only_output_path}")
    print(f"  summary: {summary_path}")
    for label, values in summary.items():
        if label in {"plot", "alpha_only_plot"}:
            continue
        print(
            f"  {label}: alpha_end={values['alpha']['cumulative_end']:.6f}, "
            f"profit_end={values['profit']['cumulative_end']:.6f}"
        )


if __name__ == "__main__":
    main()
