import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from opening_strength_fit.backtest import load_backtest_series, summarize_daily_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative alpha/profit curves from backtest csv outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="output/backtest/gbm_opening_1y_next_month",
        help="Directory that contains alpha.csv and profit.csv.",
    )
    parser.add_argument(
        "--output",
        default="output/backtest/gbm_opening_1y_next_month/cumulative_curves.png",
        help="PNG path for the generated figure.",
    )
    parser.add_argument(
        "--summary",
        default="output/backtest/gbm_opening_1y_next_month/curve_summary.json",
        help="JSON path for the plotted curve summary.",
    )
    parser.add_argument(
        "--baseline-output",
        default="output/backtest/gbm_opening_1y_next_month/profit_vs_baseline.png",
        help="PNG path for the cumulative model-vs-baseline figure.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional display label. Defaults to the input directory name.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    label = args.label or input_dir.name
    alpha = load_backtest_series(input_dir / "alpha.csv", "alpha")
    profit = load_backtest_series(input_dir / "profit.csv", "profit")
    aligned_index = profit.index.intersection(alpha.index)
    alpha = alpha.loc[aligned_index]
    profit = profit.loc[aligned_index]
    baseline = (profit - alpha).rename("baseline")

    alpha_cum = alpha.cumsum()
    profit_cum = profit.cumsum()
    baseline_cum = baseline.cumsum()

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(alpha_cum.index, alpha_cum.values, color="#1f77b4", linewidth=1.8)
    axes[0].set_title(f"{label}: Cumulative Alpha")
    axes[0].set_ylabel("Cumulative Alpha")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(profit_cum.index, profit_cum.values, color="#2ca02c", linewidth=1.8)
    axes[1].set_title(f"{label}: Cumulative Profit")
    axes[1].set_ylabel("Cumulative Profit")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline_fig, baseline_ax = plt.subplots(figsize=(14, 6))
    baseline_ax.plot(
        profit_cum.index,
        profit_cum.values,
        color="#2ca02c",
        linewidth=1.8,
        label="model profit",
    )
    baseline_ax.plot(
        baseline_cum.index,
        baseline_cum.values,
        color="#ff7f0e",
        linewidth=1.8,
        label="baseline profit",
    )
    baseline_ax.set_title(f"{label}: Model vs Baseline")
    baseline_ax.set_ylabel("Cumulative Return")
    baseline_ax.set_xlabel("Date")
    baseline_ax.grid(True, alpha=0.25)
    baseline_ax.legend()
    baseline_fig.tight_layout()

    baseline_output_path = Path(args.baseline_output)
    baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_fig.savefig(baseline_output_path, dpi=180, bbox_inches="tight")
    plt.close(baseline_fig)

    summary = {
        "alpha": summarize_daily_series(alpha),
        "profit": summarize_daily_series(profit),
        "baseline": summarize_daily_series(baseline),
        "plot": str(output_path),
        "baseline_plot": str(baseline_output_path),
    }
    summary_path = Path(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("plot_backtest_curves_complete:")
    print(f"  plot: {output_path}")
    print(f"  baseline_plot: {baseline_output_path}")
    print(f"  summary: {summary_path}")
    print(f"  alpha_end: {summary['alpha']['cumulative_end']:.6f}")
    print(f"  profit_end: {summary['profit']['cumulative_end']:.6f}")


if __name__ == "__main__":
    main()
