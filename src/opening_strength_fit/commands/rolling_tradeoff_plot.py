from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.analysis import write_json

DEFAULT_INPUT = (
    "experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv"
)
DEFAULT_OUTPUT_DIR = "output/reports/rolling_alpha_conditioned_top100_validation_v1"
DEFAULT_VARIANTS = (
    "alpha_rank=Baseline",
    "gap_penalty_030_p80=Gap 0.30 p80",
    "gap_penalty_035_p80=Gap 0.35 p80",
)
REQUIRED_COLUMNS = (
    "test_month",
    "variant",
    "short_top_excess_bps",
    "next_top_excess_bps",
)
COLORS = ("#35699a", "#df8f16", "#9b3f4e", "#607a52", "#6c5aa7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot rolling Top100 short-vs-next tradeoff from archived "
            "rolling_month_summary.csv evidence."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="rolling_short_vs_next_tail.png")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant mapping as variant=label. Defaults to alpha_rank, "
            "gap_penalty_030_p80, and gap_penalty_035_p80."
        ),
    )
    parser.add_argument(
        "--title",
        default="09:31-09:40 Rolling Validation: Short Signal vs. Next Tail",
    )
    return parser.parse_args()


def parse_variant_labels(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value in values or list(DEFAULT_VARIANTS):
        if "=" not in value:
            raise SystemExit(f"variant mapping must be formatted as variant=label: {value!r}")
        variant, label = value.split("=", 1)
        variant = variant.strip()
        label = label.strip()
        if not variant or not label:
            raise SystemExit(f"variant mapping must include variant and label: {value!r}")
        labels[variant] = label
    return labels


def load_month_summary(path: Path, variant_labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    frame = frame.loc[frame["variant"].isin(variant_labels)].copy()
    if frame.empty:
        raise SystemExit(f"{path} has no requested variants")
    frame["variant_label"] = frame["variant"].map(variant_labels)
    frame["test_month"] = frame["test_month"].astype(str)
    frame["variant"] = pd.Categorical(
        frame["variant"],
        categories=list(variant_labels),
        ordered=True,
    )
    return frame.sort_values(["test_month", "variant"])


def build_plot_data(month_summary: pd.DataFrame) -> pd.DataFrame:
    return month_summary[
        [
            "test_month",
            "variant",
            "variant_label",
            "short_top_excess_bps",
            "next_top_excess_bps",
        ]
    ].copy()


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        offset = 3 if height >= 0 else -5
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:+.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold",
            color=bar.get_facecolor(),
        )


def plot_tradeoff(
    month_summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    months = sorted(month_summary["test_month"].unique())
    variants = [
        variant
        for variant in month_summary["variant"].cat.categories
        if variant in set(month_summary["variant"].astype(str))
    ]
    labels = {
        str(row.variant): str(row.variant_label)
        for row in month_summary[["variant", "variant_label"]].drop_duplicates().itertuples()
    }
    x = np.arange(len(months))
    width = min(0.24, 0.78 / max(1, len(variants)))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width

    fig, axes = plt.subplots(2, 1, figsize=(13.6, 8.4), dpi=180, sharex=True)
    panels = (
        ("Short label: Top100 excess per selected sample", "short_top_excess_bps"),
        ("Next-close check: Top100 excess per selected sample", "next_top_excess_bps"),
    )
    color_by_variant = {
        variant: COLORS[index % len(COLORS)] for index, variant in enumerate(variants)
    }
    handles = []
    for ax, (panel_title, value_col) in zip(axes, panels, strict=True):
        for index, variant in enumerate(variants):
            subset = (
                month_summary.loc[month_summary["variant"].astype(str).eq(variant)]
                .set_index("test_month")
                .reindex(months)
            )
            values = subset[value_col].astype(float).to_numpy()
            bars = ax.bar(
                x + offsets[index],
                values,
                width=width,
                color=color_by_variant[variant],
                label=labels[variant],
            )
            annotate_bars(ax, bars)
            if ax is axes[0]:
                handles.append(bars[0])
        values = month_summary[value_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            lower = min(0.0, float(values.min()))
            upper = max(0.0, float(values.max()))
            padding = max((upper - lower) * 0.16, 4.0)
            ax.set_ylim(lower - padding, upper + padding)
        ax.set_title(panel_title, loc="left", fontsize=14, fontweight="bold")
        ax.set_ylabel("bps")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(months)
    axes[-1].set_xlabel("Rolling test month")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.legend(
        handles=handles,
        labels=[labels[variant] for variant in variants],
        loc="upper center",
        ncol=len(variants),
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.text(
        0.02,
        0.018,
        (
            "Note: Top100 excess = selected Top100 mean - same date x minute "
            "cross-section mean. Each month retrains models; score rules are fixed."
        ),
        fontsize=9.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    variant_labels = parse_variant_labels(args.variant)
    month_summary = load_month_summary(input_path, variant_labels)
    plot_data = build_plot_data(month_summary)
    plot_data_path = output_dir / "rolling_short_vs_next_tail_plot_data.csv"
    trace_path = output_dir / "rolling_short_vs_next_tail_trace.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(plot_data_path, index=False)
    plot_tradeoff(month_summary, output_path, title=args.title)
    write_json(
        trace_path,
        {
            "input": str(input_path),
            "figure": str(output_path),
            "plot_data": str(plot_data_path),
            "variants": variant_labels,
        },
        ensure_ascii=True,
    )
    print(f"figure: {output_path}")
    print(f"plot_data: {plot_data_path}")
    print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
