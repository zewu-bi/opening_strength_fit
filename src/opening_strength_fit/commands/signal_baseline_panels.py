from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    clock_range,
    write_json,
)
from opening_strength_fit.io import read_frame
from opening_strength_fit.model import corr
from opening_strength_fit.next_close_labels import (
    add_next_close_label_arguments,
    load_or_fetch_next_close_labels_from_args,
)

DEFAULT_INPUT = (
    "output/legacy/predictions/lgbm_opening_1y_next_month_delay2/predictions_all.parquet"
)
DEFAULT_OUTPUT_DIR = "output/legacy/reports/experiment0_delay2_four_panel_baseline"
STALE_OUTPUTS = ("next_close_top100_return_by_minute.png",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the four-panel signal-strength baseline: short Rank IC, "
            "short Top100 excess, next-close Rank IC, and next-close Top100 excess."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--title", default="Delay2 baseline signal panels")
    parser.add_argument("--start-clock", default="09:30")
    parser.add_argument("--end-clock", default="09:40")
    add_next_close_label_arguments(parser, include_connection=True)
    return parser.parse_args()


def load_predictions(path: Path, clocks: list[str], score_col: str) -> pd.DataFrame:
    required = [
        "date",
        "symbol",
        "decision_target_timestamp",
        score_col,
        "label",
        "buy_price",
    ]
    frame = read_frame(path, columns=required)
    frame = frame.dropna(
        subset=["date", "symbol", "decision_target_timestamp", score_col, "label"]
    ).copy()
    if score_col != "prediction":
        frame["prediction"] = pd.to_numeric(frame[score_col], errors="coerce")
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["decision_target_timestamp"] = pd.to_datetime(
        frame["decision_target_timestamp"],
        errors="coerce",
    )
    frame["clock"] = frame["decision_target_timestamp"].dt.strftime("%H:%M")
    return frame.loc[frame["clock"].isin(clocks)].copy()


def summarize_by_minute(
    frame: pd.DataFrame,
    *,
    label_col: str,
    prefix: str,
    top_n: int,
    clocks: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    valid = frame.dropna(subset=[label_col, "prediction"])
    for (date, timestamp), group in valid.groupby(
        ["date", "decision_target_timestamp"],
        sort=True,
    ):
        ordered = group.sort_values("prediction", ascending=False)
        top = ordered.head(top_n)
        all_mean = float(group[label_col].mean())
        top_mean = float(top[label_col].mean())
        rows.append(
            {
                "date": str(date),
                "decision_target_timestamp": pd.Timestamp(timestamp),
                "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                "rows": int(len(group)),
                f"{prefix}_rank_ic": corr(group[label_col], group["prediction"], "spearman"),
                f"{prefix}_ic": corr(group[label_col], group["prediction"], "pearson"),
                f"{prefix}_all_mean_bps": all_mean * 10_000,
                f"{prefix}_top100_mean_bps": top_mean * 10_000,
                f"{prefix}_top100_excess_bps": (top_mean - all_mean) * 10_000,
                f"{prefix}_top100_median_bps": float(top[label_col].median()) * 10_000,
                f"{prefix}_top100_win_rate": float((top[label_col] > 0).mean()),
                f"{prefix}_top100_trades": int(len(top)),
            }
        )

    group_metrics = pd.DataFrame(rows)
    summary = group_metrics.groupby("clock", as_index=False).agg(
        groups=(f"{prefix}_rank_ic", "size"),
        rows=("rows", "sum"),
        rank_ic_mean=(f"{prefix}_rank_ic", "mean"),
        rank_ic_std=(f"{prefix}_rank_ic", "std"),
        rank_ic_positive_rate=(
            f"{prefix}_rank_ic",
            lambda series: float((series > 0).mean()),
        ),
        ic_mean=(f"{prefix}_ic", "mean"),
        all_mean_bps=(f"{prefix}_all_mean_bps", "mean"),
        top100_mean_bps=(f"{prefix}_top100_mean_bps", "mean"),
        top100_excess_bps=(f"{prefix}_top100_excess_bps", "mean"),
        top100_excess_bps_std=(f"{prefix}_top100_excess_bps", "std"),
        top100_positive_rate=(
            f"{prefix}_top100_mean_bps",
            lambda series: float((series > 0).mean()),
        ),
        top100_excess_positive_rate=(
            f"{prefix}_top100_excess_bps",
            lambda series: float((series > 0).mean()),
        ),
        top100_median_bps=(f"{prefix}_top100_median_bps", "mean"),
        top100_win_rate=(f"{prefix}_top100_win_rate", "mean"),
    )
    summary["rank_ic_sem"] = summary["rank_ic_std"] / np.sqrt(summary["groups"])
    summary["top100_excess_bps_sem"] = summary["top100_excess_bps_std"] / np.sqrt(summary["groups"])
    summary["clock"] = pd.Categorical(summary["clock"], categories=clocks, ordered=True)
    summary = summary.sort_values("clock")
    summary["clock"] = summary["clock"].astype(str)
    return group_metrics, summary


def plot_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    y_col: str,
    sem_col: str | None,
    title: str,
    ylabel: str,
    color: str,
    value_fmt: str,
) -> None:
    x = np.arange(len(summary))
    y = summary[y_col].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, color=color)
    if sem_col:
        sem = summary[sem_col].to_numpy(dtype=float)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8)
    for xi, yi in zip(x, y, strict=True):
        if pd.isna(yi):
            continue
        is_last = xi == x[-1]
        x_offset = -5 if is_last else 5
        y_offset = 4 if yi >= 0 else -4
        ax.annotate(
            value_fmt.format(yi),
            xy=(xi, yi),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if is_last else "left",
            va="bottom" if yi >= 0 else "top",
            fontsize=7,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["clock"], rotation=0)
    ax.grid(True, axis="y", alpha=0.3)


def remove_stale_outputs(output_dir: Path) -> None:
    for name in STALE_OUTPUTS:
        path = output_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(output_dir)

    clocks = clock_range(args.start_clock, args.end_clock)
    predictions = load_predictions(input_path, clocks, args.score_col)
    next_close_labels = load_or_fetch_next_close_labels_from_args(
        predictions,
        args=args,
        output_dir=output_dir,
    )
    frame = predictions.merge(
        next_close_labels,
        on=["date", "symbol", "decision_target_timestamp"],
        how="left",
    )

    short_group, short_summary = summarize_by_minute(
        frame,
        label_col="label",
        prefix="short",
        top_n=int(args.top_n),
        clocks=clocks,
    )
    next_group, next_summary = summarize_by_minute(
        frame,
        label_col="alpha_return_next_close",
        prefix="next_close",
        top_n=int(args.top_n),
        clocks=clocks,
    )

    short_group.to_csv(output_dir / "short_delay2_group_metrics.csv", index=False)
    short_summary.to_csv(output_dir / "short_delay2_minute_summary.csv", index=False)
    next_group.to_csv(output_dir / "next_close_group_metrics.csv", index=False)
    next_summary.to_csv(output_dir / "next_close_minute_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=170, sharex=True)
    plot_panel(
        axes[0, 0],
        short_summary,
        y_col="rank_ic_mean",
        sem_col="rank_ic_sem",
        title="Short-horizon Rank IC",
        ylabel="Mean Rank IC",
        color="#2563a7",
        value_fmt="{:.3f}",
    )
    plot_panel(
        axes[0, 1],
        short_summary,
        y_col="top100_excess_bps",
        sem_col="top100_excess_bps_sem",
        title="Short-horizon Top100 excess",
        ylabel="Top100 excess (bps)",
        color="#c46a1a",
        value_fmt="{:.1f}",
    )
    plot_panel(
        axes[1, 0],
        next_summary,
        y_col="rank_ic_mean",
        sem_col="rank_ic_sem",
        title="Next-close Rank IC",
        ylabel="Mean Rank IC",
        color="#4d7c0f",
        value_fmt="{:.3f}",
    )
    plot_panel(
        axes[1, 1],
        next_summary,
        y_col="top100_excess_bps",
        sem_col="top100_excess_bps_sem",
        title="Next-close Top100 excess",
        ylabel="Top100 excess (bps)",
        color="#9333ea",
        value_fmt="{:.1f}",
    )
    for ax in axes[1, :]:
        ax.set_xlabel("Decision minute")
    fig.suptitle(args.title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    panel_path = output_dir / "signal_baseline_four_panel.png"
    fig.savefig(panel_path)
    plt.close(fig)

    trace = {
        "experiment": "experiment0_delay2_four_panel_baseline",
        "input": str(input_path),
        "top_n": int(args.top_n),
        "score_col": args.score_col,
        "clocks": clocks,
        "cross_section": "date x decision_target_timestamp",
        "short_label": "label",
        "next_close_label": "alpha_return_next_close",
        "main_metrics": [
            "short rank IC",
            "short Top100 excess bps",
            "next-close rank IC",
            "next-close Top100 excess bps",
        ],
        "outputs": {
            "four_panel": str(panel_path),
            "short_summary": str(output_dir / "short_delay2_minute_summary.csv"),
            "next_close_summary": str(output_dir / "next_close_minute_summary.csv"),
            "short_group_metrics": str(output_dir / "short_delay2_group_metrics.csv"),
            "next_close_group_metrics": str(output_dir / "next_close_group_metrics.csv"),
            "next_close_labels": str(output_dir / "clickhouse_next_close_labels.parquet"),
        },
    }
    write_json(output_dir / "trace.json", trace)

    print("short")
    print(
        short_summary[["clock", "rank_ic_mean", "top100_excess_bps", "top100_mean_bps"]].to_string(
            index=False,
            formatters={
                "rank_ic_mean": "{:.6f}".format,
                "top100_excess_bps": "{:.3f}".format,
                "top100_mean_bps": "{:.3f}".format,
            },
        )
    )
    print("\nnext_close")
    print(
        next_summary[["clock", "rank_ic_mean", "top100_excess_bps", "top100_mean_bps"]].to_string(
            index=False,
            formatters={
                "rank_ic_mean": "{:.6f}".format,
                "top100_excess_bps": "{:.3f}".format,
                "top100_mean_bps": "{:.3f}".format,
            },
        )
    )
    print(f"\nfour_panel: {panel_path}")


if __name__ == "__main__":
    main()
