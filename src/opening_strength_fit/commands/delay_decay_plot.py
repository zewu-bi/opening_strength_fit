from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_REPLAY_ROOT = "output/reports/opening_intraday_lgbm_delay_replays"
DEFAULT_SCENARIO = "proxy_top20"
RUN_COLORS = {
    "Universe": "#1f77b4",
    "Strong": "#d17a22",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the LightGBM no-constraint replay decay across delay0/1/2 "
            "from osf-run-lgbm-delay-replays outputs."
        )
    )
    parser.add_argument("--replay-root", default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--output",
        default="",
        help=("Output PNG path. Defaults to <replay-root>/delay_decay_<scenario>.png."),
    )
    parser.add_argument(
        "--table-output",
        default="",
        help=(
            "Output CSV path for plotted values. Defaults to "
            "<replay-root>/delay_decay_<scenario>.csv."
        ),
    )
    return parser.parse_args()


def branch_from_run(run: object) -> str:
    return "Strong" if "strong" in str(run) else "Universe"


def delay_number(delay: object) -> int:
    return int(str(delay).removeprefix("delay"))


def cycle_sem(replay_root: Path, scenario: str) -> pd.DataFrame:
    rows = []
    for delay_dir in sorted(replay_root.glob("delay*")):
        path = delay_dir / scenario / "intraday_cycles.csv"
        if not path.exists():
            continue
        cycles = pd.read_csv(path)
        for run, group in cycles.groupby("run", sort=False):
            values = pd.to_numeric(group["cycle_return_bps"], errors="coerce").dropna()
            sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "delay": delay_dir.name,
                    "run": run,
                    "cycle_return_bps_sem": sem,
                }
            )
    return pd.DataFrame(rows)


def build_plot_table(replay_root: Path, scenario: str) -> pd.DataFrame:
    summary_path = replay_root / "scenario_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"missing replay summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    table = summary.loc[summary["scenario"].astype(str) == scenario].copy()
    if table.empty:
        raise SystemExit(f"scenario {scenario!r} not found in {summary_path}")
    table["branch"] = table["run"].map(branch_from_run)
    table["delay_num"] = table["delay"].map(delay_number)
    sem = cycle_sem(replay_root, scenario)
    if not sem.empty:
        table = table.merge(sem, on=["delay", "run"], how="left")
    else:
        table["cycle_return_bps_sem"] = 0.0
    table["cycle_return_bps_sem"] = table["cycle_return_bps_sem"].fillna(0.0)
    branch_order = {"Universe": 0, "Strong": 1}
    table["branch_order"] = table["branch"].map(branch_order).fillna(99)
    return table.sort_values(["branch_order", "delay_num"]).drop(columns=["branch_order"])


def plot_delay_decay(table: pd.DataFrame, output_path: Path, *, scenario: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for branch, branch_frame in table.groupby("branch", sort=False):
        branch_frame = branch_frame.sort_values("delay_num")
        ax.errorbar(
            branch_frame["delay_num"],
            branch_frame["mean_cycle_return_bps"],
            yerr=branch_frame["cycle_return_bps_sem"],
            marker="o",
            linewidth=2.4,
            capsize=4,
            color=RUN_COLORS.get(branch, "#333333"),
            label=branch,
        )
        for _, row in branch_frame.iterrows():
            ax.annotate(
                f"{row['mean_cycle_return_bps']:.1f}",
                (row["delay_num"], row["mean_cycle_return_bps"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=RUN_COLORS.get(branch, "#333333"),
            )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["delay0", "delay1", "delay2"])
    ax.set_ylabel("Mean cycle return (bps)")
    ax.set_title(f"{scenario} replay decays with execution delay")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    replay_root = Path(args.replay_root)
    output_path = (
        Path(args.output) if args.output else replay_root / f"delay_scan_{args.scenario}.png"
    )
    table_path = (
        Path(args.table_output)
        if args.table_output
        else replay_root / f"delay_scan_{args.scenario}.csv"
    )
    table = build_plot_table(replay_root, args.scenario)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_delay_decay(table, output_path, scenario=args.scenario)
    print(
        json.dumps(
            {
                "figure": str(output_path),
                "table": str(table_path),
                "scenario": args.scenario,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
