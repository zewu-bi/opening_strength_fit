from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.evaluation import (
    resolve_group_cols,
    score_bucket_returns,
    top_score_trades,
)
from opening_strength_fit.horizons import HorizonLike, label_column_name

RUN_COLORS = {
    "Universe": "#1f77b4",
    "Strong": "#d17a22",
}


def rank_ic_by_group(
    frame: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    group_cols: tuple[str, ...],
) -> tuple[float, float, float, int]:
    resolved = resolve_group_cols(frame, group_cols)
    if not resolved:
        valid = frame[[label_col, score_col]].dropna()
        if len(valid) < 2:
            return float("nan"), float("nan"), float("nan"), 0
        corr = valid[label_col].rank(method="average").corr(valid[score_col].rank(method="average"))
        return float(corr), float("nan"), float("nan"), 1
    values = []
    for _, group in frame.groupby(list(resolved), sort=False, observed=True):
        valid = group[[label_col, score_col]].dropna()
        if len(valid) < 3:
            continue
        if valid[label_col].nunique() < 2 or valid[score_col].nunique() < 2:
            continue
        corr = valid[label_col].rank(method="average").corr(valid[score_col].rank(method="average"))
        if pd.notna(corr):
            values.append(float(corr))
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    mean = float(series.mean())
    ir = mean / std if std and np.isfinite(std) else float("nan")
    return mean, std, ir, int(len(series))


def select_bottom_score_trades(
    frame: pd.DataFrame,
    *,
    top_n: int,
    label_col: str,
    score_col: str,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    work = frame.copy()
    work["_inverse_prediction"] = -pd.to_numeric(work[score_col], errors="coerce")
    return top_score_trades(
        work,
        top_n=top_n,
        label_col=label_col,
        score_col="_inverse_prediction",
        group_cols=group_cols,
    ).drop(columns=["_inverse_prediction"], errors="ignore")


def group_return_sem(
    trades: pd.DataFrame,
    *,
    label_col: str,
    group_cols: tuple[str, ...],
) -> float:
    resolved = resolve_group_cols(trades, group_cols)
    if not resolved or trades.empty:
        return float("nan")
    group_means = trades.groupby(list(resolved), observed=True)[label_col].mean()
    if len(group_means) < 2:
        return float("nan")
    return float(group_means.std(ddof=1) / np.sqrt(len(group_means)))


def summarize_horizon(
    frame: pd.DataFrame,
    *,
    branch: str,
    horizon: HorizonLike,
    label_col: str,
    score_col: str,
    top_n: int,
    group_cols: tuple[str, ...],
) -> dict[str, object]:
    valid = frame.loc[
        pd.to_numeric(frame[label_col], errors="coerce").notna()
        & pd.to_numeric(frame[score_col], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return {
            "branch": branch,
            "horizon": horizon.name,
            "horizon_label": horizon.label,
            "horizon_seconds": horizon.seconds,
            "label_col": label_col,
            "rows": 0,
        }
    valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce")
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    resolved_groups = resolve_group_cols(valid, group_cols)
    top = top_score_trades(
        valid,
        top_n=top_n,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    bottom = select_bottom_score_trades(
        valid,
        top_n=top_n,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    rank_ic_mean, rank_ic_std, rank_ic_ir, rank_ic_groups = rank_ic_by_group(
        valid,
        label_col=label_col,
        score_col=score_col,
        group_cols=group_cols,
    )
    top_mean = float(top[label_col].mean()) if not top.empty else float("nan")
    bottom_mean = float(bottom[label_col].mean()) if not bottom.empty else float("nan")
    return {
        "branch": branch,
        "horizon": horizon.name,
        "horizon_label": horizon.label,
        "horizon_seconds": horizon.seconds,
        "label_col": label_col,
        "rows": int(len(valid)),
        "dates": int(valid["date"].nunique()) if "date" in valid else 0,
        "symbols": int(valid["symbol"].nunique()) if "symbol" in valid else 0,
        "groups": int(valid.groupby(list(resolved_groups)).ngroups) if resolved_groups else 1,
        "group_cols": ",".join(resolved_groups) if resolved_groups else "global",
        "top_n": int(top_n),
        "top_trades": int(len(top)),
        "top_groups": int(top.groupby(list(resolved_groups)).ngroups)
        if resolved_groups and not top.empty
        else (1 if not top.empty else 0),
        "mean_alpha_return": top_mean,
        "mean_alpha_return_bps": top_mean * 10_000.0,
        "mean_alpha_return_sem": group_return_sem(
            top,
            label_col=label_col,
            group_cols=group_cols,
        ),
        "mean_alpha_return_bps_sem": group_return_sem(
            top,
            label_col=label_col,
            group_cols=group_cols,
        )
        * 10_000.0,
        "median_alpha_return": float(top[label_col].median()) if not top.empty else float("nan"),
        "top_win_rate": float((top[label_col] > 0).mean()) if not top.empty else float("nan"),
        "all_mean_return": float(valid[label_col].mean()),
        "all_mean_return_bps": float(valid[label_col].mean() * 10_000.0),
        "bottom_mean_return": bottom_mean,
        "bottom_mean_return_bps": bottom_mean * 10_000.0,
        "top_bottom_spread": top_mean - bottom_mean,
        "top_bottom_spread_bps": (top_mean - bottom_mean) * 10_000.0,
        "group_rank_ic_mean": rank_ic_mean,
        "group_rank_ic_std": rank_ic_std,
        "group_rank_ic_ir": rank_ic_ir,
        "group_rank_ic_groups": rank_ic_groups,
    }


def build_summary_tables(
    predictions: pd.DataFrame,
    *,
    horizons: list[HorizonLike],
    top_n: int,
    score_bins: int,
    group_cols: tuple[str, ...],
    allow_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_rows = []
    bucket_frames = []
    missing: list[str] = []
    for branch, branch_frame in predictions.groupby("branch", sort=False):
        for spec in horizons:
            label_col = label_column_name(spec.name)
            if label_col not in branch_frame.columns:
                missing.append(f"{branch}:{spec.name}")
                continue
            non_null = pd.to_numeric(branch_frame[label_col], errors="coerce").notna()
            if not non_null.any():
                missing.append(f"{branch}:{spec.name}")
                continue
            summary_rows.append(
                summarize_horizon(
                    branch_frame,
                    branch=branch,
                    horizon=spec,
                    label_col=label_col,
                    score_col="prediction",
                    top_n=top_n,
                    group_cols=group_cols,
                )
            )
            buckets = score_bucket_returns(
                branch_frame,
                bins=score_bins,
                label_col=label_col,
                score_col="prediction",
                group_cols=group_cols,
            )
            buckets.insert(0, "horizon", spec.name)
            buckets.insert(0, "branch", branch)
            bucket_frames.append(buckets)

    if missing and not allow_missing:
        raise SystemExit(
            "missing requested horizon labels: "
            + ", ".join(missing)
            + ". Provide --tick-input, --label-input/--horizon-label, or use "
            "--allow-missing-horizons for partial output."
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = add_decay_retention(summary, horizons)
    buckets = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
    return summary, buckets, missing


def add_decay_retention(summary: pd.DataFrame, horizons: list[HorizonLike]) -> pd.DataFrame:
    out = summary.copy()
    horizon_order = {spec.name: index for index, spec in enumerate(horizons)}
    branch_order = {branch: index for index, branch in enumerate(out["branch"].drop_duplicates())}
    out["horizon_order"] = out["horizon"].map(horizon_order)
    out["branch_order"] = out["branch"].map(branch_order)
    out = out.sort_values(["branch_order", "horizon_order"])
    out["retention_vs_first"] = np.nan
    out["retention_vs_60s"] = np.nan
    for _branch, branch_frame in out.groupby("branch", sort=False):
        first = branch_frame["mean_alpha_return_bps"].dropna()
        first_value = first.iloc[0] if not first.empty else np.nan
        sixty = branch_frame.loc[
            branch_frame["horizon"] == "60s",
            "mean_alpha_return_bps",
        ].dropna()
        sixty_value = sixty.iloc[0] if not sixty.empty else np.nan
        branch_index = branch_frame.index
        if pd.notna(first_value) and first_value != 0:
            out.loc[branch_index, "retention_vs_first"] = (
                out.loc[branch_index, "mean_alpha_return_bps"] / first_value
            )
        if pd.notna(sixty_value) and sixty_value != 0:
            out.loc[branch_index, "retention_vs_60s"] = (
                out.loc[branch_index, "mean_alpha_return_bps"] / sixty_value
            )
    return out


def plot_mean_alpha_return(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    if summary.empty:
        return
    sort_cols = (
        ["horizon_order", "branch_order"]
        if "branch_order" in summary
        else ["horizon_order", "branch"]
    )
    table = summary.sort_values(sort_cols)
    horizons = list(table["horizon"].drop_duplicates())
    branch_sort_cols = ["branch_order"] if "branch_order" in table else ["branch"]
    branches = list(table.sort_values(branch_sort_cols)["branch"].drop_duplicates())
    x = np.arange(len(horizons))
    width = 0.78 / max(1, len(branches))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, branch in enumerate(branches):
        branch_table = table.loc[table["branch"] == branch].set_index("horizon")
        values = [
            branch_table.loc[h, "mean_alpha_return_bps"] if h in branch_table.index else np.nan
            for h in horizons
        ]
        errors = [
            branch_table.loc[h, "mean_alpha_return_bps_sem"] if h in branch_table.index else np.nan
            for h in horizons
        ]
        offset = (idx - (len(branches) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=branch,
            color=RUN_COLORS.get(branch, None),
            alpha=0.9,
        )
        for bar, value in zip(bars, values, strict=True):
            if pd.isna(value):
                continue
            va = "bottom" if value >= 0 else "top"
            y = value + (1.2 if value >= 0 else -1.2)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.1f}",
                ha="center",
                va=va,
                fontsize=8,
            )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([h.replace("_", "\n") for h in horizons])
    ax.set_ylabel("Mean alpha return (bps)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rank_ic(summary: pd.DataFrame, output_path: Path, *, title: str) -> None:
    if summary.empty or "group_rank_ic_mean" not in summary:
        return
    sort_cols = (
        ["horizon_order", "branch_order"]
        if "branch_order" in summary
        else ["horizon_order", "branch"]
    )
    table = summary.sort_values(sort_cols)
    horizons = list(table["horizon"].drop_duplicates())
    branch_sort_cols = ["branch_order"] if "branch_order" in table else ["branch"]
    branches = list(table.sort_values(branch_sort_cols)["branch"].drop_duplicates())
    x = np.arange(len(horizons), dtype="float64")
    width = 0.78 / max(1, len(branches))

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for idx, branch in enumerate(branches):
        branch_table = table.loc[table["branch"] == branch].set_index("horizon")
        values = [
            float(branch_table.loc[horizon, "group_rank_ic_mean"])
            if horizon in branch_table.index
            else np.nan
            for horizon in horizons
        ]
        offset = (idx - (len(branches) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=branch,
            color=RUN_COLORS.get(branch, None),
            alpha=0.9,
        )
        for bar, value in zip(bars, values, strict=True):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.004 if value >= 0 else -0.004),
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0.0, color="#666666", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([horizon.replace("_", "\n") for horizon in horizons])
    ax.set_ylabel("Group rank IC mean")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
