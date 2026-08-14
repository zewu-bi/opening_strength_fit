from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

WINDOWS = (
    ("09:31-09:40", "0931_0940"),
    ("10:01-10:10", "1001_1010"),
    ("14:01-14:10", "1401_1410"),
)
HORIZONS = (("1m", "short1m"), ("3m", "short3m"))
RUN_TEMPLATE = "nn_v6_w{window}_{horizon}_corrected_nextclose_36m_grouped_gated_v2_mse"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_row(path: Path, pool: str) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["test_month"].astype(str).eq("Mean") & frame["pool"].astype(str).eq(pool)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one Mean/{pool} row in {path}, got {len(selected)}")
    return selected.iloc[0]


def build_summary(backtests_root: Path, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for window_label, window_key in WINDOWS:
        for horizon_label, horizon_key in HORIZONS:
            run_id = RUN_TEMPLATE.format(window=window_key, horizon=horizon_key)
            run_dir = backtests_root / run_id
            short_path = run_dir / "short_excess_rank_ic_plot_data.csv"
            next_path = run_dir / "next_excess_rank_ic_plot_data.csv"
            short_universe = _mean_row(short_path, "universe")
            short_pool = _mean_row(short_path, "pool_L")
            next_pool = _mean_row(next_path, "pool_L")
            rows.append(
                {
                    "window": window_label,
                    "holding_horizon": horizon_label,
                    "run_id": run_id,
                    "universe_short_rank_ic": float(short_universe["short_rank_ic"]),
                    "pool_L_top100_short_excess_bps": float(
                        short_pool["short_internal_excess_bps"]
                    ),
                    "pool_L_top100_overnight_excess_bps": float(
                        next_pool["next_internal_excess_bps"]
                    ),
                    "pool_L_overnight_rank_ic": float(next_pool["next_rank_ic"]),
                }
            )
            for path in (short_path, next_path):
                relative = path.as_posix()
                sources[relative] = {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "corrected_label_3x2_summary.csv"
    summary.to_csv(summary_path, index=False, lineterminator="\n")

    metrics = [
        "universe_short_rank_ic",
        "pool_L_top100_short_excess_bps",
        "pool_L_top100_overnight_excess_bps",
        "pool_L_overnight_rank_ic",
    ]
    effects: list[dict[str, object]] = []
    for window_label, _ in WINDOWS:
        pair = summary.loc[summary["window"].eq(window_label)].set_index("holding_horizon")
        for metric in metrics:
            effects.append(
                {
                    "effect": "holding_horizon_3m_minus_1m",
                    "slice": window_label,
                    "metric": metric,
                    "delta": float(pair.loc["3m", metric] - pair.loc["1m", metric]),
                }
            )
    for horizon_label, _ in HORIZONS:
        windowed = summary.loc[summary["holding_horizon"].eq(horizon_label)].set_index("window")
        for later_window in ("10:01-10:10", "14:01-14:10"):
            for metric in metrics:
                effects.append(
                    {
                        "effect": "window_minus_09:31-09:40",
                        "slice": f"{later_window} @ {horizon_label}",
                        "metric": metric,
                        "delta": float(
                            windowed.loc[later_window, metric] - windowed.loc["09:31-09:40", metric]
                        ),
                    }
                )
    effects_frame = pd.DataFrame(effects)
    effects_path = output_dir / "corrected_label_3x2_effects.csv"
    effects_frame.to_csv(effects_path, index=False, lineterminator="\n")

    markdown = summary[
        [
            "window",
            "holding_horizon",
            "universe_short_rank_ic",
            "pool_L_top100_short_excess_bps",
            "pool_L_top100_overnight_excess_bps",
            "pool_L_overnight_rank_ic",
        ]
    ].copy()
    markdown.columns = [
        "时间窗口",
        "持有期",
        "短期 IC",
        "短期超额(bps)",
        "隔夜超额(bps)",
        "隔夜 IC",
    ]
    markdown_path = output_dir / "corrected_label_3x2_summary.md"
    header = "| " + " | ".join(markdown.columns) + " |"
    separator = "| " + " | ".join(["---", "---", "---:", "---:", "---:", "---:"]) + " |"
    body = [
        "| "
        + " | ".join(
            [str(row.iloc[0]), str(row.iloc[1])] + [f"{float(value):.6f}" for value in row.iloc[2:]]
        )
        + " |"
        for _, row in markdown.iterrows()
    ]
    markdown_path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")

    outputs = (summary_path, effects_path, markdown_path)
    optional_outputs = tuple(
        path
        for path in (
            output_dir / "02_top100_cumulative_3x2.csv",
            output_dir / "02_top100_cumulative_3x2.svg",
            output_dir / "02_top100_cumulative_3x2_trace.json",
        )
        if path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "corrected-label three-window by two-short-horizon rolling-OOS grid",
        "period": "2022-01 through 2025-12",
        "definitions": {
            "short_ic": "Mean universe short Rank IC across quarterly OOS summaries",
            "short_excess": "Mean pool_L Top100 short internal excess in bps",
            "overnight_excess": "Mean pool_L Top100 next-close internal excess in bps",
            "overnight_ic": "Mean pool_L next-close Rank IC",
        },
        "sources": sources,
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (*outputs, *optional_outputs)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtests-root", type=Path, default=Path("experiments/results/backtests")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/backtests/corrected_label_3x2_grid_2022_2025_v1"),
    )
    args = parser.parse_args()
    build_summary(args.backtests_root, args.output_dir)
    print(f"wrote corrected-label 3x2 summary: {args.output_dir}")


if __name__ == "__main__":
    main()
