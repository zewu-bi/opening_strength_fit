from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.artifact_catalog import file_sha256
from opening_strength_fit.optimization_acceptance_plots import (
    add_cumulative_percent_display_columns,
    combine_net_alpha_cumulative_data,
    ensure_plot_colors,
)
from opening_strength_fit.optimization_direction_data import (
    DirectionSpec,
    line_axis,
    line_step,
    load_realized_cumulative_plot_data,
)
from opening_strength_fit.pool_internal_plot_svg import write_two_panel_line_svg

DIRECTIONS = tuple(
    DirectionSpec(
        key=f"w{window}_{horizon}",
        label=f"{label} / {horizon}",
        run_id=(
            f"nn_v6_w{window}_{'short1m' if horizon == '1m' else 'short3m'}_"
            "corrected_nextclose_36m_grouped_gated_v2_mse"
        ),
    )
    for window, label in (
        ("0931_0940", "09:31-09:40"),
        ("1001_1010", "10:01-10:10"),
        ("1401_1410", "14:01-14:10"),
    )
    for horizon in ("1m", "3m")
)


def build(backtests_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = tuple(direction.key for direction in DIRECTIONS)
    realized = load_realized_cumulative_plot_data(
        backtests_root=backtests_root,
        directions=DIRECTIONS,
        pool="pool_L",
        include_baseline_pool=False,
        include_baseline_universe=False,
        baseline_run_id=DIRECTIONS[0].run_id,
        fee_bps=8.0,
        pool_turnover_path="auto",
        pool_fee_mode="stock_pool_membership",
    )
    cumulative = combine_net_alpha_cumulative_data(realized)
    plot_data = add_cumulative_percent_display_columns(cumulative)
    csv_path = output_dir / "02_top100_cumulative_3x2.csv"
    svg_path = output_dir / "02_top100_cumulative_3x2.svg"
    trace_path = output_dir / "02_top100_cumulative_3x2_trace.json"
    cumulative[
        [
            "pool",
            "pool_label",
            "week_start",
            "next_cumulative_net_return_bps",
            "pool_next_cumulative_net_return_bps",
            "next_cumulative_alpha_bps",
        ]
    ].to_csv(csv_path, index=False, float_format="%.6f")

    ensure_plot_colors(keys)
    net_values = pd.to_numeric(plot_data["next_cumulative_net_return_pct"], errors="coerce")
    alpha_values = pd.to_numeric(plot_data["next_cumulative_alpha_pct"], errors="coerce")
    write_two_panel_line_svg(
        plot_data,
        title="2022-2025 corrected-label 3×2 fee 8bps pool_L Top100 隔夜累和",
        panels=[
            {
                "title": "扣除手续费累和收益",
                "ylabel": "%",
                "column": "next_cumulative_net_return_pct",
                "default_ylim": line_axis(net_values),
                "tick_step": line_step(net_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": "相对各自窗口 pool_L 的费后累和超额",
                "ylabel": "%",
                "column": "next_cumulative_alpha_pct",
                "default_ylim": line_axis(alpha_values),
                "tick_step": line_step(alpha_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=svg_path,
        pools=keys,
        x_label_mode="years_only",
        line_width=2.1,
        line_marker_count=0,
    )

    endpoints = (
        cumulative.sort_values("week_start")
        .groupby(["pool", "pool_label"], sort=False)
        .tail(1)[
            [
                "pool",
                "pool_label",
                "week_start",
                "next_cumulative_net_return_bps",
                "pool_next_cumulative_net_return_bps",
                "next_cumulative_alpha_bps",
            ]
        ]
        .to_dict(orient="records")
    )
    trace_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "fee_bps": 8.0,
                "pool": "pool_L",
                "pool_fee_mode": "stock_pool_membership",
                "next_close_capital_fraction": 0.5,
                "series": [direction.__dict__ for direction in DIRECTIONS],
                "endpoints": endpoints,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.setdefault("outputs", {})
        for path in (csv_path, svg_path, trace_path):
            outputs[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(svg_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtests-root", type=Path, default=Path("experiments/results/backtests")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/evidence/backtests/corrected_label_3x2_grid_2022_2025_v1"),
    )
    args = parser.parse_args()
    build(args.backtests_root, args.output_dir)


if __name__ == "__main__":
    main()
