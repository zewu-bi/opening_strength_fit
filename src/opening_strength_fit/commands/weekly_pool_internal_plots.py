from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.pool_internal_plots import (
    slug_label,
    write_weekly_pool_internal_rolling_plot,
)
from opening_strength_fit.pool_internal_weekly import (
    POOL_CHOICES,
    build_weekly_pool_internal_summaries,
    normalize_pools,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render weekly pool-internal diagnostics from pool_internal_group_metrics.csv. "
            "The weekly and rolling summaries are equal weighted by trading day."
        )
    )
    parser.add_argument("--group-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="baseline")
    parser.add_argument(
        "--plot-variant-label",
        default="",
        help="Display label used in the generated SVG title. Defaults to --output-prefix.",
    )
    parser.add_argument(
        "--pool",
        action="append",
        choices=POOL_CHOICES,
        help="Pools to include. Defaults to universe, S, M, and L.",
    )
    parser.add_argument("--rolling-weeks", type=int, default=4)
    parser.add_argument(
        "--top-worst",
        type=int,
        default=5,
        help="Worst single-week and rolling windows to include per pool/horizon.",
    )
    return parser.parse_args()


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    args = parse_args()
    pools = normalize_pools(args.pool)
    group_metrics_path = Path(args.group_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_metrics = pd.read_csv(group_metrics_path)
    daily, weekly, overall, worst = build_weekly_pool_internal_summaries(
        group_metrics,
        pools=pools,
        rolling_weeks=args.rolling_weeks,
        top_worst=args.top_worst,
    )

    daily_path = output_dir / "daily_pool_internal_summary.csv"
    weekly_path = output_dir / "weekly_pool_internal_summary.csv"
    overall_path = output_dir / "weekly_pool_internal_overall_summary.csv"
    worst_path = output_dir / "weekly_worst_windows.csv"
    _csv_ready(daily).to_csv(daily_path, index=False, float_format="%.6f")
    _csv_ready(weekly).to_csv(weekly_path, index=False, float_format="%.6f")
    overall.to_csv(overall_path, index=False, float_format="%.6f")
    worst.to_csv(worst_path, index=False, float_format="%.6f")

    output_prefix = slug_label(args.output_prefix)
    variant_label = args.plot_variant_label or args.output_prefix
    plot_paths = write_weekly_pool_internal_rolling_plot(
        weekly,
        output_dir,
        input_path=weekly_path,
        output_prefix=output_prefix,
        variant_label=variant_label,
        pools=pools,
        rolling_weeks=args.rolling_weeks,
    )

    trace_path = output_dir / "weekly_pool_internal_trace.json"
    write_json(
        trace_path,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "input": str(group_metrics_path),
            "daily_summary": str(daily_path),
            "weekly_summary": str(weekly_path),
            "overall_summary": str(overall_path),
            "worst_windows": str(worst_path),
            "plot_paths": plot_paths,
            "pools": list(pools),
            "rolling_weeks": args.rolling_weeks,
            "weighting": (
                "date x pool is first averaged across decision clocks; weekly summaries and "
                "rolling windows are equal weighted by trading day"
            ),
        },
        ensure_ascii=True,
    )

    print("weekly_pool_internal_overall_summary:")
    print(overall.to_string(index=False))
    print("\nweekly_pool_internal_outputs:")
    for label, path in {
        "daily": daily_path,
        "weekly": weekly_path,
        "overall": overall_path,
        "worst": worst_path,
        "trace": trace_path,
        **{key: Path(value) for key, value in plot_paths.items()},
    }.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
