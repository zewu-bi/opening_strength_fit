import argparse

import _bootstrap  # noqa: F401
from opening_strength_fit.dataset import build_labeled_feature_frame, load_ticks
from opening_strength_fit.io import write_frame
from opening_strength_fit.reports import dataset_summary, print_mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature and trade-label rows from opening tick data."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--buy-price-col", default="ask_price_1")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--turnover-col", default="turnover")
    parser.add_argument("--hold-seconds", type=int, default=60)
    parser.add_argument("--sell-window-seconds", type=int, default=60)
    parser.add_argument("--volume-unit-multiplier", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-tick-delay", type=int, default=0)
    parser.add_argument(
        "--entry-max-gap-seconds",
        type=int,
        default=None,
        help="Maximum adjacent tick gap on the decision-to-entry path.",
    )
    parser.add_argument("--sample-start-time", default="09:30:00")
    parser.add_argument("--sample-end-time", default="09:40:00")
    parser.add_argument("--max-future-gap-seconds", type=int, default=None)
    parser.add_argument("--no-preopen", action="store_true")
    args = parser.parse_args()

    ticks = load_ticks(args.input)
    labeled = build_labeled_feature_frame(
        ticks,
        buy_price_col=args.buy_price_col,
        volume_col=args.volume_col,
        turnover_col=args.turnover_col,
        hold_seconds=args.hold_seconds,
        sell_window_seconds=args.sell_window_seconds,
        volume_unit_multiplier=args.volume_unit_multiplier,
        fee_bps=args.fee_bps,
        entry_tick_delay=args.entry_tick_delay,
        entry_max_gap_seconds=args.entry_max_gap_seconds,
        sample_start_time=args.sample_start_time,
        sample_end_time=args.sample_end_time,
        include_preopen=not args.no_preopen,
        max_future_gap_seconds=args.max_future_gap_seconds,
    )
    write_frame(labeled, args.output)
    print_mapping("labeled_dataset", dataset_summary(labeled))
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
