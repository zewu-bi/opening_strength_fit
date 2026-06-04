import argparse

from opening_strength_fit.evaluation import (
    format_group_cols,
    group_cols_for_mode,
    score_bucket_returns,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.io import read_frame
from opening_strength_fit.model import evaluate_prediction_frame
from opening_strength_fit.reports import print_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction parquet/csv.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--bucket-mode",
        default="daily",
        choices=["global", "daily", "symbol_day", "cross_section"],
        help="Grouping used before assigning score buckets.",
    )
    parser.add_argument(
        "--selection-mode",
        default="symbol_day",
        choices=["global", "daily", "symbol_day", "cross_section"],
        help=(
            "Grouping used before selecting top score trades. "
            "symbol_day means top ticks per symbol/day; cross_section means "
            "top symbols per decision point."
        ),
    )
    parser.add_argument(
        "--ic-mode",
        default="daily",
        choices=["global", "daily", "symbol_day", "cross_section"],
        help="Grouping used to compute mean IC/rank IC.",
    )
    args = parser.parse_args()

    predictions = read_frame(args.input)
    bucket_group_cols = group_cols_for_mode(args.bucket_mode)
    selection_group_cols = group_cols_for_mode(args.selection_mode)
    ic_group_cols = group_cols_for_mode(args.ic_mode)
    metrics = evaluate_prediction_frame(
        predictions,
        label_col=args.label_col,
        score_col=args.score_col,
        group_cols=ic_group_cols,
    )
    buckets = score_bucket_returns(
        predictions,
        bins=args.bins,
        label_col=args.label_col,
        score_col=args.score_col,
        group_cols=bucket_group_cols,
    )
    top_trades = top_score_trades(
        predictions,
        top_n=args.top_n,
        label_col=args.label_col,
        score_col=args.score_col,
        group_cols=selection_group_cols,
    )
    print_mapping(
        "evaluation_settings",
        {
            "score_bucket_mode": args.bucket_mode,
            "score_bucket_group_cols": format_group_cols(bucket_group_cols),
            "selection_mode": args.selection_mode,
            "selection_group_cols": format_group_cols(selection_group_cols),
            "ic_mode": args.ic_mode,
            "ic_group_cols": format_group_cols(ic_group_cols),
            "top_n": args.top_n,
        },
    )
    print_mapping("prediction_metrics", metrics)
    print("\nscore_buckets:")
    print(buckets.to_string(index=False))
    print_mapping(
        f"top_score_summary[top_n={args.top_n}]",
        summarize_trades(top_trades, group_cols=selection_group_cols),
    )


if __name__ == "__main__":
    main()
