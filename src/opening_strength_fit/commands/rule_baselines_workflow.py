import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import config_int, config_str, load_toml
from opening_strength_fit.evaluation import (
    format_group_cols,
    group_cols_for_mode,
    score_bucket_returns,
    summarize_trades,
    top_score_trades,
)
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.model import evaluate_prediction_frame
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.rules import available_rule_scores, rule_prediction_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple rule scores before model training."
    )
    parser.add_argument("--input", required=True, help="Labeled research parquet/csv path.")
    parser.add_argument("--config", default="", help="Optional run TOML for evaluation settings.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rules", nargs="*", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = load_toml(args.config) if args.config else {}
    frame = read_frame(args.input)
    if args.start_date:
        frame = frame.loc[frame["date"].astype(str) >= args.start_date].copy()
    if args.end_date:
        frame = frame.loc[frame["date"].astype(str) <= args.end_date].copy()
    if "valid_label" in frame.columns:
        frame = frame.loc[frame["valid_label"]].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_mode = config_str(config, "evaluation", "bucket_mode", "daily")
    selection_mode = config_str(config, "evaluation", "selection_mode", "symbol_day")
    ic_mode = config_str(config, "evaluation", "ic_mode", bucket_mode)
    bucket_group_cols = group_cols_for_mode(bucket_mode)
    selection_group_cols = group_cols_for_mode(selection_mode)
    ic_group_cols = group_cols_for_mode(ic_mode)
    bins = config_int(config, "evaluation", "score_bins", 10)
    top_n = config_int(config, "evaluation", "top_n", 20)

    metrics_rows = []
    bucket_frames = []
    prediction_frames = []
    scores = available_rule_scores(frame, rules=args.rules)
    if not scores:
        raise SystemExit("no requested rule score columns are available in input")

    for rule_name, score in scores.items():
        predictions = rule_prediction_frame(frame, rule_name=rule_name, score=score)
        metrics = evaluate_prediction_frame(predictions, group_cols=ic_group_cols)
        buckets = score_bucket_returns(
            predictions,
            bins=bins,
            group_cols=bucket_group_cols,
        )
        top_trades = top_score_trades(
            predictions,
            top_n=top_n,
            group_cols=selection_group_cols,
        )
        top_summary = summarize_trades(top_trades, group_cols=selection_group_cols)
        metrics_rows.append(
            {
                "rule_name": rule_name,
                **metrics,
                **{f"top_score_{key}": value for key, value in top_summary.items()},
            }
        )
        bucket_frames.append(buckets.assign(rule_name=rule_name))
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("group_rank_ic_mean", ascending=False)
    buckets_df = pd.concat(bucket_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_df.to_csv(output_dir / "rule_baseline_metrics.csv", index=False)
    buckets_df.to_csv(output_dir / "rule_baseline_buckets.csv", index=False)
    write_frame(predictions_df, output_dir / "rule_baseline_predictions.parquet")

    print_mapping("labeled_dataset", dataset_summary(frame))
    print_mapping(
        "rule_baseline_settings",
        {
            "rules": ",".join(scores),
            "score_bucket_group_cols": format_group_cols(bucket_group_cols),
            "selection_group_cols": format_group_cols(selection_group_cols),
            "ic_group_cols": format_group_cols(ic_group_cols),
            "bins": bins,
            "top_n": top_n,
            "output_dir": str(output_dir),
        },
    )
    print("\nrule_baseline_metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
