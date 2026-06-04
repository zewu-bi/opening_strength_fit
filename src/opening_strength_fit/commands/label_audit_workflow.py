import argparse
from pathlib import Path

from opening_strength_fit.io import read_frame
from opening_strength_fit.label_audit import (
    DEFAULT_FEE_BPS,
    summarize_label_distribution,
)
from opening_strength_fit.reports import dataset_summary, print_mapping


def _parse_fees(value: str) -> list[float]:
    return [float(part) for part in value.replace(",", " ").split() if part]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit opening short-horizon labels before model training."
    )
    parser.add_argument("--input", required=True, help="Labeled research parquet/csv path.")
    parser.add_argument("--output", required=True, help="Audit CSV output path.")
    parser.add_argument(
        "--fees-bps",
        default=",".join(str(value) for value in DEFAULT_FEE_BPS),
        help="Comma/space separated fee bps values, e.g. 0,5,10,13.",
    )
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=["year", "month", "minute_bucket"],
        help="Audit grouping columns after adding year/month/minute_bucket.",
    )
    args = parser.parse_args()

    frame = read_frame(args.input)
    audit = summarize_label_distribution(
        frame,
        fee_bps_values=_parse_fees(args.fees_bps),
        group_cols=tuple(args.group_cols),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)

    print_mapping("labeled_dataset", dataset_summary(frame))
    print_mapping(
        "label_audit",
        {
            "rows": len(audit),
            "fees_bps": args.fees_bps,
            "group_cols": ",".join(args.group_cols),
            "output": str(output),
        },
    )


if __name__ == "__main__":
    main()
