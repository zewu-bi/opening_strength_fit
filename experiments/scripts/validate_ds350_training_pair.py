from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
LABEL_COLUMNS = (*KEY_COLUMNS, "label_short", "label_next_close", "target_label")


def _key_hashes(path: Path) -> np.ndarray:
    parts = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=list(KEY_COLUMNS), batch_size=500_000):
        frame = batch.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
        frame["symbol"] = frame["symbol"].astype(str)
        frame["decision_target_timestamp"] = pd.to_datetime(
            frame["decision_target_timestamp"], errors="raise"
        )
        parts.append(
            pd.util.hash_pandas_object(frame.loc[:, list(KEY_COLUMNS)], index=False).to_numpy(
                dtype="uint64", copy=False
            )
        )
    return np.concatenate(parts) if parts else np.empty(0, dtype="uint64")


def validate_year(
    feature_root: Path,
    label_roots: list[Path],
    *,
    year: int,
    expected_feature_count: int,
) -> dict[str, object]:
    feature_year = feature_root / f"year={year}"
    feature_path = feature_year / "features.parquet"
    if not feature_path.exists() or not (feature_year / "_SUCCESS").exists():
        raise SystemExit(f"incomplete feature year: {feature_year}")
    feature_file = pq.ParquetFile(feature_path)
    feature_names = feature_file.schema_arrow.names
    if feature_names[: len(KEY_COLUMNS)] != list(KEY_COLUMNS):
        raise SystemExit(f"unexpected feature keys: {feature_names[: len(KEY_COLUMNS)]}")
    if len(feature_names) - len(KEY_COLUMNS) != expected_feature_count:
        raise SystemExit(
            f"feature count mismatch year={year}: "
            f"expected={expected_feature_count} actual={len(feature_names) - len(KEY_COLUMNS)}"
        )
    if set(LABEL_COLUMNS[3:]).intersection(feature_names):
        raise SystemExit(f"feature file contains label columns: {feature_path}")

    feature_hashes = _key_hashes(feature_path)
    if len(np.unique(feature_hashes)) != len(feature_hashes):
        raise SystemExit(f"duplicate feature keys year={year}")
    feature_key_digest = hashlib.sha256(feature_hashes.tobytes()).hexdigest()
    reports = []
    for label_root in label_roots:
        label_year = label_root / f"year={year}"
        label_path = label_year / "labels.parquet"
        if not label_path.exists() or not (label_year / "_SUCCESS").exists():
            raise SystemExit(f"incomplete label year: {label_year}")
        label_file = pq.ParquetFile(label_path)
        if label_file.schema_arrow.names != list(LABEL_COLUMNS):
            raise SystemExit(
                f"unexpected label schema: {label_path} {label_file.schema_arrow.names}"
            )
        if label_file.metadata.num_rows != feature_file.metadata.num_rows:
            raise SystemExit(
                f"feature/label row mismatch year={year} root={label_root}: "
                f"features={feature_file.metadata.num_rows} labels={label_file.metadata.num_rows}"
            )
        label_hashes = _key_hashes(label_path)
        if not np.array_equal(feature_hashes, label_hashes):
            raise SystemExit(f"feature/label key order mismatch year={year} root={label_root}")
        target = pq.read_table(label_path, columns=["target_label"])["target_label"]
        valid_target_rows = int(pc.sum(pc.is_valid(target)).as_py())
        reports.append(
            {
                "label_root": str(label_root),
                "rows": int(label_file.metadata.num_rows),
                "valid_target_rows": valid_target_rows,
                "valid_target_ratio": valid_target_rows / int(label_file.metadata.num_rows),
                "key_sha256": hashlib.sha256(label_hashes.tobytes()).hexdigest(),
            }
        )

    timestamps = pq.read_table(feature_path, columns=["decision_target_timestamp"])[
        "decision_target_timestamp"
    ].to_pandas()
    clocks = sorted(pd.to_datetime(timestamps, errors="raise").dt.strftime("%H:%M:%S").unique())
    expected_clocks = [f"11:{minute:02d}:00" for minute in range(1, 11)]
    if clocks != expected_clocks:
        raise SystemExit(f"decision clocks mismatch year={year}: {clocks}")
    return {
        "year": year,
        "feature_root": str(feature_root),
        "feature_rows": int(feature_file.metadata.num_rows),
        "feature_count": expected_feature_count,
        "feature_key_sha256": feature_key_digest,
        "decision_times": clocks,
        "labels": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model-ready DS350 feature/label pairs.")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", action="append", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--expected-feature-count", type=int, default=350)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_year(
        Path(args.feature_root),
        [Path(value) for value in args.label_root],
        year=args.year,
        expected_feature_count=args.expected_feature_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f"{output.suffix}.partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    output.with_name("_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
