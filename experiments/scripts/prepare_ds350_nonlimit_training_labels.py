from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
LABEL_COLUMNS = [*KEYS, "label_short", "label_next_close", "target_label"]
VARIANTS = (
    "close_z_all",
    "mixed_nonup",
    "mixed_ordinary",
    "rank_mixed_all",
    "rank_close_all",
    "rank_close_nonup",
    "rank_close_ordinary",
)


def _date_key(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        finite = pd.to_numeric(values, errors="coerce").dropna()
        median = float(finite.median()) if not finite.empty else np.nan
        if np.isfinite(median) and median >= 10_000_000:
            parsed = pd.to_datetime(
                values.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
            )
        else:
            parsed = pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _symbol_key(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        )
    )


def _status_for_labels(labels: pd.DataFrame, reference_path: Path) -> pd.Series:
    reference = pd.read_parquet(
        reference_path,
        columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
    )
    reference["_date"] = _date_key(reference["TradingDay"])
    reference["_symbol"] = _symbol_key(reference["Symbol"])
    reference["_status"] = pd.to_numeric(reference["UpdownLimitStatus"], errors="coerce")
    reference = reference.dropna(subset=["_date", "_symbol", "_status"])
    duplicate = reference.duplicated(["_date", "_symbol"], keep=False)
    if duplicate.any():
        conflicts = (
            reference.loc[duplicate].groupby(["_date", "_symbol"], sort=False)["_status"].nunique()
        )
        if conflicts.gt(1).any():
            raise SystemExit("daily reference has conflicting limit states")
        reference = reference.drop_duplicates(["_date", "_symbol"], keep="last")

    mapping = reference.set_index(["_date", "_symbol"])["_status"]
    label_index = pd.MultiIndex.from_arrays(
        [_date_key(labels["date"]), _symbol_key(labels["symbol"])],
        names=["_date", "_symbol"],
    )
    status = pd.Series(mapping.reindex(label_index).to_numpy(), index=labels.index)
    missing = int(status.isna().sum())
    maximum_missing = max(1_000, int(len(labels) * 0.001))
    if missing > maximum_missing:
        raise SystemExit(
            "daily reference coverage is unexpectedly low: "
            f"missing={missing:,}/{len(labels):,}, allowed={maximum_missing:,}"
        )
    if missing:
        print(
            json.dumps(
                {
                    "warning": "missing_daily_limit_state",
                    "missing_rows": missing,
                    "total_rows": int(len(labels)),
                }
            )
        )
    return status


def _rank_centered(values: pd.Series, labels: pd.DataFrame, valid: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = valid & numeric.notna()
    grouped = numeric.where(usable).groupby([labels[column] for column in GROUPS], sort=False)
    rank_pct = grouped.rank(method="average", pct=True)
    count = grouped.transform("count")
    rank_mean = (count + 1.0) / (2.0 * count)
    return (rank_pct - rank_mean).where(usable)


def _zscore(values: pd.Series, labels: pd.DataFrame) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = numeric.groupby([labels[column] for column in GROUPS], sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    return ((numeric - mean) / std.where(std > 1e-12)).where(numeric.notna())


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _variant_targets(
    labels: pd.DataFrame,
    status: pd.Series,
    *,
    training_year: bool,
) -> dict[str, pd.Series]:
    all_rows = pd.Series(True, index=labels.index)
    known = status.notna()
    nonup = (known & status.ne(1)) if training_year else all_rows
    ordinary = status.eq(0) if training_year else all_rows
    mixed = pd.to_numeric(labels["target_label"], errors="coerce")
    close = pd.to_numeric(labels["label_short"], errors="coerce")
    rank_close_all = _rank_centered(close, labels, all_rows)
    return {
        "close_z_all": _zscore(close, labels),
        "mixed_nonup": mixed.where(nonup),
        "mixed_ordinary": mixed.where(ordinary),
        "rank_mixed_all": _rank_centered(mixed, labels, all_rows),
        "rank_close_all": rank_close_all,
        "rank_close_nonup": (
            _rank_centered(close, labels, nonup) if training_year else rank_close_all
        ),
        "rank_close_ordinary": (
            _rank_centered(close, labels, ordinary) if training_year else rank_close_all
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-end-year", type=int, default=2025)
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()

    summary: dict[str, object] = {
        "source_root": str(args.source_root),
        "raw_source_root": str(args.raw_source_root),
        "output_root": str(args.output_root),
        "train_end_year": args.train_end_year,
        "variants": {},
    }
    for variant in VARIANTS:
        summary["variants"][variant] = {}

    for year in args.years:
        source = args.source_root / f"year={year}" / "labels.parquet"
        reference = args.raw_source_root / f"year={year}" / "daily_reference.parquet"
        if not source.exists() or not reference.exists():
            raise SystemExit(f"missing source files for year={year}: {source}, {reference}")
        labels = pd.read_parquet(source, columns=LABEL_COLUMNS)
        if labels.duplicated(KEYS).any():
            raise SystemExit(f"label keys are not unique for year={year}")
        status = _status_for_labels(labels, reference)
        targets = _variant_targets(
            labels,
            status,
            training_year=year <= args.train_end_year,
        )
        for variant, target in targets.items():
            output = labels.copy(deep=False)
            output["target_label"] = target.astype("float32")
            year_root = args.output_root / variant / f"year={year}"
            _atomic_parquet(output, year_root / "labels.parquet")
            (year_root / "_SUCCESS").touch()
            details = {
                "rows": int(len(output)),
                "target_rows": int(output["target_label"].notna().sum()),
                "up_limit_rows": int(status.eq(1).sum()),
                "down_limit_rows": int(status.eq(-1).sum()),
                "ordinary_rows": int(status.eq(0).sum()),
                "unknown_limit_state_rows": int(status.isna().sum()),
            }
            (year_root / "manifest.json").write_text(
                json.dumps(
                    {"variant": variant, "year": year, **details},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            summary["variants"][variant][str(year)] = details
            print(json.dumps({"variant": variant, "year": year, **details}))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "_READY").touch()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
