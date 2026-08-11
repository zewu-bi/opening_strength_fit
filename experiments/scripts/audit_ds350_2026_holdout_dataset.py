from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.schema import normalize_date_series as _normalized_dates

KEY_COLUMNS = ["date", "symbol", "decision_target_timestamp"]


def _manifest(root: Path, year: int) -> dict[str, object]:
    year_root = root / f"year={year}"
    if not (year_root / "_SUCCESS").exists():
        raise RuntimeError(f"missing _SUCCESS: {year_root}")
    return json.loads((year_root / "manifest.json").read_text(encoding="utf-8"))


def _parquet(root: Path, year: int) -> Path:
    path = (
        root
        / f"year={year}"
        / ("features.parquet" if "features_350" in root.name else "labels.parquet")
    )
    if not path.exists():
        raise RuntimeError(f"missing parquet: {path}")
    return path


def _schema(path: Path) -> list[tuple[str, str]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [(field.name, str(field.type)) for field in schema]


def _key_table(path: Path):
    return pq.read_table(path, columns=KEY_COLUMNS).combine_chunks()


def _predecessors(calendar: list[str], missing: list[str]) -> list[str]:
    positions = {date: index for index, date in enumerate(calendar)}
    return [calendar[positions[date] - 1] for date in missing if positions.get(date, 0) > 0]


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    feature_root = output_root / "datasets/opening_0931_0940_features_350"
    h1m_root = output_root / "datasets/opening_0931_0940_labels_h1m_v2"
    close_root = output_root / "datasets/opening_0931_0940_labels_hclose_v1"
    raw_root = output_root / "cache/opening_0931_0940_raw_source"

    roots = {"features": feature_root, "h1m": h1m_root, "hclose": close_root}
    manifests = {
        name: {str(year): _manifest(root, year) for year in (2025, 2026)}
        for name, root in roots.items()
    }
    schemas = {
        name: {str(year): _schema(_parquet(root, year)) for year in (2025, 2026)}
        for name, root in roots.items()
    }
    schema_equal = {name: schemas[name]["2025"] == schemas[name]["2026"] for name in roots}

    paths_2026 = {name: _parquet(root, 2026) for name, root in roots.items()}
    key_tables = {name: _key_table(path) for name, path in paths_2026.items()}
    key_order_equal = {
        "features_vs_h1m": key_tables["features"].equals(key_tables["h1m"]),
        "features_vs_hclose": key_tables["features"].equals(key_tables["hclose"]),
        "h1m_vs_hclose": key_tables["h1m"].equals(key_tables["hclose"]),
    }

    keys = key_tables["features"].to_pandas()
    keys["date"] = _normalized_dates(keys["date"])
    duplicate_key_rows = int(keys.duplicated(KEY_COLUMNS, keep=False).sum())
    h1_mask = keys["date"].between("2026-01-01", "2026-06-30")
    h1_keys = keys.loc[h1_mask]

    raw_year_root = raw_root / "year=2026"
    raw_manifest = _manifest(raw_root, 2026)
    tick_dates = sorted(
        path.stem.removeprefix("date=") for path in (raw_year_root / "ticks").glob("date=*.parquet")
    )
    daily = pd.read_parquet(raw_year_root / "daily_reference.parquet", columns=["TradingDay"])
    daily_dates = sorted(
        date
        for date in _normalized_dates(daily["TradingDay"]).dropna().unique()
        if date.startswith("2026-")
    )
    h1_tick_dates = [date for date in tick_dates if date <= "2026-06-30"]
    h1_daily_dates = [date for date in daily_dates if date <= "2026-06-30"]
    daily_not_tick = sorted(set(h1_daily_dates) - set(h1_tick_dates))
    tick_not_daily = sorted(set(h1_tick_dates) - set(h1_daily_dates))
    gap_predecessors = _predecessors(h1_daily_dates, daily_not_tick)

    label_columns = ["label_short", "label_next_close", "target_label"]
    hclose = pd.read_parquet(paths_2026["hclose"], columns=["date", *label_columns])
    hclose["date"] = _normalized_dates(hclose["date"])
    hclose_h1 = hclose.loc[hclose["date"].between("2026-01-01", "2026-06-30")]
    non_null_rows = {column: int(hclose_h1[column].notna().sum()) for column in label_columns}
    excluded_rows = int(hclose_h1["date"].isin(gap_predecessors).sum())
    june30 = hclose_h1.loc[hclose_h1["date"].eq("2026-06-30")]

    result: dict[str, object] = {
        "status": "ok",
        "source_tables": {"tick": "stock.tick", "daily": "stock.daily_bar_jy"},
        "raw_schema_version": raw_manifest.get("schema_version"),
        "schema_equal_2025_2026": schema_equal,
        "key_order_equal": key_order_equal,
        "duplicate_feature_key_rows": duplicate_key_rows,
        "rows_2026": {name: int(manifests[name]["2026"]["rows"]) for name in roots},
        "h1": {
            "date_start": str(h1_keys["date"].min()),
            "date_end": str(h1_keys["date"].max()),
            "sample_dates": int(h1_keys["date"].nunique()),
            "sample_rows": int(len(h1_keys)),
            "decision_clocks": int(h1_keys["decision_target_timestamp"].dt.time.nunique()),
            "daily_calendar_dates": len(h1_daily_dates),
            "tick_dates": len(h1_tick_dates),
            "daily_not_tick": daily_not_tick,
            "tick_not_daily": tick_not_daily,
            "gap_predecessor_dates_to_exclude_for_next_close": gap_predecessors,
            "gap_predecessor_rows": excluded_rows,
            "hclose_non_null_rows": non_null_rows,
            "june30_rows": int(len(june30)),
            "june30_next_close_non_null_rows": int(june30["label_next_close"].notna().sum()),
        },
        "manifests": manifests,
    }

    expected_gaps = ["2026-03-19", "2026-04-23", "2026-05-07"]
    checks = {
        "all_schemas_equal": all(schema_equal.values()),
        "all_key_orders_equal": all(key_order_equal.values()),
        "no_duplicate_feature_keys": duplicate_key_rows == 0,
        "h1_reaches_june30": result["h1"]["date_end"] == "2026-06-30",
        "june30_has_next_close": result["h1"]["june30_next_close_non_null_rows"] > 0,
        "only_known_tick_gaps": daily_not_tick == expected_gaps and not tick_not_daily,
        "expected_feature_count": int(manifests["features"]["2026"].get("feature_count", -1))
        == 350,
    }
    result["checks"] = checks
    if not all(checks.values()):
        result["status"] = "failed"

    output_dir = Path(args.audit_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if result["status"] != "ok":
        raise RuntimeError(f"dataset audit failed: {checks}")
    (output_dir / "_SUCCESS").touch()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/mnt/output/opening_strength_fit")
    parser.add_argument(
        "--audit-output",
        default="/mnt/output/opening_strength_fit/audits/ds350_2026_holdout_dataset_v1",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
