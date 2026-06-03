from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.reports import dataset_summary


REQUIRED_LABELED_CACHE_COLUMNS = (
    "date",
    "symbol",
    "timestamp",
    "decision_target_timestamp",
    "label",
    "valid_label",
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if pd.isna(value):
        return None
    return value


def _config_fingerprint(config: dict) -> str:
    payload = json.dumps(_json_ready(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numeric_distribution(series: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _column_schema(
    frame: pd.DataFrame,
    *,
    schema_columns: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    if schema_columns is not None:
        return [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in schema_columns
        ]
    return [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in frame.dtypes.items()
    ]


def build_cache_manifest(
    frame: pd.DataFrame,
    *,
    cache_path: str | Path,
    config: dict,
    run_name: str,
    config_path: str | Path | None = None,
    schema_columns: Sequence[tuple[str, str]] | None = None,
    row_count: int | None = None,
) -> dict[str, object]:
    """Build a compact, JSON-safe manifest for a labeled cache shard."""

    schema = _column_schema(frame, schema_columns=schema_columns)
    column_names = [item["name"] for item in schema]
    summary = dataset_summary(frame)
    if row_count is not None:
        summary["rows"] = int(row_count)
    if schema_columns is not None:
        summary["columns"] = len(schema)

    missing_required = [
        column for column in REQUIRED_LABELED_CACHE_COLUMNS if column not in column_names
    ]
    decision_time_counts: dict[str, int] = {}
    if "decision_time" in frame.columns:
        counts = frame["decision_time"].astype(str).value_counts().sort_index()
        decision_time_counts = {str(key): int(value) for key, value in counts.items()}

    label_summary: dict[str, object] = {}
    if "label" in frame.columns:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        valid = (
            frame["valid_label"].fillna(False).astype(bool)
            if "valid_label" in frame.columns
            else labels.notna()
        )
        label_summary = {
            "non_null_labels": int(labels.notna().sum()),
            "valid_labels": int(valid.sum()),
            "valid_label_ratio": float(valid.mean()) if len(valid) else 0.0,
        }

    timing_summary: dict[str, object] = {}
    for column in (
        "decision_lag_seconds",
        "entry_delay_ticks",
        "entry_delay_seconds",
        "entry_max_tick_gap_seconds",
    ):
        if column in frame.columns:
            timing_summary[column] = _numeric_distribution(frame[column])

    return {
        "manifest_version": 1,
        "cache_schema_version": str(
            config.get("cache", {}).get("schema_version", "base_labeled_v1")
        ),
        "run_id": run_name,
        "cache_path": str(cache_path),
        "config_path": str(config_path or ""),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_fingerprint": _config_fingerprint(config),
        "summary": summary,
        "required_columns": {
            "expected": list(REQUIRED_LABELED_CACHE_COLUMNS),
            "missing": missing_required,
        },
        "decision_time_counts": decision_time_counts,
        "label_summary": label_summary,
        "timing_summary": timing_summary,
        "schema": schema,
        "config": {
            "data": config.get("data", {}),
            "clickhouse": config.get("clickhouse", {}),
            "universe": config.get("universe", {}),
            "sample": config.get("sample", {}),
            "labels": config.get("labels", {}),
            "features": config.get("features", {}),
            "filters": config.get("filters", {}),
            "cache": config.get("cache", {}),
        },
    }


def write_cache_manifest(manifest: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(manifest),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
