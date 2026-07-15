from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.io import frame_columns, json_safe, write_json
from opening_strength_fit.reports import dataset_summary

REQUIRED_LABELED_CACHE_COLUMNS = (
    "date",
    "symbol",
    "timestamp",
    "decision_target_timestamp",
    "label",
    "valid_label",
)
MANIFEST_VERSION = 2
FINGERPRINT_SECTIONS = (
    "data",
    "clickhouse",
    "universe",
    "sample",
    "labels",
    "features",
    "filters",
)


def cache_manifest_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_name(f"{path.name}.manifest.json")


def _cache_file_info(cache_path: str | Path) -> dict[str, object]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    stat = path.stat()
    return {"bytes": int(stat.st_size)}


def _fingerprint_config(config: dict) -> dict[str, object]:
    payload = {section: config.get(section, {}) for section in FINGERPRINT_SECTIONS}
    payload["cache"] = {
        "schema_version": config.get("cache", {}).get("schema_version", "base_labeled_v1")
    }
    return payload


def _config_fingerprint(config: dict) -> str:
    payload = json.dumps(
        json_safe(_fingerprint_config(config)),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
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
        return [{"name": str(name), "dtype": str(dtype)} for name, dtype in schema_columns]
    return [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]


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
        "manifest_version": MANIFEST_VERSION,
        "cache_schema_version": str(
            config.get("cache", {}).get("schema_version", "base_labeled_v1")
        ),
        "run_id": run_name,
        "cache_path": str(cache_path),
        "config_path": str(config_path or ""),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cache_file": _cache_file_info(cache_path),
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
        "config": _fingerprint_config(config),
    }


def write_cache_manifest(manifest: dict[str, object], path: str | Path) -> None:
    write_json(path, manifest, atomic=True)


def publish_cache_manifest(
    frame: pd.DataFrame,
    *,
    cache_path: str | Path,
    config: dict,
    run_name: str,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    manifest = build_cache_manifest(
        frame,
        cache_path=cache_path,
        config=config,
        run_name=run_name,
        config_path=config_path,
    )
    write_cache_manifest(manifest, cache_manifest_path(cache_path))
    return manifest


def validate_cache_manifest(
    cache_path: str | Path,
    config: dict,
    *,
    required: bool = False,
) -> dict[str, object] | None:
    """Validate the cache boundary before a persisted labeled frame is reused."""

    manifest_path = cache_manifest_path(cache_path)
    if not manifest_path.exists():
        if required:
            raise SystemExit(f"labeled cache manifest does not exist: {manifest_path}")
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"cannot read labeled cache manifest {manifest_path}: {error}") from error

    errors: list[str] = []
    version = int(manifest.get("manifest_version", 0))
    if version not in {1, MANIFEST_VERSION}:
        errors.append(f"unsupported manifest_version={version}")

    expected_schema = str(config.get("cache", {}).get("schema_version", "base_labeled_v1"))
    actual_schema = str(manifest.get("cache_schema_version", ""))
    if actual_schema != expected_schema:
        errors.append(f"schema version {actual_schema!r} != {expected_schema!r}")

    schema = manifest.get("schema", [])
    manifest_columns = {
        str(item.get("name")) for item in schema if isinstance(item, dict) and item.get("name")
    }
    missing_columns = sorted(set(REQUIRED_LABELED_CACHE_COLUMNS) - manifest_columns)
    if missing_columns:
        errors.append(f"required columns missing: {', '.join(missing_columns)}")

    cache_path = Path(cache_path)
    cache_file = manifest.get("cache_file", {})
    if isinstance(cache_file, dict) and cache_file.get("bytes") is not None:
        try:
            actual_bytes = cache_path.stat().st_size
        except OSError as error:
            errors.append(f"cannot stat cache file: {error}")
        else:
            try:
                expected_bytes = int(cache_file["bytes"])
            except (TypeError, ValueError):
                errors.append(f"invalid manifest cache_file.bytes={cache_file['bytes']!r}")
            else:
                if actual_bytes != expected_bytes:
                    errors.append(
                        f"cache file bytes {actual_bytes} != manifest bytes {expected_bytes}"
                    )

    try:
        actual_columns = frame_columns(cache_path)
    except SystemExit as error:
        errors.append(f"cannot inspect cache file schema: {error}")
    else:
        missing_actual_columns = sorted(set(REQUIRED_LABELED_CACHE_COLUMNS) - actual_columns)
        if missing_actual_columns:
            errors.append(
                "required columns missing from cache file: " + ", ".join(missing_actual_columns)
            )
        if manifest_columns and actual_columns != manifest_columns:
            missing_from_cache = sorted(manifest_columns - actual_columns)
            extra_in_cache = sorted(actual_columns - manifest_columns)
            detail_parts = []
            if missing_from_cache:
                detail_parts.append("missing in cache: " + ", ".join(missing_from_cache))
            if extra_in_cache:
                detail_parts.append("extra in cache: " + ", ".join(extra_in_cache))
            errors.append(
                "manifest schema columns do not match cache file"
                + (f" ({'; '.join(detail_parts)})" if detail_parts else "")
            )

    if version >= MANIFEST_VERSION:
        actual_fingerprint = str(manifest.get("config_fingerprint", ""))
        expected_fingerprint = _config_fingerprint(config)
        if actual_fingerprint != expected_fingerprint:
            errors.append("cache-building config fingerprint does not match")

    if errors:
        detail = "; ".join(errors)
        raise SystemExit(f"incompatible labeled cache manifest {manifest_path}: {detail}")
    return manifest
