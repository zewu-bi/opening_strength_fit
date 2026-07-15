from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePath

import numpy as np
import pandas as pd


def json_safe(value: object) -> object:
    """Return a JSON-compatible representation of common project values."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (pd.Series, pd.Index)):
        return json_safe(value.tolist())
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return None if pd.isna(value) else pd.Timedelta(value).isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(
    path: str | Path,
    payload: object,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    atomic: bool = False,
) -> None:
    """Write an indented, standards-compliant JSON artifact ending in a newline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        + "\n"
    )
    if not atomic:
        path.write_text(content, encoding="utf-8")
        return

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
