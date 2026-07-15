from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    json_safe as analysis_json_safe,
)
from opening_strength_fit.analysis import (
    write_json as analysis_write_json,
)
from opening_strength_fit.io import json_safe, write_json


def test_json_safe_normalizes_project_scalar_and_container_types(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path / "artifact.json",
        "timestamp": pd.Timestamp("2026-07-10 09:30:00"),
        "values": (
            np.int64(3),
            np.bool_(True),
            np.array([np.float64(1.5), np.nan, np.inf, -np.inf]),
            pd.Series([1.0, pd.NA], dtype="Float64"),
            pd.NaT,
            np.datetime64("NaT"),
            np.timedelta64(3, "s"),
            np.timedelta64("NaT"),
        ),
    }

    normalized = json_safe(payload)

    assert normalized == {
        "path": str(tmp_path / "artifact.json"),
        "timestamp": "2026-07-10T09:30:00",
        "values": [
            3,
            True,
            [1.5, None, None, None],
            [1.0, None],
            None,
            None,
            "P0DT0H0M3S",
            None,
        ],
    }
    json.dumps(normalized, allow_nan=False)


def test_write_json_supports_atomic_sorted_ascii_output(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.json"
    path.parent.mkdir(parents=True)
    path.write_text("old", encoding="utf-8")

    write_json(
        path,
        {"路径": Path("输出.json"), "a": np.float64(np.nan)},
        ensure_ascii=True,
        sort_keys=True,
        atomic=True,
    )

    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert not content.endswith("\n\n")
    assert "NaN" not in content
    assert "\\u8def\\u5f84" in content
    assert content.index('"a"') < content.index('"\\u8def\\u5f84"')
    assert json.loads(content) == {"a": None, "路径": "输出.json"}
    assert not list(path.parent.glob(".*.tmp"))


def test_analysis_reexports_shared_json_helpers() -> None:
    assert analysis_json_safe is json_safe
    assert analysis_write_json is write_json
