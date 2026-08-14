from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parents[1] / "experiments/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "build_ds350_four_window_limit_tables.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("four_window_limit_tables", SCRIPT_PATH)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
TABLES = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(TABLES)


def _metrics(selected_pct: float = 2.4) -> dict[str, object]:
    return {
        "selected_final_limit_pct": selected_pct,
        "outcomes": {
            "own_label": {
                "final_limit_contribution_bps": 2.78,
                "excess_bps": 11.76,
            },
            "same_day_close": {
                "final_limit_contribution_bps": 23.76,
                "excess_bps": 17.83,
            },
        },
    }


def test_display_cells_show_contribution_denominators_and_pool_enrichment() -> None:
    metrics = _metrics()

    assert TABLES.contribution_cell(metrics) == "2.78/11.76 (23.76/17.83)"
    assert TABLES.limit_rate_cell(metrics, 0.96) == "2.400% (2.50x)"


def test_build_tables_uses_window_training_and_label_order() -> None:
    rows = [
        {"窗口": "11:01-11:10", "训练": "无涨跌停", "Label": "3m", "metrics": _metrics()},
        {"窗口": "09:31-09:40", "训练": "Baseline", "Label": "1m", "metrics": _metrics()},
        {"窗口": "11:01-11:10", "训练": "Baseline", "Label": "1m", "metrics": _metrics()},
    ]

    table_1, table_2, numeric_1, numeric_2 = TABLES.build_tables(rows, 0.96)

    expected = [
        ("09:31-09:40", "Baseline", "1m"),
        ("11:01-11:10", "Baseline", "1m"),
        ("11:01-11:10", "无涨跌停", "3m"),
    ]
    for frame in (table_1, table_2, numeric_1, numeric_2):
        assert list(frame[["窗口", "训练", "Label"]].itertuples(index=False, name=None)) == expected
