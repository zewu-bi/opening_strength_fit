from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.pool_internal_plot_svg import PLOT_POOLS


def slug_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "pool_internal"


def pool_group_slug(pools: tuple[str, ...]) -> str:
    if pools == PLOT_POOLS:
        return "universe_sml"
    return slug_label("_".join(pools))


def pool_group_title(pools: tuple[str, ...]) -> str:
    if pools == PLOT_POOLS:
        return "universe / S / M / L"
    labels = {
        "pool_S": "S",
        "pool_M": "M",
        "pool_L": "L",
    }
    return " / ".join(labels.get(pool, pool) for pool in pools)


def summary_months(month_summary: pd.DataFrame) -> list[str]:
    return sorted(month_summary["test_month"].astype(str).unique())


def write_plot_trace(
    path: Path,
    *,
    input_path: Path | None,
    plot_data: Path,
    figure: Path,
    variant_label: str,
    pools: tuple[str, ...] = PLOT_POOLS,
    **metadata: object,
) -> None:
    write_json(
        path,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(pools),
            **metadata,
        },
        ensure_ascii=True,
    )
