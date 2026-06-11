from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

TIME_HORIZON_SUFFIXES = {"s", "m", "h"}


class HorizonLike(Protocol):
    name: str
    label: str
    seconds: int | None


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    label: str
    seconds: int | None = None


def normalize_horizon_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def label_column_name(horizon: str) -> str:
    normalized = normalize_horizon_name(horizon)
    return f"alpha_return_{normalized}"


def key_columns_for_merge(frame: pd.DataFrame) -> list[str]:
    for time_col in ("decision_target_timestamp", "timestamp"):
        if time_col in frame.columns:
            return ["date", "symbol", time_col]
    raise SystemExit("inputs need decision_target_timestamp or timestamp merge key")


def parse_seconds_horizon(value: str) -> int | None:
    horizon = normalize_horizon_name(value)
    if horizon in {"close", "next_open", "next_close"}:
        return None
    aliases = {
        "30sec": "30s",
        "60sec": "60s",
        "5min": "5m",
    }
    horizon = aliases.get(horizon, horizon)
    if len(horizon) < 2 or horizon[-1] not in TIME_HORIZON_SUFFIXES:
        raise argparse.ArgumentTypeError(
            f"unknown horizon {value!r}; use Ns, Nm, Nh, close, next_open, or next_close"
        )
    amount = int(horizon[:-1])
    unit = horizon[-1]
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    return amount * multiplier


def horizon_specs(values: Iterable[str]) -> list[HorizonSpec]:
    specs = []
    for value in values:
        name = normalize_horizon_name(value)
        name = {"30sec": "30s", "60sec": "60s", "5min": "5m"}.get(name, name)
        seconds = parse_seconds_horizon(name)
        label = name.replace("_", " ")
        specs.append(HorizonSpec(name=name, label=label, seconds=seconds))
    return specs

