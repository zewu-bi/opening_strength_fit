from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import (
    config_int,
    config_optional_int,
    config_str,
    config_value,
)
from opening_strength_fit.dataset import load_ticks
from opening_strength_fit.training_labeled import (
    build_labeled_frame_from_config,
    filter_labeled_frame,
    looks_labeled,
)
from opening_strength_fit.training_windows import (
    period_end_date,
    resolve_window_mode,
    test_year_from_args,
)


def input_kind(args: argparse.Namespace, config: dict) -> str:
    return args.input_kind or config_str(config, "data", "input_kind", "auto")


def load_training_frame(path: str, args: argparse.Namespace, config: dict) -> pd.DataFrame:
    frame = load_ticks(path)
    kind = input_kind(args, config)
    if kind == "labeled" or (kind == "auto" and looks_labeled(frame)):
        return filter_labeled_frame(frame, config)
    if kind not in {"auto", "raw_ticks"}:
        raise SystemExit(f"unknown data.input_kind={kind!r}; expected auto, raw_ticks, or labeled")
    return build_labeled_frame_from_config(frame, config)


def resolve_data_source(args: argparse.Namespace, config: dict, tick_path: str) -> str:
    if args.input:
        return "path"
    if getattr(args, "labeled_input", None):
        return "labeled_pvc"
    feature_input = getattr(args, "feature_input", None)
    label_input = getattr(args, "label_input", None)
    if bool(feature_input) != bool(label_input):
        raise SystemExit("pass --feature-input and --label-input together")
    if feature_input and label_input:
        return "labeled_pvc"
    source = args.data_source or config_str(config, "data", "source", "auto")
    source = source.strip().lower()
    if source == "auto":
        labeled_path = os.environ.get("OPENING_STRENGTH_LABELED_PATH", "") or config_value(
            config, "data", "labeled_path", ""
        )
        feature_path = config_value(config, "data", "feature_path", "")
        label_path = config_value(config, "data", "label_path", "")
        if labeled_path or (feature_path and label_path):
            return "labeled_pvc"
        return "path" if tick_path else "clickhouse"
    if source in {"path", "clickhouse", "labeled_pvc"}:
        return source
    raise SystemExit(
        f"unknown data.source={source!r}; expected auto, path, clickhouse, or labeled_pvc"
    )


def clickhouse_date_bounds(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    explicit_start = config_value(
        config,
        "data",
        "start_date",
        config_value(config, "clickhouse", "start_date", None),
    )
    explicit_end = config_value(
        config,
        "data",
        "end_date",
        config_value(config, "clickhouse", "end_date", None),
    )
    if explicit_start and explicit_end:
        return str(pd.Timestamp(explicit_start).date()), str(pd.Timestamp(explicit_end).date())

    window_mode = resolve_window_mode(args, config)
    if window_mode == "rolling_monthly":
        train_months = (
            args.train_months
            if args.train_months is not None
            else config_int(config, "window", "train_months", 12)
        )
        first_test_month = args.test_start_month or config_value(
            config, "window", "test_start_month", None
        )
        last_test_month = args.test_end_month or config_value(
            config, "window", "test_end_month", None
        )
        if not first_test_month or not last_test_month:
            raise SystemExit(
                "ClickHouse rolling_monthly source needs [window].test_start_month "
                "and [window].test_end_month, or CLI overrides."
            )
        first_period = pd.Period(first_test_month, freq="M")
        last_period = pd.Period(last_test_month, freq="M")
        start_period = first_period - int(train_months)
        return str(start_period.to_timestamp().date()), period_end_date(last_period)

    if window_mode == "rolling_annual":
        train_start_year = (
            args.train_start_year
            if args.train_start_year is not None
            else config_optional_int(config, "window", "train_start_year", None)
        )
        first_test_year = test_year_from_args(args, config, "start")
        last_test_year = test_year_from_args(args, config, "end")
        if train_start_year is None or first_test_year is None or last_test_year is None:
            raise SystemExit(
                "ClickHouse rolling_annual source needs train_start_year and test start/end years."
            )
        return f"{int(train_start_year):04d}-01-01", f"{int(last_test_year):04d}-12-31"

    train_start_date = config_value(
        config,
        "window",
        "train_start_date",
        config_value(config, "data", "train_start_date", None),
    )
    test_start_date = args.test_start_date or config_value(
        config,
        "window",
        "test_start_date",
        None,
    )
    test_end_date = args.test_end_date or config_value(
        config,
        "window",
        "test_end_date",
        None,
    )
    start_date = explicit_start or train_start_date
    end_date = explicit_end or test_end_date
    if not start_date or not end_date:
        raise SystemExit(
            "ClickHouse chronological source needs [data].start_date/[data].end_date "
            "or [window].train_start_date plus test_start_date/test_end_date."
        )
    if test_start_date and str(pd.Timestamp(start_date).date()) >= str(
        pd.Timestamp(test_start_date).date()
    ):
        raise SystemExit("ClickHouse chronological train source must start before test_start_date")
    return str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date())


def clickhouse_setting(
    args: argparse.Namespace,
    config: dict,
    arg_name: str,
    config_key: str,
    env_name: str,
    default,
):
    arg_value = getattr(args, arg_name)
    if arg_value not in (None, ""):
        return arg_value
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    return config_value(config, "clickhouse", config_key, default)


def resolve_cache_path(config: dict) -> Path | None:
    raw = config_value(
        config,
        "cache",
        "labeled_path",
        config_value(config, "cache", "path", ""),
    )
    if raw in (None, ""):
        return None
    return Path(str(raw))
