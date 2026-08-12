from __future__ import annotations

import argparse

import pandas as pd

from opening_strength_fit.config import (
    config_float,
    config_int,
    config_optional_int,
    config_str,
)
from opening_strength_fit.config import config_value as get
from opening_strength_fit.evaluation import format_group_cols, group_cols_for_mode
from opening_strength_fit.rolling import (
    annual_rolling_date_splits,
    chronological_date_split,
    monthly_rolling_date_splits,
)
from opening_strength_fit.stock_pool import (
    stock_pool_config_from_mapping,
    stock_pool_evaluation_settings,
)


def period_end_date(period: pd.Period) -> str:
    return str(((period + 1).to_timestamp() - pd.Timedelta(days=1)).date())


def _arg_value(args: argparse.Namespace, name: str, default=None):
    return getattr(args, name, default)


def _window_arg(args: argparse.Namespace, config: dict, name: str, default=None):
    value = _arg_value(args, name)
    return value if value is not None else get(config, "window", name, default)


def _optional_int_window_arg(args: argparse.Namespace, config: dict, name: str) -> int | None:
    value = _arg_value(args, name)
    return value if value is not None else config_optional_int(config, "window", name)


def rolling_monthly_date_bounds(
    args: argparse.Namespace,
    config: dict,
) -> tuple[str, str] | None:
    train_months = _window_arg(args, config, "train_months", 12)
    first_test_month = _window_arg(args, config, "test_start_month")
    last_test_month = _window_arg(args, config, "test_end_month")
    if not first_test_month or not last_test_month:
        return None
    first_period = pd.Period(first_test_month, freq="M")
    last_period = pd.Period(last_test_month, freq="M")
    start_period = first_period - int(train_months)
    return str(start_period.to_timestamp().date()), period_end_date(last_period)


def test_year_from_args(args: argparse.Namespace, config: dict, key: str) -> int | None:
    year_name = f"test_{key}_year"
    if (year := _arg_value(args, year_name)) is not None:
        return year
    date_name = f"test_{key}_date"
    test_date = _arg_value(args, date_name) or get(config, "window", date_name, None)
    explicit = config_optional_int(config, "window", year_name, None)
    if explicit is not None:
        return explicit
    return int(pd.Timestamp(test_date).year) if test_date else None


def date_splits(labeled: pd.DataFrame, args: argparse.Namespace, config: dict):
    window_mode = resolve_window_mode(args, config)
    if window_mode == "rolling_monthly":
        return monthly_rolling_date_splits(
            labeled,
            train_months=int(_window_arg(args, config, "train_months", 12)),
            test_months=int(_window_arg(args, config, "test_months", 1)),
            test_stride_months=_window_arg(args, config, "test_stride_months"),
            first_test_month=_window_arg(args, config, "test_start_month"),
            last_test_month=_window_arg(args, config, "test_end_month"),
        )
    if window_mode == "rolling_annual":
        return annual_rolling_date_splits(
            labeled,
            train_start_year=_optional_int_window_arg(args, config, "train_start_year"),
            first_test_year=test_year_from_args(args, config, "start"),
            last_test_year=test_year_from_args(args, config, "end"),
            min_train_years=config_int(config, "window", "min_train_years", 1),
        )
    return [
        chronological_date_split(
            labeled,
            test_start_date=_window_arg(args, config, "test_start_date"),
            test_end_date=_window_arg(args, config, "test_end_date"),
            train_fraction=config_float(config, "window", "train_fraction", 0.8),
        )
    ]


def resolve_window_mode(args: argparse.Namespace, config: dict) -> str:
    window_mode = _arg_value(args, "split_mode") or config_str(
        config, "window", "mode", "chronological"
    )
    if _arg_value(args, "rolling_monthly", False):
        return "rolling_monthly"
    if _arg_value(args, "rolling_annual", False):
        return "rolling_annual"
    return window_mode


def build_evaluation_settings(config: dict, args: argparse.Namespace) -> dict[str, object]:
    bucket_mode = config_str(config, "evaluation", "bucket_mode", "daily")
    selection_mode = config_str(config, "evaluation", "selection_mode", "symbol_day")
    ic_mode = config_str(config, "evaluation", "ic_mode", bucket_mode)
    bucket_group_cols = group_cols_for_mode(bucket_mode)
    selection_group_cols = group_cols_for_mode(selection_mode)
    ic_group_cols = group_cols_for_mode(ic_mode)
    settings = {
        "score_bucket_mode": bucket_mode,
        "score_bucket_group_cols": format_group_cols(bucket_group_cols),
        "selection_mode": selection_mode,
        "selection_group_cols": format_group_cols(selection_group_cols),
        "ic_mode": ic_mode,
        "ic_group_cols": format_group_cols(ic_group_cols),
        "top_n": (
            args.top_n if args.top_n is not None else config_int(config, "evaluation", "top_n", 20)
        ),
        "score_bins": config_int(config, "evaluation", "score_bins", 5),
        "_bucket_group_cols": bucket_group_cols,
        "_selection_group_cols": selection_group_cols,
        "_ic_group_cols": ic_group_cols,
    }
    settings.update(stock_pool_evaluation_settings(stock_pool_config_from_mapping(config)))
    return settings
