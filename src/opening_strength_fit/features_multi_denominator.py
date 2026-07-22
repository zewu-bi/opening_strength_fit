from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.feature_utils import safe_divide


def _positive_numeric(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(values > 0.0)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = safe_divide(numerator, denominator)
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=numerator.index, dtype="float32")


def add_multi_denominator_ratio_features(
    frame: pd.DataFrame,
    *,
    turnover_columns: tuple[str, ...],
    volume_columns: tuple[str, ...],
    depth_columns: tuple[str, ...],
    cross_sectional_median_columns: tuple[str, ...],
    cross_sectional_group_cols: tuple[str, ...] = (
        "date",
        "decision_target_timestamp",
    ),
    historical_window: int = 60,
    prefix: str = "multi_den_ratio_",
    min_features: int = 0,
    max_features: int = 40,
) -> pd.DataFrame:
    """Add a small, explicit family of dimensionless activity/liquidity ratios.

    The inputs are deliberately configured as exact column lists. This keeps the
    feature family auditable and prevents a broad prefix match from multiplying
    every order-book level by every denominator.
    """

    out = frame.copy(deep=False)
    hist_turnover_col = f"hist_avg_daily_turnover_{int(historical_window)}d"
    hist_volume_col = f"hist_avg_daily_volume_{int(historical_window)}d"
    denominators = {
        "float_market_cap": _positive_numeric(out, "float_market_cap"),
        hist_turnover_col: _positive_numeric(out, hist_turnover_col),
        "float_shares": _positive_numeric(out, "float_shares"),
        hist_volume_col: _positive_numeric(out, hist_volume_col),
    }
    new_columns: dict[str, pd.Series] = {}

    def add_ratio(source_col: str, denominator_key: str) -> None:
        denominator = denominators.get(denominator_key)
        if source_col not in out.columns or denominator is None:
            return
        numerator = pd.to_numeric(out[source_col], errors="coerce")
        name = f"{prefix}{source_col}_to_{denominator_key}"
        new_columns[name] = _ratio(numerator, denominator)

    for column in dict.fromkeys(turnover_columns):
        add_ratio(column, "float_market_cap")
        add_ratio(column, hist_turnover_col)
    for column in dict.fromkeys(volume_columns):
        add_ratio(column, "float_shares")
        add_ratio(column, hist_volume_col)
    for column in dict.fromkeys(depth_columns):
        add_ratio(column, "float_shares")
        add_ratio(column, hist_volume_col)
    median_columns = list(dict.fromkeys(cross_sectional_median_columns))
    available_median_columns = [column for column in median_columns if column in out.columns]
    available_group_cols = [
        column for column in cross_sectional_group_cols if column in out.columns
    ]
    if available_median_columns and len(available_group_cols) == len(cross_sectional_group_cols):
        medians = out.groupby(available_group_cols, sort=False, observed=True)[
            available_median_columns
        ].transform("median")
        for column in available_median_columns:
            numerator = pd.to_numeric(out[column], errors="coerce")
            denominator = pd.to_numeric(medians[column], errors="coerce").where(
                lambda values: values > 0.0
            )
            new_columns[f"{prefix}{column}_to_xs_median"] = _ratio(
                numerator,
                denominator,
            )

    feature_count = len(new_columns)
    if feature_count < int(min_features):
        raise SystemExit(
            "multi-denominator feature generation produced "
            f"{feature_count} columns; expected at least {int(min_features)}"
        )
    if feature_count > int(max_features):
        raise SystemExit(
            "multi-denominator feature generation produced "
            f"{feature_count} columns; configured maximum is {int(max_features)}"
        )
    if not new_columns:
        return out
    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
