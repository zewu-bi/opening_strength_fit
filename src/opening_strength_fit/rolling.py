from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    train_dates: list[str]
    test_dates: list[str]

    @property
    def train_start_date(self) -> str:
        return self.train_dates[0]

    @property
    def train_end_date(self) -> str:
        return self.train_dates[-1]

    @property
    def test_start_date(self) -> str:
        return self.test_dates[0]

    @property
    def test_end_date(self) -> str:
        return self.test_dates[-1]


def chronological_date_split(
    frame: pd.DataFrame,
    *,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    train_fraction: float = 0.8,
) -> DateSplit:
    dates = sorted(str(date) for date in frame["date"].dropna().unique())
    if len(dates) < 2:
        raise SystemExit("need at least two dates for train/test split")

    if test_start_date:
        train_dates = [date for date in dates if date < test_start_date]
        test_dates = [
            date
            for date in dates
            if date >= test_start_date and (test_end_date is None or date <= test_end_date)
        ]
    else:
        cut = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
        train_dates = dates[:cut]
        test_dates = dates[cut:]

    if not train_dates:
        raise SystemExit("train split is empty")
    if not test_dates:
        raise SystemExit("test split is empty")
    return DateSplit(train_dates=train_dates, test_dates=test_dates)


def annual_rolling_date_splits(
    frame: pd.DataFrame,
    *,
    train_start_year: int | None = None,
    first_test_year: int | None = None,
    last_test_year: int | None = None,
    min_train_years: int = 1,
) -> list[DateSplit]:
    dates = sorted(str(date) for date in frame["date"].dropna().unique())
    if len(dates) < 2:
        raise SystemExit("need at least two dates for train/test split")

    date_years = pd.Series(pd.to_datetime(dates).year, index=dates)
    years = sorted(int(year) for year in date_years.unique())
    if len(years) < 2:
        raise SystemExit("need at least two calendar years for rolling annual split")

    train_start_year = train_start_year or years[0]
    first_test_year = first_test_year or max(years[1], train_start_year + min_train_years)
    last_test_year = last_test_year or years[-1]

    splits: list[DateSplit] = []
    for test_year in range(int(first_test_year), int(last_test_year) + 1):
        train_years = [year for year in years if train_start_year <= year < test_year]
        if len(train_years) < min_train_years:
            continue
        train_dates = [
            date for date in dates if train_start_year <= int(date_years.loc[date]) < test_year
        ]
        test_dates = [date for date in dates if int(date_years.loc[date]) == test_year]
        if train_dates and test_dates:
            splits.append(DateSplit(train_dates=train_dates, test_dates=test_dates))

    if not splits:
        raise SystemExit(
            "rolling annual split produced no train/test windows; "
            "check train_start_year/test years and available dates"
        )
    return splits


def monthly_rolling_date_splits(
    frame: pd.DataFrame,
    *,
    train_months: int = 12,
    test_months: int = 1,
    test_stride_months: int | None = None,
    first_test_month: str | None = None,
    last_test_month: str | None = None,
) -> list[DateSplit]:
    train_months = int(train_months)
    test_months = int(test_months)
    test_stride_months = test_months if test_stride_months is None else int(test_stride_months)
    if train_months < 1:
        raise SystemExit("rolling monthly split needs train_months >= 1")
    if test_months < 1:
        raise SystemExit("rolling monthly split needs test_months >= 1")
    if test_stride_months < 1:
        raise SystemExit("rolling monthly split needs test_stride_months >= 1")

    dates = sorted(str(date) for date in frame["date"].dropna().unique())
    if len(dates) < 2:
        raise SystemExit("need at least two dates for train/test split")

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    months = sorted(date_index.to_period("M").unique())
    if len(months) < 2:
        raise SystemExit("need at least two calendar months for rolling monthly split")

    default_first_index = min(max(train_months, 1), len(months) - 1)
    first_period = (
        pd.Period(first_test_month, freq="M") if first_test_month else months[default_first_index]
    )
    last_period = pd.Period(last_test_month, freq="M") if last_test_month else months[-1]

    splits: list[DateSplit] = []
    test_month = first_period
    while test_month <= last_period:
        test_end = test_month + test_months - 1
        if test_end > last_period:
            break
        train_start = test_month - train_months
        train_end = test_month - 1
        train_dates = [
            date
            for date in dates
            if train_start <= pd.Period(pd.Timestamp(date), freq="M") <= train_end
        ]
        test_dates = [
            date
            for date in dates
            if test_month <= pd.Period(pd.Timestamp(date), freq="M") <= test_end
        ]
        if train_dates and test_dates:
            splits.append(DateSplit(train_dates=train_dates, test_dates=test_dates))
        test_month += test_stride_months

    if not splits:
        raise SystemExit(
            "rolling monthly split produced no train/test windows; "
            "check train_months/test months and available dates"
        )
    return splits
