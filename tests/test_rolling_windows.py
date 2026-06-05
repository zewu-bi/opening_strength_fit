from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.rolling import monthly_rolling_date_splits


class RollingWindowTest(unittest.TestCase):
    def test_halfyear_windows_roll_from_2018_to_2024(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [
                    str(period.to_timestamp().date())
                    for period in pd.period_range("2015-01", "2024-12", freq="M")
                ]
            }
        )

        splits = monthly_rolling_date_splits(
            frame,
            train_months=36,
            test_months=6,
            test_stride_months=6,
            first_test_month="2018-01",
            last_test_month="2024-12",
        )

        self.assertEqual(len(splits), 14)
        self.assertEqual(splits[0].train_start_date, "2015-01-01")
        self.assertEqual(splits[0].train_end_date, "2017-12-01")
        self.assertEqual(splits[0].test_start_date, "2018-01-01")
        self.assertEqual(splits[0].test_end_date, "2018-06-01")
        self.assertEqual(splits[-1].train_start_date, "2021-07-01")
        self.assertEqual(splits[-1].train_end_date, "2024-06-01")
        self.assertEqual(splits[-1].test_start_date, "2024-07-01")
        self.assertEqual(splits[-1].test_end_date, "2024-12-01")


if __name__ == "__main__":
    unittest.main()
