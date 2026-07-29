from __future__ import annotations

import unittest
from unittest.mock import Mock

from opening_strength_fit.commands.daily_return_label_cache import (
    query_daily_open_close_bars,
)


class DailyReturnLabelCacheTest(unittest.TestCase):
    def test_query_projects_only_daily_open_close_fields(self) -> None:
        client = Mock()
        client.query_df.return_value = Mock()

        query_daily_open_close_bars(
            client,
            table="stock.daily_bar_jy",
            start_date="2025-01-02",
            end_date="2025-01-17",
        )

        sql = client.query_df.call_args.args[0]
        parameters = client.query_df.call_args.kwargs["parameters"]
        self.assertIn("OpenPrice as open_price", sql)
        self.assertIn("ClosePrice as close_price", sql)
        self.assertIn("PreClosePrice as preclose_price", sql)
        self.assertNotIn("select *", sql.lower())
        self.assertEqual(
            parameters,
            {"start_date": "2025-01-02", "end_date": "2025-01-17"},
        )


if __name__ == "__main__":
    unittest.main()
