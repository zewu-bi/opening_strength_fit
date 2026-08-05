from __future__ import annotations

import argparse

from opening_strength_fit.config import load_toml


def load_run_config(path: str) -> dict:
    if not path:
        return {}
    return load_toml(path)


def build_training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="", help="Optional TOML run config.")
    parser.add_argument("--input", default=None, help="Tick parquet/csv path override.")
    parser.add_argument(
        "--data-source",
        choices=["auto", "path", "clickhouse", "labeled_pvc"],
        default=None,
        help="Override [data].source. --input always uses a local/path source.",
    )
    parser.add_argument(
        "--labeled-input",
        default=None,
        help="PVC/local labeled parquet/csv path override for data.source=labeled_pvc.",
    )
    parser.add_argument(
        "--feature-input",
        default=None,
        help="Model-ready feature dataset root used with --label-input.",
    )
    parser.add_argument(
        "--label-input",
        default=None,
        help="Final label dataset root used with --feature-input.",
    )
    parser.add_argument("--run-id", default=None, help="Runtime run id override.")
    parser.add_argument(
        "--input-kind",
        choices=["auto", "raw_ticks", "labeled"],
        default=None,
        help="Whether --input is raw ticks or an already labeled research dataset.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--test-end-year", type=int, default=None)
    parser.add_argument("--train-months", type=int, default=None)
    parser.add_argument("--test-months", type=int, default=None)
    parser.add_argument("--test-stride-months", type=int, default=None)
    parser.add_argument("--test-start-month", default=None)
    parser.add_argument("--test-end-month", default=None)
    parser.add_argument(
        "--rolling-annual",
        action="store_true",
        help="Use train <= test_year-1 and test = calendar year splits.",
    )
    parser.add_argument(
        "--rolling-monthly",
        action="store_true",
        help="Use rolling N-month train windows and calendar month test windows.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "rolling_annual", "rolling_monthly"],
        default=None,
        help="Override [window].mode.",
    )
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override [evaluation].top_n for top-score summaries.",
    )
    parser.add_argument(
        "--clickhouse-host",
        default=None,
        help="ClickHouse host override. Defaults to CLICKHOUSE_HOST or config.",
    )
    parser.add_argument(
        "--clickhouse-port",
        type=int,
        default=None,
        help="ClickHouse port override. Defaults to CLICKHOUSE_PORT or config.",
    )
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-table", default=None)
    parser.add_argument("--start-offset-us", type=int, default=None)
    parser.add_argument("--end-offset-us", type=int, default=None)
    parser.add_argument(
        "--pool",
        choices=["L", "M", "S", "l", "m", "s"],
        default=None,
        help=(
            "Use mentor stock pool L/M/S as a selection mask. "
            "By default this keeps full-universe training and restricts TopN selection."
        ),
    )
    parser.add_argument(
        "--pool-path",
        default=None,
        help="Explicit stock-pool parquet path, e.g. lml.bzw@ssd/data/pool_S.parquet.",
    )
    parser.add_argument(
        "--pool-date-lag-sessions",
        type=int,
        default=None,
        help="Use the pool from this many prior pool sessions; set 1 for conservative no-lookahead checks.",
    )
    parser.add_argument(
        "--pool-filter-train",
        action="store_true",
        help="Also restrict training rows to the selected stock pool. Default only restricts TopN selection.",
    )
    parser.add_argument(
        "--pool-add-feature",
        action="store_true",
        help="Add stock_pool_member as a model feature. Default only annotates predictions.",
    )
    return parser
