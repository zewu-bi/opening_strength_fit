from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
from build_ds350_clip_tables import evaluate_model, limit_states, outcome_labels

from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

YEARS = range(2022, 2026)
WINDOW_ORDER = {
    "09:31-09:40": 0,
    "10:01-10:10": 1,
    "11:01-11:10": 2,
    "14:01-14:10": 3,
}
LABEL_ORDER = {"1m": 0, "3m": 1}
EXPERIMENT_ORDER = {"Baseline": 0, "无涨跌停": 1}

LEGACY_CASES = (
    ("09:31-09:40", "1m", "Baseline", "clip", "baseline_1m"),
    ("09:31-09:40", "1m", "无涨跌停", "w0931", "ordinary_train_1m"),
    ("09:31-09:40", "3m", "Baseline", "two_window", "w0931_0940_h3m_baseline"),
    ("09:31-09:40", "3m", "无涨跌停", "two_window", "w0931_0940_h3m_no_limit"),
    ("10:01-10:10", "1m", "Baseline", "w1001", "baseline"),
    ("10:01-10:10", "1m", "无涨跌停", "w1001", "no_limit_training"),
    ("10:01-10:10", "3m", "Baseline", "two_window", "w1001_1010_h3m_baseline"),
    ("10:01-10:10", "3m", "无涨跌停", "two_window", "w1001_1010_h3m_no_limit"),
    ("14:01-14:10", "1m", "Baseline", "w1401", "w1401_1410_h1m_baseline"),
    ("14:01-14:10", "1m", "无涨跌停", "w1401", "w1401_1410_h1m_no_limit"),
    ("14:01-14:10", "3m", "Baseline", "w1401", "w1401_1410_h3m_baseline"),
    ("14:01-14:10", "3m", "无涨跌停", "w1401", "w1401_1410_h3m_no_limit"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified four-window limit contribution and enrichment tables."
    )
    parser.add_argument("--legacy-analysis-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--no-limit-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pool-reference-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def _read_metrics(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"missing legacy metrics: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"invalid metrics mapping: {path}")
    return value


def load_legacy_results(root: Path) -> list[dict[str, object]]:
    sources = {
        "clip": _read_metrics(root / "ds350_clip_tables_v1/metrics.json"),
        "w0931": _read_metrics(root / "ds350_w0931_three_experiment_tables_v1/metrics.json"),
        "w1001": _read_metrics(root / "ds350_w1001_h1m_limit_train_tables_v1/metrics.json"),
        "two_window": _read_metrics(
            root / "ds350_two_window_h3m_h10m_limit_tables_v1/metrics.json"
        ),
        "w1401": _read_metrics(root / "ds350_w1401_limit_tables_v1/metrics.json"),
    }
    rows = []
    for window, label, experiment, source, key in LEGACY_CASES:
        if key not in sources[source]:
            raise SystemExit(f"legacy metrics {source} missing key: {key}")
        rows.append(
            {
                "窗口": window,
                "训练": experiment,
                "Label": label,
                "metrics": sources[source][key],
            }
        )
    return rows


def evaluate_w1101(args: argparse.Namespace) -> list[dict[str, object]]:
    close_root = args.dataset_root / "opening_1101_1110_labels_hclose_v1"
    raw_root = args.cache_root / "opening_1101_1110_raw_source"
    labels = outcome_labels(close_root, YEARS)
    daily = limit_states(raw_root, YEARS)
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    rows = []
    for label in ("1m", "3m"):
        case = f"w1101_1110_h{label}"
        roots = (
            ("Baseline", args.baseline_root / case),
            ("无涨跌停", args.no_limit_root / case),
        )
        for experiment, root in roots:
            print(
                f"evaluate window=11:01 label={label} training={experiment} root={root}", flush=True
            )
            metrics = evaluate_model(
                root,
                labels=labels,
                daily=daily,
                pool=pool,
                top_n=args.top_n,
                rank_ic_outcomes=(),
            )
            rows.append(
                {
                    "窗口": "11:01-11:10",
                    "训练": experiment,
                    "Label": label,
                    "metrics": metrics,
                }
            )
            gc.collect()
    return rows


def common_pool_final_limit_pct(raw_root: Path) -> tuple[float, int]:
    daily = limit_states(raw_root, YEARS)
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    mask = stock_pool_membership_mask(daily, pool, date_lag_sessions=0)
    eligible = daily.loc[mask & daily["limit_state"].notna()]
    if eligible.empty:
        raise SystemExit("empty Pool L daily reference for final-limit denominator")
    return float(eligible["limit_state"].eq(1).mean() * 100.0), int(len(eligible))


def _number(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def contribution_cell(metrics: dict[str, object]) -> str:
    outcomes = metrics["outcomes"]
    own = outcomes["own_label"]
    close = outcomes["same_day_close"]
    return (
        f"{_number(own['final_limit_contribution_bps'])}/{_number(own['excess_bps'])} "
        f"({_number(close['final_limit_contribution_bps'])}/{_number(close['excess_bps'])})"
    )


def limit_rate_cell(metrics: dict[str, object], pool_pct: float) -> str:
    selected_pct = float(metrics["selected_final_limit_pct"])
    enrichment = selected_pct / pool_pct
    return f"{selected_pct:.3f}% ({enrichment:.2f}x)"


def _sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_window"] = out["窗口"].map(WINDOW_ORDER)
    out["_training"] = out["训练"].map(EXPERIMENT_ORDER)
    out["_label"] = out["Label"].map(LABEL_ORDER)
    return (
        out.sort_values(["_window", "_training", "_label"], kind="stable")
        .drop(columns=["_window", "_training", "_label"])
        .reset_index(drop=True)
    )


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_tables(
    rows: list[dict[str, object]], pool_pct: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contribution_display = []
    rate_display = []
    contribution_numeric = []
    rate_numeric = []
    for row in rows:
        keys = {name: row[name] for name in ("窗口", "训练", "Label")}
        metrics = row["metrics"]
        own = metrics["outcomes"]["own_label"]
        close = metrics["outcomes"]["same_day_close"]
        selected_pct = float(metrics["selected_final_limit_pct"])
        contribution_display.append(
            {
                **keys,
                "Label涨停贡献/总超额（收盘涨停贡献/总超额）": contribution_cell(metrics),
            }
        )
        rate_display.append(
            {
                **keys,
                "Top100实际涨停率（相对Pool L富集）": limit_rate_cell(metrics, pool_pct),
            }
        )
        contribution_numeric.append(
            {
                **keys,
                "Label涨停贡献": own["final_limit_contribution_bps"],
                "Label总超额": own["excess_bps"],
                "收盘涨停贡献": close["final_limit_contribution_bps"],
                "收盘总超额": close["excess_bps"],
            }
        )
        rate_numeric.append(
            {
                **keys,
                "PoolL涨停率": pool_pct,
                "Top100实际涨停率": selected_pct,
                "相对PoolL富集": selected_pct / pool_pct,
            }
        )
    return tuple(
        _sort_rows(pd.DataFrame(values))
        for values in (
            contribution_display,
            rate_display,
            contribution_numeric,
            rate_numeric,
        )
    )


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    rows = [*load_legacy_results(args.legacy_analysis_root), *evaluate_w1101(args)]
    pool_pct, pool_rows = common_pool_final_limit_pct(args.pool_reference_raw_root)
    table_1, table_2, numeric_1, numeric_2 = build_tables(rows, pool_pct)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table_1.to_csv(output_dir / "table_1_limit_contribution_ratio.csv", index=False)
    table_2.to_csv(output_dir / "table_2_limit_rate_enrichment.csv", index=False)
    numeric_1.to_csv(output_dir / "limit_contribution_numeric.csv", index=False)
    numeric_2.to_csv(output_dir / "limit_rate_enrichment_numeric.csv", index=False)
    combined = {f"{row['窗口']}|{row['训练']}|{row['Label']}": row["metrics"] for row in rows}
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "pool_l_final_limit_pct": pool_pct,
                "pool_l_daily_rows": pool_rows,
                "cases": combined,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    readme = (
        "# 四窗口涨停依赖表\n\n"
        f"统一的 2022-2025 日度 Pool L 最终涨停率为 `{pool_pct:.4f}%` "
        f"（`{pool_rows:,}` 个 date-symbol）。\n\n"
        "## 涨停贡献/总超额\n\n"
        + _markdown(table_1)
        + "\n\n## 实际涨停率与富集\n\n"
        + _markdown(table_2)
        + "\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "_SUCCESS").touch()
    print(f"POOL_L_FINAL_LIMIT_PCT={pool_pct:.6f} rows={pool_rows}", flush=True)
    print("TABLE_1", flush=True)
    print(table_1.to_csv(index=False).strip(), flush=True)
    print("TABLE_2", flush=True)
    print(table_2.to_csv(index=False).strip(), flush=True)


if __name__ == "__main__":
    main()
