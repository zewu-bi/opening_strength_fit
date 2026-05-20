import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_ENDPOINTS = (
    "http://10.20.201.15:7777/pnl",
    "http://10.20.201.42:7777/pnl",
)
SERIES_KEYS = ("alpha", "profit", "overday", "inday", "turnover", "rent", "count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert opening-strength tick predictions into the backtest API score "
            "matrix, call the API, and persist all results and run traces."
        )
    )
    parser.add_argument(
        "--predictions",
        default="output/backtest/ridge_opening_full/predictions_all.parquet",
        help="Long-form parquet with date, symbol, and prediction columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/backtest/ridge_opening_full",
        help="Directory for score matrices, API responses, and trace metadata.",
    )
    parser.add_argument(
        "--aggregate",
        default="max",
        choices=["max", "mean", "median", "first", "last"],
        help="How to collapse multiple opening ticks into one date-symbol score.",
    )
    parser.add_argument("--tar", default="I500", choices=["I500", "large", "small"])
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--trgain", type=float, default=None)
    parser.add_argument("--vol-limit", type=float, default=None)
    parser.add_argument(
        "--fee",
        dest="fee",
        action="store_true",
        default=None,
        help="Explicitly enable fees. Omit to use API default.",
    )
    parser.add_argument(
        "--no-fee",
        dest="fee",
        action="store_false",
        help="Disable fees in the backtest request.",
    )
    parser.add_argument(
        "--return-eod",
        action="store_true",
        help="Request end-of-day weights from the API.",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        help="Optional explicit endpoint(s). If omitted, try the two documented endpoints.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def pack_df(frame: pd.DataFrame) -> dict:
    packed = frame.copy()
    packed.index = pd.to_datetime(packed.index).strftime("%Y-%m-%d")
    packed.columns = packed.columns.astype(str)
    return packed.to_dict("tight")


def aggregate_scores(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.dropna(subset=["date", "symbol", "prediction"])
    sort_cols = [column for column in ["date", "symbol", "timestamp"] if column in frame]
    if sort_cols:
        frame = frame.sort_values(sort_cols)

    grouped = frame.groupby(["date", "symbol"], sort=True)["prediction"]
    if method == "max":
        collapsed = grouped.max()
    elif method == "mean":
        collapsed = grouped.mean()
    elif method == "median":
        collapsed = grouped.median()
    elif method == "first":
        collapsed = grouped.first()
    elif method == "last":
        collapsed = grouped.last()
    else:
        raise SystemExit(f"unknown aggregate method: {method}")
    return collapsed.reset_index()


def load_score_matrix(path: Path, aggregate: str) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_parquet(path)
    required = {"date", "symbol", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"predictions parquet missing required columns: {sorted(missing)}")

    collapsed = aggregate_scores(frame.loc[:, list(required | {"timestamp"} & set(frame.columns))], aggregate)
    score = collapsed.pivot(index="date", columns="symbol", values="prediction").sort_index()
    all_nan_rows = int(score.isna().all(axis=1).sum())
    if all_nan_rows:
        raise SystemExit(f"score matrix contains {all_nan_rows} all-NaN dates")

    stats = {
        "rows": int(len(frame)),
        "collapsed_rows": int(len(collapsed)),
        "dates": int(score.index.nunique()),
        "symbols": int(score.columns.nunique()),
        "date_min": str(score.index.min().date()),
        "date_max": str(score.index.max().date()),
        "aggregate": aggregate,
        "nan_ratio": float(score.isna().sum().sum() / (score.shape[0] * score.shape[1])),
    }
    return score, stats


def create_payload(
    score: pd.DataFrame,
    tar: str,
    *,
    cap: float | None,
    trgain: float | None,
    fee: bool | None,
    vol_limit: float | None,
    return_eod: bool,
) -> bytes:
    args = {
        "score": pack_df(score),
        "tar": tar,
    }
    if cap is not None:
        args["cap"] = cap
    if trgain is not None:
        args["trgain"] = trgain
    if fee is not None:
        args["fee"] = fee
    if vol_limit is not None:
        args["vol_limit"] = vol_limit
    if return_eod:
        args["return_eod"] = True
    return pickle.dumps(args)


def call_backtest_api(payload: bytes, endpoints: list[str], timeout: float) -> tuple[str, dict]:
    import requests

    last_error = None
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            result = pickle.loads(response.content)
            for key in SERIES_KEYS:
                result[key] = pd.Series(result[key])
            result["solve_rate"] = pd.Series(**result["solve_rate"])
            if "w_eod" in result:
                result["w_eod"] = pd.DataFrame.from_dict(result["w_eod"], orient="tight")
            return endpoint, result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise SystemExit(f"backtest API failed for all endpoints: {last_error}")


def write_backtest_outputs(output_dir: Path, score: pd.DataFrame, result: dict) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score_wide.parquet"
    score.to_parquet(score_path)

    result_path = output_dir / "backtest_result.pkl"
    with result_path.open("wb") as file:
        pickle.dump(result, file)

    series_summary = {}
    for key in SERIES_KEYS:
        series = pd.Series(result[key]).sort_index()
        csv_path = output_dir / f"{key}.csv"
        series.rename(key).to_csv(csv_path, header=True)
        series_summary[key] = {
            "csv": str(csv_path),
            "count": int(series.shape[0]),
            "sum": float(series.sum()),
            "mean": float(series.mean()),
        }

    solve_rate = pd.Series(result["solve_rate"]).sort_index()
    solve_rate_path = output_dir / "solve_rate.csv"
    solve_rate.rename("solve_rate").to_csv(solve_rate_path, header=True)

    w_eod_path = None
    if "w_eod" in result:
        w_eod_path = output_dir / "w_eod.parquet"
        result["w_eod"].to_parquet(w_eod_path)

    summary = {
        "alpha_sum": float(pd.Series(result["alpha"]).sum()),
        "profit_sum": float(pd.Series(result["profit"]).sum()),
        "turnover_mean": float(pd.Series(result["turnover"]).mean()),
        "solve_rate_mean": float(solve_rate.mean()),
        "solve_rate_min": float(solve_rate.min()),
        "count_mean": float(pd.Series(result["count"]).mean()),
        "score_wide": str(score_path),
        "result_pickle": str(result_path),
        "solve_rate_csv": str(solve_rate_path),
        "w_eod_parquet": str(w_eod_path) if w_eod_path else None,
        "series": series_summary,
    }
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    score, score_stats = load_score_matrix(predictions_path, args.aggregate)
    payload = create_payload(
        score,
        args.tar,
        cap=args.cap,
        trgain=args.trgain,
        fee=args.fee,
        vol_limit=args.vol_limit,
        return_eod=args.return_eod,
    )

    endpoints = args.endpoint or list(DEFAULT_ENDPOINTS)
    endpoint_used, result = call_backtest_api(payload, endpoints, args.timeout)
    summary = write_backtest_outputs(output_dir, score, result)

    trace = {
        "backtested_at_utc": datetime.now(UTC).isoformat(),
        "predictions": str(predictions_path),
        "output_dir": str(output_dir),
        "endpoint_used": endpoint_used,
        "endpoints_tried": endpoints,
        "tar": args.tar,
        "cap": args.cap,
        "trgain": args.trgain,
        "fee": args.fee,
        "vol_limit": args.vol_limit,
        "return_eod": args.return_eod,
        "score_stats": score_stats,
        "summary": summary,
    }
    (output_dir / "backtest_trace.json").write_text(
        json.dumps(trace, indent=2),
        encoding="utf-8",
    )
    (output_dir / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("backtest_complete:")
    print(f"  predictions: {predictions_path}")
    print(f"  endpoint: {endpoint_used}")
    print(f"  output_dir: {output_dir}")
    print(f"  aggregate: {args.aggregate}")
    print(f"  alpha_sum: {summary['alpha_sum']:.6f}")
    print(f"  profit_sum: {summary['profit_sum']:.6f}")
    print(f"  turnover_mean: {summary['turnover_mean']:.6f}")
    print(f"  solve_rate_mean: {summary['solve_rate_mean']:.6f}")
    print(f"  solve_rate_min: {summary['solve_rate_min']:.6f}")


if __name__ == "__main__":
    main()
