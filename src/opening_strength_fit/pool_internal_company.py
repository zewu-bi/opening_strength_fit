from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.pool_internal_plots import (
    slug_label,
    write_company_backtest_cumulative_plot,
    write_company_backtest_neutral_comparison_plot,
)
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

COMPANY_API_TIMES = ("930", "0930", "950", "0950", "1030", "1130", "1300", "1400")
COMPANY_SCORE_AGGS = ("mean", "first", "last", "max")
COMPANY_SCORE_TRANSFORMS = ("identity", "negate")
COMPANY_SERIES_KEYS = ("alpha", "profit", "overday", "inday", "turnover", "rent", "count")
DEFAULT_COMPANY_ENDPOINTS = (
    "http://10.20.201.15:7777/pnl",
    "http://10.20.201.42:7777/pnl",
)


def normalize_clock(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise SystemExit("clock value is empty")
    if ":" in raw:
        return pd.Timestamp(f"2000-01-01 {raw}").strftime("%H:%M")
    digits = raw.zfill(4)
    if len(digits) != 4 or not digits.isdigit():
        raise SystemExit(f"invalid clock value: {value!r}")
    return f"{digits[:2]}:{digits[2:]}"


def filter_company_backtest_scores(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    clocks: list[str] | None,
    start_clock: str,
    end_clock: str,
    pool: str,
    pool_path: str,
    pool_date_lag_sessions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = predictions.loc[:, [*KEY_COLUMNS, score_col]].copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "decision_target_timestamp", score_col])
    frame["_clock"] = pd.to_datetime(
        frame["decision_target_timestamp"],
        errors="coerce",
    ).dt.strftime("%H:%M")

    requested_clocks = [normalize_clock(clock) for clock in clocks or []]
    if requested_clocks:
        frame = frame.loc[frame["_clock"].isin(requested_clocks)].copy()
    if start_clock:
        frame = frame.loc[frame["_clock"] >= normalize_clock(start_clock)].copy()
    if end_clock:
        frame = frame.loc[frame["_clock"] <= normalize_clock(end_clock)].copy()

    pool_source = ""
    if pool or pool_path:
        resolved_pool_path = pool_path or DEFAULT_STOCK_POOL_PATHS[str(pool).upper()]
        mask = stock_pool_membership_mask(
            frame,
            load_stock_pool(resolved_pool_path),
            date_lag_sessions=pool_date_lag_sessions,
        )
        frame = frame.loc[mask].copy()
        pool_source = resolved_pool_path

    stats = {
        "rows_after_filters": int(len(frame)),
        "clocks": sorted(frame["_clock"].dropna().unique().tolist()),
        "requested_clocks": requested_clocks,
        "start_clock": normalize_clock(start_clock) if start_clock else "",
        "end_clock": normalize_clock(end_clock) if end_clock else "",
        "pool": pool,
        "pool_path": pool_source,
        "pool_date_lag_sessions": pool_date_lag_sessions,
    }
    if frame.empty:
        raise SystemExit("no prediction rows remain after company API clock/pool filters")
    return frame, stats


def build_company_score_matrix(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    score_agg: str,
    score_transform: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if score_agg not in COMPANY_SCORE_AGGS:
        raise SystemExit(f"unsupported company score aggregation: {score_agg}")
    if score_transform not in COMPANY_SCORE_TRANSFORMS:
        raise SystemExit(f"unsupported company score transform: {score_transform}")
    work = predictions.sort_values(["date", "symbol", "decision_target_timestamp"]).copy()
    if score_agg in {"first", "last"}:
        grouped = work.groupby(["date", "symbol"], as_index=False, sort=True)
        aggregated = getattr(grouped, score_agg)()
    else:
        aggregated = (
            work.groupby(["date", "symbol"], as_index=False, sort=True)[score_col]
            .agg(score_agg)
            .reset_index(drop=True)
        )
    aggregated = aggregated.loc[:, ["date", "symbol", score_col]].dropna(subset=[score_col])

    if top_n > 0:
        aggregated = (
            aggregated.sort_values(["date", score_col, "symbol"], ascending=[True, False, True])
            .groupby("date", group_keys=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )

    if score_transform == "negate":
        aggregated[score_col] = -aggregated[score_col]

    score = aggregated.pivot(index="date", columns="symbol", values=score_col).sort_index()
    score.index = pd.to_datetime(score.index, errors="coerce")
    score = score.loc[score.index.notna()].sort_index()
    score.columns = score.columns.astype(str)
    all_nan_dates = int(score.isna().all(axis=1).sum())
    if all_nan_dates:
        raise SystemExit(f"company score matrix contains {all_nan_dates} all-NaN dates")
    if score.empty:
        raise SystemExit("company score matrix is empty")

    stats = {
        "score_rows_long": int(len(aggregated)),
        "score_dates": int(score.index.nunique()),
        "score_symbols": int(score.columns.nunique()),
        "score_date_min": str(score.index.min().date()),
        "score_date_max": str(score.index.max().date()),
        "score_nan_ratio": float(
            score.isna().sum().sum() / max(1, score.shape[0] * score.shape[1])
        ),
        "score_agg": score_agg,
        "score_transform": score_transform,
        "top_n": int(top_n),
    }
    return score, aggregated, stats


def build_company_neutral_score_matrix(
    score: pd.DataFrame,
    *,
    score_col: str,
    neutral_score: float,
    base_stats: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    neutral = score.where(score.isna(), float(neutral_score))
    neutral_long = neutral.stack().dropna().rename(score_col).reset_index()
    neutral_long.columns = ["date", "symbol", score_col]
    stats = {
        **base_stats,
        "score_rows_long": int(len(neutral_long)),
        "score_dates": int(neutral.index.nunique()),
        "score_symbols": int(neutral.columns.nunique()),
        "score_nan_ratio": float(
            neutral.isna().sum().sum() / max(1, neutral.shape[0] * neutral.shape[1])
        ),
        "score_agg": "neutral_pool",
        "score_transform": "neutral_constant",
        "neutral_score": float(neutral_score),
    }
    return neutral, neutral_long, stats


def pack_company_score_frame(frame: pd.DataFrame) -> dict[str, Any]:
    packed = frame.copy()
    packed.index = pd.to_datetime(packed.index).strftime("%Y-%m-%d")
    packed.columns = packed.columns.astype(str)
    return packed.to_dict("tight")


def create_company_backtest_payload(
    score: pd.DataFrame,
    *,
    api_time: str,
    daily: bool,
    tar: str,
    cap: float | None,
    trgain: float | None,
    fee: bool | None,
    vol_limit: float | None,
    return_eod: bool,
) -> bytes:
    packed_score: dict[str, Any] | dict[str, dict[str, Any]]
    packed_score = (
        pack_company_score_frame(score) if daily else {api_time: pack_company_score_frame(score)}
    )
    payload: dict[str, Any] = {"score": packed_score, "tar": tar}
    for key, value in {"cap": cap, "trgain": trgain, "fee": fee, "vol_limit": vol_limit}.items():
        if value is not None:
            payload[key] = value
    if return_eod:
        payload["return_eod"] = True
    return pickle.dumps(payload)


def decode_company_backtest_result(raw: bytes) -> dict[str, Any]:
    try:
        result = pickle.loads(raw)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(str(raw)) from exc
    for key in COMPANY_SERIES_KEYS:
        if key in result:
            result[key] = pd.Series(result[key])
    if "solve_rate" in result:
        result["solve_rate"] = pd.Series(**result["solve_rate"])
    if "w_eod" in result:
        result["w_eod"] = pd.DataFrame.from_dict(result["w_eod"], orient="tight")
    return result


def call_company_backtest_api(
    payload: bytes,
    *,
    endpoints: list[str],
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            return endpoint, decode_company_backtest_result(response.content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise SystemExit(f"company backtest API failed for all endpoints: {last_error}")


def write_company_score_inputs(
    output_dir: Path,
    score: pd.DataFrame,
    score_long: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score_wide.parquet"
    score_long_path = output_dir / "score_long.parquet"
    score.to_parquet(score_path)
    score_long.to_parquet(score_long_path, index=False)
    return {"score_wide": str(score_path), "score_long": str(score_long_path)}


def write_company_api_outputs(output_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    result_path = output_dir / "backtest_result.pkl"
    with result_path.open("wb") as file:
        pickle.dump(result, file)

    series_summary = {}
    for key in COMPANY_SERIES_KEYS:
        if key not in result:
            continue
        series = pd.Series(result[key]).sort_index()
        csv_path = output_dir / f"{key}.csv"
        series.rename(key).to_csv(csv_path, header=True)
        series_summary[key] = {
            "csv": str(csv_path),
            "count": int(series.shape[0]),
            "sum": float(series.sum()),
            "mean": float(series.mean()),
        }

    solve_rate_path = None
    solve_rate_summary = {}
    if "solve_rate" in result:
        solve_rate = pd.Series(result["solve_rate"]).sort_index()
        solve_rate_path = output_dir / "solve_rate.csv"
        solve_rate.rename("solve_rate").to_csv(solve_rate_path, header=True)
        solve_rate_summary = {
            "solve_rate_mean": float(solve_rate.mean()),
            "solve_rate_min": float(solve_rate.min()),
        }

    w_eod_path = None
    if "w_eod" in result:
        w_eod_path = output_dir / "w_eod.parquet"
        result["w_eod"].to_parquet(w_eod_path)

    return {
        "result_pickle": str(result_path),
        "solve_rate_csv": str(solve_rate_path) if solve_rate_path else None,
        "w_eod_parquet": str(w_eod_path) if w_eod_path else None,
        "series": series_summary,
        **solve_rate_summary,
    }


def company_backtest_plot_data(
    result: dict[str, Any],
    *,
    series_key: str,
    series_label: str,
) -> pd.DataFrame:
    alpha = pd.Series(result["alpha"], name="alpha").sort_index()
    profit = pd.Series(result["profit"], name="profit").sort_index()
    aligned = profit.index.intersection(alpha.index)
    out = pd.DataFrame(
        {
            "pool": series_key,
            "pool_label": series_label,
            "week_start": pd.to_datetime(aligned, errors="coerce"),
            "profit": pd.to_numeric(profit.loc[aligned], errors="coerce").to_numpy(),
            "alpha": pd.to_numeric(alpha.loc[aligned], errors="coerce").to_numpy(),
        }
    )
    out["profit_bps"] = out["profit"] * 10_000.0
    out["alpha_bps"] = out["alpha"] * 10_000.0
    out["profit_cumulative_bps"] = out["profit_bps"].fillna(0.0).cumsum()
    out["alpha_cumulative_bps"] = out["alpha_bps"].fillna(0.0).cumsum()
    for key in ("turnover", "rent", "count", "solve_rate"):
        if key not in result:
            continue
        series = pd.Series(result[key]).sort_index()
        out[key] = pd.to_numeric(series.reindex(aligned), errors="coerce").to_numpy()
    return out


def company_backtest_relative_plot_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    series_key: str,
    series_label: str,
) -> pd.DataFrame:
    left_data = left.set_index("week_start").sort_index()
    right_data = right.set_index("week_start").sort_index()
    aligned = left_data.index.intersection(right_data.index)
    out = pd.DataFrame(
        {
            "pool": series_key,
            "pool_label": series_label,
            "week_start": aligned,
            "profit": left_data.loc[aligned, "profit"].to_numpy()
            - right_data.loc[aligned, "profit"].to_numpy(),
            "alpha": left_data.loc[aligned, "alpha"].to_numpy()
            - right_data.loc[aligned, "alpha"].to_numpy(),
        }
    )
    out["profit_bps"] = out["profit"] * 10_000.0
    out["alpha_bps"] = out["alpha"] * 10_000.0
    out["profit_cumulative_bps"] = out["profit_bps"].fillna(0.0).cumsum()
    out["alpha_cumulative_bps"] = out["alpha_bps"].fillna(0.0).cumsum()
    return out


def company_backtest_neutral_comparison_plot_data(
    model: pd.DataFrame,
    neutral: pd.DataFrame,
    *,
    model_key: str,
    model_label: str,
    neutral_key: str,
    neutral_label: str,
    delta_key: str,
    delta_label: str,
) -> pd.DataFrame:
    columns = [
        "pool",
        "pool_label",
        "week_start",
        "profit_cumulative_bps",
        "incremental_cumulative_bps",
    ]
    comparison = []
    for frame, key, label in (
        (model, model_key, model_label),
        (neutral, neutral_key, neutral_label),
    ):
        item = frame.assign(pool=key, pool_label=label, incremental_cumulative_bps=pd.NA)
        comparison.append(item[columns])

    delta = company_backtest_relative_plot_data(
        model,
        neutral,
        series_key=delta_key,
        series_label=delta_label,
    )
    delta["profit_cumulative_bps"] = pd.NA
    delta = delta.rename(columns={"alpha_cumulative_bps": "incremental_cumulative_bps"})
    comparison.append(delta[columns])
    return pd.concat(comparison, ignore_index=True)


def run_company_score_backtest(
    score: pd.DataFrame,
    score_long: pd.DataFrame,
    output_dir: Path,
    plot_dir: Path,
    *,
    args: argparse.Namespace,
    series_key: str,
    series_label: str,
    series_color: str,
    output_prefix: str,
    output_name: str,
) -> dict[str, Any]:
    paths = write_company_score_inputs(output_dir, score, score_long)
    api_summary: dict[str, Any] = {}
    plot_paths: dict[str, str] = {}
    endpoint_used = ""
    result: dict[str, Any] | None = None
    plot_data = pd.DataFrame()

    if not args.company_skip_api:
        payload = create_company_backtest_payload(
            score,
            api_time=args.company_api_time,
            daily=args.company_daily,
            tar=args.company_tar,
            cap=args.company_cap,
            trgain=args.company_trgain,
            fee=args.company_fee,
            vol_limit=args.company_vol_limit,
            return_eod=args.company_return_eod,
        )
        endpoints = args.company_endpoint or list(DEFAULT_COMPANY_ENDPOINTS)
        endpoint_used, result = call_company_backtest_api(
            payload,
            endpoints=endpoints,
            timeout=args.company_timeout,
        )
        api_summary = write_company_api_outputs(output_dir, result)
        plot_data = company_backtest_plot_data(
            result,
            series_key=series_key,
            series_label=series_label,
        )
        color = {series_key: series_color} if series_color else None
        plot_paths = write_company_backtest_cumulative_plot(
            plot_data,
            plot_dir,
            input_path=Path(paths["score_wide"]),
            output_prefix=output_prefix,
            output_name=output_name,
            variant_label=series_label,
            pools=(series_key,),
            series_colors=color,
            x_label_mode="years_only",
        )

    return {
        "paths": paths,
        "api": {
            "called": not args.company_skip_api,
            "endpoint_used": endpoint_used,
            "api_time": args.company_api_time,
            "daily": bool(args.company_daily),
            "tar": args.company_tar,
            "cap": args.company_cap,
            "trgain": args.company_trgain,
            "fee": args.company_fee,
            "vol_limit": args.company_vol_limit,
            "return_eod": args.company_return_eod,
            "summary": api_summary,
        },
        "plot_paths": plot_paths,
        "plot_data": plot_data,
        "result": result,
    }


def run_company_backtest_analysis(
    predictions: pd.DataFrame,
    output_dir: Path,
    plot_dir: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    company_top_n = int(args.company_top_n if args.company_top_n is not None else args.top_n)
    filtered, filter_stats = filter_company_backtest_scores(
        predictions,
        score_col=args.score_col,
        clocks=args.company_clock,
        start_clock=args.company_start_clock,
        end_clock=args.company_end_clock,
        pool=args.company_pool,
        pool_path=args.company_pool_path,
        pool_date_lag_sessions=args.pool_date_lag_sessions,
    )
    score, score_long, score_stats = build_company_score_matrix(
        filtered,
        score_col=args.score_col,
        score_agg=args.company_score_agg,
        score_transform=args.company_score_transform,
        top_n=company_top_n,
    )
    series_key = args.company_series_key or slug_label(args.variant or args.run_id or "company")
    series_label = (
        args.company_series_label or args.plot_variant_label or args.variant or series_key
    )
    model_job = run_company_score_backtest(
        score,
        score_long,
        output_dir,
        plot_dir,
        args=args,
        series_key=series_key,
        series_label=series_label,
        series_color=args.company_series_color,
        output_prefix=args.plot_prefix or series_key,
        output_name=f"{slug_label(args.plot_prefix or series_key)}_company_backtest",
    )

    neutral_trace: dict[str, Any] = {}
    neutral_delta_plot_paths: dict[str, str] = {}
    if args.company_neutral_baseline:
        neutral_score, neutral_score_long, neutral_score_stats = build_company_neutral_score_matrix(
            score,
            score_col=args.score_col,
            neutral_score=args.company_neutral_score,
            base_stats=score_stats,
        )
        neutral_key = "neutral_pool"
        neutral_label = "neutral_pool"
        neutral_output_dir = output_dir.with_name(f"{output_dir.name}_neutral_pool")
        neutral_plot_dir = plot_dir.with_name(f"{plot_dir.name}_neutral_pool")
        neutral_job = run_company_score_backtest(
            neutral_score,
            neutral_score_long,
            neutral_output_dir,
            neutral_plot_dir,
            args=args,
            series_key=neutral_key,
            series_label=neutral_label,
            series_color="#5d6674",
            output_prefix=f"{args.plot_prefix or series_key}_neutral_pool",
            output_name=f"{slug_label(args.plot_prefix or series_key)}_neutral_pool_company_backtest",
        )
        neutral_trace = {
            "score_stats": neutral_score_stats,
            "paths": neutral_job["paths"],
            "api": neutral_job["api"],
            "plot_paths": neutral_job["plot_paths"],
        }
        if not model_job["plot_data"].empty and not neutral_job["plot_data"].empty:
            delta_key = series_key
            delta_label = series_label
            comparison = company_backtest_neutral_comparison_plot_data(
                model_job["plot_data"],
                neutral_job["plot_data"],
                model_key=series_key,
                model_label=series_label,
                neutral_key=neutral_key,
                neutral_label=neutral_label,
                delta_key=delta_key,
                delta_label=delta_label,
            )
            neutral_delta_plot_paths = write_company_backtest_neutral_comparison_plot(
                comparison,
                plot_dir,
                input_path=Path(model_job["paths"]["score_wide"]),
                output_prefix=f"{args.plot_prefix or series_key}_minus_neutral",
                output_name=(
                    f"{slug_label(args.plot_prefix or series_key)}"
                    "_neutral_comparison_company_backtest"
                ),
                variant_label=f"{series_label} vs neutral_pool",
                pools=(series_key, neutral_key),
                series_colors={
                    series_key: args.company_series_color,
                    neutral_key: "#5d6674",
                },
                x_label_mode="years_only",
            )

    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_rows": int(len(predictions)),
        "filter_stats": filter_stats,
        "score_stats": score_stats,
        "paths": model_job["paths"],
        "api": model_job["api"],
        "neutral_baseline": neutral_trace,
        "neutral_delta_plot_paths": neutral_delta_plot_paths,
        "plot_paths": {
            **model_job["plot_paths"],
            **{
                f"neutral_{key}": value
                for key, value in neutral_trace.get("plot_paths", {}).items()
            },
            **{f"neutral_delta_{key}": value for key, value in neutral_delta_plot_paths.items()},
        },
    }
    write_json(output_dir / "company_backtest_trace.json", trace, ensure_ascii=True)
    return trace
