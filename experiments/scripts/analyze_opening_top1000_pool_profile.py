from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import newey_west_mean_ci
from opening_strength_fit.feature_utils import finite_numeric as finite

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "output/diagnostics/opening_h1m_maxpath_deep_audit_v1"
OUTPUT_DIR = ROOT / "output/diagnostics/opening_h1m_top1000_pool_profile_v1"
SELECTOR = "max_z_10clock_top1000"


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float(wealth.div(wealth.cummax()).sub(1.0).min())


def performance_row(name: str, returns_bps: pd.Series) -> dict[str, float | int | str]:
    returns = finite(returns_bps).dropna().div(10_000.0)
    compounded = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + compounded) ** (252.0 / len(returns)) - 1.0)
    annualized_volatility = float(returns.std() * np.sqrt(252.0))
    return {
        "series": name,
        "days": len(returns),
        "mean_bps": returns.mean() * 10_000.0,
        "median_bps": returns.median() * 10_000.0,
        "p05_bps": returns.quantile(0.05) * 10_000.0,
        "p95_bps": returns.quantile(0.95) * 10_000.0,
        "positive_days_pct": returns.gt(0).mean() * 100.0,
        "annualized_return_pct": annualized_return * 100.0,
        "annualized_volatility_pct": annualized_volatility * 100.0,
        "sharpe_rf0": returns.mean() / returns.std() * np.sqrt(252.0),
        "max_drawdown_pct": max_drawdown(returns) * 100.0,
    }


def active_row(
    name: str,
    selected_bps: pd.Series,
    market_bps: pd.Series,
) -> dict[str, float | int | str]:
    selected = finite(selected_bps).div(10_000.0)
    market = finite(market_bps).div(10_000.0)
    valid = selected.notna() & market.notna()
    selected = selected.loc[valid]
    market = market.loc[valid]
    active = selected.sub(market)
    n = len(active)
    ci_low, ci_high = newey_west_mean_ci(active)
    return {
        "series": name,
        "days": n,
        "selected_mean_bps": selected.mean() * 10_000.0,
        "market_mean_bps": market.mean() * 10_000.0,
        "active_mean_bps": active.mean() * 10_000.0,
        "nw5_ci_low_bps": ci_low * 10_000.0,
        "nw5_ci_high_bps": ci_high * 10_000.0,
        "beta_to_equal_weight_market": selected.cov(market) / market.var(),
        "correlation_to_equal_weight_market": selected.corr(market),
        "tracking_error_ann_pct": active.std() * np.sqrt(252.0) * 100.0,
        "information_ratio": active.mean() / active.std() * np.sqrt(252.0),
        "active_win_days_pct": active.gt(0).mean() * 100.0,
    }


def quantile_row(
    metric: str,
    values: pd.Series,
    multiplier: float = 1.0,
    unit: str = "",
) -> dict[str, float | int | str]:
    values = finite(values).dropna().mul(multiplier)
    return {
        "metric": metric,
        "unit": unit,
        "valid_rows": len(values),
        "mean": values.mean(),
        "p01": values.quantile(0.01),
        "p10": values.quantile(0.10),
        "p25": values.quantile(0.25),
        "p50": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
        "p99": values.quantile(0.99),
    }


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(AUDIT_DIR / "current_selected_members.parquet")
    daily_all = pd.read_csv(AUDIT_DIR / "parameter_daily_metrics.csv")
    daily = daily_all.loc[daily_all["selector"].eq(SELECTOR)].copy()
    daily["date"] = daily["date"].astype(str)
    summary = (
        pd.read_csv(AUDIT_DIR / "parameter_summary.csv")
        .loc[lambda x: x["selector"].eq(SELECTOR)]
        .iloc[0]
    )
    by_year = pd.read_csv(AUDIT_DIR / "current_by_year.csv").copy()
    industry = pd.read_csv(AUDIT_DIR / "current_industry_exposure.csv").copy()
    board = pd.read_csv(AUDIT_DIR / "current_board_exposure.csv").copy()
    numeric = pd.read_csv(AUDIT_DIR / "current_numeric_exposure.csv").copy()
    adjacent = pd.read_csv(AUDIT_DIR / "current_adjacent_overlap_summary.csv").iloc[0]
    frequency_summary = pd.read_csv(AUDIT_DIR / "current_symbol_frequency_summary.csv").iloc[0]
    style = pd.read_csv(OUTPUT_DIR / "style_factor_exposure.csv").copy()
    style_members = pd.read_parquet(OUTPUT_DIR / "selected_style_factors.parquet")
    st_exposure = pd.read_csv(OUTPUT_DIR / "st_exposure.csv").iloc[0]
    market_regular = pd.read_csv(OUTPUT_DIR / "market_regular_daily.csv")
    market_regular["date"] = market_regular["date"].astype(str)

    performance = pd.DataFrame(
        [
            performance_row("pool_same_day_close_preclose", daily["selected_daily_return_bps"]),
            performance_row("market_same_day_close_preclose", daily["market_daily_return_bps"]),
            performance_row("pool_next_day_close_close", daily["selected_next_day_return_bps"]),
            performance_row("market_next_day_close_close", daily["market_next_day_return_bps"]),
        ]
    )
    performance.to_csv(OUTPUT_DIR / "performance_timeseries_summary.csv", index=False)
    active = pd.DataFrame(
        [
            active_row(
                "same_day_close_preclose",
                daily["selected_daily_return_bps"],
                daily["market_daily_return_bps"],
            ),
            active_row(
                "next_day_close_close",
                daily["selected_next_day_return_bps"],
                daily["market_next_day_return_bps"],
            ),
        ]
    )
    active.to_csv(OUTPUT_DIR / "performance_active_summary.csv", index=False)

    return_distribution = pd.DataFrame(
        [
            quantile_row("stock_same_day_return", selected["daily_return_bps"], unit="bps"),
            quantile_row("stock_next_day_return", selected["next_day_return_bps"], unit="bps"),
            quantile_row(
                "stock_intraday_high_low_range",
                selected["daily_high_return_bps"].sub(selected["daily_low_return_bps"]),
                unit="bps",
            ),
        ]
    )
    return_distribution.to_csv(OUTPUT_DIR / "return_distribution.csv", index=False)

    absolute_same_day_return = selected["daily_return_bps"].abs()
    regular_mask = absolute_same_day_return.le(3_000.0)
    nonstandard_mask = absolute_same_day_return.gt(3_000.0)
    selected_regular_daily = (
        selected.loc[regular_mask].groupby("date", sort=False)["daily_return_bps"].mean()
    )
    regular_compare = market_regular.set_index("date").join(
        selected_regular_daily.rename("selected_regular_return_bps"), how="inner"
    )
    nonstandard_selected = int(nonstandard_mask.sum())
    nonstandard_market = int(market_regular["nonstandard_rows"].sum())
    market_rows = int(market_regular["market_rows"].sum())
    tail_rows = [
        {
            "event": "close_limit_up",
            "selected_rows": int(selected["limit_up"].sum()),
            "selected_rate_pct": summary["selected_limit_up_rate_pct"],
            "market_rows": int(round(summary["market_limit_up_per_day"] * summary["days"])),
            "market_rate_pct": summary["market_limit_up_rate_pct"],
            "market_event_coverage_pct": summary["recall_limit_up_pooled_pct"],
            "density_enrichment": summary["enrichment_limit_up"],
        },
        {
            "event": "close_limit_down",
            "selected_rows": int(selected["limit_down"].sum()),
            "selected_rate_pct": summary["selected_limit_down_rate_pct"],
            "market_rows": int(round(summary["market_limit_down_per_day"] * summary["days"])),
            "market_rate_pct": summary["market_limit_down_rate_pct"],
            "market_event_coverage_pct": summary["recall_limit_down_pooled_pct"],
            "density_enrichment": summary["enrichment_limit_down"],
        },
        {
            "event": "close_limit_combined",
            "selected_rows": int(selected["limit_event"].sum()),
            "selected_rate_pct": summary["selected_limit_event_rate_pct"],
            "market_rows": int(round(summary["market_limit_event_per_day"] * summary["days"])),
            "market_rate_pct": summary["market_limit_event_rate_pct"],
            "market_event_coverage_pct": summary["recall_limit_event_pooled_pct"],
            "density_enrichment": summary["enrichment_limit_event"],
        },
        {
            "event": "absolute_close_move_over_30pct",
            "selected_rows": nonstandard_selected,
            "selected_rate_pct": nonstandard_selected / len(selected) * 100.0,
            "market_rows": nonstandard_market,
            "market_rate_pct": nonstandard_market / market_rows * 100.0,
            "market_event_coverage_pct": nonstandard_selected / nonstandard_market * 100.0,
            "density_enrichment": (nonstandard_selected / len(selected))
            / (nonstandard_market / market_rows),
        },
    ]
    tail_profile = pd.DataFrame(tail_rows)
    tail_profile.to_csv(OUTPUT_DIR / "tail_event_profile.csv", index=False)

    outcome_rows = []
    for name, mask in (
        ("limit_up", selected["limit_up"]),
        ("limit_down", selected["limit_down"]),
        ("non_limit", ~selected["limit_event"]),
    ):
        part = selected.loc[mask]
        outcome_rows.append(
            {
                "outcome": name,
                "rows": len(part),
                "share_pct": len(part) / len(selected) * 100.0,
                "same_day_return_mean_bps": part["daily_return_bps"].mean(),
                "next_day_return_mean_bps": part["next_day_return_bps"].mean(),
                "same_day_mean_contribution_bps": part["daily_return_bps"].sum() / len(selected),
                "next_day_mean_contribution_bps": part["next_day_return_bps"].sum() / len(selected),
            }
        )
    pd.DataFrame(outcome_rows).to_csv(OUTPUT_DIR / "outcome_return_decomposition.csv", index=False)

    industry = industry.loc[industry["industry"].ne("missing")].copy()
    industry["active_weight_pp"] = industry["selected_share_pct"] - industry["market_share_pct"]
    industry.to_csv(OUTPUT_DIR / "industry_profile.csv", index=False)
    industry_daily = (
        selected.assign(industry=selected["industry"].fillna("missing"))
        .groupby(["date", "industry"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    industry_daily_share = industry_daily.div(industry_daily.sum(axis=1), axis=0)
    industry_daily_profile = pd.DataFrame(
        {
            "industry": industry_daily_share.columns,
            "daily_share_mean_pct": industry_daily_share.mean().mul(100.0).to_numpy(),
            "daily_share_std_pct": industry_daily_share.std().mul(100.0).to_numpy(),
            "daily_share_p10_pct": industry_daily_share.quantile(0.10).mul(100.0).to_numpy(),
            "daily_share_p50_pct": industry_daily_share.quantile(0.50).mul(100.0).to_numpy(),
            "daily_share_p90_pct": industry_daily_share.quantile(0.90).mul(100.0).to_numpy(),
        }
    ).sort_values("daily_share_mean_pct", ascending=False)
    industry_daily_profile.to_csv(OUTPUT_DIR / "industry_daily_profile.csv", index=False)
    board["active_weight_pp"] = board["selected_share_pct"] - board["market_share_pct"]
    board.to_csv(OUTPUT_DIR / "board_profile.csv", index=False)

    hhi_selected = float(np.square(industry["selected_share_pct"].div(100.0)).sum())
    hhi_market = float(np.square(industry["market_share_pct"].div(100.0)).sum())
    daily_hhi = industry_daily_share.pow(2).sum(axis=1)
    daily_top5 = pd.Series(
        np.sort(industry_daily_share.to_numpy(), axis=1)[:, -5:].sum(axis=1),
        index=industry_daily_share.index,
    )
    concentration = pd.DataFrame(
        [
            {
                "scope": "pooled_selected_stock_days",
                "industry_hhi": hhi_selected,
                "effective_industries": 1.0 / hhi_selected,
                "top3_industry_share_pct": industry["selected_share_pct"].nlargest(3).sum(),
                "top5_industry_share_pct": industry["selected_share_pct"].nlargest(5).sum(),
                "top10_industry_share_pct": industry["selected_share_pct"].nlargest(10).sum(),
            },
            {
                "scope": "pooled_all_a_stock_days",
                "industry_hhi": hhi_market,
                "effective_industries": 1.0 / hhi_market,
                "top3_industry_share_pct": industry["market_share_pct"].nlargest(3).sum(),
                "top5_industry_share_pct": industry["market_share_pct"].nlargest(5).sum(),
                "top10_industry_share_pct": industry["market_share_pct"].nlargest(10).sum(),
            },
            {
                "scope": "selected_daily_mean",
                "industry_hhi": daily_hhi.mean(),
                "effective_industries": (1.0 / daily_hhi).mean(),
                "top3_industry_share_pct": np.nan,
                "top5_industry_share_pct": daily_top5.mean() * 100.0,
                "top10_industry_share_pct": np.nan,
            },
        ]
    )
    concentration.to_csv(OUTPUT_DIR / "industry_concentration.csv", index=False)

    size_liquidity = pd.DataFrame(
        [
            quantile_row("total_market_cap", selected["market_cap"], 10_000.0 / 1e9, "CNY bn"),
            quantile_row(
                "float_market_cap", selected["float_market_cap"], 10_000.0 / 1e9, "CNY bn"
            ),
            quantile_row("daily_amount", selected["daily_amount"], 1_000.0 / 1e6, "CNY mn"),
            quantile_row("turnover_rate", selected["daily_turnover_rate"], unit="pct"),
            quantile_row("free_turnover_rate", selected["free_turnover_rate"], unit="pct"),
            quantile_row("close_price", selected["close_price"], unit="CNY"),
        ]
    )
    size_liquidity.to_csv(OUTPUT_DIR / "size_liquidity_distribution.csv", index=False)
    amount_mn = selected["daily_amount"].mul(1_000.0 / 1e6)
    total_cap_bn = selected["market_cap"].mul(10_000.0 / 1e9)
    tradability = pd.DataFrame(
        [
            {
                "metric": "daily_amount_below_50m_pct",
                "value": amount_mn.lt(50).mean() * 100.0,
            },
            {
                "metric": "daily_amount_below_100m_pct",
                "value": amount_mn.lt(100).mean() * 100.0,
            },
            {
                "metric": "total_market_cap_below_3bn_pct",
                "value": total_cap_bn.lt(3).mean() * 100.0,
            },
            {
                "metric": "close_price_below_2_pct",
                "value": selected["close_price"].lt(2).mean() * 100.0,
            },
            {
                "metric": "st_share_pct",
                "value": st_exposure["selected_st_share_pct"],
            },
        ]
    )
    tradability.to_csv(OUTPUT_DIR / "tradability_flags.csv", index=False)

    frequency = selected.groupby("symbol", sort=False).size().sort_values(ascending=False)
    composition_rows = [
        {"metric": "days", "value": selected["date"].nunique()},
        {"metric": "names_per_day", "value": len(selected) / selected["date"].nunique()},
        {"metric": "unique_symbols", "value": selected["symbol"].nunique()},
        {"metric": "adjacent_day_retention_pct", "value": adjacent["retention_mean_pct"]},
        {
            "metric": "adjacent_day_one_way_replacement_pct",
            "value": 100.0 - adjacent["retention_mean_pct"],
        },
        {"metric": "adjacent_day_jaccard_pct", "value": adjacent["jaccard_mean_pct"]},
        {
            "metric": "symbol_selected_days_median",
            "value": frequency_summary["selected_days_p50"],
        },
        {"metric": "symbol_selected_days_p90", "value": frequency_summary["selected_days_p90"]},
    ]
    for top_n in (10, 50, 100, 500, 1_000):
        composition_rows.append(
            {
                "metric": f"top_{top_n}_frequent_names_membership_share_pct",
                "value": frequency.head(top_n).sum() / len(selected) * 100.0,
            }
        )
    pd.DataFrame(composition_rows).to_csv(OUTPUT_DIR / "composition_stability.csv", index=False)

    style_corrected = style.copy()
    for index, row in style_corrected.iterrows():
        factor = row["factor"]
        factor_pct = finite(style_members[f"{factor}_pct"]).dropna()
        style_corrected.loc[index, "selected_bottom_decile_pct"] = (
            factor_pct.le(0.10).mean() * 100.0
        )
        style_corrected.loc[index, "selected_top_decile_pct"] = factor_pct.ge(0.90).mean() * 100.0
    style_corrected.to_csv(OUTPUT_DIR / "style_factor_exposure.csv", index=False)

    by_year_keep = by_year[
        [
            "year",
            "days",
            "selected_daily_return_bps",
            "daily_return_excess_bps",
            "selected_next_day_return_bps",
            "next_day_return_excess_bps",
            "selected_limit_event_rate_pct",
            "market_limit_event_rate_pct",
            "recall_limit_event_pooled_pct",
            "enrichment_limit_event",
        ]
    ].copy()
    by_year_keep.to_csv(OUTPUT_DIR / "profile_by_year.csv", index=False)

    same_active = active.loc[active["series"].eq("same_day_close_preclose")].iloc[0]
    next_active = active.loc[active["series"].eq("next_day_close_close")].iloc[0]
    next_perf = performance.loc[performance["series"].eq("pool_next_day_close_close")].iloc[0]
    market_next_perf = performance.loc[
        performance["series"].eq("market_next_day_close_close")
    ].iloc[0]
    selected_numeric = numeric.set_index("exposure")
    top_industries = industry.nlargest(10, "selected_share_pct")
    over_industries = industry.nlargest(5, "active_weight_pp")
    under_industries = industry.nsmallest(5, "active_weight_pp")
    style_index = style_corrected.set_index("factor")
    size_index = size_liquidity.set_index("metric")
    regular_excess = (
        regular_compare["selected_regular_return_bps"]
        - regular_compare["market_regular_return_bps"]
    ).mean()
    years_forward_negative = int(by_year_keep["next_day_return_excess_bps"].lt(0).sum())

    report = (
        f"""# 开盘 09:31–09:40 Top1000 股池画像

## 一句话结论

这不是一个“平均收益与全 A 相同的中性股池”，而是一个**高动量、高波动、高换手、偏成长估值、
行业略集中、显著富集价格极端事件**的状态股池。它对收盘涨跌停的历史覆盖确实达到
**{summary["recall_limit_event_pooled_pct"]:.2f}%**，但纯前瞻的下一交易日收益相对全 A 为
**{next_active["active_mean_bps"]:.2f} bp/日**，四个自然年全部为负，当前不满足收益中性目标。

## 口径

- 样本：2022-01-04 至 2025-12-31，969 个 rolling-OOS 交易日，共 969,000 个入池 stock-day；每日固定 1,000 只。
- 建池信息截至 09:40。池子默认等权描述，不把模型分数当作股池性质。
- 基准：同一日期、同一 A 股代码范围内的全市场股票等权 stock-day / 日收益。
- `当日收益` 是 Close/PreClose，其中 09:40 前收益已经发生，只用于描述池中股票，不是可交易回测。
- `下一日收益` 是入池日收盘到下一交易日收盘，完全位于建池之后，但也不是 09:40 精确成交回报。
- 收益未计交易成本；在每日约 50% 的成分更换率下，不能把年化数直接当可实现策略业绩。

## 标准股池画像总表

| 维度 | 当前 Top1000 | 全 A / 相对结论 |
| --- | ---: | --- |
| 当日等权平均收益 | {same_active["selected_mean_bps"]:.2f} bp | 全 A {same_active["market_mean_bps"]:.2f} bp；含已发生行情 |
| 下一日等权平均收益 | {next_active["selected_mean_bps"]:.2f} bp | 全 A {next_active["market_mean_bps"]:.2f} bp；超额 {next_active["active_mean_bps"]:.2f} bp |
| 下一日年化波动 | {next_perf["annualized_volatility_pct"]:.2f}% | 全 A {market_next_perf["annualized_volatility_pct"]:.2f}% |
| 下一日市场 beta / 相关性 | {next_active["beta_to_equal_weight_market"]:.3f} / {next_active["correlation_to_equal_weight_market"]:.3f} | beta 高于 1 |
| 池内收盘涨停率 | {summary["selected_limit_up_rate_pct"]:.3f}% | 全 A {summary["market_limit_up_rate_pct"]:.3f}%；{summary["enrichment_limit_up"]:.2f}x |
| 池内收盘跌停率 | {summary["selected_limit_down_rate_pct"]:.3f}% | 全 A {summary["market_limit_down_rate_pct"]:.3f}%；{summary["enrichment_limit_down"]:.2f}x |
| 池内涨跌停合计率 | {summary["selected_limit_event_rate_pct"]:.3f}% | 全 A {summary["market_limit_event_rate_pct"]:.3f}%；{summary["enrichment_limit_event"]:.2f}x |
| 全市场涨跌停覆盖 | {summary["recall_limit_event_pooled_pct"]:.2f}% | 涨停 {summary["recall_limit_up_pooled_pct"]:.2f}%；跌停 {summary["recall_limit_down_pooled_pct"]:.2f}% |
| 总市值 | 中位 {size_index.loc["total_market_cap", "p50"] * 10:.2f} 亿元 | 每日市场百分位均值 {selected_numeric.loc["market_cap_pct", "selected_mean_pct"]:.2f}，大小盘中性 |
| 日成交额 | 中位 {size_index.loc["daily_amount", "p50"]:.1f} 百万元 | 市场百分位 {selected_numeric.loc["daily_amount_pct", "selected_mean_pct"]:.2f}，明显偏高 |
| 换手率 | 中位 {size_index.loc["turnover_rate", "p50"]:.2f}% | 市场百分位 {selected_numeric.loc["daily_turnover_rate_pct", "selected_mean_pct"]:.2f}，明显偏高 |
| 20 日动量 | 中位 {style_index.loc["momentum_20d", "selected_p50"] * 100:.2f}% | 全 A {style_index.loc["momentum_20d", "market_p50"] * 100:.2f}%；市场百分位 {style_index.loc["momentum_20d", "selected_mean_market_pct"]:.2f} |
| 20 日年化波动 | 中位 {style_index.loc["volatility_20d_ann", "selected_p50"] * 100:.2f}% | 全 A {style_index.loc["volatility_20d_ann", "market_p50"] * 100:.2f}%；市场百分位 {style_index.loc["volatility_20d_ann", "selected_mean_market_pct"]:.2f} |
| 正 PE 中位数 | {style_index.loc["pe_positive", "selected_p50"]:.2f}x | 全 A {style_index.loc["pe_positive", "market_p50"]:.2f}x；偏贵 |
| 正 PB 中位数 | {style_index.loc["pb_positive", "selected_p50"]:.2f}x | 全 A {style_index.loc["pb_positive", "market_p50"]:.2f}x；偏贵 |
| 创业板占比 | {board.loc[board["board"].eq("chinext"), "selected_share_pct"].iloc[0]:.2f}% | 全 A {board.loc[board["board"].eq("chinext"), "market_share_pct"].iloc[0]:.2f}% |
| ST 占比 | {st_exposure["selected_st_share_pct"]:.2f}% | 全 A {st_exposure["market_st_share_pct"]:.2f}%；{st_exposure["st_representation_ratio"]:.2f}x |
| 相邻日保留 / 更换 | {adjacent["retention_mean_pct"]:.2f}% / {100 - adjacent["retention_mean_pct"]:.2f}% | 成分很不稳定 |
| 行业有效数量 | {1 / hhi_selected:.2f} | 全 A {1 / hhi_market:.2f}；略更集中 |

> 表中市值原始单位已换算：数据库 `TotalMarketValue × 10,000` 元；此处中位数实际为
> {size_index.loc["total_market_cap", "p50"]:.2f} 十亿元，即 {size_index.loc["total_market_cap", "p50"] * 10:.2f} 亿元。

## 1. 收益和风险

最重要的结论不是“当日涨得好”，而是**形成股池以后存在均值回归**：

- 当日池均收益 +{same_active["selected_mean_bps"]:.2f} bp，全 A +{same_active["market_mean_bps"]:.2f} bp，表面超额 +{same_active["active_mean_bps"]:.2f} bp；但这包含 09:40 前已经观察到的价格路径。
- 下一交易日 close-to-close：池子 {next_active["selected_mean_bps"]:.2f} bp，全 A +{next_active["market_mean_bps"]:.2f} bp，超额 {next_active["active_mean_bps"]:.2f} bp。
- 下一日超额的 Newey-West(5) 95% 区间为 [{next_active["nw5_ci_low_bps"]:.2f}, {next_active["nw5_ci_high_bps"]:.2f}] bp，不是“统计上和市场差不多”。
- 下一日 beta={next_active["beta_to_equal_weight_market"]:.3f}、相关性={next_active["correlation_to_equal_weight_market"]:.3f}；高 beta 不能解释负超额。
- 下一日超额在 2022、2023、2024、2025 四年中有 {years_forward_negative}/4 年为负，分别见 `profile_by_year.csv`。

即使删除绝对涨跌超过 30% 的非标准交易日代理样本，当日池均收益仍为
{regular_compare["selected_regular_return_bps"].mean():.2f} bp，全 A 为
{regular_compare["market_regular_return_bps"].mean():.2f} bp，超额 {regular_excess:.2f} bp；
所以当日强势不完全由 IPO/恢复上市等大跳变造成，但仍然不是 09:40 后的收益。

## 2. 涨跌停和尾部

四年里池中共有 {int(selected["limit_up"].sum()):,} 个收盘涨停、{int(selected["limit_down"].sum()):,} 个收盘跌停，
合计 {int(selected["limit_event"].sum()):,} 个，占全部池成员的 {summary["selected_limit_event_rate_pct"]:.3f}%。
随机按市场占比取 1,000 只，事件密度约为 {summary["market_limit_event_rate_pct"]:.3f}%；当前池是 {summary["enrichment_limit_event"]:.2f} 倍。

这里必须同时报告两个不同概念：

- **覆盖率**：全市场发生的涨跌停中，有 {summary["recall_limit_event_pooled_pct"]:.2f}% 落在池里。
- **池内纯度**：池里的股票中，只有 {summary["selected_limit_event_rate_pct"]:.2f}% 最终收盘涨停或跌停，另外 {100 - summary["selected_limit_event_rate_pct"]:.2f}% 不是涨跌停。

池内单股当日收益 P10/P50/P90 为
{selected["daily_return_bps"].quantile(0.10):.0f}/{selected["daily_return_bps"].quantile(0.50):.0f}/{selected["daily_return_bps"].quantile(0.90):.0f} bp；
绝对涨跌超过 5% 的两侧比例分别为
{selected["daily_return_bps"].ge(500).mean() * 100:.2f}% 和 {selected["daily_return_bps"].le(-500).mean() * 100:.2f}%。
绝对收盘涨跌超过 30% 的非标准状态代理，池中覆盖 {nonstandard_selected}/{nonstandard_market}={nonstandard_selected / nonstandard_market * 100:.2f}%，
密度是市场的 {(nonstandard_selected / len(selected)) / (nonstandard_market / market_rows):.2f} 倍；这是一个此前容易漏掉的重要性质。

## 3. 行业与板块

池中权重最高的十个申万一级行业：

| 行业 | 池内占比 | 全 A 占比 | 主动偏离 |
| --- | ---: | ---: | ---: |
"""
        + "\n".join(
            f"| {row.industry} | {row.selected_share_pct:.2f}% | {row.market_share_pct:.2f}% | {row.active_weight_pp:+.2f} pp |"
            for row in top_industries.itertuples()
        )
        + f"""

最大超配是 {", ".join(f"{r.industry} {r.active_weight_pp:+.2f}pp" for r in over_industries.itertuples())}；
最大低配是 {", ".join(f"{r.industry} {r.active_weight_pp:+.2f}pp" for r in under_industries.itertuples())}。
行业 HHI 为 {hhi_selected:.4f}，全 A 为 {hhi_market:.4f}；对应有效行业数 {1 / hhi_selected:.2f} vs {1 / hhi_market:.2f}。
因此它不是行业高度押注池，但确有 TMT/高弹性行业偏置，行业不能视为中性。

板块方面，创业板 {board.loc[board["board"].eq("chinext"), "selected_share_pct"].iloc[0]:.2f}%
（全 A {board.loc[board["board"].eq("chinext"), "market_share_pct"].iloc[0]:.2f}%）；
沪主板 {board.loc[board["board"].eq("sh_main"), "selected_share_pct"].iloc[0]:.2f}%、
深主板 {board.loc[board["board"].eq("sz_main"), "selected_share_pct"].iloc[0]:.2f}%、
科创板 {board.loc[board["board"].eq("star"), "selected_share_pct"].iloc[0]:.2f}%。

## 4. 规模、风格与状态

- 规模基本中性：总市值市场百分位均值 {selected_numeric.loc["market_cap_pct", "selected_mean_pct"]:.2f}，流通市值 {selected_numeric.loc["float_market_cap_pct", "selected_mean_pct"]:.2f}。
- 价格略高：收盘价市场百分位 {selected_numeric.loc["close_price_pct", "selected_mean_pct"]:.2f}。
- 动量明显偏高：20 日/60 日动量市场百分位分别为 {style_index.loc["momentum_20d", "selected_mean_market_pct"]:.2f}/{style_index.loc["momentum_60d", "selected_mean_market_pct"]:.2f}。
- 波动明显偏高：20 日年化波动市场百分位 {style_index.loc["volatility_20d_ann", "selected_mean_market_pct"]:.2f}，Top10% 高波动股在池中占 {style_index.loc["volatility_20d_ann", "selected_top_decile_pct"]:.2f}%。
- 更靠近 52 周高位：52 周位置市场百分位 {style_index.loc["position_52w", "selected_mean_market_pct"]:.2f}。
- 估值偏贵：正 PE/PB 市场百分位 {style_index.loc["pe_positive", "selected_mean_market_pct"]:.2f}/{style_index.loc["pb_positive", "selected_mean_market_pct"]:.2f}；这更像成长/题材活跃池，不是价值池。
- ST 略超配：{st_exposure["selected_st_share_pct"]:.2f}% vs 全 A {st_exposure["market_st_share_pct"]:.2f}%，但不是主要构成。

## 5. 流动性与可交易性

- 日成交额中位数 {size_index.loc["daily_amount", "p50"]:.1f} 百万元，P10/P90 为 {size_index.loc["daily_amount", "p10"]:.1f}/{size_index.loc["daily_amount", "p90"]:.1f} 百万元。
- 成交额市场百分位均值 {selected_numeric.loc["daily_amount_pct", "selected_mean_pct"]:.2f}；Top10% 成交额股票占池 {selected_numeric.loc["daily_amount_pct", "selected_top_decile_pct"]:.2f}%。
- 换手率中位数 {size_index.loc["turnover_rate", "p50"]:.2f}%，自由流通换手率中位数 {size_index.loc["free_turnover_rate", "p50"]:.2f}%。
- {amount_mn.lt(50).mean() * 100:.2f}% 的 stock-day 日成交额低于 5,000 万元，{amount_mn.lt(100).mean() * 100:.2f}% 低于 1 亿元。
- 单股流动性总体不错，但池子约一半名字每天更换，整体交易容量最终取决于资金规模、参与率、涨跌停封单和 09:40 冲击成本；仅凭日成交额不能认定可交易。

## 6. 成分稳定性与集中度

- 相邻交易日平均只保留 {adjacent["intersection_mean"]:.1f}/1000 只，即单边更换 {100 - adjacent["retention_mean_pct"]:.2f}%；Jaccard {adjacent["jaccard_mean_pct"]:.2f}%。
- 四年累计出现 {int(frequency_summary["unique_symbols"]):,} 只不同股票；单只入池天数中位 {frequency_summary["selected_days_p50"]:.0f} 天，P90 {frequency_summary["selected_days_p90"]:.0f} 天。
- 最常出现的前 100 只只贡献全部 membership 的 {frequency.head(100).sum() / len(selected) * 100:.2f}%，不是少数妖股长期霸占。
- 每日 1000 只等权时单名权重天然是 0.1%，个股权重集中度不高；真正的集中来自共同的高动量、高波动、高换手状态，以及 TMT 行业偏置。

## 对项目目标的判断

当前这一步已经解决了一半“尾部召回”：1000 只覆盖约 {summary["recall_limit_event_pooled_pct"]:.2f}% 的收盘涨跌停，
且涨、跌两端都在 2.4 倍以上富集。但它没有满足另一个约束——收益中性；下一日超额为
{next_active["active_mean_bps"]:.2f} bp/日，而且换手、行业、动量、波动、估值暴露都不小。

因此三路线合并时，不能只优化涨跌停覆盖，还要至少同时冻结五个约束：

1. 固定总池不超过 1000，只计算去重后的增量召回；
2. 以 09:40 后的明确前瞻区间约束平均收益，而不是当日 Close/PreClose；
3. 约束行业主动偏离、20/60 日动量、20 日波动和 PB/PE 暴露；
4. 控制相邻日单边更换率与低成交额占比；
5. 分普通交易日、涨跌停日、绝对涨跌超过 30% 的非标准状态分别报告，防止少量 IPO/退市样本扭曲结论。

## 为什么采用这些维度

这是按主流指数与组合评价框架整理的：FTSE Russell 的股票指数方法把自由流通市值、规模、
成交金额、交易频率和行业分类列为可投资性核心；Morningstar 用规模、价值/成长因子描述股票组合风格；
CFA 的组合评价框架要求把收益、风险、基准比较与归因分开。这里再加入本项目特有的涨跌停覆盖/纯度和 09:40 可交易时间口径。

- FTSE Russell: https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/fr-global-equity-indices-ground-rules.pdf
- Morningstar Equity Style Box Methodology: https://advisor.morningstar.com/Enterprise/VTC/MorningstarEquityStyleBoxMethodology.pdf
- CFA Institute Portfolio Performance Evaluation: https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/portfolio-performance-evaluation
"""
    )
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
