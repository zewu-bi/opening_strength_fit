from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.optimization_acceptance_plots import (
    CUMULATIVE_MARKET_ALPHA_PANEL_TITLE,
    CUMULATIVE_MODE_CAPACITY,
    CUMULATIVE_MODE_REALISTIC,
    CUMULATIVE_MODE_TOP100,
    CUMULATIVE_MODES,
    CUMULATIVE_NET_RETURN_PANEL_TITLE,
    _attach_capacity_fraction_to_market_source,
    _panel_values,
    _replace_capacity_pool_source,
    add_background_cumulative_data,
    add_cumulative_baseline_relative_data,
    add_cumulative_market_relative_data,
    add_cumulative_percent_display_columns,
    add_market_cumulative_data,
    apply_display_labels,
    capacity_label,
    combine_net_alpha_cumulative_data,
    combine_overlay_acceptance_data,
    default_plot_directions,
    ensure_plot_colors,
    realistic_label,
    validate_plot_directions,
)
from opening_strength_fit.optimization_direction_data import (
    CUMULATIVE_DECISION_NORMALIZER,
    DEFAULT_POOL_FEE_MODE,
    DEFAULT_REALIZED_FEE_BPS,
    NEXT_CLOSE_CAPITAL_DIVISOR,
    RETURN_BPS_DENOMINATOR,
    DirectionSpec,
    line_axis,
    line_step,
    load_capacity_cumulative_plot_data,
    load_horizon_plot_data,
    load_realized_cumulative_plot_data,
    source_files,
)
from opening_strength_fit.pool_internal_plot_svg import (
    write_two_panel_bar_svg,
    write_two_panel_line_svg,
)


def write_optimization_direction_plots(
    *,
    backtests_root: Path,
    output_dir: Path,
    directions: tuple[DirectionSpec, ...] | None = None,
    pool: str = "pool_L",
    include_baseline_pool_cumulative: bool = True,
    include_baseline_universe_cumulative: bool = False,
    baseline_run_id: str = "baseline_2022_2025_cluster",
    baseline_label: str = "baseline",
    realized_fee_bps: float = DEFAULT_REALIZED_FEE_BPS,
    pool_turnover_path: str | Path | None = "auto",
    pool_fee_mode: str = DEFAULT_POOL_FEE_MODE,
    cumulative_mode: str = CUMULATIVE_MODE_TOP100,
    capacity_total_notional: float | None = None,
    capacity_decision_notional: float | None = None,
    capacity_baseline_run_id: str | None = None,
    capacity_run_ids: dict[str, str] | None = None,
    realistic_baseline_run_id: str | None = None,
    realistic_run_ids: dict[str, str] | None = None,
    title_prefix: str = "2022-2025",
    top_n: int = 100,
) -> dict[str, str]:
    if cumulative_mode not in CUMULATIVE_MODES:
        raise ValueError(
            f"unknown cumulative_mode {cumulative_mode!r}; expected {CUMULATIVE_MODES}"
        )
    if directions is None:
        plot_directions = default_plot_directions()
    else:
        plot_directions = validate_plot_directions(tuple(directions))
    if not include_baseline_pool_cumulative:
        raise ValueError("baseline pool cumulative series is required for cumulative plots")

    series = tuple(direction.key for direction in plot_directions)
    model_cumulative_series = ("baseline_pool_l", *series)
    top_cumulative_series = ("market", "background", *model_cumulative_series)
    alpha_cumulative_series = ("background", *model_cumulative_series)
    acceptance_directions = (
        DirectionSpec(key="baseline", label=baseline_label, run_id=baseline_run_id),
        *plot_directions,
    )
    acceptance_series = tuple(direction.key for direction in acceptance_directions)
    ensure_plot_colors((*acceptance_series, *top_cumulative_series, *alpha_cumulative_series))

    short_universe_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool="universe",
            horizon="short",
        )
    )
    next_pool_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool=pool,
            horizon="next",
        )
    )
    if cumulative_mode in {CUMULATIVE_MODE_CAPACITY, CUMULATIVE_MODE_REALISTIC}:
        if capacity_total_notional is None or capacity_total_notional <= 0:
            raise ValueError(f"{cumulative_mode} cumulative mode requires capacity_total_notional")
        summary_filename = "capacity_acceptance_daily_summary.csv"
        source_label = "capacity"
        mode_baseline_run_id = capacity_baseline_run_id
        mode_run_ids = capacity_run_ids or {}
        if cumulative_mode == CUMULATIVE_MODE_REALISTIC:
            summary_filename = "realistic_acceptance_daily_summary.csv"
            source_label = "realistic"
            mode_baseline_run_id = realistic_baseline_run_id
            mode_run_ids = realistic_run_ids or {}
        if not mode_baseline_run_id:
            raise ValueError(f"{cumulative_mode} cumulative mode requires baseline run id")
        missing_capacity = sorted(
            direction.key for direction in plot_directions if direction.key not in mode_run_ids
        )
        if missing_capacity:
            raise ValueError(
                f"missing {cumulative_mode} run ids for directions: {missing_capacity}"
            )
        capacity_directions = (
            DirectionSpec(
                key="baseline_pool_l",
                label=baseline_label,
                run_id=mode_baseline_run_id,
            ),
            *(
                DirectionSpec(
                    key=direction.key,
                    label=direction.label,
                    run_id=mode_run_ids[direction.key],
                )
                for direction in plot_directions
            ),
        )
        capacity_cumulative_data = load_capacity_cumulative_plot_data(
            backtests_root=backtests_root,
            capacity_directions=capacity_directions,
            pool=pool,
            capacity_total_notional=capacity_total_notional,
            summary_filename=summary_filename,
            source_label=source_label,
        )
        realized_source = load_realized_cumulative_plot_data(
            backtests_root=backtests_root,
            directions=(),
            pool=pool,
            include_baseline_pool=True,
            include_baseline_universe=True,
            baseline_run_id=baseline_run_id,
            baseline_label=baseline_label,
            fee_bps=realized_fee_bps,
            pool_turnover_path=pool_turnover_path,
            pool_fee_mode=pool_fee_mode,
        )
        capacity_cumulative_data = _replace_capacity_pool_source(
            capacity_cumulative_data,
            realized_source,
        )
        market_source = realized_source.loc[
            realized_source["pool"].astype(str).eq("baseline_universe")
        ].copy()
        realized_cumulative_data = pd.concat(
            [
                capacity_cumulative_data,
                _attach_capacity_fraction_to_market_source(
                    market_source,
                    capacity_cumulative_data,
                ),
            ],
            ignore_index=True,
        )
    else:
        realized_cumulative_data = load_realized_cumulative_plot_data(
            backtests_root=backtests_root,
            directions=plot_directions,
            pool=pool,
            include_baseline_pool=include_baseline_pool_cumulative,
            include_baseline_universe=True,
            baseline_run_id=baseline_run_id,
            baseline_label=baseline_label,
            fee_bps=realized_fee_bps,
            pool_turnover_path=pool_turnover_path,
            pool_fee_mode=pool_fee_mode,
        )
    realized_cumulative_output = apply_display_labels(
        realized_cumulative_data.drop(
            columns=[
                "pool_short_mean_bps",
                "selected_short_mean_bps",
                "short_internal_excess_bps",
                "short_net_return_bps",
                "short_cumulative_net_return_bps",
            ],
            errors="ignore",
        )
    )
    overlay_acceptance_data = combine_overlay_acceptance_data(
        short_universe_data,
        next_pool_data,
    )
    net_alpha_cumulative_data = combine_net_alpha_cumulative_data(
        realized_cumulative_output,
    )
    net_alpha_cumulative_data = add_cumulative_baseline_relative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
        comparison_keys=series,
    )
    net_alpha_cumulative_data = add_background_cumulative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
    )
    net_alpha_cumulative_data = add_market_cumulative_data(net_alpha_cumulative_data)
    net_alpha_cumulative_data = add_cumulative_market_relative_data(
        net_alpha_cumulative_data,
        market_key="market",
        comparison_keys=alpha_cumulative_series,
    )

    overlay_acceptance_csv = output_dir / "optimization_directions_overlay_acceptance_plot_data.csv"
    overlay_acceptance_svg = output_dir / "optimization_directions_overlay_acceptance.svg"
    net_alpha_cumulative_csv = (
        output_dir / "optimization_directions_net_alpha_cumulative_plot_data.csv"
    )
    net_alpha_cumulative_svg = output_dir / "optimization_directions_net_alpha_cumulative.svg"
    trace_path = output_dir / "optimization_directions_trace.json"
    top_n_label = f"Top{top_n}"
    next_excess_panel_title = (
        f"next {pool} excess"
        if cumulative_mode in {CUMULATIVE_MODE_CAPACITY, CUMULATIVE_MODE_REALISTIC}
        else f"next {pool} {top_n_label} excess"
    )
    if cumulative_mode == CUMULATIVE_MODE_CAPACITY:
        cumulative_capacity_definition = (
            "In capacity mode, model lines are read from capacity_acceptance_daily_summary.csv. "
            "That summary is produced by joining capacity_audit_selected.csv allocations with "
            "next-close labels and weighting each stock by allocated_notional. Capacity audit "
            "daily summaries remain pure fill/depth diagnostics and do not carry returns."
        )
        cumulative_absolute_definition = (
            "top panel plots market, pool background, baseline capacity portfolio, "
            "and comparison capacity portfolio cumulative next-close returns. "
            "Pool/model lines subtract their realized fee before applying the daily "
            "capital fraction and cumulative summation; market uses universe "
            "pool_next_mean_bps without a trading fee. Figure axis displays "
            "cumulative bps divided by 100 as percent"
        )
    elif cumulative_mode == CUMULATIVE_MODE_REALISTIC:
        cumulative_capacity_definition = (
            "In realistic mode, model lines are read from realistic_acceptance_daily_summary.csv. "
            "That summary replays capacity-selected child orders with practical constraints "
            "such as non-duplicated rolling turnover, daily symbol caps, optional fill haircuts, "
            "and cash drag for unfilled notional."
        )
        cumulative_absolute_definition = (
            "top panel plots market, pool background, baseline realistic-constrained portfolio, "
            "and comparison realistic-constrained portfolio cumulative next-close returns. "
            "Pool/model lines subtract their realized fee before applying the daily "
            "capital fraction and cumulative summation; market uses universe "
            "pool_next_mean_bps without a trading fee. Figure axis displays "
            "cumulative bps divided by 100 as percent"
        )
    else:
        cumulative_capacity_definition = (
            "In TopN mode, model lines use selected TopN pool-internal summaries."
        )
        cumulative_absolute_definition = (
            "top panel plots market, pool background, baseline selected TopN, and "
            "comparison selected TopN cumulative next-close returns. Pool/model lines "
            "subtract their realized fee before dividing by next_close_capital_divisor "
            "and cumulative summation; market uses universe pool_next_mean_bps without "
            "a trading fee. Figure axis displays cumulative bps divided by 100 as percent"
        )
    capacity_title_label = capacity_label(
        capacity_total_notional=capacity_total_notional,
        capacity_decision_notional=capacity_decision_notional,
    )
    realistic_title_label = realistic_label(capacity_total_notional=capacity_total_notional)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_acceptance_data.to_csv(overlay_acceptance_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        overlay_acceptance_data,
        title=f"{title_prefix} rank IC / pool_L超额",
        panels=[
            {
                "title": "short universe rank IC",
                "ylabel": "Rank IC",
                "column": "short_rank_ic",
                "default_ylim": (0.12, 0.17),
                "tick_step": 0.01,
                "tick_decimals": 3,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": False,
                "target_ticks": 6,
                "min_tick_step": 0.005,
            },
            {
                "title": next_excess_panel_title,
                "ylabel": "bps",
                "column": "next_internal_excess_bps",
                "default_ylim": (0.0, 12.0),
                "tick_step": 2.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 1.0,
            },
        ],
        output_path=overlay_acceptance_svg,
        pools=acceptance_series,
    )

    net_alpha_cumulative_data.to_csv(
        net_alpha_cumulative_csv,
        index=False,
        float_format="%.6f",
    )
    net_alpha_cumulative_plot_data = add_cumulative_percent_display_columns(
        net_alpha_cumulative_data
    )
    if cumulative_mode == CUMULATIVE_MODE_CAPACITY:
        cumulative_subject = f"{capacity_title_label or '容量'}隔夜净收益累和"
    elif cumulative_mode == CUMULATIVE_MODE_REALISTIC:
        cumulative_subject = f"{realistic_title_label}隔夜净收益累和"
    else:
        cumulative_subject = f"池内{top_n_label}隔夜净收益累和"
    if capacity_title_label and cumulative_mode == CUMULATIVE_MODE_TOP100:
        cumulative_subject = f"{cumulative_subject} ({capacity_title_label})"
    cumulative_title = f"{title_prefix} fee {realized_fee_bps:g}bps {cumulative_subject}"
    cumulative_net_values = _panel_values(
        net_alpha_cumulative_plot_data,
        pools=top_cumulative_series,
        column="next_cumulative_net_return_pct",
    )
    market_alpha_values = _panel_values(
        net_alpha_cumulative_plot_data,
        pools=alpha_cumulative_series,
        column="next_cumulative_alpha_vs_market_pct",
    )
    write_two_panel_line_svg(
        net_alpha_cumulative_plot_data,
        title=cumulative_title,
        panels=[
            {
                "title": CUMULATIVE_NET_RETURN_PANEL_TITLE,
                "ylabel": "%",
                "column": "next_cumulative_net_return_pct",
                "default_ylim": line_axis(cumulative_net_values),
                "tick_step": line_step(cumulative_net_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": CUMULATIVE_MARKET_ALPHA_PANEL_TITLE,
                "ylabel": "%",
                "column": "next_cumulative_alpha_vs_market_pct",
                "default_ylim": line_axis(market_alpha_values),
                "tick_step": line_step(market_alpha_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=net_alpha_cumulative_svg,
        pools=top_cumulative_series,
        x_label_mode="years_only",
        line_width=2.1,
        line_marker_count=0,
    )

    trace = {
        "backtests_root": str(backtests_root),
        "output_dir": str(output_dir),
        "pool": pool,
        "cumulative_decision_normalizer": CUMULATIVE_DECISION_NORMALIZER,
        "return_bps_denominator": RETURN_BPS_DENOMINATOR,
        "next_close_capital_divisor": NEXT_CLOSE_CAPITAL_DIVISOR,
        "realized_fee_bps": realized_fee_bps,
        "top_n": top_n,
        "cumulative_mode": cumulative_mode,
        "capacity_total_notional": capacity_total_notional,
        "capacity_decision_notional": capacity_decision_notional,
        "capacity_baseline_run_id": capacity_baseline_run_id,
        "capacity_run_ids": capacity_run_ids or {},
        "realistic_baseline_run_id": realistic_baseline_run_id,
        "realistic_run_ids": realistic_run_ids or {},
        "pool_turnover_path": str(pool_turnover_path) if pool_turnover_path else None,
        "pool_fee_mode": pool_fee_mode,
        "daily_cumulative_semantics": (
            "next-close labels span entry day to next trading day's close, so cumulative "
            "acceptance scales next-close bps by the daily capital fraction before "
            "linear cumulative summation. Without explicit capacity settings this "
            "falls back to 1 / next_close_capital_divisor"
        ),
        "overlay_acceptance": {
            "figure_title": f"{title_prefix} rank IC / pool_L超额",
            "panels": [
                "short universe rank IC",
                next_excess_panel_title,
            ],
            "reason": (
                "short pool excess is omitted because A-share T+1 makes short-horizon "
                "cash PnL non-tradable; next IC is omitted because this model is not "
                "trained to rank next-day returns directly"
            ),
            "baseline_run_id": baseline_run_id,
            "baseline_label": baseline_label,
        },
        "baseline_pool_cumulative": {
            "enabled": include_baseline_pool_cumulative,
            "run_id": baseline_run_id,
            "pool": pool,
            "key": "baseline_pool_l",
            "label": baseline_label,
        },
        "market_cumulative": {
            "enabled": True,
            "run_id": baseline_run_id,
            "pool": "universe",
            "key": "market",
            "definition": (
                "universe pool_next_mean_bps scaled by daily_capital_fraction and "
                "cumulatively summed"
            ),
        },
        "directions": [
            {"key": item.key, "label": item.label, "run_id": item.run_id}
            for item in plot_directions
        ],
        "plotted_series": {
            "overlay_acceptance": ["baseline", *series],
            "cumulative_top": list(top_cumulative_series),
            "cumulative_market_alpha": list(alpha_cumulative_series),
        },
        "figures": {
            "overlay_acceptance": str(overlay_acceptance_svg),
            "net_alpha_cumulative": str(net_alpha_cumulative_svg),
        },
        "plot_data": {
            "overlay_acceptance": str(overlay_acceptance_csv),
            "net_alpha_cumulative": str(net_alpha_cumulative_csv),
        },
        "cumulative_acceptance": {
            "figure_title": cumulative_title,
            "panels": [
                CUMULATIVE_NET_RETURN_PANEL_TITLE,
                CUMULATIVE_MARKET_ALPHA_PANEL_TITLE,
            ],
            "market_series": "full A-share market average overnight return",
            "background_series": "pool_L background overnight return after pool_fee_bps",
            "reason": "short cumulative is omitted because this workflow cannot trade T+0",
            "unit": "%",
            "source_unit": "bps",
            "fee_bps_per_trade": realized_fee_bps,
            "capacity_definition": cumulative_capacity_definition,
            "absolute_definition": cumulative_absolute_definition,
            "background_definition": (
                "pool_L background overnight return minus pool_fee_bps; pool fee uses "
                "equal-weight stock-pool membership turnover when available"
            ),
            "pool_turnover_source": "see pool_turnover_source column in cumulative plot data",
            "market_alpha_definition": (
                "bottom panel plots pool/background and model capital-adjusted cumulative "
                "net bps minus full-market capital-adjusted cumulative bps, displayed as "
                "percent"
            ),
            "accumulation_definition": (
                "capital-adjusted cumulative net bps = cumsum(daily_net_bps * "
                "daily_capital_fraction)"
            ),
        },
        "source_files": source_files(backtests_root, plot_directions),
    }
    trace["baseline_universe_cumulative"] = {
        "enabled": True,
        "requested_by_cli": include_baseline_universe_cumulative,
        "run_id": baseline_run_id,
        "pool": "universe",
        "key": "baseline_universe",
        "used_as": "market source",
        "panels": ["next"],
    }
    write_json(trace_path, trace, ensure_ascii=True)

    return {
        "overlay_acceptance_plot_data": str(overlay_acceptance_csv),
        "overlay_acceptance_figure": str(overlay_acceptance_svg),
        "net_alpha_cumulative_plot_data": str(net_alpha_cumulative_csv),
        "net_alpha_cumulative_figure": str(net_alpha_cumulative_svg),
        "trace": str(trace_path),
    }
