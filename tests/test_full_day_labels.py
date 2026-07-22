import json
from argparse import Namespace

import numpy as np
import pandas as pd

from opening_strength_fit.commands.full_day_label_cache import run
from opening_strength_fit.dataset import build_labeled_feature_frame
from opening_strength_fit.full_day_labels import build_full_day_temporal_labels


def _ticks(times: list[str], *, date: str = "2025-01-02") -> pd.DataFrame:
    rows = []
    for index, clock in enumerate(times):
        price = 10.0 + index * 0.01
        volume = 1_000.0 + index * 100.0
        rows.append(
            {
                "date": date,
                "symbol": "000001.SZ",
                "timestamp": pd.Timestamp(f"{date} {clock}"),
                "ask_price_1": price,
                "ask_volume_1": 1_000.0,
                "bid_price_1": price - 0.01,
                "bid_volume_1": 1_000.0,
                "volume": volume,
                "turnover": volume * price,
                "status": "TRADE",
            }
        )
    return pd.DataFrame(rows)


def test_one_minute_horizon_reproduces_fixed_clock_v4() -> None:
    ticks = _ticks(
        [
            "09:30:58",
            "09:31:02",
            "09:31:06",
            "09:31:09",
            "09:32:07",
            "09:32:10",
            "09:33:07",
            "09:33:10",
        ]
    )
    legacy = build_labeled_feature_frame(
        ticks,
        hold_seconds=60,
        sell_window_seconds=60,
        entry_tick_delay=2,
        entry_alignment="clock_state",
        entry_clock_delay_seconds=6,
        future_alignment="clock_state",
        include_preopen=False,
        sample_mode="decision_points",
        decision_times=["09:31:00"],
        decision_max_lag_seconds=5,
        require_cross_section_ready_entry=True,
        tradable_statuses=["TRADE"],
    ).iloc[0]
    temporal = build_full_day_temporal_labels(
        ticks,
        decision_times=["09:31:00"],
        horizons=["1m"],
        include_preopen=False,
        tradable_statuses=["TRADE"],
    ).iloc[0]

    assert temporal["entry_timestamp"] == legacy["entry_timestamp"]
    assert temporal["entry_source_timestamp"] == legacy["entry_source_timestamp"]
    assert temporal["sell_start_target_timestamp_1m"] == legacy["sell_start_target_timestamp"]
    assert temporal["sell_start_source_timestamp_1m"] == legacy["sell_start_source_timestamp"]
    assert temporal["sell_end_target_timestamp_1m"] == legacy["sell_end_target_timestamp"]
    assert temporal["sell_end_source_timestamp_1m"] == legacy["sell_end_source_timestamp"]
    assert temporal["buy_price"] == legacy["buy_price"]
    assert np.isclose(temporal["sell_vwap_1m"], legacy["sell_vwap"])
    assert np.isclose(temporal["alpha_return_1m"], legacy["label"])
    assert bool(temporal["valid_alpha_return_1m"]) == bool(legacy["valid_label"])


def test_lunch_crossing_horizon_is_causal_and_uses_trading_time() -> None:
    ticks = _ticks(
        [
            "11:29:03",
            "11:29:08",
            "11:29:09",
            "13:04:08",
            "13:04:10",
            "13:05:08",
            "13:05:10",
        ]
    )
    out = build_full_day_temporal_labels(
        ticks,
        decision_times=["11:29:00"],
        horizons=["5m"],
        include_preopen=False,
        tradable_statuses=["TRADE"],
    ).iloc[0]

    assert out["entry_timestamp"] == pd.Timestamp("2025-01-02 11:29:09")
    assert out["sell_start_target_timestamp_5m"] == pd.Timestamp("2025-01-02 13:04:09")
    assert out["sell_end_target_timestamp_5m"] == pd.Timestamp("2025-01-02 13:05:09")
    for source, target in (
        ("entry_source_timestamp", "entry_timestamp"),
        ("sell_start_source_timestamp_5m", "sell_start_target_timestamp_5m"),
        ("sell_end_source_timestamp_5m", "sell_end_target_timestamp_5m"),
    ):
        assert out[source] <= out[target]
    assert bool(out["valid_alpha_return_5m"])


def test_horizon_past_market_close_is_explicitly_invalid() -> None:
    ticks = _ticks(["14:59:00", "14:59:06", "15:00:00"])
    out = build_full_day_temporal_labels(
        ticks,
        decision_times=["14:59:00"],
        horizons=["5m", "30m"],
        include_preopen=False,
        tradable_statuses=["TRADE"],
    ).iloc[0]

    assert pd.isna(out["sell_start_target_timestamp_5m"])
    assert pd.isna(out["sell_start_target_timestamp_30m"])
    assert not bool(out["valid_alpha_return_5m"])
    assert not bool(out["valid_alpha_return_30m"])


def test_full_day_cache_writes_resumable_daily_shards(tmp_path) -> None:
    tick_path = tmp_path / "ticks.parquet"
    _ticks(
        [
            "09:30:58",
            "09:31:02",
            "09:31:08",
            "09:32:08",
            "09:33:08",
        ]
    ).to_parquet(tick_path, index=False)
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "artifacts"
    config = {
        "data": {"source": "path"},
        "cache": {"path": str(cache_root), "schema_version": "test_v1"},
        "full_day_labels": {
            "enabled": True,
            "decision_ranges": ["09:31:00"],
            "horizons": ["1m"],
        },
        "labels": {
            "entry_clock_delay_seconds": 6,
            "entry_tick_delay": 2,
            "sell_window_seconds": 60,
            "require_entry_after_cross_section_ready": True,
        },
        "features": {"include_preopen": False},
        "filters": {"tradable_statuses": ["TRADE"]},
    }
    args = Namespace(input=str(tick_path))

    run(args, config, run_name="test_full_day", output_dir=output_dir)
    run(args, config, run_name="test_full_day", output_dir=output_dir)

    shard = cache_root / "date=2025-01-02" / "labels.parquet"
    manifest = json.loads(
        (cache_root / "full_day_label_cache_manifest.json").read_text(encoding="utf-8")
    )
    assert shard.exists()
    assert (cache_root / "_SUCCESS").exists()
    assert manifest["total_rows"] == 1
    assert manifest["causal_timestamp_violations"] == 0
    assert manifest["days"][0]["action"] == "resumed"
