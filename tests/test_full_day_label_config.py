import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def _load(name: str) -> dict:
    with (RUNS / name).open("rb") as file:
        return tomllib.load(file)


def test_full_day_configs_keep_fixed_clock_v4_entry_contract() -> None:
    for name in (
        "build_full_day_clock6_temporal_smoke_20250102_v1.toml",
        "build_full_day_clock6_temporal_2025_v1.toml",
    ):
        config = _load(name)
        assert config["run"]["kind"] == "labeled_cache"
        assert config["full_day_labels"]["enabled"] is True
        assert config["full_day_labels"]["horizons"] == [
            "5m",
            "30m",
            "close",
            "next_close",
        ]
        assert config["labels"]["entry_alignment"] == "clock_state"
        assert config["labels"]["entry_clock_delay_seconds"] == 6
        assert config["labels"]["future_alignment"] == "clock_state"
        assert config["labels"]["require_entry_after_cross_section_ready"] is True
        assert config["clickhouse"]["end_offset_us"] == 54_000_000_000
        assert not Path(config["cache"]["path"]).suffix


def test_narrow_full_day_configs_cover_canonical_years_with_three_labels() -> None:
    names = [f"build_full_day_clock6_narrow_labels_{year}_v1.toml" for year in range(2019, 2026)]
    for year, name in zip(range(2019, 2026), names, strict=True):
        config = _load(name)
        assert config["data"]["start_date"] == f"{year}-01-01"
        assert config["data"]["end_date"] == f"{year}-12-31"
        assert config["data"]["tick_timestamp_deduplication"] == "latest_local_timestamp"
        assert config["full_day_labels"] == {
            "enabled": True,
            "output_mode": "labels_only",
            "decision_ranges": [
                "09:30:00..11:29:00",
                "13:00:00..14:59:00",
            ],
            "sessions": [
                "09:30:00-11:30:00",
                "13:00:00-15:00:00",
            ],
            "horizons": ["1m", "10m", "60m"],
        }
        assert config["labels"]["entry_clock_delay_seconds"] == 6
        assert config["labels"]["sell_window_seconds"] == 60
        assert config["cache"]["path"].endswith(f"/year={year}")
