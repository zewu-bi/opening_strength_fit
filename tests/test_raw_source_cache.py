from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from opening_strength_fit.commands.raw_source_cache import (
    CALENDAR_COLUMNS,
    CLOSE_REFERENCE_COLUMNS,
    DAILY_REFERENCE_COLUMNS,
    TICK_COLUMNS,
    _parse_windows,
    close_reference_sql,
    label_coverage,
    stream_parquet_atomic,
    tick_source_sql,
)
from opening_strength_fit.config import load_toml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Stream:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return iter((self.payload[:31], self.payload[31:]))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class _RawStreamClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object]]] = []

    def raw_stream(self, query: str, *, parameters: dict[str, object]):
        self.calls.append((query, parameters))
        return _Stream(self.payload)


def _parquet_bytes(columns: tuple[str, ...]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({column: [1] for column in columns}), sink)
    return sink.getvalue().to_pybytes()


def test_tick_contract_contains_only_raw_feature_and_label_inputs() -> None:
    assert TICK_COLUMNS[:10] == (
        "TradingDay",
        "Symbol",
        "ExchTimeOffsetUs",
        "HighPrice",
        "LowPrice",
        "LastPrice",
        "TradeNum",
        "Volume",
        "Turnover",
        "Status",
    )
    assert len(TICK_COLUMNS) == 77
    assert not {
        "LocalTimeStamp",
        "label",
        "target_label",
        "feature",
    }.intersection(TICK_COLUMNS)
    assert DAILY_REFERENCE_COLUMNS == (
        "TradingDay",
        "Symbol",
        "OpenPrice",
        "ClosePrice",
        "PreClosePrice",
        "TradeStatus",
        "STStatus",
        "UpdownLimitStatus",
        "TotalMarketValue",
        "TotalFloatMarketValue",
        "TotalShareToday",
        "FloatAShare",
        "FreeShareToday",
    )
    assert CLOSE_REFERENCE_COLUMNS == (
        "TradingDay",
        "Symbol",
        "ClosePrice",
        "CloseSourceOffsetUs",
    )


def test_tick_sql_projects_contract_and_deduplicates_at_source_boundary() -> None:
    windows = ((33_300_000_000, 33_930_000_000), (35_400_000_000, 36_900_000_000))
    sql = tick_source_sql("stock.tick", windows)

    assert "limit 1 by Symbol, ExchTimeOffsetUs" in sql
    assert "arrayMax(mapValues(LocalTimeStamp)) desc" in sql
    assert "{window_start_0:UInt64}" in sql
    assert "{window_end_1:UInt64}" in sql
    assert "format Parquet" in sql
    assert "select *" not in sql.lower()
    assert "IOPV" in sql
    assert "Withdraw" not in sql


def test_close_reference_is_a_price_observation_not_a_label() -> None:
    sql = close_reference_sql("stock.tick")

    assert "argMax(" in sql
    assert "LastPrice" in sql
    assert "ClosePrice" in sql
    assert "group by TradingDay, Symbol" in sql
    assert "label" not in sql.lower()


def test_stream_parquet_atomic_validates_and_reuses_complete_file(tmp_path: Path) -> None:
    payload = _parquet_bytes(CALENDAR_COLUMNS)
    client = _RawStreamClient(payload)
    output = tmp_path / "calendar.parquet"

    first = stream_parquet_atomic(
        client,
        query="select 1 format Parquet",
        parameters={"year": 2025},
        output_path=output,
        expected_columns=CALENDAR_COLUMNS,
        overwrite=False,
    )
    second = stream_parquet_atomic(
        client,
        query="select 1 format Parquet",
        parameters={"year": 2025},
        output_path=output,
        expected_columns=CALENDAR_COLUMNS,
        overwrite=False,
    )

    assert first == {"rows": 1, "bytes": len(payload), "reused": False}
    assert second == {"rows": 1, "bytes": len(payload), "reused": True}
    assert len(client.calls) == 1
    assert not list(tmp_path.glob("*.partial.*"))


@pytest.mark.parametrize(
    ("config_name", "expected_windows", "expected_root"),
    [
        (
            "opening_0931_0940_raw_source.toml",
            ((33_300_000_000, 35_400_000_000),),
            "opening_0931_0940_raw_source",
        ),
        (
            "opening_1001_1010_raw_source.toml",
            ((33_300_000_000, 33_930_000_000), (35_400_000_000, 37_200_000_000)),
            "opening_1001_1010_raw_source",
        ),
        (
            "opening_1401_1410_raw_source.toml",
            ((33_300_000_000, 33_930_000_000), (49_800_000_000, 51_600_000_000)),
            "opening_1401_1410_raw_source",
        ),
    ],
)
def test_raw_source_configs(
    config_name: str,
    expected_windows: tuple[tuple[int, int], ...],
    expected_root: str,
) -> None:
    config = load_toml(PROJECT_ROOT / "experiments" / "runs" / config_name)

    assert _parse_windows(config) == expected_windows
    assert config["raw_source"]["years"] == list(range(2019, 2026))
    assert config["raw_source"]["output_root"].endswith(expected_root)
    assert config["k8s"]["shard_parallelism"] == 2
    coverage = label_coverage(config)
    assert coverage["short_label_horizons_seconds"] == [60, 180, 300]
    assert coverage["entry_delay_seconds"] == 6
    assert coverage["sell_window_seconds"] == 60
    assert coverage["tail_buffer_seconds"] == 234.0
    assert coverage["next_close_reference"] is True


def test_label_coverage_rejects_a_window_that_cannot_build_5m() -> None:
    config = {
        "raw_source": {
            "tick_windows": [[33_300_000_000, 35_100_000_000]],
            "decision_end_offset_us": 34_800_000_000,
            "entry_delay_seconds": 6,
            "short_label_horizons_seconds": [60, 180, 300],
            "short_label_sell_window_seconds": 60,
        }
    }

    with pytest.raises(SystemExit, match="do not continuously cover"):
        label_coverage(config)
