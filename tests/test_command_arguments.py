from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from opening_strength_fit.commands.arguments import CommandArguments
from opening_strength_fit.commands.capacity_acceptance_analysis import (
    main as capacity_acceptance_main,
)
from opening_strength_fit.commands.realistic_acceptance_analysis import (
    main as realistic_acceptance_main,
)


def test_cli_values_take_precedence_over_config() -> None:
    arguments = CommandArguments(
        argparse.Namespace(
            inputs=["cli.parquet"],
            label_col="cli_label",
            fee_bps=1.5,
            top_n=20,
        ),
        {
            "analysis": {
                "inputs": ["config.parquet"],
                "label_col": "config_label",
                "fee_bps": 2.5,
                "top_n": 100,
            }
        },
        "analysis",
    )

    assert arguments.list("inputs") == ["cli.parquet"]
    assert arguments.tuple("inputs") == ("cli.parquet",)
    assert arguments.string("label_col") == "cli_label"
    assert arguments.float("fee_bps", 0.0) == pytest.approx(1.5)
    assert arguments.integer("top_n", 10) == 20


def test_config_alias_and_legacy_fallback() -> None:
    args = argparse.Namespace(tradable_status=None, exposure_col=None)
    current = CommandArguments(
        args,
        {
            "analysis": {
                "tradable_statuses": ["TRADE", "AUCTION"],
                "tradable_status": ["LEGACY"],
                "exposure_cols": ["market_cap"],
            }
        },
        "analysis",
    )
    legacy = CommandArguments(
        args,
        {"analysis": {"tradable_status": ["TRADE"]}},
        "analysis",
    )
    cli = CommandArguments(
        argparse.Namespace(
            tradable_status=["CLI_TRADE"],
            exposure_col=["cli_exposure"],
            output_dir="cli-output",
        ),
        {
            "analysis": {
                "tradable_statuses": ["CONFIG_TRADE"],
                "exposure_cols": ["config_exposure"],
                "local_dir": "config-output",
            }
        },
        "analysis",
    )

    assert current.aliased_tuple("tradable_status", "tradable_statuses") == (
        "TRADE",
        "AUCTION",
    )
    assert legacy.aliased_tuple("tradable_status", "tradable_statuses") == ("TRADE",)
    assert current.list("exposure_col", config_name="exposure_cols") == ["market_cap"]
    assert cli.aliased_tuple("tradable_status", "tradable_statuses") == ("CLI_TRADE",)
    assert cli.list("exposure_col", config_name="exposure_cols") == ["cli_exposure"]
    assert cli.string("output_dir", config_name="local_dir") == "cli-output"


@pytest.mark.parametrize(
    ("cli_value", "config_value", "expected"),
    [
        (True, False, True),
        (False, "yes", True),
        (False, False, False),
    ],
)
def test_flag_preserves_store_true_semantics(
    cli_value: bool,
    config_value: bool | str,
    expected: bool,
) -> None:
    arguments = CommandArguments(
        argparse.Namespace(include_decision_timestamp=cli_value),
        {"exposure_input": {"include_decision_timestamp": config_value}},
        "exposure_input",
    )

    assert arguments.flag("include_decision_timestamp") is expected


def _write_acceptance_inputs(tmp_path: Path) -> tuple[Path, Path]:
    selected = tmp_path / "selected.csv"
    labels = tmp_path / "labels.csv"
    keys = {
        "date": ["2024-01-02"],
        "symbol": ["000001.SZ"],
        "decision_target_timestamp": ["2024-01-02 09:31:00"],
    }
    pd.DataFrame(
        {
            "pool": ["pool_L"],
            **keys,
            "allocated_notional": [100.0],
            "target_notional": [100.0],
        }
    ).to_csv(selected, index=False)
    pd.DataFrame({**keys, "alpha_return_next_close": [0.01]}).to_csv(labels, index=False)
    return selected, labels


@pytest.mark.parametrize(
    ("main", "section", "artifact"),
    [
        (
            capacity_acceptance_main,
            "capacity_acceptance",
            "capacity_acceptance_trace.json",
        ),
        (
            realistic_acceptance_main,
            "realistic_acceptance",
            "realistic_acceptance_trace.json",
        ),
    ],
)
def test_acceptance_commands_read_output_dir_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    main: Callable[[], None],
    section: str,
    artifact: str,
) -> None:
    selected, labels = _write_acceptance_inputs(tmp_path)
    output_dir = tmp_path / "output"
    config = tmp_path / "run.toml"
    config.write_text(
        "\n".join(
            [
                "[output]",
                f"local_dir = {json.dumps(str(output_dir))}",
                "",
                f"[{section}]",
                f"selected_input = [{json.dumps(str(selected))}]",
                f"label_input = [{json.dumps(str(labels))}]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["acceptance-command", "--config", str(config)])

    main()

    assert (output_dir / artifact).exists()


@pytest.mark.parametrize(
    ("main", "section"),
    [
        (capacity_acceptance_main, "capacity_acceptance"),
        (realistic_acceptance_main, "realistic_acceptance"),
    ],
)
def test_acceptance_commands_require_an_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    main: Callable[[], None],
    section: str,
) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        "\n".join(
            [
                f"[{section}]",
                'selected_input = ["unused-selected.csv"]',
                'label_input = ["unused-labels.csv"]',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["acceptance-command", "--config", str(config)])

    with pytest.raises(SystemExit, match=r"pass --output-dir or set \[output\]\.local_dir"):
        main()
