from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_str,
    config_tuple,
    load_toml,
    run_id,
)


def command_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="")
    return parser


def add_arguments(parser: argparse.ArgumentParser, names: str, **kwargs: Any) -> None:
    for name in names.split():
        parser.add_argument(f"--{name}", **kwargs)


def add_options(parser: argparse.ArgumentParser, **options: dict[str, Any]) -> None:
    for name, kwargs in options.items():
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)


@dataclass(frozen=True, slots=True)
class CommandArguments:
    """Resolve one command's CLI arguments over its TOML section."""

    args: argparse.Namespace
    config: dict[str, Any]
    section: str

    def list(
        self,
        name: str,
        default: Iterable[str] = (),
        *,
        config_name: str | None = None,
    ) -> list[str]:
        value = getattr(self.args, name, None)
        if value:
            return list(value)
        return config_list(self.config, self.section, config_name or name, tuple(default))

    def tuple(self, name: str, default: Iterable[str] = ()) -> tuple[str, ...]:
        return tuple(self.list(name, default))

    def optional_tuple(self, name: str, default: Iterable[str] = ()) -> tuple[str, ...]:
        value = getattr(self.args, name, None)
        return (
            config_tuple(self.config, self.section, name, tuple(default))
            if value is None
            else tuple(value)
        )

    def aliased_tuple(self, name: str, config_name: str) -> tuple[str, ...]:
        values = self.list(name, (), config_name=config_name)
        return tuple(values or config_list(self.config, self.section, name, ()))

    def string(
        self,
        name: str,
        default: str = "",
        *,
        config_name: str | None = None,
    ) -> str:
        value = getattr(self.args, name, None)
        if value not in (None, ""):
            return str(value)
        return config_str(self.config, self.section, config_name or name, default)

    def _number(self, name: str, default: int | float, loader):
        value = getattr(self.args, name, None)
        return (
            type(default)(value)
            if value is not None
            else loader(self.config, self.section, name, default)
        )

    def float(self, name: str, default: float) -> float:
        return self._number(name, default, config_float)

    def integer(self, name: str, default: int) -> int:
        return self._number(name, default, config_int)

    def flag(self, name: str, default: bool = False) -> bool:
        return bool(getattr(self.args, name, None)) or config_bool(
            self.config,
            self.section,
            name,
            default,
        )

    def resolve_dataclass[T](
        self,
        defaults: T,
        *,
        tuple_aliases: dict[str, str] | None = None,
    ) -> T:
        aliases = tuple_aliases or {}
        values = {}
        for field in fields(defaults):
            name = field.name
            default = getattr(defaults, name)
            if isinstance(default, bool):
                value = self.flag(name, default)
            elif isinstance(default, int):
                value = self.integer(name, default)
            elif isinstance(default, float):
                value = self.float(name, default)
            elif isinstance(default, tuple):
                alias = aliases.get(name, name)
                value = (
                    self.aliased_tuple(alias, name) if alias != name else self.tuple(name, default)
                )
            else:
                value = self.string(name, default)
            values[name] = value
        return replace(defaults, **values)


def command_config(args: argparse.Namespace, default_run_name: str) -> tuple[dict[str, Any], str]:
    config = load_toml(args.config) if args.config else {}
    name = getattr(args, "run_id", "") or (
        run_id(config, args.config) if args.config else default_run_name
    )
    return config, name


def required_io_paths(
    args: argparse.Namespace,
    config: dict[str, Any],
    section: str,
    *,
    input_fallback: tuple[str, str] | None = None,
) -> tuple[Path, Path]:
    input_value = args.input or config_str(config, section, "input_path", "")
    if not input_value and input_fallback:
        input_value = config_str(config, *input_fallback, "")
    output_value = args.output or config_str(config, section, "output_path", "")
    for name, value in (("input", input_value), ("output", output_value)):
        if not value:
            raise SystemExit(f"missing {name} path: pass --{name} or [{section}].{name}_path")
    return Path(input_value), Path(output_value)


def command_context(
    args: argparse.Namespace,
    section: str,
    *,
    default_run_name: str = "",
) -> tuple[dict[str, Any], CommandArguments, str]:
    config, name = command_config(args, default_run_name or section)
    arguments = CommandArguments(args, config, section)
    return config, arguments, name


def run_config_year_command(splitter, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    splitter(
        load_toml(config_path),
        config_path,
        year=int(args.year),
        overwrite=bool(args.overwrite),
    )
