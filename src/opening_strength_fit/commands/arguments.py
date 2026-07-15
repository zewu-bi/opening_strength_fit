from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_int,
    config_list,
    config_str,
)


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
        value = getattr(self.args, name)
        if value:
            return list(value)
        return config_list(self.config, self.section, config_name or name, tuple(default))

    def tuple(self, name: str, default: Iterable[str] = ()) -> tuple[str, ...]:
        value = getattr(self.args, name)
        if value:
            return tuple(value)
        return tuple(config_list(self.config, self.section, name, tuple(default)))

    def aliased_tuple(self, name: str, config_name: str) -> tuple[str, ...]:
        value = getattr(self.args, name)
        if value:
            return tuple(str(item) for item in value)
        configured = config_list(self.config, self.section, config_name, ())
        if configured:
            return tuple(configured)
        return tuple(config_list(self.config, self.section, name, ()))

    def string(
        self,
        name: str,
        default: str = "",
        *,
        config_name: str | None = None,
    ) -> str:
        value = getattr(self.args, name)
        if value not in (None, ""):
            return str(value)
        return config_str(self.config, self.section, config_name or name, default)

    def float(self, name: str, default: float) -> float:
        value = getattr(self.args, name)
        if value is not None:
            return float(value)
        return config_float(self.config, self.section, name, default)

    def integer(self, name: str, default: int) -> int:
        value = getattr(self.args, name)
        if value is not None:
            return int(value)
        return config_int(self.config, self.section, name, default)

    def flag(self, name: str, default: bool = False) -> bool:
        return bool(getattr(self.args, name)) or config_bool(
            self.config,
            self.section,
            name,
            default,
        )
