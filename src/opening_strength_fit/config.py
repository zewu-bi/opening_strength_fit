from __future__ import annotations

import os
import re
import shlex
import tomllib
from pathlib import Path
from typing import Any, cast

from opening_strength_fit.sampling import parse_clock_times


def load_env_file(path: str | Path = ".env", *, search_parents: bool = False) -> None:
    env_path = Path(path)
    if search_parents and not env_path.is_absolute() and not env_path.exists():
        env_path = next(
            (
                parent / env_path
                for parent in (Path.cwd(), *Path.cwd().parents)
                if (parent / env_path).exists()
            ),
            env_path,
        )
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        parts = shlex.split(line, comments=True)
        if parts and "=" in parts[0]:
            key, value = parts[0].split("=", 1)
            os.environ.setdefault(key, value)


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def config_value[T](config: dict[str, Any], section: str, key: str, default: T) -> T:
    values = config.get(section, {})
    if not isinstance(values, dict):
        return default
    value = values.get(key, default)
    return default if value is None else cast(T, value)


def config_int(config: dict, section: str, key: str, default: int) -> int:
    return int(config_value(config, section, key, default))


def config_optional_int(
    config: dict,
    section: str,
    key: str,
    default: int | None = None,
) -> int | None:
    value = config_value(config, section, key, default)
    return None if value in (None, "") else int(value)


def config_float(config: dict, section: str, key: str, default: float) -> float:
    return float(config_value(config, section, key, default))


def config_str(config: dict, section: str, key: str, default: str) -> str:
    return str(config_value(config, section, key, default))


def prepare_output_dir(config: dict, override: str | None, run_name: str) -> Path:
    path = Path(
        override or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def config_bool(config: dict, section: str, key: str, default: bool) -> bool:
    return coerce_bool(config_value(config, section, key, default))


def coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    parts = value.replace(",", " ").split() if isinstance(value, str) else map(str, value)
    return [part.strip() for part in parts if part and part.strip()]


def config_list(
    config: dict,
    section: str,
    key: str,
    default: list[str] | tuple[str, ...],
) -> list[str]:
    return coerce_str_list(config_value(config, section, key, default))


def config_clock_list(
    config: dict,
    section: str,
    key: str,
    default: list[str] | tuple[str, ...],
) -> list[str]:
    return parse_clock_times(config_value(config, section, key, default))


def config_float_mapping(config: dict, section: str, key: str) -> dict[str, float]:
    value = config_value(config, section, key, {})
    if not value:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"[{section}].{key} must be a table of column = value")
    return {
        str(column): float(threshold)
        for column, threshold in value.items()
        if threshold not in (None, "")
    }


def config_tuple(
    config: dict, section: str, key: str, default: tuple, convert: type = str
) -> tuple:
    value = config_value(config, section, key, default)
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value)
    parsed = tuple(convert(item) for item in raw if str(item).strip())
    return parsed or default


def config_int_tuple(
    config: dict,
    section: str,
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    return config_tuple(config, section, key, default, int)


def config_float_tuple(
    config: dict,
    section: str,
    key: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    return config_tuple(config, section, key, default, float)


def run_id(config: dict, config_path: str | Path) -> str:
    return str(config_value(config, "run", "id", Path(config_path).stem))


def slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9-]+", "-", value.lower().replace("_", "-"))
    return out.strip("-")[:45]
