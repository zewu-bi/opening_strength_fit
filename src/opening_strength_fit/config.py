from __future__ import annotations

from pathlib import Path
import re
import tomllib

from opening_strength_fit.sampling import parse_clock_times


def load_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def config_value(config: dict, section: str, key: str, default):
    value = config.get(section, {}).get(key, default)
    return default if value is None else value


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


def config_bool(config: dict, section: str, key: str, default: bool) -> bool:
    value = config_value(config, section, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def config_list(
    config: dict,
    section: str,
    key: str,
    default: list[str] | tuple[str, ...],
) -> list[str]:
    value = config_value(config, section, key, default)
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    return [part.strip() for part in parts if part and part.strip()]


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


def config_int_tuple(
    config: dict,
    section: str,
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    value = config_value(config, section, key, default)
    if value is None:
        return default
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value)
    parsed = tuple(int(item) for item in raw if str(item).strip())
    return parsed or default


def run_id(config: dict, config_path: str | Path) -> str:
    return str(config_value(config, "run", "id", Path(config_path).stem))


def slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9-]+", "-", value.lower().replace("_", "-"))
    return out.strip("-")[:45]
