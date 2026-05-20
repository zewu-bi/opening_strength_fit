from __future__ import annotations

from pathlib import Path
import re
import tomllib


def load_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def config_value(config: dict, section: str, key: str, default):
    value = config.get(section, {}).get(key, default)
    return default if value is None else value


def run_id(config: dict, config_path: str | Path) -> str:
    return str(config_value(config, "run", "id", Path(config_path).stem))


def slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9-]+", "-", value.lower().replace("_", "-"))
    return out.strip("-")[:45]

