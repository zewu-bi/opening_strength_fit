from __future__ import annotations

import argparse

from opening_strength_fit.config import config_int, config_list


def feature_filters_from_config(config: dict) -> dict[str, tuple[str, ...]]:
    drop_columns = tuple(config_list(config, "features", "drop_feature_columns", []))
    model_only_drop_columns = tuple(
        config_list(config, "features", "exclude_model_feature_columns", [])
    )
    return {
        "include_columns": tuple(config_list(config, "features", "include_feature_columns", [])),
        "include_prefixes": tuple(config_list(config, "features", "include_feature_prefixes", [])),
        "include_patterns": tuple(config_list(config, "features", "include_feature_regexes", [])),
        "drop_columns": tuple(dict.fromkeys((*drop_columns, *model_only_drop_columns))),
        "drop_prefixes": tuple(config_list(config, "features", "drop_feature_prefixes", [])),
        "drop_patterns": tuple(config_list(config, "features", "drop_feature_regexes", [])),
    }


def feature_limit(args: argparse.Namespace, config: dict) -> int | None:
    raw = (
        args.feature_limit
        if args.feature_limit is not None
        else config_int(config, "data", "feature_limit", 0)
    )
    return raw if raw and raw > 0 else None
