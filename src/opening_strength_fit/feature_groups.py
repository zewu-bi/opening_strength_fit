from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()

    def matches(self, feature: str) -> bool:
        if self.exclude_prefixes and feature.startswith(self.exclude_prefixes):
            return False
        if feature in self.columns:
            return True
        if self.prefixes and feature.startswith(self.prefixes):
            return True
        return bool(self.contains and any(token in feature for token in self.contains))


DEFAULT_FEATURE_GROUPS = (
    FeatureGroup("preopen", prefixes=("preopen_",)),
    FeatureGroup("postopen_v2", prefixes=("postopen_v2_",)),
    FeatureGroup(
        "postopen_v1",
        prefixes=("postopen_",),
        exclude_prefixes=("postopen_v2_",),
    ),
    FeatureGroup("raw_cumulative_trade", columns=("volume", "turnover")),
    FeatureGroup(
        "trade_flow",
        prefixes=("volume_diff_", "turnover_diff_", "trade_vwap_"),
    ),
    FeatureGroup(
        "orderbook_depth",
        prefixes=(
            "ask_volume_",
            "bid_volume_",
            "ask_depth_",
            "bid_depth_",
            "depth_imbalance_",
            "ask_gap_",
            "bid_gap_",
        ),
        columns=(
            "ask_price_1",
            "bid_price_1",
            "mid_price",
            "spread_abs",
            "spread_bps",
            "ask1_to_limit_up_bps",
        ),
    ),
    FeatureGroup(
        "momentum",
        prefixes=("return_",),
        columns=("return_vs_prev_close", "return_vs_open"),
    ),
    FeatureGroup(
        "historical_surprise",
        prefixes=("hist_surprise_", "norm_hist_surprise_", "xs_rel_hist_surprise_"),
    ),
    FeatureGroup(
        "path_shape",
        prefixes=("path_shape_", "norm_path_shape_", "xs_rel_path_shape_"),
    ),
    FeatureGroup(
        "price_scale",
        prefixes=("price_scale_", "norm_price_scale_", "xs_rel_price_scale_"),
    ),
    FeatureGroup("cross_sectional_relative", prefixes=("xs_rel_", "norm_")),
)


def _tuple_config(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace(",", " ").split() if part)
    return tuple(str(item) for item in value if str(item))


def feature_groups(config: dict) -> list[FeatureGroup]:
    groups = {group.name: group for group in DEFAULT_FEATURE_GROUPS}
    custom_groups = config.get("feature_audit", {}).get("groups", {})
    if isinstance(custom_groups, dict):
        for name, spec in custom_groups.items():
            if not isinstance(spec, dict):
                continue
            groups[str(name)] = FeatureGroup(
                name=str(name),
                columns=_tuple_config(spec.get("columns")),
                prefixes=_tuple_config(spec.get("prefixes")),
                contains=_tuple_config(spec.get("contains")),
                exclude_prefixes=_tuple_config(spec.get("exclude_prefixes")),
            )

    enabled = _tuple_config(config.get("feature_audit", {}).get("enabled_groups"))
    ordered = list(groups.values())
    if enabled:
        enabled_set = set(enabled)
        ordered = [group for group in ordered if group.name in enabled_set]
    return ordered


def matching_features(features: list[str], group: FeatureGroup) -> list[str]:
    return [feature for feature in features if group.matches(feature)]


def feature_group_name(feature: str, groups: list[FeatureGroup]) -> str:
    for group in groups:
        if group.matches(feature):
            return group.name
    return "other"
