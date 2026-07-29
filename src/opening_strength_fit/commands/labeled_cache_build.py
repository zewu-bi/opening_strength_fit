from __future__ import annotations

import os
from pathlib import Path

from opening_strength_fit.cache_manifest import (
    cache_manifest_path,
    publish_cache_manifest,
    validate_cache_manifest,
    write_cache_manifest,
)
from opening_strength_fit.config import config_list, config_str, config_value, load_toml, run_id
from opening_strength_fit.io import read_frame, write_json
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.training_args import build_training_parser
from opening_strength_fit.training_data import (
    load_clickhouse_labeled_frame,
    resolve_cache_path,
)


def _apply_cache_year_shard(config: dict, *, index: int | None = None) -> int | None:
    years = [int(value) for value in config_list(config, "cache_shards", "years", [])]
    if not years:
        return None

    if index is None:
        raw_index = os.environ.get("JOB_COMPLETION_INDEX", "").strip()
        if not raw_index:
            raise SystemExit(
                "[cache_shards].years requires JOB_COMPLETION_INDEX or an explicit shard index"
            )
        try:
            index = int(raw_index)
        except ValueError as error:
            raise SystemExit(f"invalid JOB_COMPLETION_INDEX={raw_index!r}") from error
    if index < 0 or index >= len(years):
        raise SystemExit(
            f"cache shard index {index} is outside configured range 0..{len(years) - 1}"
        )

    year = years[index]
    path_template = config_str(config, "cache_shards", "cache_path_template", "").strip()
    if not path_template:
        raise SystemExit("[cache_shards].cache_path_template is required")
    reuse_template = config_str(
        config,
        "cache_shards",
        "reuse_labeled_path_template",
        "",
    ).strip()

    config.setdefault("data", {})["start_date"] = f"{year}-01-01"
    config["data"]["end_date"] = f"{year}-12-31"
    config.setdefault("cache", {})["path"] = path_template.format(year=year)
    if reuse_template:
        config.setdefault("features", {})["reuse_labeled_path"] = reuse_template.format(year=year)
    base_run_id = str(config.setdefault("run", {}).get("id", "labeled_cache"))
    config["run"]["id"] = f"{base_run_id}_{year}"
    return year


def main() -> None:
    parser = build_training_parser(
        "Build a labeled opening cache from ClickHouse without training a model."
    )
    args = parser.parse_args()
    config = load_toml(args.config) if args.config else {}
    shard_year = _apply_cache_year_shard(config)
    run_name = run_id(config, args.config) if args.config else "labeled_cache"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = resolve_cache_path(config)
    if cache_path is None:
        raw = config_value(config, "target_cache", "output_path", "")
        if raw:
            cache_path = Path(str(raw))
            config.setdefault("cache", {})["enabled"] = True
            config["cache"]["path"] = str(cache_path)
        else:
            raise SystemExit("missing cache path: set [cache].path or [cache].labeled_path")
    config.setdefault("cache", {})["enabled"] = True
    config["cache"].setdefault("read", True)
    config["cache"].setdefault("write", True)

    labeled = load_clickhouse_labeled_frame(args, config)
    summary = {
        "run_id": run_name,
        "cache_shard_year": shard_year,
        "cache_path": str(cache_path),
        **dataset_summary(labeled),
    }
    print_mapping("labeled_cache_summary", summary)
    manifest_path = cache_manifest_path(cache_path)
    output_manifest_path = output_dir / "labeled_cache_manifest.json"
    manifest = validate_cache_manifest(cache_path, config, required=False)
    if manifest is None:
        manifest = publish_cache_manifest(
            read_frame(cache_path),
            cache_path=cache_path,
            config=config,
            run_name=run_name,
            config_path=args.config or "",
        )
    write_cache_manifest(manifest, output_manifest_path)

    trace = {
        "run_id": run_name,
        "cache_shard_year": shard_year,
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "data": config.get("data", {}),
        "clickhouse": config.get("clickhouse", {}),
        "sample": config.get("sample", {}),
        "labels": config.get("labels", {}),
        "features": config.get("features", {}),
        "summary": summary,
    }
    write_json(output_dir / "labeled_cache_trace.json", trace)
    print(f"\nwrote cache: {cache_path}")
    print(f"manifest: {manifest_path}")
    print(f"trace: {output_dir / 'labeled_cache_trace.json'}")


if __name__ == "__main__":
    main()
