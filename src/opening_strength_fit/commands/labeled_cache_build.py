from __future__ import annotations

from pathlib import Path

from opening_strength_fit.cache_manifest import (
    cache_manifest_path,
    publish_cache_manifest,
    validate_cache_manifest,
    write_cache_manifest,
)
from opening_strength_fit.config import config_str, config_value, load_toml, run_id
from opening_strength_fit.io import read_frame, write_json
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.training_args import build_training_parser
from opening_strength_fit.training_data import (
    load_clickhouse_labeled_frame,
    resolve_cache_path,
)


def main() -> None:
    parser = build_training_parser(
        "Build a labeled opening cache from ClickHouse without training a model."
    )
    args = parser.parse_args()
    config = load_toml(args.config) if args.config else {}
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
