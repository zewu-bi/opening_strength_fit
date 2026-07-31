# Canonical opening baseline

The machine-readable source of truth is [opening.toml](opening.toml).

Current short names:

| Name | Meaning |
| --- | --- |
| `opening_base` | Base labeled cache built from the last state already visible at each decision clock |
| `opening_cache` | Mixed-w030 training cache consumed by the baseline model |
| `opening_model` | Current 09:31-09:40 signal baseline |

The long `source_run_id`, schema, PVC cache directory, and model directory in the registry are
immutable provenance. They identify the completed jobs and existing external artifacts; they are
not names to copy into new files or tasks.

New files and tasks use `opening_<window>_<semantic_change>`. Omit parameters already fixed by the
baseline and do not add generic `v1`, `v2`, or `v6` suffixes. Examples:

```text
opening_cache_1001
opening_model_1001
opening_model_longhold
opening_model_no_auction
```

When a second run would collide with an existing semantic name, add the actual changed concept or
the experiment date. Do not use an uninformative numeric version.
