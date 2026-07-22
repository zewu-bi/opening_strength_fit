# Evidence

This directory is the tracked, reviewable evidence layer. It contains only compact outputs that
support a recorded decision: aggregate metrics, robustness summaries, success markers, and traces
that identify the source run and inputs.

The boundary is deliberate:

- tracked here: summaries, plots, trace JSON, and small audit tables;
- ignored in `experiments/results/`: local mirrors and exploratory outputs;
- external/PVC only: labeled caches, predictions, row-level replay tables, model binaries, and other
  large or reconstructable artifacts.

Create or refresh evidence with:

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --artifacts --record
```

`--record` selects a bounded file set; it does not copy row-level strategy outputs. Every evidence
directory must map to a tracked run TOML and, when executed on Kubernetes, its tracked Job manifest.
