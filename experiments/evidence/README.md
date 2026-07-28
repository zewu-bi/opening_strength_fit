# Evidence

This directory is the tracked, reviewable evidence layer. It contains only compact outputs that
support a recorded decision: aggregate metrics, robustness summaries, success markers, and traces
that identify the source run and inputs.

Current canonical evidence:

- [promoted multiden four-figure acceptance](backtests/nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/)
- [promoted multiden unified strategy acceptance](backtests/strategy_acceptance_clock6_v4_multiden_2022_2025_v1/)
- [control unified strategy acceptance](backtests/strategy_acceptance_clock6_v4_control_2022_2025_v1/) — ablation baseline only

Current diagnostic evidence:

- [all-A 1m TCN limit-up continuation attribution](backtests/temporal_nn_36m_2022_2025_all_a_rank_1m_tcn_mse_v1/) —
  late-day/holding diagnostic only; not a tradable close-entry or general opening-strength result

The tracked tail, bootstrap, overlap, and concentration outputs are diagnostics. In particular,
the one-sided P95/P99 upper-tail cap is not an automatic promotion gate.

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

Refresh the canonical presentation bundle from an existing local result mirror with
`make evidence-four-figures`.
