# Evidence

This directory is the tracked, reviewable evidence layer. It contains only compact outputs that
support a recorded decision: aggregate metrics, robustness summaries, success markers, and traces
that identify the source run and inputs.

Current canonical evidence:

- [opening_model baseline](baselines/opening_model/) — current signal/model baseline and short-name entry point
- [opening_model source acceptance bundle](backtests/nn_delay6_v6_decision_clock_state_36m_2022_2025_w0931_0940_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/) —
  immutable source-run archive with four figures, compact CSV, traces, and checksums

Current diagnostic evidence:

- [ds350 historical max-10 15-label matrix](backtests/nn_ds350_label12_36m_grouped_gated_v2_mse_v1/) —
  completed 120-fold historical budget archive and the source of the quarter-equal
  `09:31-09:40 / close = 24.87 bps` result; retained as a superseded comparison only
- [ds350 max-30 15-label matrix](backtests/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/) —
  authoritative 15-label result archive; includes all 120 rolling folds plus the standard four-figure
  acceptance bundle for `09:31-09:40 / 1m`
- [ds350 long-label and strict-2026H1 diagnostics](backtests/ds350_long_label_2026h1_diagnostics_v1/) —
  compact closeout of future-information, limit-tail, Top100 overlap, return-path, strict holdout,
  loss-function, Pool-L availability, and corrected capacity audits, with cluster logs and checksums
- [archived v4 multiden four-figure acceptance](backtests/nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/) —
  previous signal baseline
- [archived v4 multiden unified strategy acceptance](backtests/strategy_acceptance_clock6_v4_multiden_2022_2025_v1/) —
  current downstream capacity/refill reference until rerun on `opening_model`
- [v4 control unified strategy acceptance](backtests/strategy_acceptance_clock6_v4_control_2022_2025_v1/) —
  ablation baseline only
- [opening-window point-in-time limit-up audit](backtests/opening_limit_audit_clock6_v4_multiden_2022_2025_v1/) —
  separates stocks already sealed at decision/entry from stocks predicted to reach the limit later
- [10:01-10:10 intraday-window decay checkpoint](backtests/nn_delay6_clock_state_36m_2022_2025_w1001_1010_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/) —
  first same-model, same-target rolling-OOS measurement after moving the ten-minute window later
- [corrected-label 3×2 window/horizon grid](backtests/corrected_label_3x2_grid_2022_2025_v1/) —
  six-run closeout table and controlled window/horizon effects

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

Refresh the `opening_model` source presentation bundle from an existing local result mirror with
`make evidence-four-figures`. Current short names and their immutable source paths are in
[`experiments/canonical/opening.toml`](../canonical/opening.toml).
