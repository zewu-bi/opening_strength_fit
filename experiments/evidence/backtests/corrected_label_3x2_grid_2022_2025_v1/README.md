# Corrected-label 3×2 window/horizon grid

This is the compact closeout bundle for six grouped-gated NN rolling-OOS runs: three fixed
ten-minute decision windows crossed with 1m and 3m short-label horizons. All other training and
pool-internal evaluation settings are held fixed.

- [Main table](corrected_label_3x2_summary.md)
- [Fee-adjusted cumulative comparison](02_top100_cumulative_3x2.svg)
- [Machine-readable summary](corrected_label_3x2_summary.csv)
- [Holding-horizon and window deltas](corrected_label_3x2_effects.csv)
- [Source hashes and metric definitions](manifest.json)

The economic winner is `09:31-09:40 / 3m`: short and overnight Top100 internal excess are
`12.9162 / 20.2087 bps`. Later windows have higher universe short Rank IC but sharply weaker
overnight excess, so `10:01-10:10` and `14:01-14:10` stop at signal-layer archival under the
runbook's window-decay rule.

The matching experiment definition is
[corrected_label_3x2_grid_2022_2025_v1.toml](../../../runs/corrected_label_3x2_grid_2022_2025_v1.toml).
