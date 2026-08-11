# ds350 historical max-10 archive

This bundle preserves the historical `nn_ds350_label12_36m_grouped_gated_v2_mse_v1` result that produced the reported
`09:31-09:40 / close` quarter-equal values **short IC 0.03593, short excess 24.87 bps,
overnight excess 25.66 bps**. It is a completed 15-case × 8-fold rolling-OOS experiment,
but its model-selection budget is superseded by the separately archived max-30 run.

The distinction matters: 24.87 is not the max-30 result (20.85 bps under the later limit audit),
and neither number should be mixed with the 2026H1 strict holdout result.

## Historical report table (quarter-equal)

| window | label | short_ic | short_excess_bps | overnight_excess_bps |
| --- | --- | --- | --- | --- |
| 09:31-09:40 | 1m | 0.16055 | 11.89653 | 18.56767 |
| 09:31-09:40 | 3m | 0.10758 | 12.73472 | 20.33997 |
| 09:31-09:40 | 10m | 0.06597 | 14.98908 | 21.50789 |
| 09:31-09:40 | 1h | 0.03410 | 18.10618 | 22.94653 |
| 09:31-09:40 | close | 0.03593 | 24.86658 | 25.66347 |
| 10:01-10:10 | 1m | 0.26179 | 8.50641 | 7.64430 |
| 10:01-10:10 | 3m | 0.18936 | 8.56334 | 10.41555 |
| 10:01-10:10 | 10m | 0.11559 | 9.00646 | 11.85065 |
| 10:01-10:10 | 1h | 0.04771 | 9.98701 | 12.44702 |
| 10:01-10:10 | close | 0.04694 | 17.45203 | 13.84597 |

`matrix_group_pooled.csv` and `matrix_quarter_equal.csv` retain both aggregation conventions.
`max10_vs_max30_quarter_equal.csv` is an explicit budget comparison. Fold metrics, year metrics,
half-year/quarter summaries and all 15 analysis traces are included; row-level predictions and
model binaries remain on the PVC.
