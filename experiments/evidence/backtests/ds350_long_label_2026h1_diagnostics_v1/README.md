# ds350 long-label and strict-2026H1 diagnostic archive

This compact archive closes the unarchived ds350 investigation chain through 2026-08-11. It
contains original summaries/traces, compact holdout metrics, and Kubernetes logs. Large row-level
predictions, labels, and model binaries remain on the PVC. Failed/superseded attempts are retained
as provenance: `top100_capacity_v1` has a turnover-unit bug and is superseded by v2; limit-nextclose
v3 is the corrected contribution decomposition; failed Top1000 v1/v2 jobs are visible in the job
inventory and v3 is the completed result.

## Strict 2026H1 dataset boundary

The strict holdout uses ClickHouse `stock.tick` plus `stock.daily_bar_jy`, trains on 2023-2025,
purges one session, and evaluates 110 days / 1,100 decision groups in 2026H1. The generated 2026
feature and label schemas match 2025, key order matches, and duplicate feature keys are zero.
There are 5,856,610 rows over 113 tick dates.
Three missing tick dates and their predecessor dates are explicitly recorded in the dataset audit.
Pool L ends at 2025-12-31, so no strict 2026H1 Pool-L result exists; 2026H1 tables are all-A.

## Strict 2026H1 model comparison

| model | own_label_rank_ic | same_day_close_rank_ic | same_day_close_excess_bps | selected_limit_share_pct | limit_contribution_bps | nonlimit_contribution_bps | no_limit_reselect_excess_bps | close_to_next_open_excess_bps | next_open_to_next_close_excess_bps | close_to_next_close_excess_bps | entry_return_median_pct | entry_room_to_limit_median_pct | entry_ask1_notional_p10_cny | entry_ask1_notional_median_cny |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1m_all_a | 0.1078 | 0.0226 | 33.6708 | 10.2273 | 47.8439 | -14.1731 | -13.2718 | -34.2403 | 34.2258 | -1.0481 | 5.5070 | 4.8890 | 8325.8001 | 118599.9966 |
| close_all_a | 0.0116 | 0.0116 | 36.5139 | 6.6655 | 36.6335 | -0.1196 | 1.8477 | -40.7450 | 33.2844 | -8.5662 | 4.2148 | 6.1758 | 9081.5999 | 119050.9966 |

The close model's 36.51 bps is a different sample from the historical rolling-OOS max-10 24.87
bps and max-30 20.85 bps. Its 36.63 bps limit contribution and -0.12 bps non-limit contribution
show that the paper result is almost entirely a final-limit tail effect. The causal-selection,
purge-one-session and missing-return-as-zero audits in `raw/future_info/` do not make the historical
signal disappear, but retrained scores are not bit-identical to the original chain; the trace files
preserve that qualification.

## Loss-only comparison

| loss | rank_ic | same_day_close_excess_bps | selected_limit_share_pct | limit_contribution_bps | nonlimit_contribution_bps | no_limit_reselect_excess_bps | close_to_next_close_excess_bps | top100_overlap_with_mse_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mse | 0.0116 | 36.5139 | 6.6655 | 36.6335 | -0.1196 | 1.8477 | -8.5662 | 100.0000 |
| huber | 0.0251 | 28.0456 | 6.9382 | 29.2647 | -1.2191 | -0.0858 | 5.4760 | 32.7673 |
| huber80_mse20 | 0.0234 | 32.4658 | 6.5673 | 32.5688 | -0.1030 | 1.1030 | -2.3274 | 40.2700 |

Huber raises rank IC and reduces same-day tail dependence, but it does not establish a deployable
replacement by itself. The data/features/split/architecture/seed are held fixed in this comparison.

## Capacity check (corrected v2 units)

| model | ask1_notional_p10_cny | ask1_notional_median_cny | ask10_notional_median_cny | fixed_top100_fill_ask1_plus_turnover_pct | fixed_top100_fill_ask10_plus_turnover_pct | old_turnover_only_fill_pct | eventual_limit_fill_share_ask1_pct | eventual_limit_fill_share_ask10_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1m | 8612.76 | 106433.50 | 2992773.26 | 17.22 | 82.68 | 95.89 | 11.22 | 10.63 |
| close | 8358.00 | 94074.40 | 2329687.98 | 14.60 | 74.32 | 90.74 | 7.71 | 7.17 |

The earlier turnover-only convention substantially overstates capacity. With no rank-101 refill,
25% displayed-depth participation, a 500k CNY/name cap and turnover participation, mean fixed-Top100
fill is only about 17.22%/14.60% on ask1 for the 1m/close models. This is an execution-risk result,
not evidence that the causal label or OOS split is invalid.

## Contents

| section | files | bytes | archive_path |
| --- | --- | --- | --- |
| audits | 37 | 167505 | raw/audits |
| future_info | 21 | 405379 | raw/future_info |
| holdout_models | 20 | 122140 | raw/holdout_models |
| k8s_logs | 24 | 362893 | raw/k8s_logs |
| max30_label_analyses | 20 | 21842 | raw/max30_label_analyses |

`strict_2026h1_*` files are presentation tables derived directly from the archived raw JSON.
`k8s_job_inventory.csv` records the cluster state observed at archive time. `manifest.json` hashes
every retained file.
