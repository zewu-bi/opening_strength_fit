# Top1000 Bucket Diagnostics

## 2026-07-09 old NN overlay bucket check

Scope: this diagnostic uses the old NN overlay set only: `nn_mlp_base`,
`nn_mlp_base_plus_mse`, and `nn_deep_gelu_mse`. The sample is 2022-2025
`pool_L`, with next-close pool-internal excess as the label. It should not be
read as a `grouped_gated` / `grouped_gated_v2` result.

Headline return numbers use decision-group equal weighting: each
`date x decision_target_timestamp` TopK / bucket portfolio gets one equal
weight. Month-level files are stability checks only, not the headline return
source.

Artifacts:

```text
experiments/results/backtests/local_ic_topk_nn_overlay_pool_l_v1/
experiments/results/backtests/ic_bucket_diagnostics_nn_overlay_pool_l_v1/
experiments/results/backtests/old_nn_multiscale_bucket_diag_v1/
```

Use these files as the canonical readout:

- `old_nn_multiscale_bucket_diag_v1/topk_shape_summary.csv`: Top50/100/150/200
  head-shape acceptance.
- `old_nn_multiscale_bucket_diag_v1/bucket_width_distribution_summary.csv`:
  headline Top1000 bucket returns and right-tail hit rates.
- `old_nn_multiscale_bucket_diag_v1/topk_internal_ic_summary.csv` and
  `bucket_width_within_ic_summary.csv`: pointwise / within-bucket IC checks.
- `*_month_summary.csv` files: month stability only. They are not the headline
  return convention.

Conclusion: the alpha has a reasonable coarse Top1000 bucket shape, but weak
fine ranking inside buckets. The old-three-model mean Top1000 bucket excess is
`12.71, 6.18, 4.08, 2.98, 2.36, 1.76, 1.58, 1.56, 1.37, 1.31 bps`, so score
buckets are ordered in the right direction. However, bucket medians are negative
and the positive rate is only about `44%`, which means the positive mean is
pulled by right-tail winners rather than the typical stock. The earlier
`13.03, 6.20, ...` table is the same sample under 48-month equal weighting; it
is not the headline convention.

The bucket-level signal is much clearer than pointwise IC: Top1000 bucket group
IC is about `0.038-0.045`, while TopK internal next IC is negative, around
`-0.025` at Top200 and `-0.018` at Top1000. The Top100 within-bucket Spearman IC
is also negative, around `-0.028`. This supports the read that the old NN overlay
works more like a bucket / head-region selector than a stable single-stock fine
ranker, which explains why pointwise next IC is low despite a visible bucket
gradient.

Multiscale follow-up on the same old three-model sample confirms the head-region
shape. The old-three-model mean TopK next-close pool-internal excess is
`15.50, 12.71, 10.82, 9.45, 5.66, 3.59 bps` for Top50/100/150/200/500/1000.
The 100-name and 200-name Top1000 buckets are monotone by mean excess; the
50-name buckets are strongly ordered in the head and only become noisy after the
first few hundred names.

The right-tail diagnosis is also ordered. In the 100-name Top1000 buckets,
realized pool top-10% hit rate falls from `18.12%` in ranks `1-100` to `9.52%`
in ranks `901-1000`; `>=300 bps` winner rate falls from `18.48%` to `9.97%`.
This is why bucket means can be positive while bucket medians stay negative: the
signal increases the frequency and size of large positive outcomes, not the
median stock's next-close return.

The IC diagnostics still say the same thing: Top50/100/150/200 internal Spearman
IC is about `-0.029, -0.029, -0.028, -0.026`. Sliding local IC is most negative
at the very head (`1-100` about `-0.029`) and fades toward zero after the first
few hundred ranks. This is not a contradiction with the bucket gradient; it
means the score is useful for detecting a high-score state / region, while the
fine ordering inside that region is weak or mildly reversed.

One-line interpretation: bucket IC / bucket returns measure the coarse location
of the score region, while within-bucket IC measures the local slope inside that
region, so a globally ordered head region can coexist with weak or negative
fine ranking inside each rank slice.

Decision: for this old-NN overlay sample, the bucket-diagnostic question is
settled. Additional validation should move to the current model family or to
trading checks such as costs, turnover, capacity, and weighting; it should not
repeat more Top1000 / TopK IC variants on the old three models unless the pool,
label, or weighting convention changes.
