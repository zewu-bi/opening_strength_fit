# Top1000 Bucket Diagnostics

## 2026-07-09 old NN overlay bucket check

Scope: this diagnostic uses the old NN overlay set only: `nn_mlp_base`,
`nn_mlp_base_plus_mse`, and `nn_deep_gelu_mse`. The sample is 2022-2025
`pool_L`, with next-close pool-internal excess as the label. It should not be
read as a `grouped_gated` / `grouped_gated_v2` result.

Artifacts:

```text
experiments/results/backtests/local_ic_topk_nn_overlay_pool_l_v1/
experiments/results/backtests/ic_bucket_diagnostics_nn_overlay_pool_l_v1/
```

Conclusion: the alpha has a reasonable coarse Top1000 bucket shape, but weak
fine ranking inside buckets. The averaged 10-bucket Top1000 mean excess is
roughly `13.03, 6.20, 4.10, 2.88, 2.34, 1.73, 1.61, 1.55, 1.35, 1.31 bps`,
so score buckets are ordered in the right direction. However, bucket medians are
negative and the positive rate is only about `44%`, which means the positive
mean is pulled by right-tail winners rather than the typical stock.

The bucket-level signal is much clearer than pointwise IC: Top1000 bucket group
IC is about `0.038-0.045`, while TopK internal next IC is negative, around
`-0.025` at Top200 and `-0.018` at Top1000. The Top100 within-bucket Spearman IC
is also negative, around `-0.028`. This supports the read that the old NN overlay
works more like a bucket / head-region selector than a stable single-stock fine
ranker, which explains why pointwise next IC is low despite a visible bucket
gradient.
