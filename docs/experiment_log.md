# Experiment Log

| run | status | notes |
| --- | --- | --- |
| `ridge_opening_full` | queued | 通用 opening Ridge full config。工作流已对齐 `xy_fit`：full config 本地 smoke、training/reader Job、`metrics_by_year.csv` 拉回、predictions 拉回、backtest 和 `record_experiment.py` 归档。 |
| `ridge_opening_1y_next_month` | running | Baseline A：全 A 股 universe，2021 训练、2022-01 测试，Ridge 横截面排序。 |
| `ridge_opening_1y_next_month_strong` | running | Baseline B：除开盘强势候选池过滤外，参数与 `ridge_opening_1y_next_month` 一致，用于检验强势/活跃样本内排序是否更稳定。 |
| `gbm_opening_1y_next_month` | running | 与 Baseline A 同口径，用 GBM 检验非线性模型是否优于 Ridge。 |
| `ridge_opening_1m_3d` | completed | 小窗 Baseline A：2021-12 训练、2022-01-04 至 2022-01-06 测试。原始 metrics 已归档；用修正后的 `date x decision_target_timestamp` 横截面口径重算后，decision rank IC = 0.0824，Top20 mean = +16.26 bps。 |
| `ridge_opening_1m_3d_strong` | completed | 小窗 Baseline B：开盘强势候选池过滤后训练/测试。修正横截面口径下 decision rank IC = 0.1087，Top20 mean = +6.78 bps；bucket lift 最强但 Top20 胜率低于 50%。 |
| `gbm_opening_1m_3d` | completed | 小窗 GBM：同 `ridge_opening_1m_3d` 口径。修正横截面口径下 decision rank IC = 0.1426，Top20 mean = +41.92 bps，Top20 win rate = 62.1%，三组小窗里最强。 |

## 2026-05-20 小窗结果

三组小窗实验均已从 K8s PVC 拉回 `metrics_by_year.csv` 并用
`record_experiment.py` 归档到 `experiments/results/metrics/`。由于集群任务使用的是旧镜像，
原始 metrics 中 `cross_section` 仍按实际 tick `timestamp` 分组；本地额外拉回 predictions，
用当前代码按整分钟 `decision_target_timestamp` 重算横截面指标，并归档轻量对比表：

- `experiments/results/metrics/opening_1m3d_corrected_cross_section_summary.csv`
- `experiments/results/metrics/opening_1m3d_corrected_score_buckets.csv`

修正口径摘要：

| run | overall rank IC | decision rank IC | B1 mean bps | B5 mean bps | B5-B1 bps | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gbm_opening_1m_3d` | 0.1631 | 0.1426 | -20.64 | +0.39 | +21.03 | +41.92 | 62.1% |
| `ridge_opening_1m_3d_strong` | 0.1083 | 0.1087 | -22.38 | +1.01 | +23.39 | +6.78 | 49.1% |
| `ridge_opening_1m_3d` | 0.1070 | 0.0824 | -15.79 | -3.56 | +12.23 | +16.26 | 50.3% |

初步判断：三组都有正向排序信号，GBM 在 rank IC、bucket 单调性和 Top20 分布上最强。
但测试期只有 2022-01-04 至 2022-01-06 三天，只能说明值得继续跑
`1y_next_month` 和更长 rolling，不作为最终策略结论。
