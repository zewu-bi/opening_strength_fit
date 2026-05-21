# Experiment Log

| run | status | notes |
| --- | --- | --- |
| `ridge_opening_1y_next_month` | completed | Baseline A：全 A 股 universe，2021 训练、2022-01 测试。修正横截面口径下 decision rank IC = 0.0799，Top20 mean = +18.96 bps；I500 回测 alpha/profit 为负。 |
| `ridge_opening_1y_next_month_strong` | completed | Baseline B：开盘强势候选池过滤后训练/测试。修正横截面口径下 decision rank IC = 0.1156，Top20 mean = +9.63 bps；I500 回测 alpha/profit 为负。 |
| `gbm_opening_1y_next_month` | completed | 与 Baseline A 同口径，GBM 非线性模型。修正横截面口径下 decision rank IC = 0.1831，Top20 mean = +34.33 bps；四组中 metrics 最强但 I500 回测为负。 |
| `gbm_opening_1y_next_month_strong` | completed | 开盘强势候选池过滤后的 GBM。修正横截面口径下 decision rank IC = 0.1454，Top20 mean = +18.78 bps；四组中回测亏损最小但仍为负。 |
| `gbm_opening_rolling_2022h1_strong` | running | 月度 rolling 12 个月训练、2022-H1 测试的 strong GBM；2026-05-21 检查时集群 sharded Job 仍在运行，尚未归档。 |
| `ridge_opening_1m_3d` | completed | 小窗 Baseline A：2021-12 训练、2022-01-04 至 2022-01-06 测试。原始 metrics 已归档；用修正后的 `date x decision_target_timestamp` 横截面口径重算后，decision rank IC = 0.0824，Top20 mean = +16.26 bps。 |
| `ridge_opening_1m_3d_strong` | completed | 小窗 Baseline B：开盘强势候选池过滤后训练/测试。修正横截面口径下 decision rank IC = 0.1087，Top20 mean = +6.78 bps；bucket lift 最强但 Top20 胜率低于 50%。 |
| `gbm_opening_1m_3d` | completed | 小窗 GBM：同 `ridge_opening_1m_3d` 口径。修正横截面口径下 decision rank IC = 0.1426，Top20 mean = +41.92 bps，Top20 win rate = 62.1%，三组小窗里最强。 |

## 2026-05-21 1y next-month 结果

四个 2021 训练、2022-01 测试实验均已从 K8s PVC 拉回 `metrics_by_year.csv`
和 predictions，并按 runbook 生成 I500 回测。原始 metrics 与 backtest JSON 已用
`record_experiment.py` 归档到 `experiments/results/metrics/` 和
`experiments/results/backtests/`。由于预测中实际 tick `timestamp` 可能比整分钟决策点
滞后 0-5 秒，本地额外按 `date x decision_target_timestamp` 重算横截面指标，并归档：

- `experiments/results/metrics/opening_1y_next_month_corrected_cross_section_summary.csv`
- `experiments/results/metrics/opening_1y_next_month_corrected_score_buckets.csv`

修正口径和 I500 回测摘要：

| run | decision rank IC | rank IC IR | Top20 mean bps | Top20 win rate | alpha end | profit end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | 0.1831 | 2.7548 | +34.33 | 60.7% | -0.0093 | -0.1151 |
| `gbm_opening_1y_next_month_strong` | 0.1454 | 1.9402 | +18.78 | 53.8% | -0.0021 | -0.1079 |
| `ridge_opening_1y_next_month_strong` | 0.1156 | 1.5987 | +9.63 | 51.2% | -0.0035 | -0.1094 |
| `ridge_opening_1y_next_month` | 0.0799 | 1.4788 | +18.96 | 54.4% | -0.0066 | -0.1125 |

初步判断：横截面排序信号在 2022-01 单月上均为正，普通 GBM 的 rank IC、
bucket lift 和 Top20 收益最强；但当前 `aggregate=max`、`tar=I500` 回测下四组
alpha/profit 全为负，说明 tick-level 预测信号尚未直接转化为可交易组合收益。
下一步应继续等 rolling H1 strong 任务完成，并排查回测聚合方式、交易约束和候选池规模。

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
