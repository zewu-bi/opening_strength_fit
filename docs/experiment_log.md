# Experiment Log

Note: 本地 `experiments/jobs`、`experiments/runs` 和 `experiments/results` 当前保留：
`1m3d` 小窗口 Ridge/GBM 对比、`1y_next_month` Ridge/GBM/strong 对比，以及
`1y_next_month` CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。已归档的
Ridge/GBM baseline metrics/opening replay 使用无成交延迟旧口径（`entry_tick_delay = 0`，当前 tick ask1 买入）；
LightGBM delay 分支使用各自 PVC labeled cache 中的延迟成交 label。

| run | status | notes |
| --- | --- | --- |
| `ridge_opening_1y_next_month` | completed | Baseline A：全 A 股 universe，2021 训练、2022-01 测试。修正横截面口径下 decision rank IC = 0.0799，Top20 mean = +18.96 bps。 |
| `ridge_opening_1y_next_month_strong` | completed | Baseline B：开盘强势候选池过滤后训练/测试。修正横截面口径下 decision rank IC = 0.1156，Top20 mean = +9.63 bps。 |
| `gbm_opening_1y_next_month` | completed | 与 Baseline A 同口径，GBM 非线性模型。修正横截面口径下 decision rank IC = 0.1831，Top20 mean = +34.33 bps；四组中 metrics 最强。 |
| `gbm_opening_1y_next_month_strong` | completed | 开盘强势候选池过滤后的 GBM。修正横截面口径下 decision rank IC = 0.1454，Top20 mean = +18.78 bps。 |
| `ridge_opening_1m_3d` | completed | 小窗 Baseline A：2021-12 训练、2022-01-04 至 2022-01-06 测试。原始 metrics 已归档；用修正后的 `date x decision_target_timestamp` 横截面口径重算后，decision rank IC = 0.0824，Top20 mean = +16.26 bps。 |
| `ridge_opening_1m_3d_strong` | completed | 小窗 Baseline B：开盘强势候选池过滤后训练/测试。修正横截面口径下 decision rank IC = 0.1087，Top20 mean = +6.78 bps；bucket lift 最强但 Top20 胜率低于 50%。 |
| `gbm_opening_1m_3d` | completed | 小窗 GBM：同 `ridge_opening_1m_3d` 口径。修正横截面口径下 decision rank IC = 0.1426，Top20 mean = +41.92 bps，Top20 win rate = 62.1%，三组小窗里最强。 |
| `lgbm_opening_1y_next_month_delay0` | completed | CPU LightGBM 普通 universe，2021 训练、2022-01 测试，PVC delay0 cache；group rank IC = 0.2044，Top20 mean = +50.05 bps。 |
| `lgbm_opening_1y_next_month_delay1` | completed | CPU LightGBM 普通 universe，主执行口径 delay1；group rank IC = 0.1515，Top20 mean = +40.29 bps。 |
| `lgbm_opening_1y_next_month_delay2` | completed | CPU LightGBM 普通 universe，delay2 执行敏感性；group rank IC = 0.1360，Top20 mean = +36.75 bps。 |
| `lgbm_opening_1y_next_month_strong_delay0` | completed | CPU LightGBM strong candidate 分支，PVC delay0 cache；group rank IC = 0.1729，Top20 mean = +29.28 bps。 |
| `lgbm_opening_1y_next_month_strong_delay1` | completed | CPU LightGBM strong candidate 分支，主执行口径 delay1；group rank IC = 0.1389，Top20 mean = +17.17 bps。 |
| `lgbm_opening_1y_next_month_strong_delay2` | completed | CPU LightGBM strong candidate 分支，delay2 执行敏感性；group rank IC = 0.1298，Top20 mean = +12.60 bps。 |

## 2026-05-26 CPU LightGBM delay 结果

delay0/1/2 的 one-year labeled cache 已在 PVC 完整落盘，路径为
`/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet`。
三份 cache 均为 12,308,573 行，并包含 `entry_delay_seconds`、`entry_max_tick_gap_seconds`
和 `entry_delay_ticks`。按已渲染 Job YAML 提交并完成六个 CPU LightGBM 训练 Job，
reader Job 已完成，年度 metrics 已归档到 `experiments/results/metrics/`，predictions
已拉回到 `output/predictions/<run_id>/predictions_all.parquet`。`run_lgbm_delay_replays.py --check-interface-only`
已通过。随后完成标准 delay replay 网格，轻量证据归档到：

- `experiments/results/backtests/opening_intraday_lgbm_delay_replays_scenario_summary.csv`
- `experiments/results/backtests/opening_intraday_lgbm_delay_replays_delay_scan_proxy_top20.csv`
- `experiments/results/backtests/opening_intraday_lgbm_delay_replays_trace.json`

汇报图留在 `output/reports/opening_intraday_lgbm_delay_replays/`：

- `delay_scan_proxy_top20.png`
- `replay_l3_l5_single_tradable_delay2.png`

| run | group rank IC | rank IC IR | Top20 mean bps | Top20 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lgbm_opening_1y_next_month_delay0` | 0.2044 | 3.0007 | +50.05 | 64.1% | 940,747 |
| `lgbm_opening_1y_next_month_delay1` | 0.1515 | 2.4233 | +40.29 | 60.4% | 940,744 |
| `lgbm_opening_1y_next_month_delay2` | 0.1360 | 2.1260 | +36.75 | 59.4% | 940,748 |
| `lgbm_opening_1y_next_month_strong_delay0` | 0.1729 | 2.3898 | +29.28 | 56.8% | 153,687 |
| `lgbm_opening_1y_next_month_strong_delay1` | 0.1389 | 1.8749 | +17.17 | 52.8% | 153,659 |
| `lgbm_opening_1y_next_month_strong_delay2` | 0.1298 | 1.7452 | +12.60 | 51.7% | 153,637 |

无约束 Top20 replay 显示 delay 衰减明确：

| branch | delay0 cycle bps | delay1 cycle bps | delay2 cycle bps |
| --- | ---: | ---: | ---: |
| universe | +57.04 | +45.37 | +39.00 |
| strong | +42.04 | +26.41 | +18.97 |

delay2 仍有正信号，因此作为保守成交假设检查 6 个约束场景。关键结果：

| branch | proxy bps | liquidity bps | L3/1m bps | L5/2m bps |
| --- | ---: | ---: | ---: | ---: |
| universe delay2 | +39.00 | +28.74 | +21.88 | +18.38 |
| strong delay2 | +18.97 | +8.97 | +4.95 | +1.80 |

阶段结论：普通 universe 分支强于 strong candidate 分支；delay 越长，排序和 replay 均衰减。
基础可交易和 liquidity 约束下，delay1/delay2 universe 仍为正；小容量 L3/L5 sweep 后收益继续下降但没有跳崖。
delay2 group rank IC 在 L3/L5 约为 0.130/0.132，说明排序信号仍在。Short-horizon alpha discovery 第一阶段可归档，下一阶段转向
更长周期衰减、已有库存/日频候选 overlay，或更细的执行容量研究；当前 tick replay 不应直接
解释为可放大的实盘策略收益。更长周期衰减结果见下一节。

## 2026-05-26 Alpha horizon decay 与日频候选 overlay 收尾

在 delay2 保守 opening score 口径下，补充了更长周期衰减检查。实验分两种 cohort：
固定 `09:30` 这一批 opening score，以及 `09:30-09:39` 十个开盘分钟的平均口径。
timed horizon 使用 ClickHouse 拉取的未来整分钟 bid/ask mid point return，close / next close 使用
ClickHouse close label；`1m` 这里不再混用旧 60s VWAP proxy label。

轻量证据已归档到 `experiments/results/backtests/`：

- `opening_alpha_horizon_decay_delay2_0930_summary.csv`
- `opening_alpha_horizon_decay_delay2_0930_trace.json`
- `opening_alpha_horizon_decay_delay2_open10_summary.csv`
- `opening_alpha_horizon_decay_delay2_open10_trace.json`
- `opening_alpha_horizon_decay_delay2_0930_vs_open10_summary.csv`
- `opening_alpha_horizon_decay_delay2_close_next_close_by_decision_minute.csv`

汇报图保留在 `output/reports/opening_alpha_horizon_decay_delay2_compare_selected/`：

- `rank_ic_0930_vs_open10.png`
- `mean_return_0930_vs_open10.png`
- `close_next_close_ic_by_decision_minute.png`
- `close_next_close_return_by_decision_minute.png`

固定 `09:30` cohort 的 group rank IC：

| horizon | Universe | Strong |
| --- | ---: | ---: |
| 1m | 0.199 | 0.214 |
| 2m | 0.188 | 0.178 |
| 5m | 0.143 | 0.140 |
| 10m | 0.123 | 0.122 |
| close | 0.071 | 0.077 |
| next close | 0.039 | 0.058 |

`09:30-09:39` 十分钟平均口径的 group rank IC：

| horizon | Universe | Strong |
| --- | ---: | ---: |
| 1m | 0.162 | 0.163 |
| 2m | 0.116 | 0.108 |
| 5m | 0.073 | 0.076 |
| 10m | 0.045 | 0.052 |
| close | 0.001 | 0.012 |
| next close | -0.021 | 0.009 |

阶段判断：

- 固定 `09:30` opening score 从 1m 到 10m 明显衰减，但到 close / next close 仍有弱正排序 IC。
- 固定 `09:30` 的 next close Top20 mean alpha return 为负，Universe 约 `-66.75 bps`、Strong 约 `-38.16 bps`；因此不能把 weak positive next close Rank IC 解释成“隔夜 Top20 已能赚钱”。
- `09:30-09:39` 简单平均后，close / next close 的排序效果基本消失，不支持直接把整个开盘 10 分钟均值当成日频特征。
- Alpha horizon decay 这条研究路线可以归档。下一步 active work 应转向“已有日频候选池内的开盘重排序 / 辅助排序信号”：优先用固定 `09:30` 或最早可交易 opening score，检查它能否改善上游日频候选池的最终排序、选股命中率或入场优先级。

## 2026-05-21 1y next-month 结果

四个 2021 训练、2022-01 测试实验均已从 K8s PVC 拉回 `metrics_by_year.csv`
和 predictions。原始 metrics 已用 `record_experiment.py` 归档到
`experiments/results/metrics/`。由于预测中实际 tick `timestamp` 可能比整分钟决策点
滞后 0-5 秒，本地额外按 `date x decision_target_timestamp` 重算横截面指标，并归档：

- `experiments/results/metrics/opening_1y_next_month_corrected_cross_section_summary.csv`
- `experiments/results/metrics/opening_1y_next_month_corrected_score_buckets.csv`

修正横截面口径摘要：

| run | decision rank IC | rank IC IR | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | 0.1831 | 2.7548 | +34.33 | 60.7% |
| `gbm_opening_1y_next_month_strong` | 0.1454 | 1.9402 | +18.78 | 53.8% |
| `ridge_opening_1y_next_month_strong` | 0.1156 | 1.5987 | +9.63 | 51.2% |
| `ridge_opening_1y_next_month` | 0.0799 | 1.4788 | +18.96 | 54.4% |

随后补充更贴近项目的开盘短周期 Top20 回测：每天本金从 1.0 开始，
在 `09:30/09:32/09:34/09:36/09:38` 五个非重叠两分钟 cycle 上按预测分 Top20 等权满仓，
用 label 均值复利滚动。逐日曲线在
`output/reports/opening_intraday_top20_1y_next_month/daily_curves/`，轻量摘要归档到：

- `experiments/results/backtests/opening_intraday_top20_1y_next_month_summary.csv`
- `experiments/results/backtests/opening_intraday_top20_1y_next_month_daily_summary.csv`
- `experiments/results/backtests/opening_intraday_top20_1y_next_month_cycles.csv`
- `experiments/results/backtests/opening_intraday_top20_1y_next_month_trace.json`

开盘短周期回测摘要：

| run | mean cycle bps | cycle win rate | mean day bps | positive days | compounded month |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | +42.21 | 82.1% | +212.61 | 19/19 | +49.1% |
| `gbm_opening_1y_next_month_strong` | +27.91 | 72.6% | +140.01 | 19/19 | +30.2% |
| `ridge_opening_1y_next_month` | +23.96 | 73.7% | +120.32 | 18/19 | +25.4% |
| `ridge_opening_1y_next_month_strong` | +14.66 | 64.2% | +73.51 | 15/19 | +14.9% |

初步判断：横截面排序信号在 2022-01 单月上均为正，且在匹配 label 的开盘短周期回测中
能转化为明显正收益；普通 GBM 最强。

## 2026-05-22 本地实验清理

按当时 PVC/研究口径，本地实验注册表只保留两组结果：

- `1m3d` 小窗口对比：`ridge_opening_1m_3d`、`ridge_opening_1m_3d_strong`、`gbm_opening_1m_3d`
- `1y_next_month` 对比：`ridge_opening_1y_next_month`、`ridge_opening_1y_next_month_strong`、`gbm_opening_1y_next_month`、`gbm_opening_1y_next_month_strong`

已清理本地未进入这两组归档的 LightGBM delay/materialize Job YAML 和 run config。
`experiments/results/` 按 runbook 只保留轻量证据：单 run metrics 位于 `experiments/results/metrics/*_metrics_by_year.csv`，
修正横截面对比表位于 `experiments/results/metrics/`，开盘 Top20 replay 摘要位于
`experiments/results/backtests/`；大体积 predictions、模型和临时图表仍留在 `output/`，不进入 git。

同日重新按实时 PVC 校准状态：

- 当时 PVC 可分析结果仍只有已归档的 Ridge/GBM baseline 目录；没有可拉回的 LightGBM delay0/1/2 结果目录。
- 本地已删除过期的 `output/cache/opening_1y_next_month/` cache snapshot 和 `output/k8s/pvc_snapshot/remote_tree.txt`。这类快照不是实时事实源，后续以 `hfcli kubectl exec ... find /mnt/output/opening_strength_fit` 为准。
- `opening_1y_next_month_delay{0,1,2}_labeled.parquet` 与 `opening_2013_2024_delay1_labeled.parquet` 当前属于 PVC cache 目标路径；只有最终 `*.parquet` 落盘后才可用于训练，`.tmp.parquet`、lock 和 heartbeat 只表示 materialize 正在进行或被中断。
- 当前正式训练路径是 CPU LightGBM + PVC labeled cache；GPU 仅保留为显式配置能力，当前没有活跃 GPU run/job。

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
`1y_next_month` 的 GBM / GBM strong 主线，不作为最终策略结论。
