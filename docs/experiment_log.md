# Experiment Log

本文件是实验事实源；README 和 project brief 只保留摘要。当前本地 `experiments/runs`、
`experiments/jobs` 和 `experiments/results` 保留三类归档：

- `1m3d` 小窗口 Ridge/GBM 对比。
- `1y_next_month` Ridge/GBM/strong 对比。
- `1y_next_month` CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。

旧 Ridge/GBM baseline 使用无成交延迟旧口径（`entry_tick_delay = 0`）；LightGBM delay 分支使用各自
PVC labeled cache 中的延迟成交 label。不同口径不要直接横向混比。

## Run 索引

| run | status | notes |
| --- | --- | --- |
| `ridge_opening_1m_3d` | completed | 2021-12 训练、2022-01-04 至 2022-01-06 测试；decision rank IC = 0.0824，Top20 mean = +16.26 bps。 |
| `ridge_opening_1m_3d_strong` | completed | 小窗 strong 分支；decision rank IC = 0.1087，Top20 mean = +6.78 bps。 |
| `gbm_opening_1m_3d` | completed | 小窗 GBM；decision rank IC = 0.1426，Top20 mean = +41.92 bps。 |
| `ridge_opening_1y_next_month` | completed | 2021 训练、2022-01 测试；decision rank IC = 0.0799，Top20 mean = +18.96 bps。 |
| `ridge_opening_1y_next_month_strong` | completed | strong Ridge；decision rank IC = 0.1156，Top20 mean = +9.63 bps。 |
| `gbm_opening_1y_next_month` | completed | sklearn GBM；decision rank IC = 0.1831，Top20 mean = +34.33 bps。 |
| `gbm_opening_1y_next_month_strong` | completed | strong GBM；decision rank IC = 0.1454，Top20 mean = +18.78 bps。 |
| `lgbm_opening_1y_next_month_delay0` | completed | CPU LightGBM universe delay0；group rank IC = 0.2044，Top20 mean = +50.05 bps。 |
| `lgbm_opening_1y_next_month_delay1` | completed | CPU LightGBM universe delay1；group rank IC = 0.1515，Top20 mean = +40.29 bps。 |
| `lgbm_opening_1y_next_month_delay2` | completed | CPU LightGBM universe delay2；group rank IC = 0.1360，Top20 mean = +36.75 bps。 |
| `lgbm_opening_1y_next_month_strong_delay0` | completed | CPU LightGBM strong delay0；group rank IC = 0.1729，Top20 mean = +29.28 bps。 |
| `lgbm_opening_1y_next_month_strong_delay1` | completed | CPU LightGBM strong delay1；group rank IC = 0.1389，Top20 mean = +17.17 bps。 |
| `lgbm_opening_1y_next_month_strong_delay2` | completed | CPU LightGBM strong delay2；group rank IC = 0.1298，Top20 mean = +12.60 bps。 |

## 2026-05-26 CPU LightGBM Delay

delay0/1/2 one-year labeled cache 已在 PVC 完整落盘：

```text
/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
```

三份 cache 均为 12,308,573 行，并包含 `entry_delay_seconds`、`entry_max_tick_gap_seconds` 和
`entry_delay_ticks`。六个 CPU LightGBM 训练 Job 和 reader Job 已完成，metrics 已归档到
`experiments/results/metrics/`，predictions 已拉回到 `output/predictions/<run_id>/predictions_all.parquet`。

年度 metrics：

| run | group rank IC | rank IC IR | Top20 mean bps | Top20 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lgbm_opening_1y_next_month_delay0` | 0.2044 | 3.0007 | +50.05 | 64.1% | 940,747 |
| `lgbm_opening_1y_next_month_delay1` | 0.1515 | 2.4233 | +40.29 | 60.4% | 940,744 |
| `lgbm_opening_1y_next_month_delay2` | 0.1360 | 2.1260 | +36.75 | 59.4% | 940,748 |
| `lgbm_opening_1y_next_month_strong_delay0` | 0.1729 | 2.3898 | +29.28 | 56.8% | 153,687 |
| `lgbm_opening_1y_next_month_strong_delay1` | 0.1389 | 1.8749 | +17.17 | 52.8% | 153,659 |
| `lgbm_opening_1y_next_month_strong_delay2` | 0.1298 | 1.7452 | +12.60 | 51.7% | 153,637 |

标准 replay 归档文件：

```text
experiments/results/backtests/opening_intraday_lgbm_delay_replays_scenario_summary.csv
experiments/results/backtests/opening_intraday_lgbm_delay_replays_delay_scan_proxy_top20.csv
experiments/results/backtests/opening_intraday_lgbm_delay_replays_trace.json
```

无约束 Top20 replay：

| branch | delay0 cycle bps | delay1 cycle bps | delay2 cycle bps |
| --- | ---: | ---: | ---: |
| universe | +57.04 | +45.37 | +39.00 |
| strong | +42.04 | +26.41 | +18.97 |

delay2 约束场景：

| branch | proxy bps | liquidity bps | L3/1m bps | L5/2m bps |
| --- | ---: | ---: | ---: | ---: |
| universe delay2 | +39.00 | +28.74 | +21.88 | +18.38 |
| strong delay2 | +18.97 | +8.97 | +4.95 | +1.80 |

阶段结论：普通 universe 分支强于 strong candidate；delay 越长，排序和 replay 均衰减。delay2 universe
在基础可交易、liquidity 和小容量 sweep 下仍为正，但容量可扩展性不足。Short-horizon alpha discovery
第一阶段可归档。

## 2026-05-26 Alpha Horizon Decay

在 delay2 保守 opening score 口径下，补充 longer-horizon 衰减检查。timed horizon 使用 ClickHouse
未来整分钟 bid/ask mid point return，close / next close 使用 ClickHouse close label；`1m` 不再混用旧
60s VWAP proxy label。

轻量证据：

```text
experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2_open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_open10_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_vs_open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_close_next_close_by_decision_minute.csv
```

group rank IC：

| cohort | branch | 1m | 2m | 5m | 10m | close | next close |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `09:30` | Universe | 0.199 | 0.188 | 0.143 | 0.123 | 0.071 | 0.039 |
| `09:30` | Strong | 0.214 | 0.178 | 0.140 | 0.122 | 0.077 | 0.058 |
| `09:30-09:39` avg | Universe | 0.162 | 0.116 | 0.073 | 0.045 | 0.001 | -0.021 |
| `09:30-09:39` avg | Strong | 0.163 | 0.108 | 0.076 | 0.052 | 0.012 | 0.009 |

固定 `09:30` 的 next close Top20 mean alpha return 为负，Universe 约 `-66.75 bps`、Strong 约
`-38.16 bps`。因此 weak positive next close Rank IC 不能解释成“隔夜 Top20 已能赚钱”。
`09:30-09:39` 简单平均后，close / next close 排序效果基本消失。

阶段结论：alpha horizon decay 路线可以归档。固定 `09:30` 虽然强，但可能受集合竞价特征或跨竞价边界
累计成交字段影响，不能直接作为最终信号形态。

## 2026-05-26 Mentor Re-scope

后续目标从交易约束和日频 overlay 前移为信号增强：

- 先把 opening signal 做强，再考虑 fee/slippage、同股冷却、T+1 overlay 等交易问题。
- 容量暂只看 ask1 可买量，不把 L3/L5 sweep 作为主线优化目标。
- 主评估改为 Rank IC 和 Top100 选股收益；Top20 保留为尖端 alpha 辅助观察。
- 重点放在开盘后的盘口信息，尤其是 ask/bid 档位、深度、queue 变化和成交冲击。
- 集合竞价相关 feature 应减少或作为对照组，重点排查 `09:30` 强势是否由 `preopen_*`
  或跨 09:30 的累计 `volume/turnover` 差分造成。

下一组实验应优先做 feature ablation：all features、去掉 `preopen_*`、post-open reset、
只用开盘后盘口/成交动态，并按分钟输出 Rank IC 和 Top100 收益。

## 2026-05-21 1y Next-Month Baseline

四个 2021 训练、2022-01 测试实验已从 K8s PVC 拉回 metrics 和 predictions。由于旧镜像原始 metrics
按实际 tick `timestamp` 分组，本地按 `date x decision_target_timestamp` 重算横截面指标并归档：

```text
experiments/results/metrics/opening_1y_next_month_corrected_cross_section_summary.csv
experiments/results/metrics/opening_1y_next_month_corrected_score_buckets.csv
```

| run | decision rank IC | rank IC IR | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | 0.1831 | 2.7548 | +34.33 | 60.7% |
| `gbm_opening_1y_next_month_strong` | 0.1454 | 1.9402 | +18.78 | 53.8% |
| `ridge_opening_1y_next_month_strong` | 0.1156 | 1.5987 | +9.63 | 51.2% |
| `ridge_opening_1y_next_month` | 0.0799 | 1.4788 | +18.96 | 54.4% |

旧开盘短周期 Top20 replay 轻量摘要归档到
`experiments/results/backtests/opening_intraday_top20_1y_next_month_*`。旧普通 GBM mean cycle return 约
`+42.21 bps`，19 个测试日均为正。该结果只说明短周期方向性值得继续验证，不代表 T+1 可交易收益。

## 2026-05-22 本地实验清理

当时按 PVC/研究口径，本地只保留 `1m3d` 小窗口和 `1y_next_month` Ridge/GBM baseline 归档；
未进入归档的旧 LightGBM delay/materialize Job YAML 和 run config 已清理。随后按实时 PVC 校准状态：

- 当时 PVC 可分析结果仍只有 Ridge/GBM baseline；没有可拉回的 LightGBM delay 结果目录。
- 已删除过期的本地 cache snapshot；后续以实时 PVC `find /mnt/output/opening_strength_fit` 为准。
- `*.tmp.parquet`、lock 和 heartbeat 只表示 materialize 进行中或被中断，不是可用训练输入。
- 正式训练路径校准为 CPU LightGBM + PVC labeled cache；GPU 仅保留为显式配置能力。

## 2026-05-20 小窗结果

三组小窗实验已从 K8s PVC 拉回 `metrics_by_year.csv`，并用当前代码按
`date x decision_target_timestamp` 重算横截面指标：

| run | overall rank IC | decision rank IC | B5-B1 bps | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gbm_opening_1m_3d` | 0.1631 | 0.1426 | +21.03 | +41.92 | 62.1% |
| `ridge_opening_1m_3d_strong` | 0.1083 | 0.1087 | +23.39 | +6.78 | 49.1% |
| `ridge_opening_1m_3d` | 0.1070 | 0.0824 | +12.23 | +16.26 | 50.3% |

小窗测试期只有 2022-01-04 至 2022-01-06 三天，只能作为继续跑 one-year 主线的依据。
