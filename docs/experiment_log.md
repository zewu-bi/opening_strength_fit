# Experiment Log

本文件是实验事实源；README 和 project brief 只保留摘要。当前项目保留正式归档实验和 active
feature exploration 两类记录。

正式归档实验：

- `1m3d` 小窗口 Ridge/GBM 对比。
- `1y_next_month` Ridge/GBM/strong 对比。
- `1y_next_month` CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。

旧 Ridge/GBM baseline 使用无成交延迟旧口径（`entry_tick_delay = 0`）；LightGBM delay 分支使用各自
PVC labeled cache 中的延迟成交 label。不同口径不要直接横向混比。

Active 非归档任务：

| task | kind | status | output |
| --- | --- | --- | --- |
| `lgbm_delay2_postopen_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_v1/` |
| `lgbm_delay2_postopen_no_preopen_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_no_preopen_v1/` |
| `lgbm_delay2_postopen_v2` | exploration | running | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_v2/` |
| `lgbm_delay2_feature_dependence_v1` | feature_audit | running | `/mnt/output/opening_strength_fit/lgbm_delay2_feature_dependence_v1/` |

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
| `lgbm_delay2_postopen_v2` | running | post-open v1 plus v2 queue/depth-shape/trade-impact features；Top100 evaluation。 |
| `lgbm_delay2_feature_dependence_v1` | running | grouped feature importance、permutation 和 drop-retrain ablation。 |

## Output 索引

本地 `output/` 只保留能追溯到上述 run/job 的产物：

| local path | source |
| --- | --- |
| `output/predictions/<run_id>/predictions_all.parquet` | 对应 `experiments/runs/<run_id>.toml`、K8s training job 和 sync 记录。 |
| `output/k8s/metrics/<run_id>_metrics_by_year.csv` | 从对应 PVC run output 拉回的 raw metrics。 |
| `output/reports/opening_1m3d_*` | 小窗 Ridge/GBM 归档实验对比和校正指标。 |
| `output/reports/opening_1y_next_month_*` | 一年训练、次月测试 Ridge/GBM 归档实验对比和校正指标。 |
| `output/reports/opening_intraday_top20_1y_next_month` | 旧 GBM/strong baseline replay，对应 `opening_intraday_top20_1y_next_month_*` 归档摘要。 |
| `output/reports/opening_intraday_lgbm_delay_replays` | LightGBM delay0/1/2 标准 replay，对应 `opening_intraday_lgbm_delay_replays_*` 归档摘要。 |
| `output/reports/opening_alpha_horizon_decay_delay2_*` | delay2 horizon decay，对应 `opening_alpha_horizon_decay_delay2_*` 归档摘要。 |
| `output/reports/opening_delay2_signal_baseline` | delay2 保守 baseline 的分钟四曲线，用于当前 feature-strengthening 门槛。 |

## 2026-05-26 CPU LightGBM Delay

delay0/1/2 one-year labeled cache 已在 PVC 完整落盘：

```text
/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
```

三份 cache 均为 12,308,573 行，并包含 `entry_delay_seconds`、`entry_max_tick_gap_seconds` 和
`entry_delay_ticks`。六个 CPU LightGBM 训练 Job 已完成，metrics 已归档到
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

阶段结论：alpha horizon decay 路线可以归档。固定 `09:30` 虽然强，但后续无论是否受集合竞价影响，
都应重做特征并把重点放到开盘后的盘口信息。

## 2026-05-26 Mentor Re-scope

后续目标从交易约束和日频 overlay 前移为信号增强：

- 先把 opening signal 做强，再考虑 fee/slippage、同股冷却、T+1 overlay 等交易问题。
- 容量暂只看 ask1 可买量，不把 L3/L5 sweep 作为主线优化目标。
- 主评估改为 Rank IC 和 Top100 选股收益；Top20 不再作为主评估。
- 重点放在开盘后的盘口信息，尤其是 ask/bid 档位、深度、queue 变化和成交冲击。
- 集合竞价相关 feature 不需要一刀切删除；应评估 `preopen_*`、累计成交字段和盘口特征的模型贡献，
  确认集合竞价不是主要依赖即可。

下一组实验应优先做开盘后盘口特征工程，并补充 feature importance / permutation / ablation
报告。时间段拆分只作为稳定性诊断，不作为主评估。

## 2026-05-26 Delay2 Post-Open Feature v1

按 runbook 恢复为标准实验：`experiments/runs/lgbm_delay2_postopen_v1.toml` ->
`scripts/render_k8s_job.py` -> K8s `run_experiment.py` -> `scripts/sync_experiment_artifacts.py` ->
四宫格分析。训练仍使用已有 delay2 labeled cache：

```text
/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_labeled.parquet
```

本轮只改特征，不改 label、split、模型参数或 universe。新增
`[features].include_postopen_decision = true`，在 labeled decision rows 上追加开盘后盘口动态特征：
ask/bid 一档队列变化、10 档深度变化、depth imbalance 变化、spread/mid/ask/bid 变化、成交量/成交额变化、
top depth share、gap shape 和 trade-vs-ask1 queue。Top100 只作为评估口径。

K8s 训练完成：

```text
run:     lgbm_delay2_postopen_v1
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260526-postopen-v1
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_v1
local:   output/predictions/lgbm_delay2_postopen_v1/predictions_all.parquet
report:  output/reports/lgbm_delay2_postopen_v1_four_panel/signal_baseline_four_panel.png
```

训练 metrics：

| run | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_v1` | 203 | 0.1366 | 2.1929 | +13.32 | 52.2% | 940,748 |

相对旧 delay2 baseline 四宫格的分钟平均变化：

| metric | mean delta |
| --- | ---: |
| short Rank IC | +0.0005 |
| short Top100 excess | +2.13 bps |
| next-close Rank IC | +0.0010 |
| next-close Top100 excess | +1.31 bps |

逐分钟 short Top100 excess 大多改善，尤其 `09:32-09:39`；short Rank IC 基本持平。next-close 两条线没有
系统性变强，只能算没有明显恶化。结论：decision-level post-open 动态特征方向有效但幅度小，下一轮需要
更强的 tick-level 开盘后特征，或做 preopen/auction ablation 来释放模型容量。

## 2026-05-27 Delay2 Post-Open No-Preopen v1

按 runbook 跑 `lgbm_delay2_postopen_no_preopen_v1`。本轮仍使用已有 delay2 labeled cache，只改训练特征：
保留 `postopen_v1` 决策点盘口动态特征，同时在读取 labeled cache 后删除 `preopen_*` 特征。训练仍走
`run_experiment.py`，Top100 只作为 evaluation。

```text
run:     lgbm_delay2_postopen_no_preopen_v1
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260527-no-preopen-v1
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_no_preopen_v1
local:   output/predictions/lgbm_delay2_postopen_no_preopen_v1/predictions_all.parquet
report:  output/reports/lgbm_delay2_postopen_no_preopen_v1_four_panel/signal_baseline_four_panel.png
```

训练 metrics：

| run | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_no_preopen_v1` | 197 | 0.1354 | 2.2420 | +12.94 | 52.2% | 940,748 |

相对旧 delay2 baseline 四宫格的分钟平均变化：

| metric | mean delta |
| --- | ---: |
| short Rank IC | -0.0006 |
| short Top100 excess | +1.75 bps |
| next-close Rank IC | +0.0016 |
| next-close Top100 excess | +3.04 bps |

相对 `postopen_v1`，short Rank IC 平均低 `0.0011`，short Top100 excess 平均低 `0.38 bps`；但
next-close 两条线略好。逐分钟看，去掉 preopen 后 `09:30` 明显变弱，`09:32-09:39` 的 short
Top100 excess 仍多数高于旧 baseline。结论：完全删除 `preopen_*` 不是更强的短期方案，但后续分钟没有
崩掉，说明开盘后盘口动态本身有独立信息；下一步不应一刀切删除集合竞价，而应做轻降权/特征组 ablation
或加强 tick-level post-open 特征。

## 2026-05-27 Post-Open v2 and Feature Audit Setup

已提交两条 running 任务，均继续使用 delay2 labeled cache 和 Top100 evaluation：

- `lgbm_delay2_postopen_v2`：在 `postopen_v1` 上追加 `postopen_v2_` 特征，包括 top3/top5/top10 深度、
  depth concentration、gap slope/curve、相对开盘的队列/深度/价差轨迹、短 tick trade-vs-depth 和
  trade-vwap impact。
- `lgbm_delay2_feature_dependence_v1`：同一套 v1+v2 特征上做 grouped feature importance、
  cross-section permutation，以及 drop-retrain ablation。默认组包括 `preopen`、`postopen_v1`、
  `postopen_v2`、raw cumulative trade、trade flow、orderbook depth 和 momentum。

这两项已创建 K8s job，尚未产出 metrics；完成后需要同步 PVC 输出并把结果补进本日志。

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
未进入归档的旧 LightGBM delay Job YAML 和 run config 已清理。随后按实时 PVC 校准状态：

- 当时 PVC 可分析结果仍只有 Ridge/GBM baseline；没有可拉回的 LightGBM delay 结果目录。
- 已删除过期的本地 cache snapshot；后续以实时 PVC `find /mnt/output/opening_strength_fit` 为准。
- `*.tmp.parquet`、lock 和 heartbeat 只表示进行中或被中断，不是可用训练输入。
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
