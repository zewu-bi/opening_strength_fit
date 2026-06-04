# Experiment Log

本文件是实验事实源，适合查具体 run、数字、K8s 输出和配置。想先读当前判断，请看
[project_brief.md](project_brief.md)；这个文件不适合从头顺读。

## 当前路线摘要

当前路线：

- 项目窗口仍是 `09:30-09:40`；当前优化子域是 `09:31-09:40`。
- 训练主线改为单模型 mixed label：一分钟 VWAP short label 为主，小权重加入持有到第二天收盘的 long label。
- 训练仍用 full universe；`pool_S`、`pool_M`、`pool_L` 只作为 TopN selection mask，指标在不同 mask 下分别汇总。
- mixed label 已扫 `w_long=0.10 / 0.20 / 0.30`；结合 S/M/L 池内 Top100 excess，当前暂定
  `w_long=0.30`。固定权重后进入 S/M/L 绑定下的信号增强，next close 作为约束和诊断，不作为后续主优化目标。
- 两模型 `final_score = alpha_rank - lambda * gap_risk_rank` 路线封存。它通过 rolling，说明短+长目标有信息，
  但当前不继续用两个模型定义目标。

关键分叉结果：

| stage / score | short Top100 excess bps | next Top100 excess bps | decision |
| --- | ---: | ---: | --- |
| raw post-open baseline | +22.21 | -32.21 | short 强，但 next tail 脏。 |
| hard `next_flip_guard_10t` | +6.77 | +11.88 | 可见信息能排雷，但太防守。 |
| `guard_shrunk_target_050_v1` | +14.55 | -20.98 | clean target 有效但仍不够干净。 |
| `guard_shrunk_target_075_v1` | +6.21 | +0.07 | next 接近修复，short 掉太多。 |
| alpha-conditioned `gap_penalty_030_p80` | +16.79 | +4.49 | 两模型单月 frontier 最好。 |
| 18m rolling `gap_penalty_030_p80` | +21.20 | +7.84 | 证明短+长目标跨月有信息。 |
| S/M/L pool-internal mixed `w=0.30` | +10.0 / +12.2 / +14.1 | +6.6 / +9.0 / +9.4 | 暂定单模型主线权重；三列为 pool_S/M/L。 |

## 实验时间线

| date | stage | conclusion |
| --- | --- | --- |
| 2026-05-20 | 小窗 Ridge/GBM | 小窗 short signal 为正，值得扩到 1y。 |
| 2026-05-21 | 1y next-month baseline | 2021 训、2022-01 测，GBM Top20 `+34.33 bps`。 |
| 2026-05-22 | 本地实验清理 | 正式路径校准为 CPU LightGBM + PVC labeled cache。 |
| 2026-05-26 | LightGBM delay / replay / horizon | delay2 universe 仍为正；日频衰减明显，先不做 T+1。 |
| 2026-05-26 | mentor re-scope | 从交易约束前移到信号增强，主看 Rank IC 和 Top100。 |
| 2026-05-26/27 | post-open feature engineering | v1/v2 有增量，`postopen_v2` Rank IC `0.1394`。 |
| 2026-05-27 | target / feature diagnostics | xs demean 增益小；`09:31-09:40` baseline 成为主样本域。 |
| 2026-05-28 | guard / clean target | 可见信息能修 next tail，但直接硬塞进模型代价大。 |
| 2026-05-28/29 | learned risk layer | 两模型公式可行；conditional v1 失败，alpha-conditioned v2 改善。 |
| 2026-05-29/06-02 | rolling validation | `gap_penalty_030_p80` 6 个月 rolling 通过。 |
| 2026-06-02 | mentor re-scope | 不继续两模型，改为直接训练 single mixed label。 |
| 2026-06-03 | S/M/L mixed weight scan | `w_long=0.30` 在池内保住 short，并改善 next internal excess；暂定为主线权重。 |

## 2026-05-20 小窗结果

三组小窗实验已从 K8s PVC 拉回 `metrics_by_year.csv`，并用当前代码按
`date x decision_target_timestamp` 重算横截面指标：

| run | overall rank IC | decision rank IC | B5-B1 bps | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gbm_opening_1m_3d` | 0.1631 | 0.1426 | +21.03 | +41.92 | 62.1% |
| `ridge_opening_1m_3d_strong` | 0.1083 | 0.1087 | +23.39 | +6.78 | 49.1% |
| `ridge_opening_1m_3d` | 0.1070 | 0.0824 | +12.23 | +16.26 | 50.3% |

小窗测试期只有 2022-01-04 至 2022-01-06 三天，只能作为继续跑 one-year 主线的依据。

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

## 2026-05-27 Delay2 Post-Open v2

`lgbm_delay2_postopen_v2` 已完成并同步 PVC 输出。本轮继续使用 delay2 labeled cache、Top100 evaluation，
在 `postopen_v1` 上追加 `postopen_v2_` 特征，包括 top3/top5/top10 深度、depth concentration、
gap slope/curve、相对开盘的队列/深度/价差轨迹、短 tick trade-vs-depth 和 trade-vwap impact。

```text
run:     lgbm_delay2_postopen_v2
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260527-postopen-v2-oomfix
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_v2
local:   output/predictions/lgbm_delay2_postopen_v2/predictions_all.parquet
report:  output/reports/lgbm_delay2_postopen_v2_four_panel/signal_baseline_four_panel.png
```

训练 metrics：

| run | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_v2` | 442 | 0.1394 | 2.2940 | +14.01 | 52.5% | 940,748 |

相对旧 delay2 baseline 四宫格的分钟平均变化：

| metric | mean delta |
| --- | ---: |
| short Rank IC | +0.0034 |
| short Top100 excess | +2.82 bps |
| next-close Rank IC | +0.0008 |
| next-close Top100 excess | -2.48 bps |

短周期主线明显强于旧 delay2 baseline，也好于 `postopen_v1` / no-preopen 分支；但 next-close Top100
excess 没有同步增强。结论：v2 盘口/队列/成交冲击特征对开盘短周期有效，下一步重点看
`lgbm_delay2_feature_dependence_v1` 的 grouped importance、permutation 和 drop-retrain ablation，
确认增益主要来自哪些特征组。

`lgbm_delay2_feature_dependence_v1` 轻量版已完成：只跑 feature importance 和 cross-section permutation，
跳过 drop-retrain ablation。第一次任务卡在 node7 的 PVC read，进程处于 `D (disk sleep)`；重调度到 node8
后正常读完。中途修复了全空特征被 imputer 跳过后 importance 长度不一致的问题。
归档文件在 `experiments/results/feature_audits/lgbm_delay2_feature_dependence_v1/`。

Permutation 结果显示，v2 增益不是主要来源；模型更依赖原始盘口深度和 v1 决策点动态特征：

| group | features | rank IC drop | Top100 drop bps |
| --- | ---: | ---: | ---: |
| `orderbook_depth` | 47 | 0.0484 | 4.03 |
| `postopen_v1` | 82 | 0.0319 | 10.40 |
| `postopen_v2` | 239 | 0.0140 | 4.46 |
| `preopen` | 6 | 0.0066 | 1.67 |
| `momentum` | 4 | 0.0037 | 0.70 |
| `trade_flow` | 12 | 0.0003 | 1.64 |
| `raw_cumulative_trade` | 2 | -0.0000 | 0.03 |

结论：`postopen_v2` 有增量，但新增 239 个特征换来的依赖强度有限；下一步优先修剪/重做 v2 中全空或弱贡献特征，
并重点保留 `orderbook_depth` 和 `postopen_v1` 这两组。

## 2026-05-27 Target Alignment and v3 Direction

目标对齐先做低成本版本，但不覆盖原始收益 label。`cache_transform` 任务
`build_delay2_xs_demean_cache_v1` 已完成：读取原始 delay2 labeled cache，保留原始 `label` 用于四宫格、
Top100 bps 和 replay 评估，新增横截面去均值训练目标 `target_label`。对应训练实验
`lgbm_delay2_postopen_v2_xs_demean_v1` 已完成，仍使用统一 `09:30-09:40` 模型，不做分时建模。

cache 摘要：

```text
rows:           12,308,573
valid_rows:     12,160,999
groups:          2,882
target mean:    ~0 after cross-sectional de-mean
raw label mean: -0.001013
```

训练 metrics：

| run | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_v2_xs_demean_v1` | 442 | 0.1406 | 2.2793 | +14.05 | 52.7% | 940,748 |

相对旧 delay2 baseline 四宫格的分钟平均变化：

| metric | mean delta |
| --- | ---: |
| short Rank IC | +0.0046 |
| short Top100 excess | +2.86 bps |
| next-close Rank IC | +0.0009 |
| next-close Top100 excess | -0.87 bps |

相对 `postopen_v2`，short Rank IC 平均高 `0.0012`，short Top100 excess 平均高 `0.04 bps`，提升很小。
曲线形状上，`09:30` short Top100 excess 相对 baseline 低 `1.39 bps`，`09:31-09:39` 多数高于
baseline；这与“弱化集合竞价依赖、后续分钟靠盘口/turnover 抬升”的诊断方向一致，但幅度还不够大。

本轮暂缓 minute-specialized 模型。当前更希望观察的是：在加强盘口/turnover 信息、弱化集合竞价依赖后，
分钟曲线是否自然呈现 `09:30` 下降、后续分钟显著上升；分时模型会遮蔽这个诊断。

下一条更大的主线是 true tick-window post-open feature v3。当前 v2 仍是 decision-row minute snapshot
和分钟轨迹特征；v3 应在 raw ticks 上对每个 decision point 回看 `5s/15s/30s/60s`，提取 ask/bid queue
消耗与回补、spread persistence、depth concentration slope、turnover/trade-vwap pressure、
imbalance persistence 和 micro-price drift。该类特征不能从已有 decision-row cache 补出，应作为
ClickHouse/raw-tick cache 构建阶段的新 feature transform 嵌入，再继续复用现有 `run_experiment.py`、
K8s、sync 和四宫格评估链路。

结合 `feature_dependence_v1` 和目标对齐实验，当前判断是：先不推进 tick-window v3，也不继续在 objective
上深挖。目标对齐相对 `postopen_v2` 只带来 `+0.0012` short Rank IC、`+0.04 bps` short Top100 excess；
它支持“训练目标可横截面化”的工程能力，但不是本轮信号增益的主要来源。feature dependence 反而显示模型
最依赖 `orderbook_depth`、`postopen_v1`，其次才是 239 个 `postopen_v2` 特征；`preopen` 并不主导，
`raw_cumulative_trade` 几乎无效，`trade_flow` 对 Top100 有贡献但对 IC 弱。

下一步应做一个现有结构内的受控特征重配实验，而不是新建 raw tick window：

1. 做 `lgbm_delay2_postopen_core_v1`：保留 `orderbook_depth`、`postopen_v1`、`trade_flow` 和精选
   `postopen_v2`，去掉 `preopen`、`raw_cumulative_trade` 以及全空/弱贡献 v2 特征。
2. 补一个配置级 feature group/column include-exclude 能力，避免每轮靠临时改 cache 或手工列名。
3. 再跑一组 retrain ablation：drop `preopen`、drop `postopen_v2`、drop `trade_flow`，用来验证模型重训后
   能不能自然把权重压到盘口/成交动态上。

成功标准：`09:30` short Top100 excess 不上升，`09:31-09:40` 或 `09:32-09:40` 相对 baseline 有
明显抬升；整体 short Rank IC 不低于 `0.140`，Top100 不低于 `postopen_v2`。next-close 暂不作为主目标，
但不能比 v2 明显恶化。

## 2026-05-27 Mentor Feedback and Label-First Direction

mentor 反馈后，本轮判断需要从“先做特征重配”修正为“先把 post-open label/target 做强，特征重配作为辅助”。
约束如下：

1. ClickHouse 里 6 秒间隔通常不是数据缺失，而是上一条 3 秒 tick 所有字段都没有变化；当前不应把它当作
   stale/missing tick 剔除。
2. 真实成交不是简单 delay：如果价格已经涨上去，挂在原位置的单可能无法成交；当前阶段先不建模这个执行约束。
3. 以后可以考虑 short label + overnight label 的复合目标，但当前不做。
4. 主线仍然是把 label 做强。
5. 如果 `09:30` 是特殊 opening snapshot 区间，就先不围绕它优化；量太小，主信号应从 `09:31-09:40`
   post-open decision points 找。
6. 当前 score 看起来有“短期正、隔夜负”的拥挤/反转暴露，但历史诊断显示真正的 short winner 也可能隔夜为正。
   因此目标不是放弃 short label，而是让模型少学纯交易拥挤，多学真正强的 short signal。

下一步实验建议：

| priority | run direction | purpose |
| --- | --- | --- |
| 1 | `lgbm_delay2_postopen_0931_0940_baseline_v1` | 去掉 `09:30` 主优化目标，只在 `09:31-09:40` 建立干净 baseline；验证问题是否集中在 post-open path。 |
| 2 | `build_delay2_postopen_heat_neutral_target_v1` | 生成保留 raw `label` 的 target cache：在 `date x decision_time` 横截面内，从 short label 中削弱 price / turnover / opening-impact 暴露。 |
| 3 | `lgbm_delay2_postopen_heat_neutral_v1` | 用 heat-neutral target 训练，但仍用 raw short label 和 next-close panel 评估；目标是 short 不掉太多、next-close 负暴露收敛。 |
| 4 | `lgbm_delay2_postopen_core_after_label_v1` | 在 label 实验成立后再做 feature core：保留盘口深度、postopen_v1、trade_flow、精选 postopen_v2，剔除 preopen/raw cumulative/弱 v2。 |

建议 gate：

- 主看 `09:31-09:40` 或 `09:32-09:40`，`09:30` 只做旁路观察。
- raw short Rank IC 不低于当前 v2/xs-demean 太多，Top100 至少持平或小幅上升。
- next-close Top100 excess 的负值明显收敛；不要求立刻转正，但不能继续恶化。
- 如果 heat-neutral target 大幅损伤 short Top100，说明它把真实强势也洗掉了，需要减弱 neutralization
  或只对最明显的 chase/turnover 暴露做 shrinkage。

口径修正：`09:40` 是既有实验的正式 decision point；其 short label 使用到约 `09:41-09:42` 是预期设定，
不是出界问题。后续排除的是特殊 `09:30` opening snapshot，而不是 `09:40`。

## 2026-05-27 Post-Open 09:31-09:40 Baseline

已补工程口径：`labeled_pvc` 读取已标注 cache 后会按 `[sample].decision_times` 重新过滤
decision points，避免只改 TOML 却仍混入 `09:30`。本次 K8s 日志确认过滤生效：

```text
run:       lgbm_delay2_postopen_0931_0940_baseline_v1
image:     registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260527-postopen-0931-v1
output:    /mnt/output/opening_strength_fit/lgbm_delay2_postopen_0931_0940_baseline_v1
local:     output/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/predictions_all.parquet
dataset:   11,161,615 rows, time_min 09:31:00, time_max 09:40:05
```

训练 metrics：

| run | rows | groups | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_0931_0940_baseline_v1` | 853,881 | 190 | 442 | 0.1360 | 2.3502 | +13.45 | 52.3% |

同样限制在 `09:31-09:40` 后，与旧 score 的 raw short label 评估对比：

| score | group rank IC | Top100 mean bps | Top100 win rate |
| --- | ---: | ---: | ---: |
| `postopen_v2` old score | 0.1336 | +12.71 | 52.1% |
| `postopen_v2_xs_demean` old score | 0.1348 | +12.83 | 52.4% |
| `postopen_0931_0940_baseline` retrain | 0.1360 | +13.45 | 52.3% |

结论：排除 `09:30` 后，post-open 路径模型没有崩，反而在同一 `09:31-09:40` 样本域上小幅好于旧
统一模型。这支持下一步继续做 heat-neutral / cleaner target；但增益仍小，不能把它解读成已经解决
“短期正、隔夜负”的拥挤暴露。

## 2026-05-27 Heat-Neutral Target v1

`build_delay2_postopen_heat_neutral_target_v1` 和 `lgbm_delay2_postopen_heat_neutral_v1` 已完成。cache 使用
`rank_centered` exposure、`ridge_alpha = 10`、`neutralization_strength = 0.5`，在
`date x decision_time` 横截面内对 price / turnover / opening-impact 做 residual shrink；训练只取
`09:31-09:40`，评估仍看 raw `label`。

训练 metrics：

| run | rows | groups | features | group rank IC | rank IC IR | Top100 mean bps | Top100 win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_delay2_postopen_heat_neutral_v1` | 853,881 | 190 | 442 | 0.1245 | 2.7390 | +13.64 | 51.9% |

同口径 baseline 对比：heat-neutral v1 的 Top100 mean 略高于
`lgbm_delay2_postopen_0931_0940_baseline_v1`（+13.64 bps vs +13.45 bps），但 group Rank IC
低约 0.0115（0.1245 vs 0.1360）。这说明 50% heat shrink 没有直接伤掉 Top100，但排序面被明显洗弱；
四宫格同口径对比进一步显示结果是 mixed：

| score | short Rank IC | short Top100 excess bps | next-close Rank IC | next-close Top100 excess bps |
| --- | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | 0.1360 | +22.23 | -0.0260 | -32.21 |
| `heat_neutral_v1` | 0.1245 | +22.41 | -0.0085 | -44.23 |

解读：heat-neutral v1 保住了 short Top100 excess，并显著收敛 next-close Rank IC 的负暴露；但 short
Rank IC 下降，next-close Top100 excess 反而更负。v1 不能直接通过 gate。下一步若继续这条线，应降低
`neutralization_strength`，或只中性化最强 turnover/chase 暴露，不宜把 price、turnover、opening-impact
一起 50% shrink 当作默认目标。

## 2026-05-28 Post-Open Feature Core v1

`lgbm_delay2_postopen_core_v1` 已完成。该实验仍用 raw short `label`，只训练/评估 `09:31-09:40`，但把
可训练特征从 442 个压到 242 个：

- 保留 `orderbook_depth` 47 个、`postopen_v1` 82 个、`trade_flow` 12 个、`momentum` 4 个。
- 保留精选 `postopen_v2` 97 个：trade-vwap impact、trade-to-depth、from-open price/depth trajectory、
  depth concentration、gap shape、queue response 等。
- 排除 `preopen_*`、raw cumulative `volume/turnover`、`other` raw price/count/time 字段和弱 v2 特征。

同口径对比：

| score | features | short Rank IC | short Top100 mean bps | short Top100 excess bps | next-close Rank IC | next-close Top100 excess bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | 442 | 0.1360 | +13.45 | +22.23 | -0.0260 | -32.21 |
| `heat_neutral_v1` | 442 | 0.1245 | +13.64 | +22.41 | -0.0085 | -44.23 |
| `postopen_core_v1` | 242 | 0.1311 | +11.40 | +20.17 | -0.0304 | -43.37 |

结论：feature core v1 没过 gate。它减少了特征和 raw/auction 暴露，但 short Rank IC、Top100 mean、
short excess 都低于 `09:31-09:40` baseline，next-close 两条线也没有改善。当前不应把“硬 feature core”
作为下一条主线；更合理的方向是做 soft regularization / 小幅 drop，或只针对 next-close 负暴露最明显的
turnover/chase 特征做局部约束，而不是一次性剔除所有 `other` 和大半 v2。

## 2026-05-28 Top-Tail Guard Sweep v1

在不重训的前提下，对 `lgbm_delay2_postopen_0931_0940_baseline_v1` 的 score 做可见信息 Top100 guard
sweep。输入为 baseline predictions 和同口径 next-close labels，输出在：

```text
output/reports/lgbm_delay2_postopen_tail_guards_v1/
```

关键结果：

| variant | short Top100 excess bps | next-close Top100 excess bps | next positive minutes | min minute next excess bps |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | +22.21 | -32.21 | 0 / 10 | -61.69 |
| `mid_heat_10t` | +9.17 | -3.30 | 4 / 10 | -15.69 |
| `next_flip_guard_10t` | +6.43 | +15.80 | 10 / 10 | +3.79 |
| `next_flip_guard_10t_robust` | +5.12 | +13.55 | 10 / 10 | +7.56 |

`next_flip_guard_10t` 使用 `date x decision_time` 横截面 rank guard：
`spread_bps <= p80`、`turnover_diff_10t in [p10, p80]`、`return_10t in [p20, p70]`、
`ask_depth_10 >= p40`、`depth_imbalance_10 in [p20, p70]`。这说明 next 负 tail 不是不可动的；
只要把极端追涨、低深度/失衡和高 spread tail 从 Top100 拿掉，next-close Top100 excess 可以在测试月
出现全分钟转正。但该结果来自同一 2022-01 测试月的 post-hoc sweep，必须用跨月/滚动样本验证，不能
直接当成最终交易规则。

## 2026-05-28 Heat-Neutral v2 and Regularized v1

三组 follow-up 已跑完：tail-guard sweep、gentler heat-neutral v2 cache + train、raw-label 强正则 LGBM。

训练 metrics：

| run | features | group Rank IC | Rank IC IR | Top100 mean bps | Top100 win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | 442 | 0.1360 | 2.3502 | +13.45 | 52.3% |
| `lgbm_delay2_postopen_heat_neutral_v2` | 442 | 0.1362 | 2.6906 | +13.74 | 52.4% |
| `lgbm_delay2_postopen_regularized_v1` | 442 | 0.1341 | 2.2872 | +12.65 | 52.1% |

不画四宫格，只用同一个 cached next-close label 做最小本地 sanity check：

| score | short Top100 excess bps | next-close Rank IC | next-close Top100 excess bps | next positive minutes |
| --- | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | +22.21 | -0.0260 | -32.21 | 0 / 10 |
| `heat_neutral_v1` | +22.38 | -0.0085 | -44.23 | 0 / 10 |
| `heat_neutral_v2` | +22.48 | -0.0166 | -28.77 | 0 / 10 |
| `regularized_v1` | +21.37 | -0.0268 | -34.18 | 0 / 10 |

结论：heat-neutral v2 是比 v1 更健康的方向，short 端略高于 baseline，同时 next-close Top100 负值从
`-32.21` 收敛到 `-28.77`，但没有质变；强正则没有解决问题。真正的质变来自 Top-tail guard，而不是
这两个模型本身：同一条 `next_flip_guard_10t` 套在 heat-neutral v2 score 上，next-close Top100 excess
仍为 `+15.90 bps`，10 / 10 分钟为正。

## 2026-05-28 Guard-In-Model Attempts

为验证“把候选池构造练进树里”，跑了五个 follow-up：

1. `guard_filtered_v1`：只在固定 `next_flip_guard_10t` 候选池内训练/评估。
2. `guard_weighted_025_v1` / `guard_weighted_050_v1`：全样本训练，guard-fail 样本分别降权到
   0.25 / 0.50。
3. `guard_features_v1`：显式加入 `spread_bps`、`turnover_diff_10t`、`return_10t`、
   `ask_depth_10`、`depth_imbalance_10` 的横截面 rank 特征和 `guard_pass`。
4. `guard_feature_weighted_025_v1`：rank/pass 特征 + guard-fail 权重 0.25。

结果：

| score | features | short Rank IC | short Top100 mean bps | short Top100 excess bps | next Rank IC | next Top100 excess bps | Top100 guard-pass count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | 442 | 0.1360 | +13.43 | +22.21 | -0.0260 | -32.21 | 0.55 |
| `heat_neutral_v2` | 442 | 0.1361 | +13.70 | +22.48 | -0.0166 | -28.77 | 0.61 |
| `guard_filtered_v1` | 442 | 0.1167 | -2.11 | +5.00 | -0.0304 | -6.82 | 4.59 |
| `guard_weighted_025_v1` | 442 | 0.1365 | +12.89 | +21.67 | -0.0252 | -32.54 | 0.59 |
| `guard_weighted_050_v1` | 442 | 0.1362 | +13.14 | +21.92 | -0.0256 | -33.06 | 0.55 |
| `guard_features_v1` | 448 | 0.1333 | +12.25 | +21.03 | -0.0267 | -34.36 | 0.65 |
| `guard_feature_weighted_025_v1` | 448 | 0.1340 | +12.77 | +21.54 | -0.0270 | -34.64 | 0.81 |

结论：

- 硬候选池不是主路线：它明显削弱 short alpha，Top100 mean 变成负值。
- 单纯把 guard-fail 降权没有让树自然多选 guard-pass；Top100 里通过 guard 的股票仍不到 1 只。
- 显式加入横截面 rank / `guard_pass` 特征也没有解决，说明 raw short label 对追涨/热度 tail 的奖励仍压过
  guard 约束。
- 当前 evidence 支持：tailguard 是有效的后处理/诊断排雷器，但如果要“练进模型”，需要更强的 target
  改造，例如直接构造 clean short target 或 bad-tail penalty，而不是只加 feature 或 sample weight。

## 2026-05-28 Clean Target and Risk Sweep

本轮把 guard 信息做成两类 target，并补了 existing-score risk sweep：

- `guard_shrunk` target：以 `date x decision_time` 横截面 median 为 base，只回缩 guard-fail 且
  `label > median` 的正 excess。guard-pass 样本不动，guard-fail 但 short 不强的样本也不动。
- `guard_risk_shrunk` target：把 spread、turnover heat、return chase、ask depth、depth imbalance
  的横截面 rank threshold 转成连续 dirty risk，再按 `lambda * risk * positive_excess` 回缩。
- `score_risk_sweep_guard_shrunk_v1`：不重训模型，对已有 score 计算 `alpha_rank - penalty * dirty_risk`，
  同时扫 `next_flip_guard_10t` hard gate 和 `dirty_risk <= threshold` hard gate。

相关工程变更：

- 新增 `src/opening_strength_fit/targets.py`，统一生成 `target_label`、保留 raw `label`，并输出
  `label_xs_*` / guard / risk 诊断列。
- 新增 `scripts/build_target_label_cache.py` 和 `scripts/run_score_risk_sweep.py`；`render_k8s_job.py`、
  `audit_experiments.py`、`check_workflow_coverage.py` 已识别 `cache_transform`、`target_cache`、
  `score_risk_sweep` run kind。
- 训练链路支持 `[model].target_col`、`sample_weight_col`、feature include/drop 过滤、guard rank/pass
  特征、guard-fail sample weight，以及 labeled PVC 读取后的 decision-time 过滤。
- candidate filter 支持 rank upper bound 和可复用 mask；K8s helper pod 名称支持 63 字符截断哈希，
  避免长 run id 导致 pod name 非法。
- 测试补了 target-label、candidate guard、feature filter、K8s helper、labeled PVC time-filter 覆盖。

目标公式：

```text
base = median(label | date, decision_time)
positive_excess = max(label - base, 0)

guard_shrunk:
  target_label = label - penalty * 1[dirty] * positive_excess

guard_risk_shrunk:
  target_label = label - lambda * dirty_risk * positive_excess
```

模型级四宫格摘要：

| score | short Top100 excess bps | next Top100 excess bps | next positive minutes | Top100 guard-pass count |
| --- | ---: | ---: | ---: | ---: |
| `postopen_0931_0940_baseline` | +22.21 | -32.21 | 0 / 10 | ~1 |
| `guard_shrunk_target_050_v1` | +14.55 | -20.98 | 0 / 10 | ~36 |
| `guard_shrunk_target_060_v1` | +10.47 | -13.13 | 0 / 10 | ~57 |
| `guard_shrunk_target_065_v1` | +8.49 | -8.92 | 1 / 10 | ~66 |
| `guard_shrunk_target_075_v1` | +6.21 | +0.07 | 5 / 10 | ~80 |
| `guard_risk_shrunk_target_075_v1` | +19.95 | -25.60 | 0 / 10 | ~5 |
| `guard_risk_shrunk_target_100_v1` | +18.80 | -16.87 | 1 / 10 | ~9 |

解读：

- 二元 clean target 方向有效，但代价清楚：penalty 越强，next 越收敛、guard-pass 越多，short excess
  同步从 +22 bps 掉到 +6 bps。它证明 guard 信息能改变模型排序，但不适合作为下一轮 alpha 主目标。
- 连续 risk-shrunk target 更温和，保住了 short Rank IC 和 Top100，但 Top100 仍没有明显进入干净风险区，
  next-close 仍为负。当前这版 risk target 不够像一个否决层。
- clean target 和 risk penalty 叠加会双重惩罚 dirty tail，容易变成过度防守；它可以当诊断，不应作为主线。

`score_risk_sweep_guard_shrunk_v1` 的关键结果：

| base score | variant | short Top100 excess bps | next Top100 excess bps | next positive minutes | Top100 guard-pass count |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | `alpha_rank` | +22.21 | -32.21 | 0 / 10 | 0.98 |
| baseline | `risk_penalty_075` | +10.91 | +1.99 | 6 / 10 | 27.58 |
| baseline | `risk_penalty_100` | +9.88 | +1.92 | 7 / 10 | 34.11 |
| baseline | `hard_gate_next_flip_guard_10t` | +6.77 | +11.88 | 9 / 10 | 100.00 |
| `guard_shrunk_050` | `alpha_rank` | +14.54 | -20.98 | 0 / 10 | 36.31 |
| `guard_shrunk_050` | `risk_penalty_075` | +5.86 | +17.61 | 10 / 10 | 82.22 |
| `guard_shrunk_075` | `alpha_rank` | +6.21 | +0.07 | 5 / 10 | 79.86 |
| `guard_shrunk_075` | `risk_penalty_075` | +4.36 | +19.11 | 10 / 10 | 94.46 |

结论：manual dirty-risk penalty 叠在 baseline score 上，已经能把 next-close 从系统性负值拉到小正，同时 short
excess 仍保留约 +10 bps；hard gate 更干净但 short 更弱。这个结果不是最终策略，因为 risk 还是手工公式，
但它支持下一步优先验证 baseline alpha + learned risk layer；clean target / risk-shrunk target 暂时作为
诊断和对照，不继续作为主 alpha target 扩大。

## 2026-05-28 Learned Risk Layer v1

本轮按 runbook 在集群完成三步：

```text
alpha_model = raw short-label post-open baseline
risk_model  = learned dirty-risk / next-flip layer
final_score = alpha_score - lambda * risk_score
```

新增能力：

- 新增 `scripts/run_learned_risk_layer.py`，训练只用 decision point 当时及以前可见特征的 risk model。
- `scripts/run_score_risk_sweep.py` 支持外部 learned-risk predictions，并可对 risk score 做 group rank transform。
- `scripts/sync_experiment_artifacts.py` 支持 `score_risk_sweep` artifact sync，不再需要临时拉取脚本。

三个 run：

| run | kind | result artifact |
| --- | --- | --- |
| `learned_risk_layer_guard_teacher_v1` | learned_risk_layer | `experiments/results/metrics/learned_risk_layer_guard_teacher_v1_metrics_by_year.csv` |
| `learned_risk_layer_bad_tail_v1` | learned_risk_layer | `experiments/results/metrics/learned_risk_layer_bad_tail_v1_metrics_by_year.csv` |
| `score_learned_risk_sweep_v1` | score_risk_sweep | `experiments/results/backtests/score_learned_risk_sweep_v1_summary.csv` |

Risk-model metrics：

| run | target | group rank IC | overall rank IC | note |
| --- | --- | ---: | ---: | --- |
| `learned_risk_layer_guard_teacher_v1` | `target_dirty_risk` | 0.9768 | 0.9733 | 几乎完整复现手工 dirty-risk teacher。 |
| `learned_risk_layer_bad_tail_v1` | `target_bad_tail_risk` | 0.1028 | 0.1014 | learnable，但只是弱到中等强度。 |

`target_bad_tail_risk` 的定义：

```text
short_rank = rank(label | date, decision_time)
next_rank  = rank(alpha_return_next_close | date, decision_time)
bad_tail   = max((short_rank - 0.50) / 0.50, 0)
           * max((0.50 - next_rank) / 0.50, 0)
```

`score_learned_risk_sweep_v1` 的关键结果：

| score variant | short Top100 excess bps | next Top100 excess bps | next positive minutes |
| --- | ---: | ---: | ---: |
| `alpha_rank` | +22.21 | -32.21 | 0 / 10 |
| `guard_teacher_penalty_025` | +11.13 | -2.21 | 4 / 10 |
| `guard_teacher_penalty_050` | +9.05 | +3.28 | 6 / 10 |
| `guard_teacher_penalty_075` | +8.23 | +6.45 | 8 / 10 |
| `guard_teacher_penalty_100` | +7.57 | +7.85 | 9 / 10 |
| `bad_tail_penalty_025` | +8.13 | +21.05 | 10 / 10 |
| `bad_tail_penalty_050` | +6.00 | +28.74 | 10 / 10 |
| `bad_tail_penalty_075` | +5.13 | +31.51 | 10 / 10 |
| `bad_tail_penalty_100` | +4.67 | +34.87 | 10 / 10 |

辅助诊断：

- group Spearman：`alpha_score` vs short label = +0.1360，vs next close = -0.0260。
- group Spearman：`bad_tail_score` vs short label = +0.0476，vs next close = -0.1080。
- group Spearman：`alpha_score` vs `bad_tail_score` = +0.4725，说明 dirty-tail risk 和 raw alpha 有明显重叠。
- baseline Top100 内按 `bad_tail_score` 分 5 桶，最低风险桶 short = +11.62 bps、next = +8.50 bps；
  最高风险桶 short = +46.15 bps、next = -104.70 bps。这个结果只说明 Top100 里脏尾集中，
  不能把低风险桶当作完整 Top100 模型比较。
- 小 lambda 诊断显示 `bad_tail` 只有 0.15 到 0.20 附近可能接近可用 tradeoff；lambda 0.25
  已经让 next = +21.05 bps、short = +8.13 bps，过于 next-close 化。

结论：

- 两层公式方向有效：`alpha_rank - lambda * learned_risk_rank` 可以把 baseline 的 next tail 拉回。
- `guard_teacher` 是比较平衡的 learned risk baseline，但它主要是在复现手工规则。
- `bad_tail` v1 证明有一部分 short-positive / next-negative 成分可被学习，但不能证明“短期强 alpha 能隔夜”。
  它更像在学哪些开盘后盘口和成交状态对应更好的 next close。
- 因此下一步不应奖励 B 类“next-close 好”的股票，而应在 A 类短期强势中扣掉更容易回吐的那部分。

## 2026-05-28 Next Direction: Conditional Risk Layer

下一步实验要把 bad-tail 从全样本 next-close selector 改成条件反转风险层：

```text
candidate = high_alpha or high_short_rank
risk      = reversal_risk among candidate rows
final     = alpha_rank - lambda * conditional_risk_rank
```

建议任务：

| step | run direction | purpose |
| --- | --- | --- |
| 1 | `conditional_bad_tail_risk_v1` | 只在 alpha Top 分位或 short-rank 高分位候选里训练 risk；非候选样本排除或降权。 |
| 2 | `score_conditional_risk_sweep_v1` | 小 lambda sweep：0.05、0.10、0.15、0.20、0.25；同时看 Top20 / Top50 / Top100 和 two-stage gate。 |
| 3 | `rolling_conditional_risk_validation_v1` | 跨月验证 dirty tail 和 conditional risk 是否稳定，避免 2022-01 post-hoc。 |

下一轮 gate：

- short Top100 excess 保住约 +10 bps 以上。
- next-close Top100 excess 从 -32 bps 明显收敛，目标是接近 0 或小正，不追求 next 远大于 short。
- risk 层只扣“短期强势且容易回吐”的 A 类 dirty tail；B 类 next-close 好不是 final score 的奖励项。
- learned risk 至少接近 manual risk penalty 的 short/next tradeoff，并在小 lambda 区间更平滑。

## 2026-05-28 Conditional Risk Jobs and 18m Cache

按 conditional risk 路线挂起新一轮 K8s jobs，镜像 tag：

```text
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260528-conditional-risk-v1
```

已提交 job：

| run | kind | purpose |
| --- | --- | --- |
| `conditional_bad_tail_risk_v1` | learned_risk_layer | continuous rank-gap conditional risk；非候选样本权重 0.05。 |
| `conditional_bad_tail_binary_risk_v1` | learned_risk_layer | stricter binary conditional risk；非候选样本权重 0.05。 |
| `score_conditional_risk_sweep_v1` | score_risk_sweep | 在 alpha p80 候选池内做 Top20/50/100 和 lambda 0.05-0.30 sweep；等待两个 risk prediction 输入。 |
| `build_delay2_18m_cache_v1` | labeled_cache | 纯 ClickHouse -> labeled cache，不训练模型；输出 18 个月 delay2 cache。 |

rolling validation 暂不挂。原因是当前正式 cache 只有 2021-01 至 2022-01，样本长度不适合 6 个月 rolling；
先补 2020-08 至 2022-01 的 18 个月 cache，再做 `rolling_conditional_risk_validation_v1`。

## 2026-05-29 Conditional Risk v1 Results

三项 conditional risk 任务已完成并同步到本地：

```text
experiments/results/metrics/conditional_bad_tail_risk_v1_metrics_by_year.csv
experiments/results/metrics/conditional_bad_tail_binary_risk_v1_metrics_by_year.csv
experiments/results/backtests/score_conditional_risk_sweep_v1_summary.csv
```

Risk-model metrics：

| run | target | group rank IC | overall rank IC | mean target |
| --- | --- | ---: | ---: | ---: |
| `conditional_bad_tail_risk_v1` | conditional rank gap | 0.6901 | 0.6897 | 0.0981 |
| `conditional_bad_tail_binary_risk_v1` | conditional binary | 0.4023 | 0.4021 | 0.0880 |

分数层结果没有通过。`score_conditional_risk_sweep_v1` 在 `alpha_rank >= p80` 候选池内评估：

| TopK | best acceptable? | baseline alpha short / next excess bps | best conditional observation |
| ---: | --- | ---: | --- |
| 20 | no | +50.96 / -69.43 | `conditional_binary_penalty_030` 把 next 拉到 -40.25，但 short 变成 -23.97。 |
| 50 | no | +31.55 / -50.08 | `conditional_binary_penalty_030` next 为 -47.84，short 变成 -21.73。 |
| 100 | no | +22.21 / -32.21 | baseline 本身已经是 best next；所有 conditional variants next <= -40.85，short 多数为负。 |

诊断：

- `conditional_gap_score` 在 alpha Top100 内 vs short label 的 group Spearman 约 `+0.7544`，vs next close 约 `+0.0587`。
- `conditional_binary_score` 在 alpha Top100 内 vs short label 的 group Spearman 约 `+0.7463`，vs next close 约 `+0.0584`。
- 这说明 conditional risk v1 主要学到了“短期赢家强度”，不是“短期强势中的回吐风险”。扣它会删除 short alpha，
  但不会把组合推向 next 更干净的区域。

下一步判断：

- 不继续沿用真实 `short_rank` 条件风险标签。即使 target IC 很高，也只是证明短期赢家形态可学，不代表风险层可用。
- 先等 `build_delay2_18m_cache_v1` 完成，再做 6 个月 rolling，避免继续在单个 2022-01 测试月上调参。
- 新 risk label 应该以 alpha-score / OOF-alpha 为条件，而不是以真实 short label 为条件。候选可以是
  `oof_alpha_rank >= p80`，风险可以是 `next_rank <= p40/p50`、`next residual` 或
  `next_rank - expected_next_rank(alpha_bucket)` 的负残差。
- 评估时增加一条硬门槛：risk score 在 alpha Top100 内与 raw short label 的 Spearman 不能接近
  `+0.7`；否则说明它在惩罚 alpha 本体。

18 个月 cache 随后完成，实际覆盖 `2020-08-03` 至 `2022-01-28`，365 个交易日、16,748,169 行。

## 2026-05-29 Alpha-Conditioned Risk v2 Jobs

用户指出当前仍在找规律阶段，rolling 只是验证；因此 discovery 不等待 18m cache，继续在
`opening_1y_next_month_delay2_labeled.parquet` 上做 2021 train -> 2022-01 test。

这轮改动：

- `run_learned_risk_layer.py` 新增 `alpha_conditioned_reversal` target。
- target 先在 2021 train 上拟合一个 raw short-label alpha model，再给 2021 train 和 2022-01 test 打
  `candidate_alpha_score` / `candidate_alpha_rank`。
- risk target 不再用真实 `short_rank` 定义候选，只用 `candidate_alpha_rank >= p80`。
- risk model 训练时排除 `candidate_alpha_score`、`candidate_alpha_rank` 和 sample-weight 辅助列，避免把候选定义或权重列当成特征。

已挂起镜像：

```text
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260529-alpha-conditioned-risk-v2
```

Jobs：

| run | kind | status | purpose |
| --- | --- | --- | --- |
| `alpha_conditioned_reversal_binary_risk_v2` | learned_risk_layer | completed | alpha p80 候选内学习 `next_rank <= p40` 的 hard reversal risk。 |
| `alpha_conditioned_reversal_gap_risk_v2` | learned_risk_layer | completed | alpha p80 候选内学习 bottom-half next-rank severity。 |
| `score_alpha_conditioned_risk_gate_v2` | score_risk_sweep | completed | 等待两个 risk prediction 后，在 alpha p80 候选池内做 Top20/50/100、risk gate 0.60/0.70/0.80/0.90 和 lambda 0.05-0.25 sweep。 |

期望看到：

- alpha baseline 仍是参照：Top100 short / next excess 约 `+22.21 / -32.21 bps`。
- 合格结果应保住 Top100 short 至少约 `+10 bps`，同时把 next tail 明显从 `-32 bps` 拉向 0。
- 如果 next 明显转正但 short 被打到接近 0 或负数，说明又变成 next-close selector。
- 如果 risk score 在 alpha Top100 内与 short label 相关接近 conditional v1 的 `+0.7`，说明它仍在扣 alpha 本体。

## 2026-05-29 Alpha-Conditioned Risk v2 Results

Artifacts:

```text
experiments/results/metrics/alpha_conditioned_reversal_binary_risk_v2_metrics_by_year.csv
experiments/results/metrics/alpha_conditioned_reversal_gap_risk_v2_metrics_by_year.csv
experiments/results/backtests/score_alpha_conditioned_risk_gate_v2_summary.csv
experiments/results/backtests/score_alpha_conditioned_top100_sweep_v3_summary.csv
```

Risk-model metrics:

| run | target | group rank IC | overall rank IC |
| --- | --- | ---: | ---: |
| `alpha_conditioned_reversal_binary_risk_v2` | alpha-conditioned binary next-low | 0.4121 | 0.4106 |
| `alpha_conditioned_reversal_gap_risk_v2` | alpha-conditioned next-rank gap | 0.4276 | 0.4260 |

风险分数诊断：

- 在全市场里 risk score 与 alpha prediction 仍较相关，但这是因为 alpha-conditioned target 只在 alpha 候选内激活。
- 限制到 alpha Top100/50/20 后，risk score 与 raw short label 的 group Spearman 只有约 `0.04-0.07`，
  与 next close label 稳定负相关约 `-0.10` 至 `-0.12`。
- 这与 conditional v1 明显不同；v2 不再主要惩罚 short-alpha 本体。

`score_alpha_conditioned_risk_gate_v2` 先看 Top20/50/100，结论是 Top20/50 信号很强，但策略判断应回到
Top100；hard gate 可以把 next 拉正，但会让 short 掉得过多。

Top100-only v3 随后只扫 Top100，候选池 p80/p85/p90，soft penalty 0.25-0.45，hard gate 只作对照。
关键 Top100 excess：

| score | short excess bps | next excess bps | next positive minutes |
| --- | ---: | ---: | ---: |
| raw alpha baseline | +22.21 | -32.21 | 0 / 10 |
| heat-neutral v2 + `mid_heat_10t` | +9.15 | +2.10 | 8 / 10 |
| `gap penalty 0.30`, p80 | +16.79 | +4.49 | 7 / 10 |
| `gap penalty 0.35`, p80 | +13.24 | +17.86 | 10 / 10 |
| `gap penalty 0.30`, p90 | +17.68 | +0.72 | 6 / 10 |
| `binary penalty 0.35`, p80 | +19.49 | -2.04 | 4 / 10 |

结论：

- 当前阶段主看 excess frontier，不用 actual 否定 discovery 结果；actual 留到交易成本/容量阶段。
- `gap risk + soft penalty` 是当时固定 rolling 的最好路线，`0.30` 平衡，`0.35` 防守；两者都优于简单 guard frontier。
- `binary` risk 更保 short，但 next 还没有稳定过零，只保留为 rolling 对照。
- hard gate 不是主线。
- 不再在 2022-01 单月继续调参；下一步固定参数做 rolling。

## 2026-05-29 Rolling Validation v1

18m cache 已完成：

```text
/mnt/output/opening_strength_fit/cache/opening_18m_202008_202201_delay2_labeled.parquet
date_min: 2020-08-03
date_max: 2022-01-28
rows: 16,748,169
valid_labels: 16,545,004
```

新增 rolling 入口：

```text
scripts/run_alpha_conditioned_rolling_validation.py
experiments/runs/rolling_alpha_conditioned_top100_validation_v1.toml
experiments/jobs/rolling_alpha_conditioned_top100_validation_v1_job.yaml
```

Validation design:

- Test months: `2021-08` 至 `2022-01`，共 6 个月。
- Each window: 前 12 个月训练 alpha model、gap risk model、binary risk model。
- Risk target: alpha p80 候选内的 next underperformance；gap 用 `next_rank <= p50` severity，
  binary 用 `next_rank <= p40`。
- Fixed Top100 variants:
  `alpha_rank`、`gap_penalty_030_p80`、`gap_penalty_035_p80`、
  `gap_penalty_030_p90`、`binary_penalty_035_p80`。

Rolling 通过标准：

- `gap_penalty_030_p80` 或 `gap_penalty_035_p80` 的 short Top100 excess 多数月份仍显著为正。
- next Top100 excess 相比 raw alpha baseline 大幅收敛，最好多数月份为正。
- 如果只在 2022-01 有效，说明 v3 是单月 post-hoc；回到 target 定义，不继续扩大 score sweep。

Rescue note:

- 第一版单 Job 在跑完 `2021-08`、进入 `2021-09` 后被 `OOMKilled`，原因是 6 个 rolling window 在同一
  Python 进程里串行训练，峰值/累积内存超过 256Gi。
- 采用 `xy_fit` 的 sharded 思路重渲染为 monthly shard：每个月独立 Python 子进程写
  `month_YYYY-MM/rolling_*.csv` 和 `month_YYYY-MM/predictions.parquet`，root 目录保留共享
  `clickhouse_next_close_labels.parquet`；`sync_experiment_artifacts.py --all` 负责拉取、合并 root-level
  `rolling_summary.csv` / `rolling_month_summary.csv`，并把轻量 summary 归档到 `experiments/results/backtests/`。
- 救援镜像：`registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260529-rolling-sharded-rescue`。
- Job YAML：`experiments/jobs/rolling_alpha_conditioned_top100_validation_v1_sharded_job.yaml`。
- 二次 OOM 复盘：上述 sharded YAML 仍是单 Pod 内的 shell loop，2021-08 shard 完成后同一 cgroup 进入
  2021-09，再次读全量 18m cache（约 15.2M 行、490 列）并训练，最终在 256Gi limit 下 OOMKilled。
  后续修正为 Kubernetes Indexed Job，每个月一个 Pod；PVC labeled cache 对 rolling 月份按 train+test
  日期范围做 parquet filter，并减少训练阶段的大 DataFrame copy。

## 2026-06-02 Rolling Validation v1 Results

`rolling_alpha_conditioned_top100_validation_v1` 的 6 个 monthly shard 已拉回并合并，`rolling_trace.json`
记录合并时间为 `2026-06-02T05:05:02Z`，无缺失月份。

Artifacts:

```text
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_summary.csv
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_trace.json
output/predictions/rolling_alpha_conditioned_top100_validation_v1/raw/predictions_2021-08.parquet
...
output/predictions/rolling_alpha_conditioned_top100_validation_v1/raw/predictions_2022-01.parquet
```

合并口径：

- Test months: `2021-08` 至 `2022-01`，共 6 个月。
- Groups: `1220` 个 `date x decision_time` 横截面。
- 每个测试月用前 12 个月重新训练 alpha model、gap risk model、binary risk model。
- 评估固定 Top100，score variants 不再按单月结果调参。

Overall Top100 rolling excess:

| variant | short excess bps | next excess bps | next positive months | next positive minutes | next positive group rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alpha_rank` | +24.87 | -6.91 | 2 / 6 | 2 / 10 | 48.4% |
| `gap_penalty_030_p80` | +21.20 | +7.84 | 6 / 6 | 8 / 10 | 55.8% |
| `gap_penalty_035_p80` | +17.39 | +13.25 | 6 / 6 | 10 / 10 | 58.8% |
| `gap_penalty_030_p90` | +21.77 | +6.45 | 3 / 6 | 8 / 10 | 54.5% |
| `binary_penalty_035_p80` | +22.45 | +3.64 | 3 / 6 | 5 / 10 | 53.0% |

Monthly next excess, Top100:

| test month | alpha | gap 0.30 p80 | gap 0.35 p80 | gap 0.30 p90 | binary 0.35 p80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2021-08` | -3.95 | +13.42 | +16.23 | +11.14 | +8.24 |
| `2021-09` | -34.75 | +2.67 | +16.07 | -0.54 | -10.29 |
| `2021-10` | +32.95 | +23.98 | +24.99 | +25.93 | +27.20 |
| `2021-11` | +33.37 | +9.74 | +2.81 | +13.67 | +15.15 |
| `2021-12` | -29.61 | +0.31 | +9.92 | -3.97 | -9.25 |
| `2022-01` | -33.73 | +0.15 | +13.04 | -3.75 | -4.62 |

Monthly short excess sanity:

- `alpha_rank` 每月 short excess 为 `+21.42` 至 `+29.35 bps`，说明 raw post-open alpha 在 rolling 上仍强。
- `gap_penalty_030_p80` 每月 short excess 为 `+17.05` 至 `+26.78 bps`，6 / 6 月显著为正。
- `gap_penalty_035_p80` 每月 short excess 为 `+12.81` 至 `+22.60 bps`，防守版本也没有把 short alpha 打没。

Risk-score diagnostic inside alpha Top100:

| risk score | Spearman vs short label | Spearman vs next close |
| --- | ---: | ---: |
| gap risk rank | +0.050 | -0.066 |
| binary risk rank | +0.042 | -0.055 |

该诊断按每个 `date x decision_time` 的 alpha Top100 计算后再跨月平均。相比 conditional v1 里接近
`+0.7` 的 short 相关，这轮 gap risk 没有明显学成 short-alpha 本体 proxy。

结论：

- rolling 通过。`gap_penalty_030_p80` 和 `gap_penalty_035_p80` 都不是只在 2022-01 单月有效。
- `gap_penalty_030_p80` 是两模型路线里的主候选：short excess 仍有 `+21.20 bps`，约保留 raw alpha 的 85%，且
  next excess 6 / 6 月为正。
- `gap_penalty_035_p80` 是两模型路线里的防守候选：short 降到 `+17.39 bps`，但 next excess 更稳，10 个分钟均值全部为正。
- `gap_penalty_030_p90` 和 `binary_penalty_035_p80` 更保 short，但 next 月度稳定性只有 3 / 6，暂时只保留为对照。
- rolling 结果说明短+长目标有信息；复盘后主线改为直接训练单模型 mixed label，而不是继续推进两模型
  `alpha - risk` score。

## 2026-06-02 Gap-Risk Attribution v1 Results

`gap_risk_penalized_attribution_v1` 是解释性分析，不训练新模型；它用 rolling validation 的 prediction shards
解释 `gap_penalty_030_p80` / `gap_penalty_035_p80` 到底替换了哪些 Top100 股票。

Artifacts:

```text
experiments/results/backtests/gap_risk_penalized_attribution_v1_outcomes_by_month.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1_outcomes_overall.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1_feature_exposure_overall.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1_penalized_feature_delta.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1_residual_penalized_vs_kept.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1_trace.json
```

Top100 replacement summary:

| score | replaced names / group | short excess bps | next excess bps | note |
| --- | ---: | ---: | ---: | --- |
| raw `alpha_rank` | 0.0 | +24.87 | -6.91 | short 强，但 next tail 为负。 |
| `gap_penalty_030_p80` | 46.4 | +21.20 | +7.84 | 主折中：少损失 short，next 转正。 |
| `gap_penalty_035_p80` | 62.2 | +17.39 | +13.25 | 防守版：next 更干净，short 损失更大。 |

Cohort outcome:

| variant | cohort | short excess bps | next excess bps |
| --- | --- | ---: | ---: |
| `gap_penalty_030_p80` | baseline kept | +28.98 | -0.54 |
| `gap_penalty_030_p80` | penalized out | +19.41 | -12.04 |
| `gap_penalty_030_p80` | replacement in | +11.94 | +17.17 |
| `gap_penalty_035_p80` | baseline kept | +26.85 | +6.42 |
| `gap_penalty_035_p80` | penalized out | +22.95 | -12.74 |
| `gap_penalty_035_p80` | replacement in | +11.27 | +17.46 |

特征画像：

- 被 penalty 踢出的票不是 short 垃圾票；它们 short excess 仍为正，但 next tail 更差。
- `penalized_out - baseline_kept` 暴露最明显的是 `preopen_turnover`、`preopen_volume`，其次是
  `turnover_diff_30t`、`volume_diff_30t`、`turnover_diff_10t` 等开盘成交增量。
- 在控制 `turnover_diff_10t`、`return_10t`、`spread_bps`、`ask_depth_10`、`depth_imbalance_10`、
  `preopen_turnover`、`buy_price` 后，`penalized_out` 相对 `baseline_kept` 的 next residual 仍更差：
  `gap_penalty_030_p80 = -10.91 bps`，`gap_penalty_035_p80 = -17.08 bps`。

结论：dirty tail 的主要形态是“短线 alpha 高、盘前/开盘成交拥挤、next close 容易回吐”。`gap 0.30 p80`
仍是两模型诊断里的主折中，`gap 0.35 p80` 作为防守对照；这轮结果支持把经验吸收到 single mixed label，
而不是继续推进 `alpha - risk` final score。

## 2026-06-02 Mentor Re-scope: Single-Label Mainline

和 mentor 复盘后，当前主线从“两模型 score 组合”切回“先把一个单模型 label 做强”。

新的 label 口径：

```text
short_label = 持有约一分钟后用 VWAP 卖出的收益
long_label  = 持有到第二天收盘的收益
train_label = xs_norm(short_label) + w_long * xs_norm(long_label)
```

`w_long` 只作为小权重稳定性约束，起点放在 `0.10` 附近窄扫。这个选择吸收了 rolling risk-layer
实验的经验：旧 `final_score = alpha_rank - lambda * gap_risk_rank` 其实也是在构造短+长目标，只是通过
两个模型完成；现在先直接训练一个模型，减少目标定义和评估解释的复杂度。

主线执行顺序：

| step | decision | note |
| --- | --- | --- |
| 1 | 先扫 `w_long` | 短 label 是主体，长 label 只加小成分；用 short / next 的 Rank IC 和 Top100 选权重，不要把模型练成 next-close selector。 |
| 2 | 训练 full universe | 不用 S/M/L 过滤训练，也不把 membership 当特征；pool 只作为 TopN selection mask。 |
| 3 | 按 mask 汇总同一组指标 | universe / S / M / L 只是筛选口径；同一模型在每个 mask 下分别报 short / next Rank IC 和 Top100。 |
| 4 | 固定权重后做 S/M/L 信号增强 | 重点做特征工程、训练权重和模型调参，主目标回到 S/M/L 池内 Rank IC 和池内 Top100 excess。 |

画图口径：

- baseline 一组至少 3 个柱子：`S`、`M`、`L`。
- baseline + 一个改进模型至少 6 个柱子。
- 如果加 rolling 维度，例如 3 个测试月或 3 个窗口，就是 `2 models x 3 pools x 3 windows = 18`
  个柱子；优先使用分组柱、按 pool 分面或 small multiples，避免把所有标签挤在一张横轴上。

判断原则：

- 选择 `w_long` 时，short / next 的 Rank IC 和 Top100 都要看；固定 `w_long` 后，后续信号增强不再把
  next close 当主优化目标。
- universe / S / M / L 是 selection mask，不是和 Rank IC、Top100 并列的评估指标。
- 如果模型只在训练 label 上好、换一个合理评估体系就塌，说明还没有真的把 opening signal 做强。

## 2026-06-03 Mixed Label w010 Rolling

`build_delay2_18m_mixed_w010_target_v1` 已完成，输出 18 个月 mixed-label cache：

```text
/mnt/output/opening_strength_fit/cache/opening_18m_202008_202201_delay2_mixed_w010_labeled.parquet
```

随后跑 `lgbm_delay2_18m_postopen_mixed_w010_rolling_v1`，6 个 monthly shard 覆盖
`2021-08` 至 `2022-01`，训练目标是：

```text
target_label = zscore(short label) + 0.10 * zscore(next-close label)
```

K8s sharded job 已完成并同步：

```text
job: opening-strength-lgbm-delay2-18m-postopen-mi-sharded-358fb69d
metrics: experiments/results/metrics/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_metrics_by_year.csv
predictions: output/predictions/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1/predictions_all.parquet
four panel: output/reports/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_four_panel/signal_baseline_four_panel.png
gate summary: experiments/results/backtests/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_signal_gate_summary.csv
```

同口径 short / next gate：

| variant | short Rank IC | short Top100 excess bps | next Rank IC | next Top100 excess bps | next positive minutes | next positive months |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw alpha rolling baseline | n/a | +24.87 | n/a | -6.91 | 2 / 10 | 2 / 6 |
| mixed w010 single model | +0.1619 | +25.02 | -0.0050 | -4.29 | 2 / 10 | 3 / 6 |

逐月 mixed w010 short / next Top100 excess：

| test_month | short excess bps | next excess bps |
| --- | ---: | ---: |
| 2021-08 | +29.96 | +3.07 |
| 2021-09 | +26.41 | -33.19 |
| 2021-10 | +23.68 | +27.07 |
| 2021-11 | +21.34 | +36.65 |
| 2021-12 | +26.15 | -28.46 |
| 2022-01 | +21.88 | -26.95 |

universe 口径结论：`w_long=0.10` 单模型没有损伤 short，甚至略强于 raw alpha rolling baseline；
next tail 也从 `-6.91 bps` 收敛到 `-4.29 bps`，但仍没有变干净，且 `2021-09`、`2021-12`、
`2022-01` 三个月偏负。它是 mixed label 方向的正向首证，但不足以固定权重。这个判断随后被
S/M/L selection-mask 重算进一步修正：池内 next tail 已经明显不同，后续权重扫描应以池内表为主。

## 2026-06-03 Pool Selection Re-score and w020/w030 Launch

按 mentor 给的 S/M/L 股池重算 Top100 selection-only 指标。训练和打分仍是 full universe；每个
`date x decision_time` 横截面只在对应 pool 成员里取 Top100。汇总按 6 个 rolling test month
等权平均，单位是 bps：

```text
experiments/results/backtests/pool_selection_top100_w010_vs_risk_summary.csv
experiments/results/backtests/pool_selection_top100_w010_vs_risk_month_summary.csv
output/local/pool_selection_top100_w010_vs_risk/pool_selection_group_metrics.csv
```

| pool | score | avg candidates | short Top100 excess | next Top100 excess | next positive months | next positive minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| universe | raw `alpha_rank` | 4364.0 | +24.9 | -6.9 | 2 / 6 | 2 / 10 |
| universe | mixed `w=0.10` | 4361.8 | +24.9 | -3.6 | 3 / 6 | 2 / 10 |
| universe | `gap_penalty_030_p80` | 873.4 | +21.2 | +7.8 | 6 / 6 | 8 / 10 |
| universe | `gap_penalty_035_p80` | 873.4 | +17.4 | +13.2 | 6 / 6 | 10 / 10 |
| pool_S | raw `alpha_rank` | 1265.3 | +11.1 | +26.2 | 5 / 6 | 10 / 10 |
| pool_S | mixed `w=0.10` | 1264.8 | +11.1 | +28.1 | 5 / 6 | 10 / 10 |
| pool_S | `gap_penalty_030_p80` | 284.5 | +11.3 | +27.5 | 5 / 6 | 10 / 10 |
| pool_S | `gap_penalty_035_p80` | 284.5 | +11.2 | +28.2 | 6 / 6 | 10 / 10 |
| pool_M | raw `alpha_rank` | 2099.0 | +13.0 | +21.9 | 5 / 6 | 10 / 10 |
| pool_M | mixed `w=0.10` | 2098.2 | +12.9 | +24.4 | 5 / 6 | 10 / 10 |
| pool_M | `gap_penalty_030_p80` | 432.1 | +13.1 | +25.2 | 6 / 6 | 10 / 10 |
| pool_M | `gap_penalty_035_p80` | 432.1 | +12.8 | +25.2 | 6 / 6 | 10 / 10 |
| pool_L | raw `alpha_rank` | 2932.4 | +14.7 | +17.1 | 5 / 6 | 10 / 10 |
| pool_L | mixed `w=0.10` | 2931.1 | +14.4 | +19.6 | 4 / 6 | 10 / 10 |
| pool_L | `gap_penalty_030_p80` | 572.5 | +14.3 | +20.9 | 6 / 6 | 10 / 10 |
| pool_L | `gap_penalty_035_p80` | 572.5 | +13.7 | +22.9 | 6 / 6 | 10 / 10 |

结论：S/M/L 股池本身很可能已经是有效候选域。raw `alpha_rank` 在 pool 内 next tail 已经转正，
尤其 `pool_S` 的 next Top100 excess 为 `+26.2 bps`；这说明 universe 的 dirty tail 很大一部分
被股池过滤掉了。池内 `w=0.10` 相对 raw alpha 对 next 仍有小幅改善，但 short 基本不变；旧
`alpha_rank - lambda * risk_rank` 在池内的边际收益比 universe 小得多。

因此后续权重扫描应以 pool selection 作为主表，而不是只看 universe。已挂起两组更大权重：

| run | status | note |
| --- | --- | --- |
| `build_delay2_18m_mixed_w020_target_v1` | completed | 构造 `w_long=0.20` mixed cache；K8s job 完成后已清理。 |
| `build_delay2_18m_mixed_w030_target_v1` | completed | 构造 `w_long=0.30` mixed cache；K8s job 完成后已清理。 |
| `lgbm_delay2_18m_postopen_mixed_w020_rolling_v1` | completed | 6 个 monthly shard 已完成；metrics、pool selection 和 pool-internal summary 已归档。 |
| `lgbm_delay2_18m_postopen_mixed_w030_rolling_v1` | completed | 6 个 monthly shard 已完成；metrics、pool selection 和 pool-internal summary 已归档。 |

这四个 job 使用 ConfigMap `opening-strength-mixed-w020-w030-configs` 挂载新 TOML 到 `/mnt/config`，
因为当前镜像只包含 2026-06-02 之前的 run config。已清理 K8s 上完成的
`opening-strength-build-delay2-2023-cache-v1` 和
`opening-strength-lgbm-delay2-18m-postopen-mi-sharded-358fb69d`；PVC 输出和本地记录保留。

补充 pool-internal 口径，回答“模型在 mentor 股池内部是否还有排序增量”。这里的 excess 改为：

```text
pool_internal_excess = pool 内 Top100 平均收益 - 同一 date x minute pool 全体候选平均收益
```

```text
experiments/results/backtests/pool_internal_top100_w010_vs_risk_summary.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk_month_summary.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk_mean_by_pool.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk_mean_by_pool.svg
output/reports/pool_selection_top100_w010_vs_risk/sml_model_comparison_pool_internal_big.png
```

| pool | score | pool short mean | short internal excess | pool next mean | next internal excess | next positive months | next positive minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pool_S | raw `alpha_rank` | -7.1 | +9.7 | +21.3 | +5.4 | 4 / 6 | 7 / 10 |
| pool_S | mixed `w=0.10` | -7.1 | +9.7 | +21.3 | +5.5 | 3 / 6 | 6 / 10 |
| pool_S | `gap_penalty_030_p80` | -7.1 | +9.9 | +21.3 | +6.2 | 4 / 6 | 8 / 10 |
| pool_S | `gap_penalty_035_p80` | -7.1 | +9.8 | +21.3 | +6.9 | 4 / 6 | 8 / 10 |
| pool_M | raw `alpha_rank` | -7.6 | +12.0 | +16.3 | +6.0 | 3 / 6 | 7 / 10 |
| pool_M | mixed `w=0.10` | -7.6 | +11.9 | +16.3 | +6.8 | 3 / 6 | 9 / 10 |
| pool_M | `gap_penalty_030_p80` | -7.6 | +12.0 | +16.3 | +8.7 | 5 / 6 | 10 / 10 |
| pool_M | `gap_penalty_035_p80` | -7.6 | +11.7 | +16.3 | +8.5 | 6 / 6 | 9 / 10 |
| pool_L | raw `alpha_rank` | -7.9 | +14.1 | +11.2 | +6.5 | 3 / 6 | 9 / 10 |
| pool_L | mixed `w=0.10` | -7.9 | +13.8 | +11.2 | +7.1 | 3 / 6 | 10 / 10 |
| pool_L | `gap_penalty_030_p80` | -7.9 | +13.6 | +11.2 | +9.6 | 5 / 6 | 10 / 10 |
| pool_L | `gap_penalty_035_p80` | -7.9 | +12.9 | +11.2 | +11.2 | 6 / 6 | 10 / 10 |

结论进一步分解为两层：mentor 股池本身确实有明显 next-close 正暴露，pool next mean 在
`pool_S/M/L` 分别约 `+21.3 / +16.3 / +11.2 bps`；但模型在股池内部仍有排序增量，raw alpha 的
next internal excess 仍为 `+5.4 / +6.0 / +6.5 bps`。risk penalty 在 pool 内主要改善 next internal
excess，代价是 `pool_L` short internal excess 下降更明显；`w=0.10` 的池内边际增量较温和。

按月图已重画为 4 张单独图，每张图横轴是 `2021-08` 至 `2022-01` 加 6 个月 `Mean`，上半区为
“短期收益”，下半区为“隔夜收益”，纵轴为各自候选域内部 Top100 excess：

```text
output/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/universe_top100_pool_internal_with_mean.svg
output/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_S_top100_pool_internal_with_mean.svg
output/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_M_top100_pool_internal_with_mean.svg
output/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_L_top100_pool_internal_with_mean.svg
output/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/monthly_pool_internal_3models_with_mean_plot_data.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk_universe_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk_pool_S_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk_pool_M_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk_pool_L_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk_monthly_plot_data.csv
```

补充同一四图对应的 monthly Rank IC 表：

```text
experiments/results/backtests/pool_internal_monthly_rank_ic_3models.csv
```

6 个月 mean 读法：raw / mixed `w=0.10` 的 short Rank IC 在 S/M/L 内分别约
`0.1190 / 0.1308 / 0.1400` 和 `0.1222 / 0.1342 / 0.1433`，说明短期排序能力扎实；
next Rank IC 只有 `0.001-0.005` 量级，说明隔夜不是全池强单调排序。`gap 0.30 p80`
是在 `alpha_rank >= p80` 子集内算 IC，next Rank IC 为 `0.0068 / 0.0079 / 0.0091`，
更像高 alpha 子集里的隔夜修正器，而不是 full-pool short 排序器。

## 2026-06-03 w020/w030 Completed and w030 Mainline Decision

`w_long=0.20`、`w_long=0.30` rolling jobs 已完成，相关 run TOML 状态改为 `completed`。当前主线只归档
S/M/L 池内 Top100 excess 表；相对 universe 的 pool-selection 表不作为本轮判断依据。

```text
experiments/results/metrics/lgbm_delay2_18m_postopen_mixed_w020_rolling_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_18m_postopen_mixed_w030_rolling_v1_metrics_by_year.csv
experiments/results/backtests/pool_internal_top100_w020_w030_summary.csv
experiments/results/backtests/pool_internal_top100_w020_w030_month_summary.csv
```

池内 Top100 excess，单位 bps，按 1220 个 `date x minute` group 汇总：

| pool | score | short internal excess | next internal excess | next positive months | next positive minutes |
| --- | --- | ---: | ---: | ---: | ---: |
| pool_S | mixed `w=0.20` | +9.9 | +5.8 | 4 / 6 | 9 / 10 |
| pool_S | mixed `w=0.30` | +10.0 | +5.6 | 3 / 6 | 8 / 10 |
| pool_M | mixed `w=0.20` | +12.2 | +7.3 | 3 / 6 | 10 / 10 |
| pool_M | mixed `w=0.30` | +12.3 | +7.7 | 3 / 6 | 10 / 10 |
| pool_L | mixed `w=0.20` | +14.0 | +6.8 | 3 / 6 | 9 / 10 |
| pool_L | mixed `w=0.30` | +14.2 | +8.0 | 3 / 6 | 10 / 10 |

把 raw alpha、mixed `w=0.10`、mixed `w=0.30` 放到同一张四图口径下比较，新归档：

```text
experiments/results/backtests/pool_internal_top100_w010_w030_universe_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030_pool_S_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030_pool_M_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030_pool_L_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030_monthly_plot_data.csv
output/reports/pool_selection_top100_w010_w030/monthly_pool_internal_3models/
```

四图 mean：

| pool | score | short internal excess | next internal excess |
| --- | --- | ---: | ---: |
| universe | raw alpha | +24.8 | -6.0 |
| universe | mixed `w=0.10` | +24.9 | -3.6 |
| universe | mixed `w=0.30` | +24.9 | -2.0 |
| pool_S | raw alpha | +9.7 | +5.4 |
| pool_S | mixed `w=0.10` | +9.7 | +5.5 |
| pool_S | mixed `w=0.30` | +10.0 | +6.6 |
| pool_M | raw alpha | +12.0 | +6.0 |
| pool_M | mixed `w=0.10` | +11.9 | +6.8 |
| pool_M | mixed `w=0.30` | +12.2 | +9.0 |
| pool_L | raw alpha | +14.1 | +6.5 |
| pool_L | mixed `w=0.10` | +13.8 | +7.1 |
| pool_L | mixed `w=0.30` | +14.1 | +9.4 |

判断：当前先定 `w_long=0.30`。它在 universe 里没有让 next Top100 彻底转正，但比 raw / `w=0.10`
更少负；在 S/M/L 池内，`w=0.30` 相比 raw 和 `w=0.10` 保住 short internal excess，同时提高
`pool_M/L` 的 next internal excess，`pool_S` 的 next 也有小幅改善。相比 `gap 0.30 p80`，
`w=0.30` 是单模型、full-pool 打分、解释和部署更简单；gap risk 暂保留为 tail 诊断和二阶段对照。

下一步不是继续无限扫权重，也不是切到 replay / 容量，而是把 `w_long=0.30` 当作固定训练目标，
在 S/M/L 绑定口径下做信号增强：

1. 固定评估面板：每个新模型都和 raw alpha、mixed `w=0.10`、mixed `w=0.30` baseline 在
   universe / pool_S / pool_M / pool_L 同图比较，横轴仍是 6 个 rolling month + Mean。
2. 做模型内增强：优先尝试 post-open feature cleanup / regroup、LightGBM 正则和采样参数、小幅 S/M/L
   样本权重，而不是引入独立 risk layer 或二阶段 score。
3. 验收标准：S/M/L 池内 short Top100 excess 不能低于当前 `w=0.30`，next Top100 excess 不能明显回吐；
   如果只改善 universe 而不改善 S/M/L，则不算主线增量。

按这个决策，已准备下一批 `w=0.30` 信号增强候选。它们都复用
`opening_18m_202008_202201_delay2_mixed_w030_labeled.parquet`，仍跑 `2021-08` 至 `2022-01`
6 个 rolling monthly shard，并通过 ConfigMap `opening-strength-w030-regroup-sweep-v1`
挂载到 K8s Job：

| run | status | purpose |
| --- | --- | --- |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_light_v1` | running | full postopen v2 feature set；轻度 sampling / regularization。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_mid_v1` | running | full postopen v2 feature set；中度 sampling / regularization，作为优先候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_strong_v1` | running | full postopen v2 feature set；更强正则，检查 short 是否被吃掉。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_v1` | running | soft feature regroup；保留核心盘口/开盘后特征，baseline LGBM 参数。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1` | running | soft feature regroup + 轻度正则。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_mid_v1` | running | soft feature regroup + 中度正则，作为 feature cleanup 主候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_no_preopen_reg_mid_v1` | running | soft feature regroup + 中度正则，并去掉 `preopen_*`，检查集合竞价依赖。 |

## 查找索引

下面只做定位用；研究逻辑以上面的时间线为准。

### 实验清单

下面的清单和表格是检索用，不是叙事正文。

正式归档实验：

- `1m3d` 小窗口 Ridge/GBM 对比。
- `1y_next_month` Ridge/GBM/strong 对比。
- `1y_next_month` CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。

旧 Ridge/GBM baseline 使用无成交延迟旧口径（`entry_tick_delay = 0`）；LightGBM delay 分支使用各自
PVC labeled cache 中的延迟成交 label。不同口径不要直接横向混比。

近期探索和辅助任务：

| task | kind | status | output |
| --- | --- | --- | --- |
| `lgbm_delay2_postopen_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_v1/` |
| `lgbm_delay2_postopen_no_preopen_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_no_preopen_v1/` |
| `lgbm_delay2_postopen_v2` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_v2/` |
| `lgbm_delay2_feature_dependence_v1` | feature_audit | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_feature_dependence_v1/` |
| `build_delay2_xs_demean_cache_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_xs_demean_labeled.parquet` |
| `lgbm_delay2_postopen_v2_xs_demean_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_v2_xs_demean_v1/` |
| `lgbm_delay2_postopen_0931_0940_baseline_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_0931_0940_baseline_v1/` |
| `build_delay2_postopen_heat_neutral_target_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_postopen_heat_neutral_labeled.parquet` |
| `lgbm_delay2_postopen_heat_neutral_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_heat_neutral_v1/` |
| `lgbm_delay2_postopen_core_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_core_v1/` |
| `build_delay2_postopen_heat_neutral_target_v2` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_postopen_heat_neutral_v2_labeled.parquet` |
| `lgbm_delay2_postopen_heat_neutral_v2` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_heat_neutral_v2/` |
| `lgbm_delay2_postopen_regularized_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_regularized_v1/` |
| `lgbm_delay2_postopen_guard_filtered_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_guard_filtered_v1/` |
| `lgbm_delay2_postopen_guard_weighted_025_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_guard_weighted_025_v1/` |
| `lgbm_delay2_postopen_guard_weighted_050_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_guard_weighted_050_v1/` |
| `lgbm_delay2_postopen_guard_features_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_guard_features_v1/` |
| `lgbm_delay2_postopen_guard_feature_weighted_025_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_postopen_guard_feature_weighted_025_v1/` |
| `build_guard_shrunk_target_050_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_shrunk_target_050_v1_labeled.parquet` |
| `guard_shrunk_target_050_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_shrunk_target_050_v1/` |
| `build_guard_shrunk_target_060_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_shrunk_target_060_v1_labeled.parquet` |
| `guard_shrunk_target_060_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_shrunk_target_060_v1/` |
| `build_guard_shrunk_target_065_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_shrunk_target_065_v1_labeled.parquet` |
| `guard_shrunk_target_065_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_shrunk_target_065_v1/` |
| `build_guard_shrunk_target_075_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_shrunk_target_075_v1_labeled.parquet` |
| `guard_shrunk_target_075_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_shrunk_target_075_v1/` |
| `build_guard_risk_shrunk_target_075_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_risk_shrunk_target_075_v1_labeled.parquet` |
| `guard_risk_shrunk_target_075_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_risk_shrunk_target_075_v1/` |
| `build_guard_risk_shrunk_target_100_v1` | cache_transform | completed | `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay2_guard_risk_shrunk_target_100_v1_labeled.parquet` |
| `guard_risk_shrunk_target_100_v1` | exploration | completed | `/mnt/output/opening_strength_fit/guard_risk_shrunk_target_100_v1/` |
| `score_risk_sweep_guard_shrunk_v1` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_risk_sweep_guard_shrunk_v1/` |
| `learned_risk_layer_guard_teacher_v1` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/learned_risk_layer_guard_teacher_v1/` |
| `learned_risk_layer_bad_tail_v1` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/learned_risk_layer_bad_tail_v1/` |
| `score_learned_risk_sweep_v1` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_learned_risk_sweep_v1/` |
| `conditional_bad_tail_risk_v1` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/conditional_bad_tail_risk_v1/` |
| `conditional_bad_tail_binary_risk_v1` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/conditional_bad_tail_binary_risk_v1/` |
| `score_conditional_risk_sweep_v1` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_conditional_risk_sweep_v1/` |
| `build_delay2_18m_cache_v1` | labeled_cache | completed | `/mnt/output/opening_strength_fit/cache/opening_18m_202008_202201_delay2_labeled.parquet` |
| `alpha_conditioned_reversal_binary_risk_v2` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/alpha_conditioned_reversal_binary_risk_v2/` |
| `alpha_conditioned_reversal_gap_risk_v2` | learned_risk_layer | completed | `/mnt/output/opening_strength_fit/alpha_conditioned_reversal_gap_risk_v2/` |
| `score_alpha_conditioned_risk_gate_v2` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_alpha_conditioned_risk_gate_v2/` |
| `score_alpha_conditioned_top100_sweep_v3_p80` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_alpha_conditioned_top100_sweep_v3_p80/` |
| `score_alpha_conditioned_top100_sweep_v3_p85` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_alpha_conditioned_top100_sweep_v3_p85/` |
| `score_alpha_conditioned_top100_sweep_v3_p90` | score_risk_sweep | completed | `/mnt/output/opening_strength_fit/score_alpha_conditioned_top100_sweep_v3_p90/` |
| `rolling_alpha_conditioned_top100_validation_v1` | alpha_conditioned_rolling_validation | completed | `/mnt/output/opening_strength_fit/rolling_alpha_conditioned_top100_validation_v1/` |
| `gap_risk_penalized_attribution_v1` | gap_risk_attribution | completed | `/mnt/output/opening_strength_fit/gap_risk_penalized_attribution_v1/` |

### Run 索引

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
| `lgbm_delay2_postopen_v2` | completed | post-open v1 plus v2 queue/depth-shape/trade-impact features；group rank IC = 0.1394，Top100 mean = +14.01 bps。 |
| `lgbm_delay2_feature_dependence_v1` | completed | grouped feature importance 和 permutation；postopen/orderbook 依赖强于 preopen。 |
| `build_delay2_xs_demean_cache_v1` | completed | 生成 delay2 横截面去均值 `target_label` cache，原始 `label` 保留用于评估。 |
| `lgbm_delay2_postopen_v2_xs_demean_v1` | completed | v2 特征、`target_label` 训练；group rank IC = 0.1406，Top100 mean = +14.05 bps。 |
| `lgbm_delay2_postopen_0931_0940_baseline_v1` | completed | 排除特殊 `09:30`，只训练/评估 `09:31-09:40`；group rank IC = 0.1360，Top100 mean = +13.45 bps。 |
| `build_delay2_postopen_heat_neutral_target_v1` | completed | 生成 heat-neutral shrink `target_label` cache；对 price / turnover / opening-impact 暴露做 50% residual shrink。 |
| `lgbm_delay2_postopen_heat_neutral_v1` | completed | 使用 heat-neutral `target_label` 训练，评估仍看 raw short `label`；`09:31-09:40` group rank IC = 0.1245，Top100 mean = +13.64 bps。 |
| `lgbm_delay2_postopen_core_v1` | completed | 242 个核心特征；group rank IC = 0.1311，Top100 mean = +11.40 bps，未通过 gate。 |
| `build_delay2_postopen_heat_neutral_target_v2` | completed | gentler heat-neutral cache：只中性化短窗 momentum / turnover-flow 暴露，strength = 0.25。 |
| `lgbm_delay2_postopen_heat_neutral_v2` | completed | v2 heat-neutral `target_label` 训练；short group Rank IC = 0.1362，Top100 mean = +13.74 bps。 |
| `lgbm_delay2_postopen_regularized_v1` | completed | raw-label 强正则 LGBM；short group Rank IC = 0.1341，Top100 mean = +12.65 bps，未通过 gate。 |
| `lgbm_delay2_postopen_guard_filtered_v1` | completed | 只在固定 `next_flip_guard_10t` 可见候选池内训练/评估，检验硬候选域是否还能学出 short alpha。 |
| `lgbm_delay2_postopen_guard_weighted_025_v1` | completed | 全样本 raw-label 训练；Top100 mean = +12.92 bps，next Top100 excess = -32.54 bps，未把 guard 练进 Top100。 |
| `lgbm_delay2_postopen_guard_weighted_050_v1` | completed | 全样本 raw-label 训练；Top100 mean = +13.17 bps，next Top100 excess = -33.06 bps，未通过 gate。 |
| `lgbm_delay2_postopen_guard_features_v1` | completed | 显式加入 guard rank/pass 特征；Top100 mean = +12.29 bps，next Top100 excess = -34.36 bps，未通过 gate。 |
| `lgbm_delay2_postopen_guard_feature_weighted_025_v1` | completed | guard rank/pass 特征 + fail 权重 0.25；Top100 mean = +12.81 bps，next Top100 excess = -34.64 bps，未通过 gate。 |
| `build_guard_shrunk_target_050_v1` | completed | 生成二元 guard-shrunk `target_label` cache：dirty short positive excess shrink 50%。 |
| `guard_shrunk_target_050_v1` | completed | 用 50% guard-shrunk target 训练；short Top100 excess = +14.55 bps，next Top100 excess = -20.98 bps。 |
| `build_guard_shrunk_target_060_v1` | completed | 生成二元 guard-shrunk `target_label` cache：dirty short positive excess shrink 60%。 |
| `guard_shrunk_target_060_v1` | completed | 用 60% guard-shrunk target 训练；short Top100 excess = +10.47 bps，next Top100 excess = -13.13 bps。 |
| `build_guard_shrunk_target_065_v1` | completed | 生成二元 guard-shrunk `target_label` cache：dirty short positive excess shrink 65%。 |
| `guard_shrunk_target_065_v1` | completed | 用 65% guard-shrunk target 训练；short Top100 excess = +8.49 bps，next Top100 excess = -8.92 bps。 |
| `build_guard_shrunk_target_075_v1` | completed | 生成二元 guard-shrunk `target_label` cache：dirty short positive excess shrink 75%。 |
| `guard_shrunk_target_075_v1` | completed | 用 75% guard-shrunk target 训练；short Top100 excess = +6.21 bps，next Top100 excess = +0.07 bps。 |
| `build_guard_risk_shrunk_target_075_v1` | completed | 生成连续 dirty-risk shrink `target_label` cache：lambda = 0.75。 |
| `guard_risk_shrunk_target_075_v1` | completed | 连续 risk-shrunk target 训练；short Top100 excess = +19.95 bps，next Top100 excess = -25.60 bps。 |
| `build_guard_risk_shrunk_target_100_v1` | completed | 生成连续 dirty-risk shrink `target_label` cache：lambda = 1.00。 |
| `guard_risk_shrunk_target_100_v1` | completed | 连续 risk-shrunk target 训练；short Top100 excess = +18.80 bps，next Top100 excess = -16.87 bps。 |
| `score_risk_sweep_guard_shrunk_v1` | completed | 对 baseline、guard_shrunk_050、guard_shrunk_075 的 score 做 alpha-rank minus dirty-risk penalty 和 hard-gate sweep。 |
| `learned_risk_layer_guard_teacher_v1` | completed | 学手工 dirty-risk teacher；group rank IC = 0.9768，说明手工风险形态可被可见特征平滑复现。 |
| `learned_risk_layer_bad_tail_v1` | completed | 学 short-rank 高且 next-rank 低的 bad-tail risk；group rank IC = 0.1028，learnable 但不强。 |
| `score_learned_risk_sweep_v1` | completed | baseline `alpha_rank - lambda * learned_risk_rank` sweep；guard teacher 较平衡，bad_tail v1 太像 next-close selector。 |
| `conditional_bad_tail_risk_v1` | completed | 条件 rank-gap reversal risk：short-rank >= p70 候选内学习 `max(short_rank - next_rank, 0)`；group rank IC = 0.6901。 |
| `conditional_bad_tail_binary_risk_v1` | completed | 条件 hard reversal risk：short-rank >= p80 且 next-rank <= p50；group rank IC = 0.4023。 |
| `score_conditional_risk_sweep_v1` | completed | alpha p80 候选池内扫 Top20/50/100 与 lambda 0.05-0.30；结果未通过，risk penalty 吃掉 short alpha 且未改善 Top100 next tail。 |
| `build_delay2_18m_cache_v1` | completed | 从 ClickHouse 构造 2020-08 至 2022-01 的 18 个月 delay2 labeled cache，供后续 6 个月 rolling 用。 |
| `alpha_conditioned_reversal_binary_risk_v2` | completed | alpha p80 候选内学习 `next_rank <= p40` 的 hard reversal risk；group rank IC = 0.4121。 |
| `alpha_conditioned_reversal_gap_risk_v2` | completed | alpha p80 候选内学习 bottom-half next-rank severity；group rank IC = 0.4276。 |
| `score_alpha_conditioned_risk_gate_v2` | completed | Top20/50 有强信号；Top100 需要更细 soft-penalty sweep，hard gate 不是主线。 |
| `score_alpha_conditioned_top100_sweep_v3_p80` | completed | Top100-only p80 fine sweep；`gap penalty 0.30` short/next excess = +16.79 / +4.49 bps。 |
| `score_alpha_conditioned_top100_sweep_v3_p85` | completed | Top100-only p85 fine sweep；`gap penalty 0.30` short/next excess = +16.82 / +3.25 bps。 |
| `score_alpha_conditioned_top100_sweep_v3_p90` | completed | Top100-only p90 fine sweep；`gap penalty 0.30` short/next excess = +17.68 / +0.72 bps。 |
| `rolling_alpha_conditioned_top100_validation_v1` | completed | 18m cache 上完成 2021-08 至 2022-01 rolling validation；`gap_penalty_030_p80` short/next excess = +21.20 / +7.84 bps，`gap_penalty_035_p80` = +17.39 / +13.25 bps。 |
| `gap_risk_penalized_attribution_v1` | completed | 解释 rolling Top100 替换；被踢出票偏高 `preopen_turnover` / `preopen_volume` 和开盘成交增量，`gap 0.30` 是主折中。 |
| `build_delay2_18m_mixed_w010_target_v1` | completed | 18m delay2 labeled cache 转换为 mixed target，`w_long=0.10`。 |
| `lgbm_delay2_18m_postopen_mixed_w010_rolling_v1` | completed | single mixed-label rolling 首证；short / next Top100 excess = +25.02 / -4.29 bps，保住 short 但 next tail 仍未转正。 |
| `build_delay2_18m_mixed_w020_target_v1` | completed | 18m delay2 labeled cache 转换为 mixed target，`w_long=0.20`。 |
| `lgbm_delay2_18m_postopen_mixed_w020_rolling_v1` | completed | `w=0.20` single mixed-label rolling；S/M/L pool-internal short / next excess = +9.9/+5.8、+12.2/+7.3、+14.0/+6.8 bps。 |
| `build_delay2_18m_mixed_w030_target_v1` | completed | 18m delay2 labeled cache 转换为 mixed target，`w_long=0.30`。 |
| `lgbm_delay2_18m_postopen_mixed_w030_rolling_v1` | completed | 暂定主线权重；S/M/L pool-internal short / next excess = +10.0/+5.6、+12.3/+7.7、+14.2/+8.0 bps。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_light_v1` | running | 固定 `w=0.30` 后的 full postopen v2 轻正则候选；需按 S/M/L pool-internal 面板验收。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_mid_v1` | running | 固定 `w=0.30` 后的 full postopen v2 中正则候选；优先看是否保住 S/M/L short。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_strong_v1` | running | 固定 `w=0.30` 后的 full postopen v2 强正则候选；用于测正则上限。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_v1` | running | soft feature regroup baseline；减少宽泛 postopen/preopen 暴露后重测。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1` | running | soft feature regroup + 轻正则候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_mid_v1` | running | soft feature regroup + 中正则候选；feature cleanup 主候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_no_preopen_reg_mid_v1` | running | soft feature regroup + 中正则并去掉 `preopen_*`；诊断集合竞价依赖。 |

## 2026-06-03 Cache v2 Rebuild Prep

用户确认主线改为 3 年训练、12 个月滚动测试，基础数据需要覆盖 2015-2024。按新口径处理：

- 停止仍在跑的 `opening-strength-build-delay2-2024-cache-v1`。
- 提交 ClickHouse tick JSON object 序列化修复：`8149ad8 Handle JSON object fields in ClickHouse ticks`。
- PVC cache 清理：删除旧 `opening_24m_202301_202412_delay2_labeled/`、旧 1y delay0/1/2 cache、
  guard / heat-neutral / xs-demean 派生 cache、18m base cache、18m mixed w010/w020 cache；暂留
  `opening_18m_202008_202201_delay2_mixed_w030_labeled.parquet`，因为仍有 w030 rolling shard 在跑。
- 新增 `opening_strength_fit.cache_manifest`，`build_labeled_cache.py` 会写 `<cache>.manifest.json`。
- 新增 `scripts/inspect_labeled_cache.py`，用于在 PVC 上轻量检查 schema / row count / required columns。
- 新增年度基础 cache v2 run/job：
  `build_delay2_2015_cache_v2` 至 `build_delay2_2024_cache_v2`，统一写到
  `/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_base_labeled_v2/`。
- labeled PVC 读取顺序改为先在完整 cache 上构造 postopen 特征，再按实验 `decision_times` 过滤；
  因此只看 `09:31-09:40` 时仍能让 `09:31` 使用 cache 中的 `09:30` context。

## 2026-06-04 Cache v2 2021-2024 Pull

`opening-strength-fit-20260603-cache-v2` 已从干净 `HEAD` build/push，digest：

```text
sha256:612d8dcb5389e26094d5118ceeddc289e1483ef357f1a9f0b3ed116c837845b6
```

已分批启动并完成四个年度 base labeled cache v2 job：

| year | job | PVC parquet size | date range | rows | valid labels | columns |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 2021 | `build_delay2_2021_cache_v2` | 4.49 GB | 2021-01-04 -> 2021-12-31 | 11,359,920 | 11,220,250 | 168 |
| 2022 | `build_delay2_2022_cache_v2` | 4.89 GB | 2022-01-04 -> 2022-12-30 | 12,410,708 | 12,287,506 | 168 |
| 2023 | `build_delay2_2023_cache_v2` | 5.04 GB | 2023-01-03 -> 2023-12-29 | 13,108,173 | 13,018,257 | 168 |
| 2024 | `build_delay2_2024_cache_v2` | 5.07 GB | 2024-01-02 -> 2024-12-31 | 13,387,526 | 13,279,131 | 168 |

PVC 目录：

```text
/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_base_labeled_v2/
```

每个 shard 都有 `.parquet`、`.parquet.lock.done` 和 `.parquet.manifest.json`；manifest 检查
`missing_required=[]`。该目录当前约 19 GB。后续 2024 测试的 36m train + 12m rolling 已有
`2021-2024` 数据底座；`2015-2020` 仍待分批启动，用于更早测试窗口和更长历史稳健性。

## 2026-06-04 Next-Close / Mixed Target Prep

为 36m rolling 前置闭环拆出 `scripts/build_next_close_labels.py`：它只负责从 labeled decision rows
读取 `buy_price`，用 ClickHouse close price 缓存 `alpha_return_next_close`；`plot_signal_baseline_panels.py`
改为消费该缓存逻辑，画图不再拥有 label 获取逻辑。

新增年度前置 run/job：

- `build_delay2_2021_next_close_labels_v1` 至 `build_delay2_2024_next_close_labels_v1`，输出到
  `/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_next_close_labels_v1/`。
- `build_delay2_2021_mixed_w030_target_v1` 至 `build_delay2_2024_mixed_w030_target_v1`，输出到
  `/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_mixed_w030_labeled_v1/`。
- `lgbm_delay2_36m_visible_mixed_w030_2024_smoke_v1` 是单月 smoke：`2021-01` 至 `2023-12`
  train，`2024-01` test。它验证 36m rolling 输入链路；正式 2024 全年 12 shard 应迁移
  2026-06-04 小缓存筛出的 `soft_core_reg_light` feature/model 口径。

执行留痕：

- `opening-strength-fit-20260604-next-close-v1` 首轮 next-close job 暴露配置 CLI 字段名问题；
  `opening-strength-fit-20260604-next-close-v2` 修复后，`2021-2024` next-close label cache 全部完成。
- `2021-2024` mixed-w030 target cache 随后全部完成。PVC metadata 检查显示 next-close 年度文件为
  4 列、`0.07-0.08 GiB`；mixed 年度文件为 184 列、`4.81-5.45 GiB`，`target_label`
  非空数分别为 `10,167,661 / 11,154,082 / 11,823,254 / 12,049,972`。
- 原 full-feature 36m smoke 在读取阶段 RSS 接近 `489 GiB` 且尚未写输出，已停止。改为
  `opening-strength-fit-20260604-next-close-v5`：labeled PVC 读取支持按 include feature filters 做列投影，
  并对目录输入按年度文件逐个 transform 后再 concat；smoke 配置同步迁移到 `soft_core_reg_light`，
  训练输入启用 `downcast_float32`。

### Output 索引

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
| `output/local/score_learned_risk_sweep_v1` | learned-risk sweep artifact；轻量 summary 归档到 `experiments/results/backtests/score_learned_risk_sweep_v1_summary.csv`。 |
| `experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_*` | 18m rolling validation 的轻量 summary / month summary / trace；可重画 short-vs-next tradeoff 图。 |
| `experiments/results/backtests/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_signal_gate_summary.csv` | mixed label `w_long=0.10` rolling short / next gate 摘要。 |
| `experiments/results/backtests/pool_internal_top100_w010_vs_risk_*` | raw alpha、mixed `w=0.10`、gap-risk score 的 pool-internal summary / month summary / mean-by-pool 图表。 |
| `experiments/results/backtests/pool_internal_top100_w020_w030_*` | `w=0.20 / 0.30` 在 S/M/L 内部的 Top100 excess summary / month summary。 |
| `experiments/results/backtests/pool_internal_top100_w010_w030_*` | raw alpha、mixed `w=0.10`、mixed `w=0.30` 的四张 monthly pool-internal SVG 和 plot data。 |
| `experiments/results/backtests/pool_internal_monthly_rank_ic_3models.csv` | raw alpha、mixed `w=0.10`、`gap 0.30 p80` 对应四图的 monthly Rank IC 表。 |
| `experiments/results/backtests/gap_risk_penalized_attribution_v1_*` | rolling gap-risk Top100 替换归因的 outcome、feature exposure 和 residual-control 证据。 |
| `output/local/<run_id>` | ignored artifact-sync buffer；不再作为唯一证据位置。 |
| `output/predictions/rolling_alpha_conditioned_top100_validation_v1/raw` | 18m rolling 各测试月 prediction shard，用于 alpha Top100 内 risk/short/next 相关诊断。 |
