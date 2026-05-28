# Experiment Log

本文件是实验事实源；README 和 project brief 只保留摘要。当前项目保留正式归档实验、近期探索和
辅助任务记录。

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
但它强烈支持下一步做 baseline alpha + learned risk layer，而不是继续把 alpha target 洗得越来越保守。

## 2026-05-28 Next Direction: Learned Risk Layer

下一轮实验目标固定为两层：

```text
alpha_model = raw short-label post-open baseline
risk_model  = learned dirty-risk / next-flip layer
final_score = alpha_score - lambda * risk_score
```

建模原则：

- alpha 模型继续用 raw short label，目标是把 1-2 分钟信号做强；不把 next-close label 混进 alpha target。
- risk 模型单独处理“短正长负”的 tail。它可以用 next-close / bad-tail label 作为监督目标，因为它的职责
  就是风险层；但 inference features 仍只允许 decision point 当时及以前可见的信息。
- `next_flip_guard_10t` 只作为 teacher、weak label 和 sanity baseline。最终希望 learned risk 比手工 guard
  更完整，能学到非线性组合和遗漏的风险形态。

建议实验顺序：

| step | run direction | purpose |
| --- | --- | --- |
| 1 | `learned_risk_layer_guard_teacher_v1` | 先用手工 guard / dirty-risk 作为 teacher，训练一个只用可见特征的 risk model，验证能否复现并平滑手工规则。 |
| 2 | `learned_risk_layer_bad_tail_v1` | 在 high-alpha 或 short-positive 样本上，用 next-close underperformance / bad-tail 作为风险标签，学习 guard 之外的回吐形态。 |
| 3 | `score_learned_risk_sweep_v1` | 对 baseline alpha 做 `alpha_rank - lambda * learned_risk` sweep，并和 manual risk sweep 同表比较。 |
| 4 | rolling / cross-month validation | 跨月份验证 guard、manual risk 和 learned risk 是否稳定，避免 2022-01 post-hoc。 |

第一轮 gate：

- baseline short Top100 excess 不能从 +22 bps 直接掉到 +5 bps 以下；优先看是否能保住约 +10 bps 以上。
- next-close Top100 excess 从 -32 bps 明显收敛，最好接近 0 或转正。
- Top100 guard-pass count 要显著高于 baseline，但不追求 100/100 的硬 gate 形态。
- clean target 暂停作为主 alpha target，只保留作对照组，避免与 risk layer 重复惩罚同一类 dirty tail。

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
