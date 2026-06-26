# Experiment Log

本文件是实验事实源：run id、数字、K8s 状态、归档路径和配置索引。当前判断只放在
[project_brief.md](project_brief.md)；命令只放在 [runbook.md](runbook.md)。这里按日期、run id
或关键词查，不适合从头顺读。

## 快速索引

- 当前 brief：`09:31-09:40`、mixed label `w_long=0.30`、baseline `soft_core_reg_light`。
- 主验收图：`experiments/results/backtests/optimization_overlay_acceptance_2022_2025/`。
- 2022-2025 baseline：`experiments/results/backtests/baseline_2022_2025_cluster/`。
- 当前信号候选：`hist_path_rank_centered`；当前干净候选：`hist_path_pruned_highdup`。
- 当前 NN 对照：3 个 `torch_mlp` MLP 变体已完成并归档；`wide_deep_h32` 待跑/待归档。
- 最新生产化证据：剪枝后 Top100 exposure、size/industry exposure、split20 capacity audits。
- 封存路线：两模型 `alpha_rank - lambda * gap_risk_rank`，只保留为历史证据。

关键分叉事实：

| stage / score | short Top100 excess bps | next Top100 excess bps | decision |
| --- | ---: | ---: | --- |
| S/M/L pool-internal mixed `w=0.30` | +10.0 / +12.2 / +14.1 | +6.6 / +9.0 / +9.4 | 固定单模型主线权重；三列为 pool_S/M/L。 |
| `soft_core_reg_light` vs `w030` baseline | +0.24 / +0.31 / +0.58 | +0.92 / -0.01 / +1.17 | 固定权重后的 feature/model 候选；三列为 pool_S/M/L 的增量。 |
| 36m `baseline` full year | +8.3 / +9.3 / +10.4 | +7.7 / +5.6 / +4.4 | 2024-01..2024-12 已同步归档；三列为 pool_S/M/L。 |
| 36m halfyear `baseline` 2020-2024 | +8.9 / +10.7 / +12.1 | +12.0 / +13.7 / +14.3 | 半年 rolling 已完成；三列为 pool_S/M/L。 |
| 2022-2025 `pool_L` second sweep | +0.017 best delta | +0.232 best delta | 9 个模型实验已完成并归档；最佳 short 增量仅 `price_path_plus` +0.017 bps，后续落实为 cross-sectional relative features 和 model ensemble。 |
| 2022-2025 fullxs batch | +0.501 best delta | +0.665 best delta | `hist_same_minute_surprise` short/next 同向改善；`path_shape_confirm` 主要改善 next；`rank_label_regression` IC 高但 Top100 变弱。 |
| 2022-2025 price / scale batch | +0.687 best delta | +0.480 best delta | `price_scale_norm` 的 `pool_L` short 增量最高；`scale_norm` 的 `pool_L` next overlay 和 capital-adjusted 累计净收益最好。 |
| 2022-2025 hist + path exact union | +0.676 best delta | +1.478 best delta | `hist_path_rank_centered` 是当前信号强度标尺。 |
| hist_path high-dup prune | -0.055 vs hist_path | +0.016 vs hist_path | 删除 26 个高重复 hist/path 特征后信号基本保留，作为 hygiene simplification 通过。 |

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
| 2026-06-03 | S/M/L mixed weight scan | `w_long=0.30` 在池内保住 short，并改善 next internal excess；固定为主线权重。 |
| 2026-06-04 | w030 feature/model sweep | 7 组 18m 小缓存对照后，`soft_core_reg_light` 晋级为当前 feature/model 候选。 |
| 2026-06-05 | 36m baseline full-year archive | 2024-01..2024-12 已同步归档；正式模型展示名定为 `baseline`，S/M/L 池内 short 和 next 均值均为正。 |
| 2026-06-05 | 36m halfyear rolling mainline running | 13y mixed cache 中 2015-2024 shard 齐备后，提交并运行 `36m train -> next 6m test` 半年 rolling，共 14 个 folds。 |
| 2026-06-08 | 36m halfyear rolling archive | PVC 上 14/14 folds 完成；补齐本地 `2024H1` shard，归档 2018-2019 universe-only 和 2020-2024 S/M/L 分析。 |
| 2026-06-09 | 2022-2025 pool_L second sweep running | 10 个 pool_L 因子增强实验和 9 个 cluster-side analysis Job 已提交；首轮 completed analysis Job 已从集群清理。 |
| 2026-06-10 | xs-relative recent-weight overnight add-on | 提交 `lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1`：横截面相对特征 + recent-regime 轻度样本权重，作为 `xs_relative` 与 `recent_regime_weight` 的交互对照。 |
| 2026-06-11 | 2022-2025 pool_L second sweep archive | 9 个模型实验训练 + pool-internal analysis 已补齐并归档；feature audit 仍在跑，不阻塞模型结论。 |
| 2026-06-11 | model ensemble archive | `lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1` 已同步归档并从集群清理；`pool_L` short / next 均低于 baseline。 |
| 2026-06-11 | xs-relative archive and retired experiment cleanup | `xs_relative_v1` / `xs_relative_recent_weight_v1` 已同步到 `experiments/results`；一个未形成正式结果的路径统计实验已删除。 |
| 2026-06-12 | fullxs and feature-audit archive | 4 个 fullxs 训练 + pool-internal analysis、feature audit、baseline prediction restore metrics 已同步归档；`hist_same_minute_surprise` 为当前最好 fullxs 候选。 |
| 2026-06-12 | mentor re-scope | 继续强化开盘短期模型；下一阶段显式做 price-regime 干预和尺度归一化特征。 |
| 2026-06-12 | price-regime / scale-normalization jobs submitted | 三组 2022-2025 fullxs 实验已挂到集群：`price_bucket`、`scale_norm`、`price_scale_norm`。 |
| 2026-06-17 | price-regime / scale-normalization archive | `price_bucket`、`scale_norm`、`price_scale_norm` 已补齐归档；`scale_norm` 综合最好，`price_scale_norm` 更偏 short。 |
| 2026-06-23 | hist + path exact-union archive | `hist_path`、`hist_path_zscore`、`rank_centered` 已补齐 metrics 和 pool-internal analysis 并归档；本批 `rank_centered` 的 short Rank IC 与 `pool_L` next excess 最好。 |
| 2026-06-23 | mentor re-scope | Top100 降为诊断口径；后续补风格暴露、风险暴露和 `10 亿` 级容量约束组合验收。 |
| 2026-06-25 | hist_path high-dup prune closeout | `hist_path_pruned_highdup` 已完成训练、pool-internal analysis、artifact sync 和 experiment alignment audit；删除 26 个高重复 hist/path 特征后信号基本保留，作为 hygiene simplification 通过。 |
| 2026-06-25 | hist_path pruned exposure audit | 已对剪枝后实验补 `pool_L` Top100 core exposure audit：选股显著偏 opening activity / turnover heat、低 spread、单票集中度不高；这被记录为开盘强势股 alpha 行为画像，而不是默认负面暴露。 |
| 2026-06-25 | hist_path pruned size / industry audit | 已接 ClickHouse 日频市值和申万行业 exposure input：`pool_L` Top100 中等偏大市值，行业上超配电子、电力设备、计算机，低配机械设备、基础化工；行业集中存在但不是单行业押注。 |
| 2026-06-25 | hist_path pruned split20 capacity audit | 正式容量口径按 `10 亿` 总资金 `/20`，即每个 `date x clock` 目标 `5000 万`。在 `pool_L`、`10% * turnover_diff_30t` 参与率和 `1%` 单票目标权重上限下，`9690/9690` 截面全满；平均吃到 top `124`、p95 top `161`、最深 top `291`。Top100 内仅 `0.7%` 截面够，top200 内 `99.3%` 够；结论是切片容量充足，但生产组合不应硬卡 Top100。 |
| 2026-06-26 | NN single-model first archive | 新增 PyTorch `torch_mlp` 训练入口和 GPU K8s 渲染；`mlp_base` / `mlp_shallow_fast` / `mlp_wide_huber` 已有 2022-2025 pool-internal 归档，`wide_deep_h32` 待跑/待归档。 |

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
`experiments/results/backtests/opening_intraday_top20_1y_next_month/`。旧普通 GBM mean cycle return 约
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
`experiments/results/metrics/`，predictions 已拉回到 `output/legacy/predictions/<run_id>/predictions_all.parquet`。

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
experiments/results/backtests/opening_intraday_lgbm_delay_replays/scenario_summary.csv
experiments/results/backtests/opening_intraday_lgbm_delay_replays/delay_scan_proxy_top20.csv
experiments/results/backtests/opening_intraday_lgbm_delay_replays/trace.json
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
experiments/results/backtests/opening_alpha_horizon_decay_delay2/0930_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2/0930_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2/open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2/open10_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2/0930_vs_open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2/close_next_close_by_decision_minute.csv
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
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260526-postopen-v1
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_v1
local:   output/legacy/predictions/lgbm_delay2_postopen_v1/predictions_all.parquet
report:  output/legacy/reports/lgbm_delay2_postopen_v1_four_panel/signal_baseline_four_panel.png
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
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260527-no-preopen-v1
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_no_preopen_v1
local:   output/legacy/predictions/lgbm_delay2_postopen_no_preopen_v1/predictions_all.parquet
report:  output/legacy/reports/lgbm_delay2_postopen_no_preopen_v1_four_panel/signal_baseline_four_panel.png
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
image:   registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260527-postopen-v2-oomfix
output:  /mnt/output/opening_strength_fit/lgbm_delay2_postopen_v2
local:   output/legacy/predictions/lgbm_delay2_postopen_v2/predictions_all.parquet
report:  output/legacy/reports/lgbm_delay2_postopen_v2_four_panel/signal_baseline_four_panel.png
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
image:     registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260527-postopen-0931-v1
output:    /mnt/output/opening_strength_fit/lgbm_delay2_postopen_0931_0940_baseline_v1
local:     output/legacy/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/predictions_all.parquet
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
output/legacy/reports/lgbm_delay2_postopen_tail_guards_v1/
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
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260528-conditional-risk-v1
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
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260529-alpha-conditioned-risk-v2
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
- 救援镜像：`registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260529-rolling-sharded-rescue`。
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
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1/summary.csv
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1/month_summary.csv
experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1/trace.json
output/legacy/predictions/rolling_alpha_conditioned_top100_validation_v1/raw/predictions_2021-08.parquet
...
output/legacy/predictions/rolling_alpha_conditioned_top100_validation_v1/raw/predictions_2022-01.parquet
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
experiments/results/backtests/gap_risk_penalized_attribution_v1/outcomes_by_month.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1/outcomes_overall.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1/feature_exposure_overall.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1/penalized_feature_delta.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1/residual_penalized_vs_kept.csv
experiments/results/backtests/gap_risk_penalized_attribution_v1/trace.json
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
| 4 | 固定权重后做 S/M/L 信号增强 | 重点做特征工程和常规模型调参，主目标回到 S/M/L 池内 Rank IC 和池内 Top100 excess。 |

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
predictions: output/legacy/predictions/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1/predictions_all.parquet
four panel: output/legacy/reports/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_four_panel/signal_baseline_four_panel.png
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
experiments/results/backtests/pool_selection_top100_w010_vs_risk/summary.csv
experiments/results/backtests/pool_selection_top100_w010_vs_risk/month_summary.csv
output/legacy/analysis/pool_selection_top100_w010_vs_risk/pool_selection_group_metrics.csv
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
experiments/results/backtests/pool_internal_top100_w010_vs_risk/summary.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk/month_summary.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk/mean_by_pool.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk/mean_by_pool.svg
output/legacy/reports/pool_selection_top100_w010_vs_risk/sml_model_comparison_pool_internal_big.png
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
output/legacy/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/universe_top100_pool_internal_with_mean.svg
output/legacy/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_S_top100_pool_internal_with_mean.svg
output/legacy/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_M_top100_pool_internal_with_mean.svg
output/legacy/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/pool_L_top100_pool_internal_with_mean.svg
output/legacy/reports/pool_selection_top100_w010_vs_risk/monthly_pool_internal_3models/monthly_pool_internal_3models_with_mean_plot_data.csv
experiments/results/backtests/pool_internal_top100_w010_vs_risk/universe_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk/pool_S_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk/pool_M_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk/pool_L_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_vs_risk/monthly_plot_data.csv
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
experiments/results/backtests/pool_internal_top100_w020_w030/summary.csv
experiments/results/backtests/pool_internal_top100_w020_w030/month_summary.csv
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
experiments/results/backtests/pool_internal_top100_w010_w030/universe_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030/pool_S_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030/pool_M_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030/pool_L_with_mean.svg
experiments/results/backtests/pool_internal_top100_w010_w030/monthly_plot_data.csv
output/legacy/reports/pool_selection_top100_w010_w030/monthly_pool_internal_3models/
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
| `lgbm_delay2_18m_postopen_mixed_w030_reg_mid_v1` | completed | full postopen v2 feature set；中度 sampling / regularization，作为优先候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_v1` | completed | soft feature regroup；保留核心盘口/开盘后特征，baseline LGBM 参数。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1` | completed | soft feature regroup + 轻度正则。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_mid_v1` | completed | soft feature regroup + 中度正则，作为 feature cleanup 主候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_no_preopen_reg_mid_v1` | completed | soft feature regroup + 中度正则，并去掉 `preopen_*`，检查集合竞价依赖。 |
| `lgbm_delay2_18m_postopen_mixed_w030_no_preopen_reg_mid_v1` | completed | full postopen v2 feature set + 中度正则，并去掉 `preopen_*`，隔离 full-feature 模型的集合竞价依赖。 |
| `lgbm_delay2_18m_postopen_mixed_w030_drop_raw_reg_mid_v1` | completed | full postopen v2 feature set + 中度正则，只丢 `volume` / `turnover` / `iopv`，隔离 raw cumulative trade 噪声。 |

## 2026-06-04 w030 Feature Regroup Sweep Results

已同步本轮保留的 7 组 completed 候选 metrics / predictions，并用同一份 next-close label cache 复评
universe / pool_S / pool_M / pool_L 池内 Top100 口径。每组都完成 `2021-08` 至 `2022-01` 的
6 / 6 rolling monthly shards；未保留的异常任务已清理，不作为 canceled 实验记录。结果文件：

```text
output/legacy/analysis/w030_regroup_analysis/pool_internal_summary.csv
output/legacy/analysis/w030_regroup_analysis/pool_internal_month_summary.csv
output/legacy/analysis/w030_regroup_analysis/pool_internal_clock_summary.csv
output/legacy/analysis/w030_regroup_analysis/pool_internal_group_metrics.csv
output/legacy/analysis/w030_regroup_analysis/pool_rank_ic_group_metrics.csv
output/legacy/analysis/w030_regroup_analysis/universe_target_metric_summary.csv
experiments/results/metrics/<run_id>_metrics_by_year.csv
output/legacy/predictions/<run_id>/predictions_all.parquet
```

主结论：`soft_core_reg_light` 是唯一值得晋级的候选。它把特征数从 442 降到 276，在 universe 和
S/M/L 池内都没有吃掉 short，同时改善 next；`reg_mid`、`soft_core_reg_mid` 和去 preopen / 只去
raw 累计成交列的变体都没有形成稳定增量。

池内 Top100 excess，单位 bps；表内为 `short / next`。baseline 是
`lgbm_delay2_18m_postopen_mixed_w030_rolling_v1`：

| variant | universe | pool_S | pool_M | pool_L |
| --- | ---: | ---: | ---: | ---: |
| baseline `w030` | +25.08 / -2.72 | +10.03 / +5.64 | +12.29 / +7.74 | +14.19 / +7.96 |
| `reg_mid` | +24.94 / -3.06 | +9.84 / +5.00 | +12.11 / +6.70 | +14.14 / +6.40 |
| `soft_core` | +25.24 / -1.33 | +10.07 / +5.84 | +12.34 / +6.76 | +14.38 / +7.60 |
| `soft_core_reg_light` | +25.75 / +0.56 | +10.27 / +6.56 | +12.60 / +7.73 | +14.77 / +9.13 |
| `soft_core_reg_mid` | +24.79 / -2.40 | +9.94 / +6.03 | +12.10 / +6.19 | +14.25 / +6.75 |
| `soft_core_no_preopen_reg_mid` | +24.80 / -2.01 | +9.81 / +6.16 | +12.10 / +6.88 | +14.10 / +5.85 |
| `full_no_preopen_reg_mid` | +24.91 / -2.41 | +9.66 / +5.77 | +11.91 / +6.28 | +14.02 / +6.12 |
| `full_drop_raw_reg_mid` | +24.95 / -2.92 | +9.96 / +5.96 | +12.09 / +6.97 | +14.13 / +6.63 |

`soft_core_reg_light` 相对 baseline 的变化是：universe `+0.66 / +3.29`，pool_S `+0.24 / +0.92`，
pool_M `+0.31 / -0.01`，pool_L `+0.58 / +1.17` bps。唯一不改善的是 pool_M next，幅度接近 0；
其它候选不是 short 被吃掉，就是 next 没改善。

universe target 指标提供同一方向的辅助证据：

| variant | features | target R2 | group rank IC | Top100 bps | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline `w030` | 442 | 0.03031 | 0.16061 | +16.53 | +0.00 |
| `reg_mid` | 442 | 0.02977 | 0.15995 | +16.39 | -0.14 |
| `soft_core` | 276 | 0.02992 | 0.15991 | +16.69 | +0.16 |
| `soft_core_reg_light` | 276 | 0.03090 | 0.16169 | +17.20 | +0.67 |
| `soft_core_reg_mid` | 276 | 0.02951 | 0.15934 | +16.24 | -0.30 |
| `soft_core_no_preopen_reg_mid` | 270 | 0.02925 | 0.15866 | +16.25 | -0.28 |
| `full_no_preopen_reg_mid` | 436 | 0.02960 | 0.15952 | +16.35 | -0.18 |
| `full_drop_raw_reg_mid` | 439 | 0.02978 | 0.15985 | +16.40 | -0.13 |

晋级配置口径：

- feature set：`include_preopen=true`，保留 `preopen_*`；保留核心盘口、depth / gap / imbalance、
  trade-flow diff、`postopen_` 和精选 `postopen_v2_*` 轨迹特征；显式丢
  `volume` / `turnover` / `iopv` 三个 raw 累计列。
- feature count：baseline 442 -> soft core 276；去 `preopen_*` 的 soft-core 版本为 270，但 short
  变弱，因此不能把集合竞价摘要整体删掉。
- model：`n_estimators=360`，`learning_rate=0.03`，`num_leaves=63`，
  `min_child_samples=300`，`subsample=0.9`，`colsample_bytree=0.9`，
  `reg_alpha=0.01`，`reg_lambda=1.0`，`max_bin=63`。

辅助判断：

- 只做 full-feature 中度正则 (`reg_mid`)：short / next 基本全线变差，说明 LGBM 正则不能单独扫。
- soft feature regroup 本身有小幅帮助，但不如 light regularization 叠加；中度正则偏强。
- 去掉 `preopen_*` 会伤 short，尤其 full-feature 去 preopen；preopen 不能整体剔除。
- 只丢 `volume` / `turnover` / `iopv` 不是主要矛盾；更有效的是保留盘口/开盘后核心结构、减少宽泛特征暴露。

下一步：把 `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1` 作为当前 feature/model 主线，
进入更长训练窗 / 更多月份验证；不要继续在 18m cache 上做同类小参数扫。

### 36m Rolling Migration Supplement

这轮 18m cache 的作用是筛选候选，不是最终稳健性证明。根据小缓存结果，下一轮主线应是：

- label：继续使用 mixed `w_long=0.30`，即 2026-06-03 定下的单模型目标。
- train/test：`36m train -> next 1m test`，`test_start_month=2024-01`，
  `test_end_month=2024-12`，每个 fold 只用测试月之前 36 个自然月训练。
- data：复用已完成的 `2021-2024` base / next-close / mixed-w030 cache v2/v1 线；
  当前足够覆盖 2024 全年 12 个 OOS 月份。
- feature/model：迁移 `soft_core_reg_light` 的 include/drop feature 规则和 light LGBM 参数；
  不沿用单月 smoke 的 full-feature + medium-reg 口径作为最终候选。
- evaluation：仍同时输出 universe / pool_S / pool_M / pool_L 的 short 与 next Rank IC、池内 Top100 excess、
  月度表和 Mean；pool 只做 TopN selection mask，不改变训练 universe。

已存在的 `lgbm_delay2_36m_visible_mixed_w030_2024_smoke_v1` 是输入链路 smoke：`2021-01` 至
`2023-12` train，`2024-01` test。它可以验证 36m cache / mixed target / sharded job 链路，但不是
这轮小缓存结论所选择的最终 feature/model 配置。正式 12-shard rolling 应新建 soft-core + light-reg
配置，必要时同时保留一个 36m full-feature baseline 作为同窗对照。

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
| `lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1/` |
| `lgbm_delay2_36m_2022_2025_pool_l_bagging_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_bagging_v1/` |
| `lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1` | exploration | completed | `/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1/` |

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
| `lgbm_delay2_18m_postopen_mixed_w030_rolling_v1` | completed | 固定主线权重；S/M/L pool-internal short / next excess = +10.0/+5.6、+12.3/+7.7、+14.2/+8.0 bps。 |
| `lgbm_delay2_18m_postopen_mixed_w030_reg_mid_v1` | completed | 固定 `w=0.30` 后的 full postopen v2 中正则候选；优先看是否保住 S/M/L short。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_v1` | completed | soft feature regroup baseline；减少宽泛 postopen/preopen 暴露后重测。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1` | completed | soft feature regroup + 轻正则；本轮晋级的 feature/model 候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_mid_v1` | completed | soft feature regroup + 中正则候选；feature cleanup 主候选。 |
| `lgbm_delay2_18m_postopen_mixed_w030_soft_core_no_preopen_reg_mid_v1` | completed | soft feature regroup + 中正则并去掉 `preopen_*`；诊断集合竞价依赖。 |
| `lgbm_delay2_18m_postopen_mixed_w030_no_preopen_reg_mid_v1` | completed | full postopen v2 + 中正则并去掉 `preopen_*`；对照 soft-core 去 preopen，隔离 full-feature 下的集合竞价依赖。 |
| `lgbm_delay2_18m_postopen_mixed_w030_drop_raw_reg_mid_v1` | completed | full postopen v2 + 中正则，只去掉 `volume` / `turnover` / `iopv`；隔离 raw cumulative trade 噪声。 |
| `lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1` | completed | 2022-2025 首轮试水强正则；pool_L short / next excess = +8.11 / +6.46 bps，系统性弱于 baseline。 |
| `lgbm_delay2_36m_2022_2025_pool_l_bagging_v1` | completed | 2022-2025 首轮试水重 bagging；pool_L short / next excess = +8.59 / +7.87 bps，最接近 baseline 但无增量。 |
| `lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1` | completed | 2022-2025 首轮试水去 `preopen_*` + 中正则；pool_L short / next excess = +8.35 / +7.39 bps，说明 preopen 不能整族删除。 |

## 2026-06-03 Cache v2 Rebuild Prep

用户确认主线改为 3 年训练、12 个月滚动测试，基础数据至少需要覆盖 2015-2024；当前统一使用 13y/2013-2025 cache 路径。按新口径处理：

- 停止仍在跑的 `opening-strength-build-delay2-2024-cache-v1`。
- 提交 ClickHouse tick JSON object 序列化修复：`8149ad8 Handle JSON object fields in ClickHouse ticks`。
- PVC cache 清理：删除旧 `opening_24m_202301_202412_delay2_labeled/`、旧 1y delay0/1/2 cache、
  guard / heat-neutral / xs-demean 派生 cache、18m base cache、18m mixed w010/w020 cache；暂留
  `opening_18m_202008_202201_delay2_mixed_w030_labeled.parquet`，因为仍有 w030 rolling shard 在跑。
- 新增 `opening_strength_fit.cache_manifest`，`build_labeled_cache.py` 会写 `<cache>.manifest.json`。
- 新增 `scripts/inspect_labeled_cache.py`，用于在 PVC 上轻量检查 schema / row count / required columns。
- 新增年度基础 cache v2 run/job：
  `build_delay2_2015_cache_v2` 至 `build_delay2_2024_cache_v2`，统一写到
  `/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/`。
- labeled PVC 读取顺序改为先在完整 cache 上构造 postopen 特征，再按实验 `decision_times` 过滤；
  因此只看 `09:31-09:40` 时仍能让 `09:31` 使用 cache 中的 `09:30` context。

## 2026-06-04 Cache v2 2021-2024 Pull

`20260603-cache-v2` 已从干净 `HEAD` build/push，digest：

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
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/
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
  `/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/`。
- `build_delay2_2021_mixed_w030_target_v1` 至 `build_delay2_2024_mixed_w030_target_v1`，输出到
  `/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/`。
- `lgbm_delay2_36m_visible_mixed_w030_2024_smoke_v1` 是单月 smoke：`2021-01` 至 `2023-12`
  train，`2024-01` test。它验证 36m rolling 输入链路；正式 2024 全年 12 shard 应迁移
  2026-06-04 小缓存筛出的 `soft_core_reg_light` feature/model 口径。

执行留痕：

- `20260604-next-close-v1` 首轮 next-close job 暴露配置 CLI 字段名问题；
  `20260604-next-close-v2` 修复后，`2021-2024` next-close label cache 全部完成。
- `2021-2024` mixed-w030 target cache 随后全部完成。PVC metadata 检查显示 next-close 年度文件为
  4 列、`0.07-0.08 GiB`；mixed 年度文件为 184 列、`4.81-5.45 GiB`，`target_label`
  非空数分别为 `10,167,661 / 11,154,082 / 11,823,254 / 12,049,972`。
- 原 full-feature 36m smoke 在读取阶段 RSS 接近 `489 GiB` 且尚未写输出，已停止。随后补上
  labeled PVC 列投影、年度文件逐个 transform、`downcast_float32`；完整参数版 v5 能进入训练但过慢。
  v6 将 smoke 限制为 `feature_limit=80`、`n_estimators=40`，已完成 `2024-01` 单月链路检查：
  train `33,144,997` 行、predict `1,103,613` 行、`80` 个特征。该结果只证明 cache/split/output
  链路可跑通，不作为正式收益对比。

## 2026-06-04 36m Full Rolling Preparation

按正式实验前 checklist 复核后，历史尾巴已收束：

- `osf-audit-experiments` 输出 `alignment_ok: yes`。
- `osf-check-project-contracts` 输出 `contracts_ok: yes`。
- `2021-2024` next-close label cache 已在 PVC：
  `/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/`。
- `2021-2024` mixed-w030 derived cache 已按年度 shard 存在：
  `/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/`。
  PVC 实查四个年度 parquet 大小约 `5.16 / 5.62 / 5.81 / 5.86 GB`。
- `lgbm_delay2_36m_visible_mixed_w030_2024_smoke_v1` 已完成并同步，只作为链路 smoke。

正式 12-shard run 配置如下：

```text
display alias:
baseline

config:
experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml

job manifest:
experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_sharded_job.yaml

rendered job name:
opening-strength-lgbm-delay2-36m-mixed-w030-sharded-a99705d1
```

该 run 使用 `36m train -> next 1m test`，`2024-01` 至 `2024-12` 共 12 个 monthly Indexed Job
shards，`shard_parallelism=1`，feature/model 迁移 18m 晋级的 `soft_core_reg_light`：
`276` 目标特征口径、`n_estimators=360`、`num_leaves=63`、`min_child_samples=300`、
`subsample=0.9`、`colsample_bytree=0.9`、`reg_alpha=0.01`、`reg_lambda=1.0`、`max_bin=63`。
它不复用 smoke 的 `feature_limit=80` / `n_estimators=40`。

### 36m baseline 全年结果归档

这次 36m 正式 rolling 是大正式实验的第一个模型配置，因此在说明文档、图表和面向 mentor 的汇报里统一称为
`baseline`。真实 run id 仍保留为：

```text
lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1
```

2026-06-05 使用 runbook 的 artifact sync 闭环重新同步并归档 12 个 OOS 月份：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml \
  --all
```

同步补齐了此前本地缺失的 `2024-11` / `2024-12` prediction shards，并重新合并
`predictions_all.parquet`。全年 training metrics：

| rows | test dates | symbols | features | group rank IC | rank IC IR | Top100 short mean bps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12,049,972 | 242 | 5,118 | 276 | 0.1601 | 1.7871 | +10.37 |

S/M/L pool-internal Top100 excess 汇总如下，单位 bps；表内为 `short / next`：

| display | pool_S | pool_M | pool_L |
| --- | ---: | ---: | ---: |
| `baseline` | +8.3 / +7.7 | +9.3 / +5.6 | +10.4 / +4.4 |

补拉月份单独看：

| month | pool_S | pool_M | pool_L |
| --- | ---: | ---: | ---: |
| 2024-11 | +7.4 / -23.7 | +8.2 / -25.0 | +8.8 / -26.5 |
| 2024-12 | +6.5 / +23.9 | +7.6 / +21.5 | +8.4 / +21.2 |

全年池内 short 12/12 个月为正；next 在 `pool_S/M/L` 分别为 `8/12`、`9/12`、`8/12`
个月为正。Universe Top100 的 short internal excess 为 `+19.8 bps`，但 next internal excess 为
`-14.2 bps`；因此当前结论仍以 S/M/L pool-internal 验收为主。

轻量归档：

```text
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1/pool_internal_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1/pool_internal_month_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1/pool_internal_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1/rank_ic_with_mean.svg
```

为长任务过程反馈补充两个入口：

- `osf-rolling-job-status`：把 Indexed Job pod index 映射回 rolling 月份，并打印每月 log 命令。
- `osf-sync-experiment-artifacts --allow-partial`：只同步已完成月份的 metrics/predictions，不归档到
  `experiments/results/`。
- `osf-analyze-pool-internal-top100`：对已同步月份 join 本地 next-close label cache 和
  universe / pool_S / pool_M / pool_L，输出 short/next Rank IC 与池内 Top100 excess 的
  summary、month、clock、group 四层 CSV。

2026-06-04 的准备记录保留在本节上方；当前事实以本小节的全年完成归档为准。

## 2026-06-05 36m Halfyear Rolling Mainline Running

PVC 当前实查确认 13y 三套年度 cache 已齐；halfyear run 使用其中 `2015-2024` 年度文件：

```text
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/
```

其中 mixed-w030 labeled cache 从 `opening_2015_delay2_mixed_w030_labeled_v1.parquet` 到
`opening_2024_delay2_mixed_w030_labeled_v1.parquet` 共 10 个年度文件用于本 run。抽查日期范围：
`2015` 为 `2015-01-05 -> 2015-12-31`，`2024` 为 `2024-01-02 -> 2024-12-31`。

在同一 `baseline` feature/model 口径上新增半年 rolling 主线任务：

```text
run_id:
lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1

config:
experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1.toml

job manifest:
experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_sharded_job.yaml

rendered job name:
os-lgbm-36m-2018-2024-w030-halfyear

image:
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260605-halfyear-v1

image digest:
sha256:70b8fb9c395d62e49466754837cd52da7ce5bec0778e7fc310f8148ad593f38b
```

该 run 使用 `36m train -> next 6m test`，`2018-01` 至 `2024-12` 共 14 个 halfyear Indexed
Job shards。每个 shard 只训练一次，预测对应 6 个月 OOS：

```text
2018-01..2018-06
2018-07..2018-12
2019-01..2019-06
2019-07..2019-12
2020-01..2020-06
2020-07..2020-12
2021-01..2021-06
2021-07..2021-12
2022-01..2022-06
2022-07..2022-12
2023-01..2023-06
2023-07..2023-12
2024-01..2024-06
2024-07..2024-12
```

为避免把半年测试误渲染成 84 个单月 shard，本次补充了 `test_months` / `test_stride_months`
支持：默认仍为单月 rolling；当 config 显式设置 `test_months=6`、`test_stride_months=6` 时，
K8s renderer 会生成 14 个窗口起点 shard，并把每个 shard 的 `--test-start-month` /
`--test-end-month` 传给训练入口。`osf-rolling-job-status` 和 artifact sync 也按窗口起点识别
`month_YYYY-MM/` shard 目录。

执行留痕：

```text
local checks:
ruff targeted files passed
pytest tests/test_rolling_windows.py tests/test_k8s_helpers.py tests/test_labeled_pvc_source.py -> 18 passed

k8s dry-run:
job.batch/os-lgbm-36m-2018-2024-w030-halfyear created (dry run)

apply:
job.batch/os-lgbm-36m-2018-2024-w030-halfyear created

initial status:
Running 0/14; first pod os-lgbm-36m-2018-2024-w030-halfyear-0-qp2k8 on node9
```

随后把本地 config / manifest 和正在运行的 Job 改为 4-way shard parallelism；确认资源可承载后继续提高到
7-way shard parallelism：

```text
config:
shard_parallelism = 7

patch:
job.batch/os-lgbm-36m-2018-2024-w030-halfyear patched to parallelism=4
job.batch/os-lgbm-36m-2018-2024-w030-halfyear patched to parallelism=7

status after patch:
0/14 completed; active shards 0..6
2018-01..2018-06 running on node9
2018-07..2018-12 running on node9
2019-01..2019-06 running on node15
2019-07..2019-12 running on node7
2020-01..2020-06 running on node9
2020-07..2020-12 running on node15
2021-01..2021-06 running on node7
```

运行中状态复核：

```text
checked: 2026-06-05
job: os-lgbm-36m-2018-2024-w030-halfyear
status: 1/14 succeeded; 7 active shards running
completed: 2018-01..2018-06 at 2026-06-05T06:55:43Z
running: 2018-07..2018-12, 2019-01..2019-06, 2019-07..2019-12,
         2020-01..2020-06, 2020-07..2020-12, 2021-01..2021-06,
         2021-07..2021-12
```

首个 shard 日志确认依赖文件均 ready，且第一窗读取范围正确：

```text
running ... shard test=2018-01..2018-06 index=0
date_start: 2015-01-01
date_end: 2018-06-30
```

### 2020 年前 universe-only 分析

2020 年之前没有可用的 S/M/L 股池文件，因此 `2018-01..2019-12` 已完成 shard 只按
`universe` 口径做 pool-internal Top100 / Rank IC 分析。输入只包含 4 个 pre-2020 prediction
shard，并 join 本地 `output/legacy/labels/next_close_labels_2018_2019/`：

| period | short internal excess bps | next internal excess bps | short Rank IC | next Rank IC | positive months short / next |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2018H1` | +33.10 | +22.29 | 0.324 | 0.041 | 6 / 5 |
| `2018H2` | +25.29 | -1.45 | 0.285 | 0.027 | 6 / 3 |
| `2019H1` | +31.68 | +6.14 | 0.275 | 0.035 | 6 / 3 |
| `2019H2` | +24.05 | +13.90 | 0.264 | 0.038 | 6 / 4 |
| `2018-2019 mean` | +28.43 | +10.07 | 0.287 | 0.035 | 24 / 15 |

结论：pre-2020 universe-only 下 short signal 很稳，24/24 个月 Top100 池内超额为正；
next-close 也转正但稳定性弱于 short，15/24 个月为正。轻量归档：

```text
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/pool_internal_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/pool_internal_halfyear_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/pool_internal_month_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/pool_internal_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/short_excess_rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/next_excess_rank_ic_with_mean.svg
```

### 2026-06-08 Halfyear Completion and 2020-2024 S/M/L

2026-06-08 复查集群时，`os-lgbm-36m-2018-2024-w030-halfyear` Job 已不在运行态；
PVC run 目录下 14 个 `month_YYYY-MM/` shard 均有 `metrics_by_year.csv` 和 `predictions.parquet`。
本地补拉此前缺失的 `2024-01..2024-06` prediction/metrics shard 后，重新合并：

```text
local raw prediction shards: 14/14
predictions_all.parquet rows: 70,380,134
date range: 2018-01-02 -> 2024-12-31
symbols: 5,315
```

S/M/L 股池当前覆盖 `2020-01-02` 至 `2025-12-31`，因此 `2020-2024` 可以做完整
universe / S / M / L 验收。2020-2024 summary：

| pool | short internal excess bps | next internal excess bps | short Rank IC | next Rank IC | positive months short / next |
| --- | ---: | ---: | ---: | ---: | ---: |
| universe | +22.15 | +1.93 | 0.174 | 0.011 | 60 / 29 |
| pool_S | +8.93 | +12.00 | 0.137 | 0.010 | 60 / 43 |
| pool_M | +10.66 | +13.73 | 0.149 | 0.010 | 60 / 44 |
| pool_L | +12.14 | +14.34 | 0.157 | 0.010 | 60 / 43 |

轻量归档：

```text
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/pool_internal_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/pool_internal_halfyear_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/pool_internal_year_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/pool_internal_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/short_excess_rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/next_excess_rank_ic_with_mean.svg
```

结论：halfyear mainline 已完成。2020-2024 S/M/L 下 short 60/60 月为正，next 43-44/60 月为正；
2018-2019 universe-only 也稳定为正。后续 2025 OOS extension 和 `2020-2025` rolling-window summary
已在下方小节完成并归档。

周度补充诊断：新增 `osf-plot-weekly-pool-internal`，从
`pool_internal_group_metrics.csv` 直接生成交易日等权的 weekly / 4-week rolling 图表。口径为先把
同一 `pool x date` 的多个决策点聚成日度均值，再按交易日数加权 4 周窗口，避免单交易日节假日周被自然周等权放大。
2020-2024 S/M/L 的 short 周度仍非常稳，positive weeks 分别为 `254/256`、`255/256`、`255/256`，
4 周滚动最差仍为正：pool_S / M / L = `+1.15 / +1.81 / +2.05 bps`。next 的周度稳定性弱于 short，
positive weeks 为 `154/256`、`161/256`、`160/256`，4 周滚动最差为
`-47.63 / -47.97 / -50.73 bps`。此前自然周等权看到的 universe next 约 `250 bps`
高点主要来自单交易日周；交易日等权后该 4 周峰值降至约 `174 bps`。

本地输出：

```text
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal/daily_pool_internal_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal/weekly_pool_internal_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal/weekly_pool_internal_overall_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal/weekly_worst_windows.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal/baseline_halfyear_2020_2024_universe_sml_weekly_rolling_4w/baseline_halfyear_2020_2024_universe_sml_weekly_rolling_4w.svg
```

### 2026-06-08 2025 Halfyear OOS Extension

`os-lgbm-36m-2025-w030-halfyear` 两个 halfyear shard 均已完成：

```text
2025-01..2025-06: Succeeded, finished 2026-06-08T06:35:32Z
2025-07..2025-12: Succeeded, finished 2026-06-08T06:36:45Z
```

已同步到本地：

```text
output/legacy/predictions/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1/raw/predictions_2025-01_2025-06.parquet
output/legacy/predictions/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1/raw/predictions_2025-07_2025-12.parquet
output/legacy/predictions/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1/predictions_all.parquet
output/legacy/labels/next_close_labels_2025/opening_2025_next_close_labels_v1.parquet
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_metrics_by_month.csv
```

Training metrics：2025 全年 `group_rank_ic_mean = 0.1316`，`group_rank_ic_ir = 1.4753`，
`test_rows = 12,311,200`，`test_dates = 243`，`symbols = 5,195`，features 仍为 276。

2025 universe / S / M / L pool-internal Top100 summary：

| pool | short internal excess bps | next internal excess bps | short Rank IC | next Rank IC | positive months short / next |
| --- | ---: | ---: | ---: | ---: | ---: |
| universe | +10.63 | -17.18 | 0.132 | 0.004 | 12 / 4 |
| pool_S | +5.09 | +7.62 | 0.116 | 0.000 | 12 / 8 |
| pool_M | +5.60 | +8.64 | 0.125 | 0.000 | 12 / 7 |
| pool_L | +6.19 | +8.19 | 0.129 | 0.001 | 12 / 8 |

半年拆分：

| pool | half | short internal excess bps | next internal excess bps | short Rank IC | next Rank IC | positive months short / next |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| universe | H1 | +12.75 | -22.76 | 0.135 | 0.000 | 6 / 1 |
| universe | H2 | +8.52 | -12.42 | 0.127 | 0.005 | 6 / 3 |
| pool_S | H1 | +5.09 | +2.78 | 0.128 | -0.002 | 6 / 3 |
| pool_S | H2 | +5.01 | +11.22 | 0.104 | 0.000 | 6 / 5 |
| pool_M | H1 | +5.64 | +4.24 | 0.134 | -0.001 | 6 / 3 |
| pool_M | H2 | +5.50 | +12.07 | 0.115 | -0.000 | 6 / 4 |
| pool_L | H1 | +6.34 | +3.36 | 0.137 | -0.000 | 6 / 4 |
| pool_L | H2 | +6.00 | +11.88 | 0.121 | 0.001 | 6 / 4 |

结论：2025 OOS 的 short leg 明显弱于 2020-2024，但仍保持 12/12 月为正；S/M/L 内 next-close
为正，且 H2 明显好于 H1。universe next-close 为负，说明 2025 的 next leg 更依赖池内选择口径。

周度补充诊断使用同一交易日等权口径生成：

| pool | weeks | short positive weeks | short 4w worst bps | next positive weeks | next 4w worst bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| universe | 53 | 51 | +5.90 | 18 | -63.82 |
| pool_S | 53 | 52 | +3.32 | 30 | -31.27 |
| pool_M | 53 | 52 | +3.73 | 30 | -41.65 |
| pool_L | 53 | 52 | +4.03 | 31 | -46.09 |

轻量归档：

```text
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_month_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_halfyear_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_year_summary.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_group_metrics.csv
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/pool_internal_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/short_excess_rank_ic_with_mean.svg
experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/next_excess_rank_ic_with_mean.svg
```

周度输出：

```text
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal/daily_pool_internal_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal/weekly_pool_internal_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal/weekly_pool_internal_overall_summary.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal/weekly_worst_windows.csv
output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal/baseline_halfyear_2025_universe_sml_weekly_rolling_4w/baseline_halfyear_2025_universe_sml_weekly_rolling_4w.svg
```

### 2020-2025 halfyear rolling 合并归档（三年训练、半年预测）

将主线 `36m train -> next 6m test` 的 `2020-2024` 与 `2025` OOS extension 合并成
`2020-2025` 统一视角。口径仍为 delay2、mixed `w=0.30` target、soft-core visible feature set、
light LightGBM regularization、Top100 pool-internal 评估；S/M/L 股池覆盖 `2020-01-02` 至
`2025-12-31`。

核心结论：

- short leg：universe 最强，2020-2024 中枢高，2025 明显降档但仍为正。
- next-close leg：universe 从 2022H2 后经常转弱，2024-2025 负段明显；S/M/L 内部 next 超额更稳。
- weekly 单周图的 2024 年末附近尖峰主要来自 `2024-09-30` 国庆长假前单日周；
  `2024-09-30` universe 单周 next 内部超额约 `+596.6 bps`，是跨长假 next-close
  标签和行情 outlier，不是合并/画图错误。
- weekly 单周图的 `2021-10-04` 周只有 `2021-10-08` 一个交易日；short 内部超额在四个池子
  均小幅转负（universe 约 `-2.9 bps`），主要由 `09:36-09:39` 的 top100 跑输拖下；
  同周 next 内部超额才是主要负异常（universe 约 `-338.9 bps`）。

整体汇总（2020-2025，date-clock group 等权）：

| pool | short excess bps | short Rank IC | next excess bps | next Rank IC |
| --- | ---: | ---: | ---: | ---: |
| universe | +20.22 | 0.167 | -1.26 | 0.010 |
| pool_S | +8.29 | 0.134 | +11.27 | 0.008 |
| pool_M | +9.81 | 0.145 | +12.88 | 0.008 |
| pool_L | +11.14 | 0.152 | +13.31 | 0.009 |

原始 `2020-2025` 合并归档按用户要求只保留 3 张核心图和 trace：

```text
experiments/results/backtests/halfyear_window_2020_2025/short_halfyear.svg
experiments/results/backtests/halfyear_window_2020_2025/next_halfyear.svg
experiments/results/backtests/halfyear_window_2020_2025/weekly.svg
experiments/results/backtests/halfyear_window_2020_2025/trace.json
```

### Mentor direction for next signal enhancement

mentor 后续指示：

- `2020`、`2021` 年做日频的人不多，下一轮信号增强重点看 `2022-2025`。
- 研究对象仍是开盘强势股本身，继续通过特征工程和常规模型参数优化来加强信号。
- 当时展示和验收主看 universe + `pool_L` 四格：universe short、`pool_L` short 和
  `pool_L` next 预期同向增强；universe next 只作为 tail 诊断，不作为否决项。当前口径已收敛为
  两视角 overlay：universe short Rank IC + `pool_L` Top100 next internal excess。
- 上一轮 `2020-2025` rolling-window summary 只保留四股池 short halfyear、next halfyear 和 weekly 单周期视图三张核心图；这里的 weekly 不是 4w rolling 诊断。

### 2022-2025 baseline 归档

按 mentor 指示切出 `2022-2025` baseline。旧本地版本曾由 halfyear mainline
`2020-2024` pool-internal group metrics 与 2025 OOS extension group metrics 拼接生成；
现在正式归档已由集群侧 `baseline_2022_2025_cluster_analysis_v1` 取代。集群 Job 直接读取
PVC 上 2022H1..2024H2 与 2025H1..2025H2 prediction shards，在容器内拼接并 join PVC
next-close labels。主展示使用 universe + `pool_L`，不放 `pool_S/M`；short / next excess +
Rank IC 按季度聚合，累计超额使用日度路径且横轴只标年份。

整体汇总（2022-2025，date-clock group 等权）：

| pool | short excess bps | short Rank IC | next excess bps | next Rank IC |
| --- | ---: | ---: | ---: | ---: |
| universe | +16.75 | 0.149 | -8.48 | 0.004 |
| pool_L | +8.63 | 0.138 | +7.97 | 0.002 |

解读：universe short 强但 next 为负，`pool_L` short 和 next 同时为正。当时判断是：
universe score 混有真实强弱和短效噪声 / 拥挤成分；`pool_L` 作为质量筛选后，同一 score
更偏向可延续的真实强弱。当前验收进一步简化为：用 universe short Rank IC 直接看短期模型，
用 `pool_L` Top100 next internal excess 看隔夜 overlay。

归档文件：

```text
experiments/runs/baseline_2022_2025_cluster_analysis_v1.toml
experiments/jobs/baseline_2022_2025_cluster_analysis_v1_pool_internal_analysis_job.yaml
experiments/results/backtests/baseline_2022_2025_cluster/pool_internal_summary.csv
experiments/results/backtests/baseline_2022_2025_cluster/pool_internal_quarter_summary.csv
experiments/results/backtests/baseline_2022_2025_cluster/daily_pool_internal_summary.csv
experiments/results/backtests/baseline_2022_2025_cluster/short_excess_rank_ic_with_mean.svg
experiments/results/backtests/baseline_2022_2025_cluster/next_excess_rank_ic_with_mean.svg
experiments/results/backtests/baseline_2022_2025_cluster/daily_cumulative.svg
output/artifacts/baseline_2022_2025_cluster_analysis_v1/
```

### 2022-2025 首轮试水优化归档

围绕 `2022-2025` / `pool_L` 先跑三组低风险特征/模型优化试水，并统一使用集群侧
pool-internal analysis 归档。三组均为 full universe 训练、`pool_L` 主验收：

1. `reg_strong`：保留 soft-core feature set，显著加强 LightGBM 正则和叶子约束
   (`num_leaves=31`, `min_child_samples=600`, `subsample=0.80`, `colsample_bytree=0.80`,
   `reg_lambda=4.0`)。
2. `bagging`：保留 baseline feature set，轻微增加树数并加重 row/column bagging
   (`n_estimators=420`, `subsample=0.75`, `colsample_bytree=0.75`, `reg_lambda=2.0`)。
3. `no_preopen_reg_mid`：去掉 `preopen_*` 特征族，并使用中等正则
   (`num_leaves=47`, `min_child_samples=400`, `subsample=0.85`, `colsample_bytree=0.85`)。

`pool_L` 总体结果如下，单位 bps；delta 均相对集群侧 2022-2025 baseline：

| variant | short excess | next excess | short IC | next IC | next positive months | short delta | next delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | +8.626 | +7.974 | 0.1380 | 0.0017 | 32 / 48 | +0.000 | +0.000 |
| `reg_strong` | +8.111 | +6.459 | 0.1369 | -0.0001 | 30 / 48 | -0.515 | -1.515 |
| `bagging` | +8.591 | +7.874 | 0.1380 | 0.0015 | 31 / 48 | -0.035 | -0.100 |
| `no_preopen_reg_mid` | +8.348 | +7.388 | 0.1373 | 0.0012 | 32 / 48 | -0.279 | -0.585 |

结论：

- 三组都没有打过 baseline；这说明当前 `soft_core_reg_light` baseline 的局部位置是健康的。
- `reg_strong` 系统性变差，强正则方向先停。
- `bagging` 最接近 baseline，但只是贴近而无增量；重 bagging 不能作为下一步主方向。
- `no_preopen_reg_mid` 小幅变差，说明 `preopen_*` 不能整族删除；后续如果动 preopen，只做子特征级筛选。

归档文件：

```text
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1.toml
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_bagging_v1.toml
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1.toml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1_sharded_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_bagging_v1_sharded_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1_sharded_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_reg_strong_v1_pool_internal_analysis_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_bagging_v1_pool_internal_analysis_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_pool_l_no_preopen_reg_mid_v1_pool_internal_analysis_job.yaml
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_pilot_sweep_summary.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_{reg_strong,bagging,no_preopen_reg_mid}_v1_*
```

未完成的 `reg_mid`、`drop_return_reg_mid`、`guard_weighted` 已按用户要求停止并移出当前归档口径；
后续不再把它们作为主线完整性的要求。

下一步已落到 2026-06-09 第二批实验：围绕 `2022-2025` 做更细粒度的
feature engineering 和常规模型优化，优先直接做强开盘短 alpha。该段记录保留当时四格验收的
历史口径；当前固定图已改为 universe short Rank IC + `pool_L` Top100 next internal excess。

### 2022-2025 pool_L 因子增强第二批实验投放

2026-06-09 提交第二批 10 个 `2022-2025` / `pool_L` 实验。口径统一为：

```text
sample: 09:31:00-09:40:00 decision points
label: mixed w_long=0.30, target_col=target_label
window: 36m train -> 6m test, stride 6m, 2022-01..2025-12
training universe: full A-share universe
analysis pools: universe + pool_L
primary objective: train a stronger opening short alpha
current acceptance: universe short Rank IC improves; pool_L Top100 next internal excess improves
legacy diagnostic: universe next-close was only a tail diagnostic
baseline reference: universe short IC 0.1489; pool_L next Top100 excess +7.974 bps
```

目标校正：训练一个更强的开盘短 alpha；用 `pool_L` 检验质量筛选后 short / next 是否同向增强。
`pool_L` 是 overlay 验收场景，不是训练域或机制本身；universe next 只作为 tail 诊断。

代码和镜像准备：

- 镜像：`registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260609-pooll-strength-v1`
  pushed digest `sha256:8e3d40d556af74dc95e100c14a7a3bc6a7a54f490a8afdf4d3e0cea7ee7355a2`。
- 2026-06-10 工程修复镜像：`registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260610-engineering-fixes-v1`
  pushed digest `sha256:8df247a3cf3d8b87eb8130f16d2bd39de603f2e11acca07674ccfb4d44692f9c`。
- `features.py` 新增 `postopen_v2_queue_*` 队列补给响应特征，以及
  `auction_price_range_bps` / `auction_last_position_in_range`。
- `training_labeled.py` 新增 `sample_weight.mode = "date_linear"`，用于 recent-regime 轻度加权。
- `feature_dependence_audit.py` 支持 stock pool selection filter，feature audit 可直接汇总 `pool_L`。
- `training.py` 的 `predictions.parquet`、bucket 和 metrics 产物改为同目录临时文件 + `os.replace`
  原子落盘，并在训练完成后写 `_SUCCESS` marker。
- analysis Job 默认等待各 shard 的 `metrics_by_year.csv`，不再仅凭 `predictions.parquet` 出现就启动；
  `analysis.pool_internal.wait_for_completion_file` 可按需覆盖。
- K8s render 支持 `[k8s].env_secrets` / `ceph_secret` 注入，并支持
  `avoid_nodes = ["node7"]` 渲染 `nodeAffinity NotIn`，training 和 analysis Job 共用。
- 本地校验：`pytest tests/test_candidate_guards.py tests/test_postopen_v2_features.py tests/test_feature_audit_groups.py`
  通过；2026-06-10 工程修复后全量 `pytest` 106 项通过、`ruff check src tests` 通过；
  `osf-audit-experiments` 显示 `alignment_ok: yes`。

已提交实验：

| run | cluster job | analysis job | purpose |
| --- | --- | --- | --- |
| `lgbm_delay2_36m_2022_2025_pool_l_feature_audit_v1` | `os-audit-36m-2225-pooll-features` | n/a | grouped feature audit，带 `pool_L` selection filter。 |
| `lgbm_delay2_36m_2022_2025_pool_l_depth_state_plus_v1` | `os-lgbm-36m-2225-depth-state` | `os-analyze-36m-2225-depth-state` | 加权/斜率 depth-gap 状态，测试盘口形态强弱。 |
| `lgbm_delay2_36m_2022_2025_pool_l_depth_trajectory_plus_v1` | `os-lgbm-36m-2225-depth-traj` | `os-analyze-36m-2225-depth-traj` | 加入价格、深度、价差的 opening trajectory。 |
| `lgbm_delay2_36m_2022_2025_pool_l_queue_response_plus_v1` | `os-lgbm-36m-2225-queue-plus` | `os-analyze-36m-2225-queue-plus` | 显式加入 ask/bid 队列补给和 spread compression response。 |
| `lgbm_delay2_36m_2022_2025_pool_l_price_path_plus_v1` | `os-lgbm-36m-2225-price-path` | `os-analyze-36m-2225-price-path` | 加入 mid/ask/bid path diff、relative path 和 spread path。 |
| `lgbm_delay2_36m_2022_2025_pool_l_trade_impact_normalized_plus_v1` | `os-lgbm-36m-2225-trade-impact` | `os-analyze-36m-2225-trade-impact` | 保留成交冲击相对深度的归一化信号，弱化 raw trade-flow diff。 |
| `lgbm_delay2_36m_2022_2025_pool_l_restore_raw_trade_state_v1` | `os-lgbm-36m-2225-raw-trade` | `os-analyze-36m-2225-raw-trade` | 恢复 raw `volume` / `turnover` / `iopv` state，看 pool_L 是否吃 liquidity/attention level。 |
| `lgbm_delay2_36m_2022_2025_pool_l_preopen_price_state_v1` | `os-lgbm-36m-2225-preopen-price` | `os-analyze-36m-2225-preopen-price` | 只保留 preopen price/range/imbalance state，去掉 preopen volume/turnover。 |
| `lgbm_delay2_36m_2022_2025_pool_l_preopen_auction_strength_v1` | `os-lgbm-36m-2225-auction` | `os-analyze-36m-2225-auction` | 加入 auction range / final position，并恢复 preopen volume/turnover。 |
| `lgbm_delay2_36m_2022_2025_pool_l_recent_regime_weight_v1` | `os-lgbm-36m-2225-recent-wt` | `os-analyze-36m-2225-recent-wt` | baseline feature/model + date-linear sample weight，2019-2025 从 0.80 线性到 1.20。 |

K8s 操作记录：

```text
deleted completed jobs:
os-analyze-36m-2225-bagging
os-analyze-36m-2225-nopreopen
os-analyze-36m-2225-regstrong

created jobs:
os-audit-36m-2225-pooll-features
os-lgbm-36m-2225-depth-state
os-lgbm-36m-2225-depth-traj
os-lgbm-36m-2225-queue-plus
os-lgbm-36m-2225-price-path
os-lgbm-36m-2225-trade-impact
os-lgbm-36m-2225-raw-trade
os-lgbm-36m-2225-preopen-price
os-lgbm-36m-2225-auction
os-lgbm-36m-2225-recent-wt
os-analyze-36m-2225-depth-state
os-analyze-36m-2225-depth-traj
os-analyze-36m-2225-queue-plus
os-analyze-36m-2225-price-path
os-analyze-36m-2225-trade-impact
os-analyze-36m-2225-raw-trade
os-analyze-36m-2225-preopen-price
os-analyze-36m-2225-auction
os-analyze-36m-2225-recent-wt
```

2026-06-10 修复后实查：

```text
completed end-to-end model experiments after 2026-06-11 archive:
depth_state_plus
depth_trajectory_plus
queue_response_plus
price_path_plus
trade_impact_normalized_plus
restore_raw_trade_state
preopen_price_state
preopen_auction_strength
recent_regime_weight

still running separately:
os-audit-36m-2225-pooll-features
```

排查事实：重建前活跃 `depth-traj` / `raw-trade` 训练 Job 仍是
`20260609-pooll-strength-v1` 镜像且没有 `affinity`，其中 `depth-traj` shard 1 被调度到
`node7`；两个 analysis Job 仍等待 `predictions.parquet`。2026-06-10 已删除并重建上述
5 个未完成 Job；新模板均为 `20260610-engineering-fixes-v1`，带
`kubernetes.io/hostname NotIn node7`，analysis `WAIT_PATHS` 均为 `metrics_by_year.csv`。
重建后 `depth-traj` / `raw-trade` 的 index 0 立即因已有 `metrics_by_year.csv` 跳过完成，
随后全部 8/8 training shards 完成；2026-06-11 补跑 `depth-traj` / `raw-trade` analysis，
并和另外 7 个模型一起同步到 `experiments/results/backtests/`。

9 个模型实验的 `pool_L` 总体结果如下，单位 bps；delta 相对集群侧 2022-2025 baseline。
完整 CSV 归档在
`experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_second_sweep_summary.csv`。

| variant | short excess | short delta | next excess | next delta | short IC | next IC | short positive months | next positive months |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | +8.626 | +0.000 | +7.974 | +0.000 | 0.1380 | 0.0017 | 48 / 48 | 32 / 48 |
| `price_path_plus` | +8.644 | +0.017 | +7.988 | +0.015 | 0.1383 | 0.0016 | 48 / 48 | 32 / 48 |
| `recent_regime_weight` | +8.634 | +0.008 | +8.205 | +0.232 | 0.1384 | 0.0017 | 48 / 48 | 31 / 48 |
| `restore_raw_trade_state` | +8.630 | +0.004 | +8.026 | +0.052 | 0.1380 | 0.0017 | 48 / 48 | 31 / 48 |
| `depth_trajectory_plus` | +8.615 | -0.011 | +8.148 | +0.174 | 0.1383 | 0.0016 | 48 / 48 | 33 / 48 |
| `queue_response_plus` | +8.583 | -0.044 | +7.981 | +0.007 | 0.1381 | 0.0019 | 48 / 48 | 32 / 48 |
| `preopen_auction_strength` | +8.581 | -0.045 | +8.006 | +0.032 | 0.1380 | 0.0017 | 48 / 48 | 31 / 48 |
| `depth_state_plus` | +8.566 | -0.060 | +7.986 | +0.013 | 0.1380 | 0.0017 | 48 / 48 | 30 / 48 |
| `trade_impact_normalized_plus` | +8.550 | -0.077 | +7.994 | +0.020 | 0.1380 | 0.0019 | 48 / 48 | 32 / 48 |
| `preopen_price_state` | +8.536 | -0.090 | +7.855 | -0.118 | 0.1376 | 0.0020 | 48 / 48 | 33 / 48 |

研究判断：`price_path_plus`、`recent_regime_weight`、`restore_raw_trade_state` 只有极小 short
正增量，最好也只有 +0.017 bps；`recent_regime_weight` 和 `depth_trajectory_plus` 对 next
略好，但 primary short 目标没有形成可用增量。说明当前 baseline 已经吃掉大部分一阶可见信号，
继续做宽泛特征族加减或轻量树模型调参，预期收益很小。后续实验仍围绕 full-universe 开盘短
alpha 本身做强，但具体落实为两个方向：

1. cross-sectional relative features：不是继续追加绝对状态，而是把核心盘口、成交流、价格路径、
   preopen / postopen 状态改写成同一 `date x decision_time` 横截面内的 rank、zscore、demean
   或相对异常表达，检验 A 股开盘短线是否更吃横截面相对强弱。
2. model ensemble：用 baseline LGBM 与低成本异质模型 / 参数变体的 OOS prediction 做横截面
   rank-level 组合，检验模型范式差异是否还能带来增量；`pool_L` 只用于检验该短 alpha 作为日频
   股池 overlay 是否更强。

2026-06-10 下班前追加一组隔夜交互对照：
`lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1`，job
`os-lgbm-36m-2225-xs-rel-wt`，analysis job `os-analyze-36m-2225-xs-rel-wt`。该组复用
`xs_relative` 的横截面 zscore/rank 特征和 `recent_regime_weight` 的 `date_linear`
样本权重，用于判断相对异常表达是否在近期 regime 加权下更稳。提交时集群上已有
`xs_relative`、`model_ensemble`、`depth_traj`、`raw_trade` 和 feature audit 在跑，因此本轮只新增这一组。

2026-06-10 `xs_relative` 首次提交的 index 0 在 `node15` 运行约 70 分钟后
`OOMKilled`，pod memory limit 为 `512Gi`。日志停在读取 labeled PVC 的早期阶段：
`projected_columns=154`，但年度 part 进入 `filter_labeled_frame()` 后列数膨胀到 `1194`，
尚未打印 `dataset` / `split_plan`，因此不是 LightGBM 训练阶段 OOM。根因是
`cross_sectional_relative_prefixes` 使用宽泛 `postopen_v2_` 等 prefix，且 xs-relative 特征在
sample / lag filter 前生成，导致每个年度 part 先扩出大量横截面相对列再 concat 多年数据。
修复：

- 代码顺序调整为 postopen/path 特征仍在 sample filter 前生成，保留 09:31 对 09:30 的路径依赖；
  `xs_rel_*` 改在 sample / lag filter 后生成。
- `xs_relative_v1` 和 `xs_relative_recent_weight_v1` 的横截面相对源列改成 compact 核心列清单，
  不再使用宽泛 `postopen_v2_` prefix。
- 新镜像：`registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260610-xs-rel-compact-v1`
  pushed digest `sha256:5c1bf994f2b189dad78bba14e56600e94439d24a9d130b086757b0667cf625e6`。
- 重提 `os-lgbm-36m-2225-xs-rel` / `os-lgbm-36m-2225-xs-rel-wt` 及对应 analysis Job；因
  `node8` 上同一路径短时不可见，两个训练 Job 的 `avoid_nodes` 扩为 `["node7", "node8"]`。

2026-06-11 两组 cross-sectional relative features 已完成并正式归档：

| variant | pool_L short excess | short delta | pool_L next excess | next delta | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `xs_relative_v1` | +8.767 | +0.141 | +7.565 | -0.408 | 无样本权重；short 小幅增强，next 变弱。 |
| `xs_relative_recent_weight_v1` | +8.840 | +0.213 | +8.023 | +0.049 | 含 recent-regime sample weight，不作为纯因子主线。 |

归档文件：

```text
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1.toml
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1.toml
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1_metrics_by_month.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_summary.csv
```

2026-06-11 `model_ensemble` 已完成并正式归档。该实验使用 LightGBM / HistGBM / Ridge
成员预测的 `date x decision_time` 横截面 rank-centered 加权平均；K8s training job
`os-ensemble-36m-2225` 为 `8/8 Complete`，analysis job `os-analyze-36m-2225-ensemble`
为 `1/1 Complete`，归档后均已从集群清理。

| variant | pool_L short excess | short delta | pool_L next excess | next delta | short IC | next IC | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `model_ensemble_v1` | +7.635 | -0.991 | +6.018 | -1.956 | 0.1393 | -0.0004 | 短线和隔夜均弱于 baseline；model ensemble 路线本轮不通过。 |

归档文件：

```text
experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1.toml
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1/
```

### 2026-06-12 fullxs feature batch and feature audit archive

2026-06-12 分批处理集群上已完成的 fullxs 实验。先同步已完成 analysis 的
`clock_segment_lgbm`，再补提并等待 `hist_same_minute_surprise` / `path_shape_confirm` /
`rank_label_regression` 三个 pool-internal analysis Job，随后统一同步 metrics、compact artifacts
和正式 `experiments/results` 归档。`baseline_2022_2025_prediction_restore_v1` 的 8/8
restore shards 已完成，本地补齐 sharded Job manifest 并同步 metrics，用作 overlap/swap
diagnostics 追溯，不作为新候选。

fullxs 主表如下，单位 bps；delta 相对集群侧 2022-2025 baseline，主看 `pool_L`。
完整 CSV 归档在
`experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_summary.csv`。

| variant | pool_L short excess | short delta | pool_L next excess | next delta | short IC | next IC | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | +8.626 | +0.000 | +7.974 | +0.000 | 0.1380 | 0.0017 | 集群侧 2022-2025 baseline。 |
| `hist_same_minute_surprise` | +9.127 | +0.501 | +8.332 | +0.358 | 0.1401 | 0.0028 | 本批最好，short/next 同向改善。 |
| `clock_segment_lgbm` | +8.711 | +0.085 | +8.335 | +0.362 | 0.1380 | 0.0021 | 小幅同向改善。 |
| `path_shape_confirm` | +8.670 | +0.044 | +8.638 | +0.665 | 0.1386 | 0.0018 | next 改善最大，short 增量很小。 |
| `rank_label_regression` | +7.135 | -1.491 | +0.872 | -7.101 | 0.1504 | 0.0151 | Rank IC 高但 Top100 excess 失败。 |

feature audit 也已完成 8 个半年 shard 并新增 sync 支持，合并归档为
`experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_v1/`。
聚合摘要在
`experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_summary.csv`。
结论：

- ablation 中，去掉 `postopen_v1` / `postopen_v2` 对 `pool_L` Top100 bps 影响最大，均值 delta
  分别为 `-0.381` / `-0.278` bps。
- permutation 中，`orderbook_depth` 对 Rank IC 影响最大，mean delta rank IC `-0.0510`；
  `postopen_v1` / `postopen_v2` 对 Top100 bps 影响最大，分别为 `-1.380` / `-1.352` bps。
- `preopen` 组仍有排序信息，但对 Top100 bps 的直接敏感度小于 postopen/orderbook 组。

归档文件：

```text
experiments/jobs/baseline_2022_2025_prediction_restore_v1_sharded_job.yaml
experiments/results/metrics/baseline_2022_2025_prediction_restore_v1_metrics_by_year.csv
experiments/results/metrics/baseline_2022_2025_prediction_restore_v1_metrics_by_month.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_*_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_*_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_*_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_summary.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_summary.csv
experiments/results/backtests/model_ensemble_vs_baseline_group_delta/
```

另一个未形成正式结果的路径统计实验，已按用户要求删除其 run config、K8s manifests、特征入口和测试。

### 2026-06-12 optimization overlay acceptance figures

验收看这两张图：

```text
experiments/results/backtests/optimization_overlay_acceptance_2022_2025/optimization_directions_overlay_acceptance.svg
experiments/results/backtests/optimization_overlay_acceptance_2022_2025/optimization_directions_net_alpha_cumulative.svg
```

图名和口径：

| figure title | data panels | purpose |
| --- | --- | --- |
| short rank IC和next pool_L 超额 | `universe short Rank IC`; `pool_L Top100 next internal excess` | 主验收：短期模型本身是否更强，以及叠加到 mentor 股池后隔夜是否同步变好。 |
| 池内Top100隔夜收益累和 | selected next net return + background; relative to baseline | 只看隔夜，保留日频累计点。 |

不再作为主验收项：`pool_L` short IC、short excess、universe next excess 和 next IC。
默认两张图展示 baseline、hist_surprise 和 path_shape；后续可替换为任意 2-3 个
comparison models。第二张上 panel 额外展示 `pool_L` background。第二张图当前没有接入
公司回测 API；仓库内未找到可调用封装。旧目录使用 `fee = 5 bps`，daily plot data 仍按
`cumulative_decision_normalizer = 1000` 缩放，图上保留日频累计点。

### 2026-06-16 optimization overlay acceptance 8bps fee refresh

根据 A 股 all-in round-trip 成本复核，默认 realized fee 从只扣卖出印花税的 `5 bps`
更新为 `8 bps`。重画输出目录：

```text
experiments/results/backtests/optimization_overlay_acceptance_2022_2025_fee8bps/
```

阶段状态：baseline 后四方向特征/模型 sweep 收尾；mentor rescope 后，下一步不急于组合定稿，
而是继续做强开盘短期模型，优先 price-regime 干预和尺度归一化特征。

### 2026-06-12 Mentor Re-scope: price ecology and scale normalization

mentor comment 判断：正确，已采纳为下一阶段研究边界。

具体理由：

1. 继续做强开盘短期模型直到收敛，与当前主线一致。首轮、第二批、xs-relative、model ensemble 和 fullxs
   批次说明常规特征族加减的增量已经变小，但 `hist_same_minute_surprise` 仍有同向改善，因此还没有到
   停止 signal discovery 的位置。
2. 贵/便宜股票需要主动干预。A 股最小价格变动为 `0.01`，对 2 元股票是约 50 bps，对 1000 元股票是
   约 0.1 bps；这会改变挂单集中在哪些档位、盘口缺口、一档占比、queue replenish 和成交冲击的含义。
   因此不能默认同一组绝对盘口特征在不同价格层级共享同一个生效模式。
3. `hist_same_minute_surprise` 不只是“历史 surprise”命名下的异常检测，它更像无量纲化：用同一股票、
   同一决策分钟的过去均值/波动来重定当前状态的尺度。`xs_relative` 则是在同一
   `date x decision_time` 横截面内找尺度。两者共同指向下一阶段：优先把价格、深度、成交和队列特征
   变成相对 tick、bps、ratio、zscore、rank 或分桶后的交互表达。

下一步实验约束：

- `primary objective` 不变：训练更强的 opening short alpha。验收用 universe short Rank IC
  看短期模型本身，用 `pool_L` Top100 next internal excess 看 overnight overlay 效果。
- 新增 price-regime 诊断和干预：按 decision price / reference price 切 cheap / mid / expensive
  bucket，分别看 feature importance、Top100 excess 和 short/next tradeoff；必要时加入 price bucket
  特征、price-bucket interaction 或 clock/price segmented model。
- 新增尺度归一化主线：优先扩展 hist-surprise、cross-sectional relative、tick-normalized depth gap、
  bps-normalized spread/path、notional/price-scaled queue 和 per-symbol historical ratio，而不是继续堆
  raw absolute orderbook state。

### 2026-06-12 price-regime / scale-normalization experiments submitted

三组实验已用新镜像提交到 research 集群：

```text
image: registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260612-price-scale-v1
digest: sha256:410746153cb4ac05f13e5f45e224ec5e83f3b4fc23ffad469be3280b3b31073f
```

| run | training job | analysis job | purpose |
| --- | --- | --- | --- |
| `lgbm_delay2_36m_2022_2025_fullxs_price_bucket_v1` | `os-lgbm-36m-2225-price-bucket` | `os-analyze-36m-2225-price-bucket` | price bucket + bucket interaction。 |
| `lgbm_delay2_36m_2022_2025_fullxs_scale_norm_v1` | `os-lgbm-36m-2225-scale-norm` | `os-analyze-36m-2225-scale-norm` | tick/bps/notional scale features + expanded hist-surprise。 |
| `lgbm_delay2_36m_2022_2025_fullxs_price_scale_norm_v1` | `os-lgbm-36m-2225-price-scale` | `os-analyze-36m-2225-price-scale` | price bucket、scale features、hist-surprise 和 compact xs-relative 组合主实验。 |

### 2026-06-17 price-regime / scale-normalization archive

`price_bucket`、`scale_norm`、`price_scale_norm` 已补齐 training metrics、pool-internal analysis
和正式 `experiments/results` 归档。`price_scale_norm` 的 analysis job
`os-analyze-36m-2225-price-scale` 于 2026-06-17 补提并完成；run config 状态改为 `completed`。

`pool_L` 主表如下，单位 bps；delta 相对集群侧 2022-2025 baseline：

| variant | pool_L short excess | short delta | pool_L next excess | next delta | short IC | next IC | next positive months |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | +8.626 | +0.000 | +7.974 | +0.000 | 0.1380 | 0.0017 | 32 / 48 |
| `price_bucket` | +8.655 | +0.029 | +8.008 | +0.035 | 0.1384 | 0.0019 | 33 / 48 |
| `scale_norm` | +9.241 | +0.615 | +8.454 | +0.480 | 0.1407 | 0.0030 | 33 / 48 |
| `price_scale_norm` | +9.313 | +0.687 | +8.396 | +0.423 | 0.1410 | 0.0033 | 33 / 48 |

解读：

- 单独 `price_bucket` 基本无增量，说明固定价格分桶/交互不是主要来源。
- `scale_norm` 明显优于 baseline，是当前综合最好候选；`pool_L` next overlay 和 8bps fee
  capital-adjusted 累计净收益都略好于 `price_scale_norm`。
- `price_scale_norm` 的 short 侧最强，但把 price bucket、scale、hist-surprise 和 compact xs-relative
  全部叠加后没有继续改善 next overlay，说明 price interaction 可能引入了一点短期拥挤/噪声。

刷新后的 price-regime acceptance 图：

```text
experiments/results/backtests/optimization_overlay_acceptance_price_regime_ready_2022_2025/
```

8bps fee、next-close capital divisor = 2.0 且线性累加口径下，期末累计净收益相对 baseline：

| variant | capital-adjusted cumulative net vs baseline | cumulative internal excess vs baseline |
| --- | ---: | ---: |
| `price_bucket` | +16.8 bps | +16.8 bps |
| `scale_norm` | +232.6 bps | +232.6 bps |
| `price_scale_norm` | +204.8 bps | +204.8 bps |

正式归档文件：

```text
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_price_scale_norm_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_price_scale_norm_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_price_scale_norm_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_summary.csv
```

### 2026-06-17 S/M/L stock-pool tradeoff check

用 `hist_same_minute_surprise_v1` 预测、Top100、`pool_S/M/L`、2022-2025 全窗口生成 S/M/L
benchmark，检查股池大小、平均 next 收益和股池成分换手之间的取舍。

```text
experiments/results/backtests/pool_sml_i500_benchmark_acceptance_2022_2025/
```

| pool | avg members | pool next mean | Top100 next mean | Top100 vs pool | stock-pool turnover | fee8 pool cost | fee8 /2 capital cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pool_S` | 1454 | 19.77 bps | 27.39 bps | 7.62 bps | 23.3% | 1.86 bps | 0.93 bps/day |
| `pool_M` | 2423 | 15.18 bps | 23.38 bps | 8.20 bps | 16.2% | 1.30 bps | 0.65 bps/day |
| `pool_L` | 3392 | 11.24 bps | 19.57 bps | 8.33 bps | 9.9% | 0.79 bps | 0.40 bps/day |

结论：股池本身呈现 S > M > L 的收益底子，同时 S > M > L 的换手成本也更高；但池内
Top100 overlay 相对各自 pool 的 next excess 并不随股池变小而提高，`pool_L` 略高。这说明
小池优势主要来自 pool selection 本身，而不是池内短期模型排序更强。

### 2026-06-18 hist + path exact-union experiments submitted

按“baseline 276 + hist_same_minute_surprise 52 + path_shape_confirm 26”口径补三组 2022-2025
fullxs rolling 实验，三者都使用 36m train -> next 6m test、8 个半年 shard，并行度
`shard_parallelism = 4`。

| run_id | training job | analysis job | feature口径 |
| --- | --- | --- | --- |
| `lgbm_delay2_36m_2022_2025_fullxs_hist_path_v1` | `os-lgbm-36m-2225-hist-path` | `os-analyze-36m-2225-hist-path` | 精确叠加 baseline + `hist_surprise_` + `path_shape_`；不加入 price-scale / bucket / xs-relative。 |
| `lgbm_delay2_36m_2022_2025_fullxs_hist_path_norm_rank_v1` | `os-lgbm-36m-2225-hist-path-norm` | `os-analyze-36m-2225-hist-path-rank` | rank-centered 对照：使用同一批源特征，按 `date,decision_target_timestamp` 生成单一 `rank_centered` 的 `norm_` 特征，模型只吃 `norm_`。后续 analysis 名和展示名改为 `hist_path_rank_centered`；training job/run_id/output 保留原名以对应已运行中的集群产物。 |
| `lgbm_delay2_36m_2022_2025_fullxs_hist_path_zscore_v1` | `os-lgbm-36m-2225-hist-path-zscore` | `os-analyze-36m-2225-hist-path-zscore` | 干净 zscore 归一化版本：同一批源特征按 `date,decision_target_timestamp` 做单一 `zscore` 的 `norm_` 特征，模型只吃 `norm_`。 |

rank-centered / zscore 版显式复用 raw 版的窄 feature selector：`postopen_v2_*` 只保留既有白名单前缀，
普通 postopen 用 `^postopen_(?!v2_)`；没有使用宽 `postopen_` 前缀，避免把未计划的
postopen_v2 派生列纳入。

2026-06-18 追加说明：`hist_path_norm_rank` 不是普通特征归一化，而是 rank-centered
对照。真正的干净归一化实验为 `hist_path_zscore_v1`，每个源特征只生成一个横截面 zscore
特征，不生成 rank 特征。

### 2026-06-23 hist + path exact-union archive

三组 2022-2025 fullxs rolling 训练 metrics 均已同步到
`experiments/results/metrics/`，pool-internal analysis 轻量结果归档到：

```text
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_zscore_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_norm_rank_v1/
```

`norm_rank` 训练 shard 已先完成，但缺少 `analysis/pool_internal_top100/`。2026-06-23
补跑 `os-analyze-36m-2225-hist-path-rank` 后完成归档；该 run 的 analysis 展示名为
`hist_path_rank_centered`。

pool-internal summary：

| model | universe short Rank IC | universe short excess bps | `pool_L` short Rank IC | `pool_L` short excess bps | `pool_L` next excess bps | `pool_L` next positive months |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hist_path` | 0.1515 | +17.60 | 0.1409 | +9.26 | +8.85 | 33 / 48 |
| `hist_path_zscore` | 0.1483 | +16.55 | 0.1392 | +8.89 | +8.42 | 32 / 48 |
| `rank_centered` | 0.1531 | +17.25 | 0.1440 | +9.30 | +9.45 | 34 / 48 |

三模型对比验收图归档到：

```text
experiments/results/backtests/optimization_overlay_acceptance_hist_path_2022_2025/
```

核心文件：

```text
optimization_directions_overlay_acceptance.svg
optimization_directions_net_alpha_cumulative.svg
optimization_directions_overlay_acceptance_plot_data.csv
optimization_directions_net_alpha_cumulative_plot_data.csv
optimization_directions_trace.json
```

这组三模型内部，`rank_centered` 的 mean short Rank IC 为 `0.1530`，`pool_L` mean next
internal excess 为 `+9.50 bps`，均为本批最高；`hist_path` 次之，`hist_path_zscore`
整体弱于 raw / rank-centered。

2026-06-23 追加重画口径：`optimization_acceptance_plots.py` 的 cumulative 图改为：
上 panel 固定画全 A 股市场平均、扣费后的 `pool_L` background、baseline Top100 和传入的
comparison models Top100；下 panel 固定画 `pool_L` background、baseline 和 comparison
models 相对全 A 股市场平均的累计 alpha。单模型 comparison 现在也允许，用于 baseline vs
`deviation+path` 的精简图：

```text
experiments/results/backtests/optimization_overlay_acceptance_baseline_deviation_path_2022_2025/
```

### 2026-06-18 baseline 276 feature hygiene / correlation audit

针对 `lgbm_delay2_36m_2022_2025_pool_l_feature_audit_v1` 的 276 个 baseline 特征补了一轮
轻量 feature hygiene / correlation audit。数据从 PVC 上
`opening_13y_201301_202512_delay2_mixed_w030_labeled_v1` 抽样，覆盖
2022-01、2022-07、2023-01、2023-07、2024-01、2024-07、2025-01、2025-07，
每月 3 天，共 24 个交易日；最终抽样 100k rows。样本实际 decision window 为
09:31:00 到 09:40:05，对应当前 baseline 配置的 09:31-09:40 decision points，而不是
09:45。

产物路径：

```text
output/artifacts/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_v1/
output/artifacts/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_corr09_v1/
/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_corr09_v1/
```

归档状态：2026-06-23 已按正式 `feature_hygiene` artifact run 归档。新增
`experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_v1.toml` 和
`experiments/runs/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_corr09_v1.toml`，
并用 k8s renderer 生成对应 job yaml；`osf-sync-experiment-artifacts --record` 会把
`feature_hygiene.csv`、相关性 pairs/clusters、prune candidates、keep/drop list 和 trace
按原名记录到 `experiments/results/backtests/<run_id>/`。此前审计把它们列为未归档，是因为
只有一份手写 hygiene job，且该 job 直接引用 feature-audit config，没有自己的 run config；
artifact sync / experiment audit / k8s contract 当时也还未把 `feature_hygiene` 定义为标准
artifact run。`corr09` 版本现在作为独立 sensitivity artifact run 归档，语义是同一抽样下把
相关性阈值从 `0.98` 放宽到 `0.90`。

主要结果：

| item | value |
| --- | ---: |
| features | 276 |
| rows | 100,000 |
| high-corr pairs, threshold 0.98 | 28 |
| corr clusters, threshold 0.98 | 14 |
| hard-drop candidates | 17 |
| review candidates | 5 |
| constant features | 1 |
| near-constant features | 0 |

`corr09` sensitivity 结果：`high_corr_pairs=51`、`corr_clusters=25`、`drop_features=16`、
`review=23`、`keep_features=260`。它用于检查 0.98 阈值外还有多少 0.90 以上的相关簇需要人工
复核，不改变主 drop-list 口径。

2026-06-23 追加 fullxs `hist_path` hygiene sensitivity 归档：为了检查
baseline + `hist_surprise_` + `path_shape_` 的精确并集是否引入新的相关簇，新增
`lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1`。该 run 使用
1 day / sampled month、50k rows 的轻量抽样，以控制历史 same-minute context 的内存占用；
run config 和 K8s job 已纳入索引：

```text
experiments/runs/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1.toml
experiments/jobs/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1_job.yaml
output/artifacts/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1/
/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1/
```

fullxs `hist_path` corr09 结果：`features=354`、`rows=50,000`、`high_corr_pairs=96`、
`corr_clusters=38`、`drop_features=19`、`review=47`、`keep_features=335`。新增 drop
主要来自 path_shape 常数/近重复项、hist_surprise zscore/ratio 同源高相关项，以及
postopen_v2 交易额/成交量深度比的近重复项；这组结果作为人工复核 sensitivity，不直接改变
当前 baseline 276 特征的主 drop-list。

### 2026-06-25 hist_path high-dup prune closeout

基于 fullxs `hist_path` corr09 hygiene sensitivity，把 `feature_prune_candidates.csv` 中所有
`hist_surprise_` / `path_shape_` 的 high-duplicate candidates 一次性从模型 feature list 删除，
用于验证“删高重复特征是否仍保留信号”。完整 drop 列表记录在 run config 的
`features.drop_feature_columns`；其中非 hist/path 的 `volume`、`turnover`、`iopv` 是原有基础
drop，真正新增删除为 26 个 `hist_surprise_` / `path_shape_` 特征。

run / job / artifact 索引：

```text
experiments/runs/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1.toml
experiments/jobs/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_sharded_job.yaml
experiments/jobs/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_pool_internal_analysis_job.yaml
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_metrics_by_year.csv
experiments/results/metrics/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_metrics_by_month.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/
output/artifacts/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/
/mnt/output/opening_strength_fit/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/
```

状态审计：

| item | status |
| --- | --- |
| run config | `status = "completed"` |
| training job | `os-lgbm-36m-2225-hist-path-prunedup`，8 个半年 shard 已完成并落 `_SUCCESS` / metrics / predictions。 |
| analysis job | `os-analyze-36m-2225-hist-path-prunedup` 已完成，pool-internal top100 artifacts 已同步。 |
| feature count | metrics 记录 `features=328`；剪枝前 `hist_path` 为 `354`。 |
| experiment alignment | `.venv/bin/osf-audit-experiments` 输出 `alignment_ok: yes`；该 run 显示 `completed`、`pool_internal_analysis,sharded_training`、`metrics=yes`。 |

与剪枝前 `hist_path` 对比：

| pool | metric | before | pruned | delta |
| --- | ---: | ---: | ---: | ---: |
| universe | short_internal_excess_bps | 17.5994 | 17.5893 | -0.0101 |
| universe | next_internal_excess_bps | -7.8433 | -7.9182 | -0.0749 |
| universe | short_rank_ic | 0.151529 | 0.151508 | -0.000021 |
| universe | next_rank_ic | 0.005227 | 0.005180 | -0.000047 |
| `pool_L` | short_internal_excess_bps | 9.2633 | 9.2080 | -0.0554 |
| `pool_L` | next_internal_excess_bps | 8.8485 | 8.8643 | +0.0158 |
| `pool_L` | short_rank_ic | 0.140879 | 0.140789 | -0.000091 |
| `pool_L` | next_rank_ic | 0.002921 | 0.002830 | -0.000092 |

稳定性没有实质退化：universe / `pool_L` 的 `short_positive_months` 均保持 `48 / 48`；
`pool_L` 的 `next_positive_months` 保持 `33 / 48`；`pool_L` short / next positive clocks
均保持 `10 / 10`。因此这次剪枝在 feature hygiene 角度成立：高重复 hist/path 特征可以删除，
信号基本保留；但它不是收益增强实验，不应宣称 performance lift。主线是否切换取决于是否更重视
更干净的特征集合和更低冗余。

#### pool_L Top100 exposure audit

对剪枝后实验补了 `pool_L` Top100 core exposure audit。输入为 PVC 同步回本地的 8 个半年
prediction shard，按 shard 运行后合并；为控制 runtime，本轮跳过 `score_exposure_spearman`，
主读 `selected_mean_rank`、`selected_mean_z` 和 `selected_top_decile_share`。审计列覆盖
price、spread/depth、opening activity、short momentum 和 depth imbalance；市值和行业字段不在
prediction parquet 中，本小节先不含，下一小节已用 ClickHouse 日频 exposure input 补齐。

归档：

```text
experiments/runs/exposure_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_core_v1.toml
experiments/jobs/exposure_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_core_v1_job.yaml
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_exposure_audit/
output/legacy/analysis/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_exposure_audit/
```

说明：这轮 core exposure audit 最初是本地按 shard 运行后合并，因此早先只记录结果目录。
现已补正式 `exposure_audit` OSF run config 和 K8s job，集群复跑时会直接用同一份 pruned
prediction 输出生成 core exposure artifact。

总体读数：

| exposure | category | selected_mean_rank | selected_mean_z | top decile share | read |
| --- | --- | ---: | ---: | ---: | --- |
| `turnover_diff_10t` | activity | 0.877 | +1.526 | 59.2% | 最强 post-open turnover heat。 |
| `turnover_diff_30t` | activity | 0.866 | +1.508 | 57.4% | turnover heat 在更长窗口仍稳定。 |
| `volume_diff_10t` | activity | 0.804 | +0.948 | 42.8% | 明显偏高成交量增量。 |
| `preopen_turnover` | activity | 0.767 | +0.902 | 41.2% | 盘前活跃度高。 |
| `preopen_volume` | activity | 0.691 | +0.523 | 29.3% | 盘前量能偏高但弱于 turnover 增量。 |
| `spread_bps` | liquidity | 0.334 | -0.212 | 3.6% | 明显避开高 spread，交易可执行性更友好。 |
| `buy_price` | price | 0.655 | +0.270 | 18.0% | 中等 higher-price tilt，需要和 price-regime 一起读。 |
| `return_10t` | momentum | 0.562 | +0.193 | 25.5% | 轻度短窗强势。 |
| `return_30t` | momentum | 0.535 | +0.061 | 27.1% | 接近中性到轻度强势。 |
| `depth_imbalance_10` | microstructure | 0.463 | -0.142 | 12.5% | 盘口 imbalance 不是主导暴露，且月度有切换。 |
| `ask_depth_10` | liquidity | 0.505 | -0.087 | 5.4% | 接近中性。 |
| `bid_depth_10` | liquidity | 0.485 | -0.109 | 3.6% | 接近中性到略低。 |

类别汇总显示 activity 是最主要的行为画像：5 个 activity 暴露的平均 `|z| = 1.081`，
平均 rank deviation `0.301`，平均 top-decile share `46.0%`。price / liquidity / momentum /
microstructure 都明显更弱；其中 liquidity 更准确的读法是“低 spread 倾向”，不是无脑追逐差流动性。

月度稳定性：`turnover_diff_10t` 的 monthly selected mean rank 在 48 个月中约为 `0.819-0.908`，
`turnover_diff_30t` 为 `0.799-0.904`，`preopen_turnover` 为 `0.682-0.812`；`spread_bps`
长期处在低分位，约 `0.242-0.380`。因此 activity heat / low-spread 画像不是少数月份造成的。

日内集中度也没有明显单票拥挤：969 个交易日、每日 10 个 decision clocks、Top100 日均约
`526` 个不同 symbols，repeat rate `47.4%`，effective symbols 约 `358`，单票最大占比均值
`0.83%`，Top5 symbols 占比均值 `3.69%`。

解读：这次 audit 不是把所有可命名暴露都定性为坏事。对开盘强势股模型来说，opening activity /
turnover heat 很可能就是 alpha mechanism 的主要载体；审计结论应写成“模型确实在买开盘活跃、
成交增量强、同时 spread 较低的股票”，这和策略定义一致。生产化剩余问题不是机械中性化 activity
heat，而是验证 heat 内仍有增量排序力，并在容量约束下确认冲击、参与率、ADV / 可成交量、
单票和行业集中度仍可接受。本次 split20 capacity audit 已补第一轮可见成交量容量证据，见下节。

#### pool_L size / industry exposure audit

为补齐市值和行业暴露，新增 `osf-build-exposure-input` 从 ClickHouse 构建日频 `date,symbol`
外部 exposure input：`stock.daily_bar_jy` 提供 `market_cap` / `float_market_cap` / log cap，
`stock.industry` 提供申万一二三级行业。对剪枝实验的 8 个 prediction shards 先按 `pool_L`
过滤，再生成 `3,281,701` 个日频 key；市值字段仅缺 `4` 个 key，申万一级行业缺 `170` 个 key。

归档：

```text
output/legacy/exposures/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_pool_l_size_industry_daily.parquet
experiments/results/backtests/exposure_input_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1/
output/legacy/analysis/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_size_industry_exposure_audit/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_size_industry_exposure_audit/
```

tracked archive 只保留 exposure input trace；完整 `exposure_input.parquet` 体积较大，保留在
`output/legacy/exposures/...` 和 PVC 侧供复跑 audit 使用。

正式 OSF 链路拆成两步是合理的：`osf-build-exposure-input` 是外部数据准备 run，负责把
ClickHouse 的日频市值/行业数据落成标准 keyed parquet；`osf-audit-exposure` 是验收 run，
只消费 prediction parquet 和这个 exposure input。这样可以复用同一份市值/行业输入，也能让
K8s job 明确等待外部 exposure input 的 trace 文件后再开始 audit。

run / job 索引：

```text
experiments/runs/exposure_input_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1.toml
experiments/jobs/exposure_input_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1_job.yaml
experiments/runs/exposure_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1.toml
experiments/jobs/exposure_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1_job.yaml
```

集群复跑时先提交 exposure input job，再提交 exposure audit job；两个 manifest 当前用
`opening-strength-fit:latest`，正式提交前应替换为包含 `osf-build-exposure-input` 的固定镜像 tag。

市值暴露总体读数：

| exposure | selected_mean_rank | selected_mean_z | top decile share | bottom decile share | read |
| --- | ---: | ---: | ---: | ---: | --- |
| `log_market_cap` | 0.650 | +0.544 | 22.2% | 3.5% | 中等偏大总市值。 |
| `log_float_market_cap` | 0.641 | +0.522 | 22.4% | 4.7% | 中等偏大流通市值。 |
| `market_cap` | 0.650 | +0.228 | 22.2% | 3.5% | 原值受极端大票影响，主读 rank/log z。 |
| `float_market_cap` | 0.641 | +0.246 | 22.4% | 4.7% | 原值同样只作辅助。 |

月度稳定性：`log_market_cap` selected mean rank 在 48 个月中约 `0.586-0.726`，
`log_float_market_cap` 约 `0.585-0.712`；top decile share 约 `16.4%-33.1%`。因此模型不是
小票暴露，反而稳定偏中大市值，这对后续容量是加分项，但也要向 mentor 说明不是市值中性策略。

申万一级行业 active share 总体读数：

| industry | candidate share | selected share | active share | read |
| --- | ---: | ---: | ---: | --- |
| 电子 | 8.8% | 12.1% | +3.2% | 最主要超配之一。 |
| 电力设备 | 6.8% | 9.9% | +3.1% | 稳定超配，尤其 2024-2025。 |
| 计算机 | 6.7% | 8.7% | +2.0% | 超配但月度有切换。 |
| 通信 | 2.4% | 3.7% | +1.3% | 小权重行业的温和超配。 |
| 有色金属 | 2.8% | 3.9% | +1.2% | 温和超配。 |
| 国防军工 | 2.7% | 3.8% | +1.2% | 温和超配。 |
| 机械设备 | 10.1% | 8.1% | -2.1% | 主要低配。 |
| 基础化工 | 7.9% | 6.0% | -1.9% | 主要低配。 |
| 轻工制造 | 3.1% | 1.7% | -1.5% | 低配。 |
| 环保 | 2.6% | 1.1% | -1.4% | 低配。 |

行业集中度：`pool_L` 候选平均约 `31.0` 个申万一级行业，Top100 平均约 `29.8` 个行业；
selected industry max share 均值 `16.9%`，Top5 industries share 均值 `53.1%`，
industry HHI 均值 `0.082`，effective industries 约 `12.9`。这说明行业偏好可见，但不是
单行业押注；最主要的 active 行业集中在电子/电力设备/计算机等更容易出现开盘活跃和成交增量的板块。

解读：市值和行业 audit 的结论比“有暴露/没暴露”更具体。市值上，模型偏中大市值，不是小票拥挤；
行业上，模型有科技制造方向的结构性偏好，尤其电子、电力设备、计算机，同时低配机械设备、基础化工。
这些不需要机械中性化，但生产验收应把它们作为组合约束候选：容量组合里监控行业 active weight、
行业 HHI、Top5 industry share，并在 mentor 汇报中明确“行业偏好是模型行为画像，但不是单一行业
收益押注”。

#### pool_L split20 capacity audit

正式容量口径按策略总资金 `10 亿`、分 `20` 份执行理解，因此每个
`date x decision_target_timestamp` 的目标 notional 为 `5000 万`，而不是每个 clock 都塞
`10 亿`。早先 `10 亿/clock` 的容量跑法只保留为功能 smoke，不作为生产容量结论。

run / job / artifact 索引：

```text
experiments/runs/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1.toml
experiments/jobs/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1_job.yaml
experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1/
output/artifacts/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1/
/mnt/output/opening_strength_fit/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1/
```

tracked archive 保留 summary / month / daily / trace；逐截面 `capacity_audit_group_metrics.csv`
和逐票 `capacity_audit_selected.csv` 体积较大，保留在 `output/artifacts/...` 和 PVC 侧用于细查，
不纳入轻量结果归档。

约束和读法：

- selection universe：`pool_L`，每个 `date x clock` 按 `prediction` 从高到低拿。
- target notional：`5000 万` / group，对应 `10 亿` 总资金 `/20`。
- 单票上限：`min(1% * target_notional, 10% * turnover_diff_30t)`；本轮未加 ask depth，
  `ask_depth_levels = 0`。
- 主读容量字段，不读收益列：`fill_success_rate`、`mean_top_depth_to_target`、
  `p95_top_depth_to_target`、`max_top_depth_to_target`。

总体结果：

| metric | value | read |
| --- | ---: | --- |
| groups | 9690 | `2022-01-04` 至 `2025-12-31`，每日 `09:31-09:40` 共 10 个 clock。 |
| target_notional | 50,000,000 | `10 亿 / 20` 的单切片容量。 |
| filled_groups | 9690 | 所有截面都能塞满。 |
| fill_success_rate | 100.0% | 切片容量充足。 |
| mean_top_depth_to_target | 124 | 平均需要吃到 top124。 |
| p50_top_depth_to_target | 120 | 中位需要吃到 top120。 |
| p90_top_depth_to_target | 149 | 90% 截面 top149 以内可塞满。 |
| p95_top_depth_to_target | 161 | 95% 截面 top161 以内可塞满。 |
| max_top_depth_to_target | 291 | 最深一次需要吃到 top291。 |

TopN 覆盖率：

| depth cap | filled groups | share |
| ---: | ---: | ---: |
| top100 | 66 / 9690 | 0.7% |
| top120 | 4977 / 9690 | 51.4% |
| top150 | 8792 / 9690 | 90.7% |
| top160 | 9181 / 9690 | 94.7% |
| top200 | 9623 / 9690 | 99.3% |
| top300 | 9690 / 9690 | 100.0% |

结论：`5000 万/切片` 在 `pool_L` 内容量完全够，但固定 Top100 不是稳健容量篮子；生产化容量组合应允许
按分数继续往后拿到约 top160/top200，极端日内截面最深到 top291。下一轮若做更严格生产验收，应补
ask-depth 约束、行业/风格上限和 next-close capacity return 口径；但“能不能塞入 `10 亿/20`”
这一问，本轮答案为可以。

supporting capacity runs：

```text
experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/
experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_t10_v1/
```

- `capacity_audit_..._pruned_highdup_v1` 是早期 `10 亿/clock` 功能 smoke，不作为生产容量结论：
  平均 fill ratio `89.9%`，min fill ratio `33.6%`，说明这个口径过重。
- `capacity_audit_..._split20_t10_v1` 是 `10 亿/20` 下把参与率基准换成
  `turnover_diff_10t` 的 sensitivity：仍然 `9690/9690` 截面全满，但平均需要吃到 top `237`，
  p95 top `449`，极端 max top `1611`，比主口径 `turnover_diff_30t` 明显更紧。

### 2026-06-26 NN single-model first archive

按 2026-06-23 mentor re-scope，先推进当前特征上的 NN 单模型，只有单模型有增量时再做
NN + LGBM ensemble。本次提交补齐 PyTorch MLP 训练入口、GPU K8s 渲染和 NN run/job scaffold；
其中 `mlp_base` / `mlp_shallow_fast` / `mlp_wide_huber` 已有 2022-2025 metrics 和
pool-internal 归档，`2025h2_mlp_smoke` 已有 smoke metrics，`wide_deep_h32` 仍待跑/待归档。

代码能力：

- `model.name = "torch_mlp"`，别名 `mlp` / `nn`；支持普通 MLP 和
  `architecture = "wide_deep_residual"`。
- 训练端对数值特征做 train-set mean/std 标准化；预测端复用同一组 mean/std。
- Docker image 可用 `INSTALL_TORCH_CUDA=1` 安装 CUDA torch wheel；CPU 环境未安装 torch 时，
  torch 单测跳过，不影响 LightGBM / K8s helper 测试。
- training job 可申请 GPU；pool-internal analysis job 不再继承 training 的 GPU node selector，
  只保留 analysis 自己的 scheduler 配置或全局 avoid_nodes。

run / job 索引：

```text
experiments/runs/nn_delay2_36m_2025h2_fullxs_hist_path_pruned_highdup_mlp_smoke_v1.toml
experiments/jobs/nn_delay2_36m_2025h2_fullxs_hist_path_pruned_highdup_mlp_smoke_v1_sharded_job.yaml

experiments/runs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1.toml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1_sharded_job.yaml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1_pool_internal_analysis_job.yaml

experiments/runs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1.toml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1_sharded_job.yaml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1_pool_internal_analysis_job.yaml

experiments/runs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1.toml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1_sharded_job.yaml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1_pool_internal_analysis_job.yaml

experiments/runs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_wide_deep_h32_v1.toml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_wide_deep_h32_v1_sharded_job.yaml
experiments/jobs/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_wide_deep_h32_v1_pool_internal_analysis_job.yaml
```

变体口径：

| run suffix | status | purpose |
| --- | --- | --- |
| `2025h2_mlp_smoke` | completed metrics | 单 epoch 2025H2 smoke，验证 GPU image / data path / sharded output。 |
| `mlp_base` | completed + pool-internal archived | 512/256/128 ReLU MLP，MSE，作为 NN 主基线。 |
| `mlp_shallow_fast` | completed + pool-internal archived | 384/192 小网络、更大 batch、更少 epoch，验证欠/快训口径。 |
| `mlp_wide_huber` | completed + pool-internal archived | 1024/512/256 GELU + Huber，验证更宽网络和抗尾部 loss。 |
| `wide_deep_h32` | queued / pending | 线性分支 + h32 residual hidden，模仿 ridge-plus-small-hidden residual 思路。 |

已归档 MLP pool-internal summary：

| variant | pool | short excess bps | next excess bps | short Rank IC | next Rank IC | positive months short / next |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `nn_mlp_base` | universe | 19.8778 | -1.9729 | 0.149818 | 0.009136 | 48 / 20 |
| `nn_mlp_base` | `pool_L` | 10.1642 | 12.4320 | 0.137205 | 0.004981 | 48 / 37 |
| `nn_mlp_shallow_fast` | universe | 18.7725 | -3.7495 | 0.146876 | 0.008550 | 48 / 19 |
| `nn_mlp_shallow_fast` | `pool_L` | 9.7789 | 11.4480 | 0.134616 | 0.004571 | 48 / 36 |
| `nn_mlp_wide_huber` | universe | 19.4997 | -3.6225 | 0.162945 | 0.019860 | 48 / 20 |
| `nn_mlp_wide_huber` | `pool_L` | 9.8349 | 10.4238 | 0.149704 | 0.014322 | 48 / 39 |

相对 `hist_path_pruned_highdup` 的 `pool_L` summary（short `9.2080`、next `8.8643`、
short Rank IC `0.140789`、next Rank IC `0.002830`）：三个 MLP 的 short excess 都略高，
next excess 也更高；`mlp_base` 的 next excess 最高，`mlp_wide_huber` 的 short / next Rank IC
最好。初步结论：NN 单模型有足够信号增量，值得补 `wide_deep_h32` 和后续 NN + LGBM ensemble；
但需注意 2025 年 metrics 里的 `model_test_r2` 出现异常大负值，应以后续诊断 loss / target scale
为准，不把 R2 作为本批验收指标。

artifact 索引：

```text
experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1_metrics_by_year.csv
experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1_metrics_by_month.csv
experiments/results/backtests/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1/

experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1_metrics_by_year.csv
experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1_metrics_by_month.csv
experiments/results/backtests/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_shallow_fast_v1/

experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1_metrics_by_year.csv
experiments/results/metrics/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1_metrics_by_month.csv
experiments/results/backtests/nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_wide_huber_v1/

experiments/results/metrics/nn_delay2_36m_2025h2_fullxs_hist_path_pruned_highdup_mlp_smoke_v1_metrics_by_year.csv
experiments/results/metrics/nn_delay2_36m_2025h2_fullxs_hist_path_pruned_highdup_mlp_smoke_v1_metrics_by_month.csv
```

本地验证：

```text
git diff --check
.venv/bin/python -m pytest tests/test_torch_mlp.py tests/test_k8s_helpers.py -q
```

结果：`git diff --check` clean；targeted pytest `14 passed, 2 skipped`。跳过项是本地
`.venv` 未安装 torch，符合可选依赖设计；集群 GPU image 需要用含 torch 的 tag 运行。

建议第一轮 hard-drop 的 17 个特征：

```text
preopen_price_min
ask_price_1
mid_price
preopen_price_max
trade_vwap_1t
trade_vwap_3t
trade_vwap_30t
postopen_volume_rel_1m
postopen_volume_rel_3m
postopen_volume_rel_5m
postopen_v2_trade_volume_to_ask_depth10_1t
postopen_v2_trade_volume_to_ask_depth10_3t
postopen_v2_trade_volume_to_ask_depth10_10t
postopen_v2_trade_volume_to_ask_depth10_30t
postopen_v2_trade_vwap_vs_mid_1t_bps
postopen_v2_trade_vwap_vs_mid_3t_bps
postopen_v2_trade_vwap_vs_mid_10t_bps
```

解释和注意事项：

- `preopen_price_min` 在当前 cache 中为 100% 常数 0；`preopen_price_max` 与
  `preopen_last_price` 相关为 1.0。当前 preopen price path 没有提供有效连续竞价路径信息，
  first-pass 应保留 `preopen_last_price`，后续若继续做集合竞价路径特征，需要改为只用
  `price > 0` 的 auction price 或换更合适的 indicative-price 字段。
- `trade_vwap_1t = turnover_diff_1t / volume_diff_1t`，代表最近 1 tick 区间成交均价。
  当窗口内无成交时 `0/0` 语义是“没有 VWAP”，保持 NaN 合理；不应填 0 或 1。LightGBM
  可以学习 NaN 分支，但若要区分“无成交”和“坏数据”，后续可显式增加 `has_trade_*`
  indicator。
- `postopen_volume_rel_3m/5m` 是 decision grid 上的 lag-relative 特征：
  `(current - lag) / abs(lag)`。当前 baseline 只用 09:31-09:40 的 10 个 decision points，
  因此前半段天然没有 3m/5m lag；5m 缺失率约 41% 属于窗口边界效应。是否保留长 lag
  应通过 per-decision-time 表现和 drop 对照确认。
- 对 LightGBM 而言，NaN 是可学习状态，不需要机械填补；但 NaN 的经济含义必须一致。
  “无成交导致 VWAP undefined”可以保留 NaN，“字段清洗错误导致 preopen min=0”则应回到
  feature construction 修。
- 相关性阈值解读：`|corr| >= 0.995` 可视为 near duplicate，适合第一轮 hard-drop；
  `0.98-0.995` 是强重复，需要 importance/backtest 确认；`0.90-0.98` 多为同家族窗口梯度，
  只 review 不机械删除。

同一抽样下放宽到 `corr_threshold = 0.90` 后：

| abs corr bin | pair count |
| --- | ---: |
| 0.90-0.95 | 15 |
| 0.95-0.98 | 8 |
| 0.98-0.995 | 5 |
| >=0.995 | 23 |

整体从 `corr>=0.98` 的 28 pairs / 14 clusters 扩展到 `corr>=0.90` 的
51 pairs / 25 clusters。新增部分主要是同家族不同窗口或不同口径特征，例如
`postopen_volume_diff_1m/3m/5m`、`postopen_ask_price_1_rel_1m/3m/5m`、
`avg_ask_price/avg_bid_price`、`postopen_v2_*_depth_3/5/10`。这些属于 review
对象，不作为自动 drop。

下一步：用 hard-drop 17 个特征跑 `hygiene_drop17` 对照，只从模型 feature list 中移除，
不要从 cache 或回测所需列中物理删除 `ask_price_1` 等基础字段。

### 归档和保留口径

- 按用户要求，`build_delay2_2024_cache_v1` 已停止；旧 2023/2024 v1 cache 和过期派生 cache 已从 PVC 清掉。
- `docs/experiment_log.md` 已记录、`experiments/results/**` 有轻量证据、或文档明确引用的 run/job/config
  作为历史证据保留。guard、clean target、two-model alpha-risk、risk penalty 和 attribution 路线都属于这类证据。
- 已运行过的 `experiments/jobs/*.yaml` 是轻量 K8s manifest trace，用来把结果追溯回可执行 Job。
- 本地 `__pycache__`、`.pytest_cache`、`*.egg-info` 可直接清理；`.venv`、`.env` 和 ignored `output/`
  通常不作为项目级瘦身目标。例外是 `output/legacy/predictions/**/*.parquet`：这些只是可重拉的
  debug 本地副本，用后可删。

### 本地结果索引

正式证据优先看 `experiments/results/**`。历史轻量摘要多为 `backtests/<prefix>_*` 平铺文件；
新的多文件 pool-internal 归档使用 `backtests/<record_prefix>/` 子目录。`output/artifacts/**`
只放当前 `2022-2025` baseline 和 pool_L 优化实验的集群侧 compact analysis 本地同步副本，方便查看完整
CSV / JSON / SVG 包；`output/legacy/**` 只保留旧本地分析和 debug 产物；prediction parquet 可缺省，
按需从 PVC 重拉。

| local path | source |
| --- | --- |
| `output/legacy/predictions/<run_id>/predictions_all.parquet` | 可选本地 debug 副本；对应 `experiments/runs/<run_id>.toml`、K8s training job 和 sync 记录，可删除后按需重拉。 |
| `experiments/results/metrics/<run_id>_metrics_by_year.csv` | 从对应 PVC run output 拉回的 raw metrics。 |
| `output/legacy/reports/opening_1m3d_*` | 小窗 Ridge/GBM 归档实验对比和校正指标。 |
| `output/legacy/reports/opening_1y_next_month_*` | 一年训练、次月测试 Ridge/GBM 归档实验对比和校正指标。 |
| `output/legacy/reports/opening_intraday_top20_1y_next_month` | 旧 GBM/strong baseline replay，对应 `experiments/results/backtests/opening_intraday_top20_1y_next_month/` 归档摘要。 |
| `output/legacy/reports/opening_intraday_lgbm_delay_replays` | LightGBM delay0/1/2 标准 replay，对应 `experiments/results/backtests/opening_intraday_lgbm_delay_replays/` 归档摘要。 |
| `output/legacy/reports/opening_alpha_horizon_decay_delay2_*` | delay2 horizon decay，对应 `experiments/results/backtests/opening_alpha_horizon_decay_delay2/` 归档摘要。 |
| `output/legacy/reports/opening_delay2_signal_baseline` | delay2 保守 baseline 的分钟四曲线，用于当前 feature-strengthening 门槛。 |
| `output/legacy/artifacts/score_learned_risk_sweep_v1` | 旧 learned-risk sweep artifact；轻量 summary 归档到 `experiments/results/backtests/score_learned_risk_sweep_v1_summary.csv`。 |
| `experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1/` | 18m rolling validation 的轻量 summary / month summary / trace；可重画 short-vs-next tradeoff 图。 |
| `experiments/results/backtests/lgbm_delay2_18m_postopen_mixed_w010_rolling_v1_signal_gate_summary.csv` | mixed label `w_long=0.10` rolling short / next gate 摘要。 |
| `experiments/results/backtests/pool_internal_top100_w010_vs_risk/` | raw alpha、mixed `w=0.10`、gap-risk score 的 pool-internal summary / month summary / mean-by-pool 图表。 |
| `experiments/results/backtests/pool_internal_top100_w020_w030/` | `w=0.20 / 0.30` 在 S/M/L 内部的 Top100 excess summary / month summary。 |
| `experiments/results/backtests/pool_internal_top100_w010_w030/` | raw alpha、mixed `w=0.10`、mixed `w=0.30` 的四张 monthly pool-internal SVG 和 plot data。 |
| `experiments/results/backtests/pool_internal_monthly_rank_ic_3models.csv` | raw alpha、mixed `w=0.10`、`gap 0.30 p80` 对应四图的 monthly Rank IC 表。 |
| `experiments/results/metrics/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_metrics_by_*` | halfyear mainline 的 14/14 shard 训练 metrics，含 yearly / monthly 汇总。 |
| `experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_pre2020_universe/` | halfyear mainline 的 2018-2019 universe-only pool-internal / Rank IC 分析。 |
| `experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_2020_2024/` | halfyear mainline 的 2020-2024 universe / S / M / L pool-internal、Rank IC 和分年/半年汇总。 |
| `output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_weekly_2020_2024_trading_day_equal` | halfyear mainline 2020-2024 的交易日等权 weekly / 4-week rolling pool-internal 诊断和 SVG。 |
| `experiments/results/backtests/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_2025/` | 2025 OOS extension 的 universe / S / M / L pool-internal、Rank IC、plot data 和分年/半年汇总。 |
| `output/legacy/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1_weekly_2025_trading_day_equal` | 2025 OOS extension 的交易日等权 weekly / 4-week rolling pool-internal 诊断和 SVG。 |
| `experiments/results/backtests/halfyear_window_2020_2025/` | 2020-2025 合并视角的三张核心 SVG：short 半年度、next 半年度、weekly 单周折线；trace 记录输入 run 和 2024-09-30 / 2021-10-04 outlier。 |
| `experiments/results/backtests/baseline_2022_2025_cluster/` | 集群侧 2022-2025 baseline 切片；主展示为 universe + `pool_L` 的季度 excess/IC 和日度累计超额曲线。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_<variant>_v1/` | 2022-2025 pool_L 优化归档；包含首轮 `reg_strong` / `bagging` / `no_preopen_reg_mid` 和第二批 9 个模型实验的集群侧 pool-internal summary / plot data / SVG；flat summary 包括 `pilot_sweep_summary.csv` 和 `second_sweep_summary.csv`。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_summary.csv` | fullxs 四组 2022-2025 universe / `pool_L` pool-internal summary 和 baseline delta。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_v1/` | grouped feature audit 的 8 个半年 shard 合并结果：metrics、permutation、feature/group importance 和 trace。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_v1/` | baseline 276 特征 hygiene / correlation 主审计：0.98 相关阈值、drop/review candidates、keep/drop list 和 trace。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_hygiene_corr09_v1/` | 同一抽样的 0.90 相关阈值 sensitivity hygiene 审计，用于人工复核更宽相关簇。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_feature_hygiene_corr09_v1/` | baseline + hist_surprise + path_shape 精确并集的 0.90 相关阈值 hygiene sensitivity；50k rows，354 features，19 个 drop candidates。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/` | `hist_path` 删除 26 个高重复 hist/path 特征后的 pool-internal closeout；328 features，信号基本保留，`hist_path_pruned_vs_before_summary.csv` 为剪枝前后 compact 对比。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_exposure_audit/` | 剪枝后实验的 `pool_L` Top100 core exposure audit；主要画像是 opening activity / turnover heat 高、spread 低、单票集中度不高，作为生产化暴露验收的第一轮证据。 |
| `experiments/results/backtests/exposure_input_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_size_industry_v1/` | 剪枝后 size/industry audit 的外部 exposure input trace；完整 parquet 留在 `output/legacy/exposures/` 和 PVC。 |
| `experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1_size_industry_exposure_audit/` | 剪枝后实验补齐市值和申万一级行业暴露：中等偏大市值，超配电子/电力设备/计算机，低配机械设备/基础化工，并输出行业 active-share 明细。 |
| `experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_v1/` | 剪枝后容量 audit 的早期 `10 亿/clock` 功能 smoke；该口径过重，不作为生产容量结论。 |
| `experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_v1/` | 剪枝后实验的正式 split20 capacity audit；按 `10 亿 / 20 = 5000 万` 每切片目标，`9690/9690` 截面全满，平均 top depth `124`，p95 top depth `161`，max top depth `291`。 |
| `experiments/results/backtests/capacity_audit_lgbm_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_split20_t10_v1/` | split20 capacity sensitivity；把参与率基准换成 `turnover_diff_10t` 后仍全满，但需要吃得更深，p95 top depth `449`。 |
| `experiments/results/backtests/gap_risk_penalized_attribution_v1/` | rolling gap-risk Top100 替换归因的 outcome、feature exposure 和 residual-control 证据。 |
| `output/artifacts/<run_id>` | 当前 2022-2025 cluster baseline / pool_L 优化实验的本地查看副本；正式摘要另归档到 `experiments/results/backtests/`。 |
| `output/legacy/artifacts/<run_id>` | 旧 artifact 拉取和 raw shard metrics，保留给 debug / history。 |
| `output/legacy/predictions/rolling_alpha_conditioned_top100_validation_v1/raw` | 18m rolling 各测试月 prediction shard，用于 alpha Top100 内 risk/short/next 相关诊断。 |

### 2026-06-23 公司 API bridge 口径纠偏

背景：为了排查公司 API `identity` / `negate` 和本地验收的矛盾，曾生成临时桥接诊断
`experiments/results/backtests/company_score_top100_local_next_bridge.csv`。其中 `path_identity`
的 `top100_excess_bps = 36.829821` 来自：

```text
top100_local_next_bps = 48.293240
pool_local_next_bps   = 11.463418
top100_excess_bps     = 36.829821
```

该数值的构造方式是：先把 `09:31-09:40` 的 prediction 按 `date x symbol` 求 mean，按这个
完整窗口 mean score 每天取一次 Top100；再把本地 next-close label 也按 `date x symbol` 对
`09:31-09:40` 求 mean，计算 Top100 减 pool。

结论：`36.829821 bps/day` 不是合法的验收收益，不能作为策略可交易性证据，也不能用来证明
高频 overlay 可以直接融入公司 API。原因是完整窗口 mean score 在 `09:31-09:39` 的买入时点
不可见，若用它解释或回测早期分钟 label，包含未来信息泄露。它最多只能作为
`hindsight stability diagnostic`，说明“事后看 10 分钟持续高分”的股票在 stock-day 层面
表现更强，但不是因果策略。

同日补算的因果/非因果拆分产物为
`experiments/results/backtests/path_shape_daily_score_causal_tradeability_variants.csv`：

| variant | selected bps/day | pool bps/day | excess bps/day | interpretation |
| --- | ---: | ---: | ---: | --- |
| `current_minute_top100_each_minute` | 19.881 | 11.255 | 8.626 | 合法，接近正式本地分钟验收。 |
| `prefix_mean_top100_each_minute_causal` | 18.331 | 11.255 | 7.076 | 合法，每分钟只用当时及以前 score。 |
| `full_window_mean_top100_all_minutes_noncausal` | 43.754 | 11.255 | 32.499 | 非法，早期分钟使用未来 score。 |
| `full_window_mean_top100_0940_only` | 14.083 | 9.450 | 4.633 | 合法，完整 10 分钟 mean 只能在 09:40 后使用。 |

后续口径要求：

- 正式高频策略验收继续使用 `date x decision_time` 的本地分钟级 Top100 / 分批买入 / 换手约束口径；
  不使用完整窗口日频聚合 label 作为 acceptance metric。
- 公司 API 不是该高频 overlay 的天然验收器。若要借用 API 交易逻辑，只能另建因果
  `score adapter`：在固定 `api_time`（如 `09:40` / `09:50` / `10:30`）用当时以前可见的高频信息
  生成 `date x symbol` 分数，并和 neutral baseline 做同口径比较。
- 不再用暴力 `09:31-09:40 mean score + mean label` 把原高频策略强行压成日频分数来解释收益。

### 2026-06-23 Mentor Re-scope: 暴露和容量验收

mentor 进一步明确：后续不能只盯固定 Top100 的收益增强。Top100 仍可作为研发阶段的快速信号诊断，
但进入更接近生产验收时，需要补三类评测：

1. 风格暴露评测：检查选股或组合收益是否主要来自已知风格偏置，而不是 opening short alpha 本身。
   待定义的风格维度包括但不限于动量、反转、波动、流动性、价格层级、成交活跃度和行业/主题集中。
2. 风险暴露评测：重点补市值、流通市值、成交额/换手、价格、波动、行业集中度和极端流动性风险。
   评测对象应同时覆盖 Top100 诊断篮子和后续容量组合。
3. 容量约束组合：从固定 Top100 推进到目标资金规模组合，例如 `10 亿`。容量本身必须有约束，
   不能为了塞满资金而接受过高单票成交占比或过度集中。

容量组合的 first-pass 约束应显式记录：

- 单票下单额 / 当日或近期 ADV 的占比上限。
- 单票下单额 / 开盘可见深度、可成交量或估计可成交量的占比上限。
- 单票权重、行业权重、风格暴露和风险暴露上限。
- 换手、费用和持仓重叠；必要时区分 gross notional、capital-adjusted notional 和实际可成交 notional。
- 若目标容量无法在合理参与率下填满，应报告 feasible capacity，而不是机械输出 `10 亿` 满仓结果。

后续研发顺序：先保留现有 fixed Top100 acceptance 图作为信号对照；同时推进当前特征 NN 单模型，
若单模型有增量再做 NN + LGBM ensemble；再新增 exposure audit 和 capacity portfolio audit，最终用
capacity-constrained next net / market-relative alpha 判断候选是否真正可推进。
