# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 cross-sectional short-horizon alpha。样本是
`trading day x symbol x opening decision time`，特征只使用 decision point 当时及以前可见的信息。

当前优化窗口是 `09:31:00-09:40:00`。`09:30` opening snapshot 单独看作 regime。

## 当前判断

目标：baseline 后四方向特征/模型优化已收尾，下一步做组合和定稿回测。

当前主线是单模型 mixed label：

```text
short_label = xs_norm(持有约 1 分钟后用 VWAP 卖出的收益 | date, decision_time)
long_label  = xs_norm(同一买入价持有到第二天收盘的收益 | date, decision_time)
train_label = short_label + 0.30 * long_label
```

核心假设：full-universe opening score 同时包含真实强弱 / 开盘承接 / 资金方向，以及很短的
microstructure fill、反弹、拥挤和临时成交优势。`pool_L` 做了质量筛选后，score 更偏向前者。

验收口径：

| metric | expectation |
| --- | --- |
| universe short | 提升 |
| `pool_L` short | 提升 |
| `pool_L` next | 跟随提升 |
| universe next | 记录 tail 方向 |

训练和打分仍在 full universe 上完成。`pool_L` 只用于应用和验收切片。

## 当前证据

| run / batch | result |
| --- | --- |
| mixed-label selection | `w_long=0.30` 在 S/M/L 复核后固定。 |
| current baseline | `soft_core_reg_light`，36m rolling，集群侧 pool-internal analysis。 |
| baseline run ids | analysis `baseline_2022_2025_cluster_analysis_v1`；prediction shards from `lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1` and `lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1`。 |
| 2020-2025 rolling summary | S/M/L 池内 short 和 next 均为正；`pool_L` short `+11.1 bps`，next `+13.3 bps`。 |
| 2022-2025 baseline | universe short `+16.8 bps`、next `-8.5 bps`；`pool_L` short `+8.6 bps`、next `+8.0 bps`。 |
| first pilot sweep | `reg_strong`、`bagging`、`no_preopen_reg_mid` 均未超过 baseline。 |
| second batch, 9 runs | 最好只是 `price_path_plus` 的 `pool_L` short `+0.017 bps` 增量；没有实质提升。 |
| cross-sectional relative features | `xs_relative_v1` / `xs_relative_recent_weight_v1` 已归档；前者 short 小幅提升但 next 变弱，后者含样本权重。 |
| model ensemble | `model_ensemble_v1` 已归档；`pool_L` short `+7.635 bps`、next `+6.018 bps`，均低于 baseline。 |
| fullxs feature batch | `hist_same_minute_surprise` 最好：`pool_L` short `+9.127 bps`（delta `+0.501`），next `+8.332 bps`（delta `+0.358`）；`path_shape_confirm` next delta `+0.665` 但 short 只 `+0.044`；`rank_label_regression` IC 高但 Top100 明显变弱。 |
| feature audit | `pool_L` grouped audit 已归档；ablation 中 postopen_v1/v2 对 Top100 最敏感，permutation 中 orderbook_depth 对 Rank IC 最敏感。 |

已归档 baseline compact artifacts：

```text
experiments/results/backtests/baseline_2022_2025_cluster/
```

第二批 9 个模型实验汇总：

```text
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_second_sweep_summary.csv
```

cross-sectional relative features 归档：

```text
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_recent_weight_v1/
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_xs_relative_summary.csv
```

model ensemble 归档：

```text
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_model_ensemble_v1/
```

fullxs / feature audit 归档：

```text
experiments/results/backtests/lgbm_delay2_36m_2022_2025_fullxs_summary.csv
experiments/results/backtests/lgbm_delay2_36m_2022_2025_pool_l_feature_audit_summary.csv
```

完整实验顺序、run id、K8s 状态和数字见 [experiment_log.md](experiment_log.md)。

## 固定口径

| item | setting |
| --- | --- |
| data source | ClickHouse `ch.db.prod.highfortfunds.com / stock.tick` |
| data window | `09:15:00-09:45:00` |
| project window | `09:30:00-09:40:00` integer-minute decision points |
| sample slice | `09:31:00-09:40:00` |
| label | mixed label, `w_long=0.30` |
| training universe | A 股 `00/30.SZ` 和 `60/68.SH` full universe |
| current baseline | archived `soft_core_reg_light` |
| main display | universe + `pool_L` |
| main metrics | Rank IC；池内 Top100 excess |
| acceptance figures | `experiments/results/backtests/optimization_direction_comparison_2022_2025/optimization_directions_daily_cumulative.svg`；`experiments/results/backtests/optimization_direction_comparison_2022_2025/optimization_directions_relative_baseline_daily_cumulative.svg`；`experiments/results/backtests/optimization_direction_comparison_2022_2025/optimization_directions_relative_baseline_yearly_mean.svg` |
| current research focus | xs_relative；hist-surprise；path-shape；clock-segment 已收尾，下一步组合/定稿 |

短线 label：

```text
decision_t = sampled decision tick
entry_t    = decision_t 之后第 entry_tick_delay 个 tick
buy_price  = ask_price_1[entry_t]
sell_vwap  = VWAP(entry_t + 60s, entry_t + 120s)
label      = sell_vwap / buy_price - 1 - fee_bps / 10000
```

外部股池来自 `lml.bzw@ssd/data/pool_{L,M,S}.parquet`，覆盖 `2020-01-02` 至 `2025-12-31`。
`pool_S ⊂ pool_M ⊂ pool_L`。

当前口径称为 decision-time visible / causal feature set。集合竞价摘要、`09:30` 开盘快照、
`09:31-09:40` 开盘后轨迹都可以作为特征，条件是在下单决策时已经可见。

`09:40` 是正式 decision point，它的 label 使用到约 `09:41-09:42` 的 VWAP。ClickHouse
原始 tick 偶尔出现 6 秒间隔时，通常表示上一条 3 秒 tick 所有字段未变。

## 术语

| term | meaning |
| --- | --- |
| `Rank IC` | 同一 `date x decision_time` 横截面内的排序能力。 |
| `Top100 excess` | 池内 Top100 均值减同一 selection mask 内全体候选均值。 |
| `selection mask` | universe / `pool_S` / `pool_M` / `pool_L` 的切片维度。 |
| `next close` | 隔夜延续性检查；当前主看 `pool_L` next。 |

## 里程碑

| stage | conclusion |
| --- | --- |
| baseline / delay | 开盘短线信号为正；delay2 后仍有可学排序。 |
| post-open features | 开盘后盘口动态有增量，`09:31-09:40` 是当前主样本域。 |
| dirty-tail diagnostics | raw short score 有短正长负 tail，后续改为 mixed label 主线。 |
| 2020-2025 mainline | S/M/L 池内 short 和 next 均为正。 |
| 2022-2025 baseline | universe + `pool_L` 集群侧分析已归档，后续信号增强聚焦这一窗口。 |
| 2022-2025 sweeps | 首轮和第二批常规增强尚未形成实质增量。 |

## 入口

- 命令、K8s、artifact sync、股池读取：见 [runbook.md](runbook.md)。
- 实验记录和归档路径：见 [experiment_log.md](experiment_log.md)。
- 代码和脚本索引：见 [project_map.md](project_map.md)。
