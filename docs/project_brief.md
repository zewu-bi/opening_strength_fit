# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 cross-sectional short-horizon alpha。样本粒度是
`trading day x symbol x opening decision time`，模型使用 decision point 当时及以前可见的信息，
预测“当前主动买入并短持有约一分钟”的收益 proxy，并检查信号在更长持有期和外部股池中的迁移表现。

项目窗口是 `09:30:00-09:40:00` 的整分钟 decision points。当前优化子域是
`09:31:00-09:40:00`，把特殊的 `09:30` opening snapshot 作为单独 regime 处理。

## 当前主线

当前主线是单模型 mixed label：

```text
short_label = xs_norm(持有约 1 分钟后用 VWAP 卖出的收益 | date, decision_time)
long_label  = xs_norm(同一买入价持有到第二天收盘的收益 | date, decision_time)
train_label = short_label + w_long * long_label
```

`short_label` 是主体，`long_label` 提供小权重稳定性约束。2026-06-03 的
`w_long=0.10 / 0.20 / 0.30` rolling 和 S/M/L selection-mask 复核后，当前固定
`w_long = 0.30`。池内 Top100 excess 的 6 个月均值为：

```text
pool_S: short +10.0 bps, next +6.6 bps
pool_M: short +12.2 bps, next +9.0 bps
pool_L: short +14.1 bps, next +9.4 bps
```

2026-06-04 的 18m feature regroup / LightGBM sampling-regularization sweep 后，当前
feature/model 候选是：

```text
lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1
```

这套口径保留 decision-time 可见的核心盘口、集合竞价和开盘后轨迹特征，减少宽泛累计特征暴露，
并使用轻度 LightGBM sampling / regularization。这组结果是已归档验证；
真实 run id 保留为 artifact 追溯键：
`lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1`。
它使用 `36m train -> next 1m test`，覆盖 `2024-01` 至 `2024-12`。该月度 rolling
已同步并归档 `2024-01` 至 `2024-12` 全年结果。

同一 feature/model 口径的 `36m train -> next 6m test` 半年 rolling 已完成：
`lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1`，
覆盖 `2018H1` 至 `2024H2` 共 14 个半年 folds。2020 年之前没有 S/M/L 股池日期，
因此 `2018-2019` 只做 universe-only 分析；`2020-2024` 已完成 universe / S / M / L
池内 Top100 验收并归档。

2025 OOS extension 也已完成，run id 为
`lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2025_halfyear_rolling_v1`，
覆盖 `2025H1` 和 `2025H2` 两个半年 folds。随后已把 `2020-2024` 主线与 `2025`
OOS extension 合并成 `2020-2025` rolling-window summary，并归档三张核心图：
short halfyear、next halfyear 和 weekly 单周期视图。

最新 mentor 指示：后续信号增强重点看 `2022-2025`；候选池展示和验收主看
`pool_L`。`2022-2025` baseline 已按 universe + `pool_L` 归档，主图为季度
short / next excess + Rank IC，以及日度累计超额曲线。下一步进入特征工程和模型优化。

2024 全年 S/M/L 池内 Top100 excess 为：

```text
pool_S: short +8.3 bps, next +7.7 bps
pool_M: short +9.3 bps, next +5.6 bps
pool_L: short +10.4 bps, next +4.4 bps
```

半年 rolling 的 2020-2024 S/M/L 池内 Top100 excess 为：

```text
pool_S: short +8.9 bps, next +12.0 bps
pool_M: short +10.7 bps, next +13.7 bps
pool_L: short +12.1 bps, next +14.3 bps
```

2018-2019 universe-only short / next internal excess 为 `+28.4 / +10.1 bps`。

2025 OOS extension 的 S/M/L 池内 Top100 excess 为：

```text
pool_S: short +5.1 bps, next +7.6 bps
pool_M: short +5.6 bps, next +8.6 bps
pool_L: short +6.2 bps, next +8.2 bps
```

2020-2025 合并 rolling-window summary 的 S/M/L 池内 Top100 excess 为：

```text
pool_S: short +8.3 bps, next +11.3 bps
pool_M: short +9.8 bps, next +12.9 bps
pool_L: short +11.1 bps, next +13.3 bps
```

2022-2025 baseline 的 universe / `pool_L` 池内 Top100 excess 和 IC 为：

```text
universe: short +16.8 bps, next -8.5 bps, short IC 0.149, next IC 0.004
pool_L:   short  +8.6 bps, next +8.0 bps, short IC 0.138, next IC 0.002
```

该 baseline 归档到 `experiments/results/backtests/baseline_2022_2025_*`。主展示保留
universe 作为参照、`pool_L` 作为验收对象；`pool_S/M` 不进主展示。

当前执行口径：

| item | current setting |
| --- | --- |
| sample slice | `09:31:00-09:40:00` |
| label | mixed label, `w_long=0.30` |
| feature/model | archived `soft_core_reg_light` validation |
| training universe | A 股 `00/30.SZ` 和 `60/68.SH` full universe |
| selection masks | 历史归档保留 universe / `pool_S` / `pool_M` / `pool_L`；后续主展示使用 universe + `pool_L`，验收聚焦 `pool_L` |
| main metrics | short Rank IC、池内 Top100 excess；next close 作为 tail 诊断和 mixed-label 定权参考 |
| completed validation | `soft_core_reg_light`，2024 全年 12 个 monthly rolling folds；run id `lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1`，已完成并归档 |
| completed mainline | 同一 feature/model 口径，`36m train -> next 6m test` 半年 rolling；`2018H1..2024H2` 主线和 `2025H1..2025H2` OOS extension 均已完成并归档 |
| current summary | `2022-2025` baseline 已归档：universe + `pool_L` 季度 excess/IC 和日度累计曲线 |
| next target | 聚焦 `2022-2025` 和 `pool_L`，进入特征工程、训练权重和模型调参 |

训练和全量打分仍使用 full universe。历史验证保留 `pool_S`、`pool_M`、`pool_L` 作为 TopN
selection mask；后续按 mentor 指示主看 `pool_L`，并保留 universe 作为主展示参照。
pool membership 作为输出标记，模型特征保持在
decision-time visible 信息集内。

## 研究口径

默认数据源和窗口：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
data window: 09:15:00 - 09:45:00
project sample window: 09:30:00 - 09:40:00 integer-minute decision points
current optimization slice: 09:31:00 - 09:40:00
```

外部候选股池来自 `lml.bzw@ssd/data/pool_{L,M,S}.parquet`，三份文件是
`date x symbol` bool 宽表，当前覆盖 `2020-01-02` 至 `2025-12-31`，
并观察为嵌套池 `pool_S ⊂ pool_M ⊂ pool_L`。读取和配置方法见
[runbook.md](runbook.md#2-外部股池)。2020 年以前没有股池日期，历史 shard 只做 universe 分析。

短线收益 label：

```text
decision_t = sampled decision tick
entry_t    = decision_t 之后第 entry_tick_delay 个 tick
buy_price  = ask_price_1[entry_t]
sell_vwap  = VWAP(entry_t + 60s, entry_t + 120s)
label      = sell_vwap / buy_price - 1 - fee_bps / 10000
```

`entry_tick_delay` 是研究用成交代理。`09:40` 是正式 decision point，它的 label 使用到约
`09:41-09:42` 的 VWAP。ClickHouse 原始 tick 偶尔出现 6 秒间隔时，通常表示上一条 3 秒 tick
所有字段未变。

术语：当前口径称为 decision-time visible / causal feature set。集合竞价摘要、`09:30` 开盘快照、
`09:31-09:40` 开盘后轨迹都可以作为特征，条件是在下单决策时已经可见。`postopen_v1/v2` 是开盘后
轨迹特征族名称。

主评估：

| term | meaning |
| --- | --- |
| `Rank IC` | 同一 `date x decision_time` 横截面内的排序能力。 |
| `Top100 excess` | 默认指池内 Top100 excess，即 Top100 均值减同一 selection mask 内全体候选均值。 |
| `selection mask` | universe / `pool_S` / `pool_M` / `pool_L` 是同一模型的切片维度；后续主看 `pool_L`。 |
| `next close` | mixed-label 定权和 dirty-tail 诊断口径。 |
| `replay` | 验证和交易约束诊断工具。 |

历史图表按 selection mask 分组或分面。已归档的 `2020-2025` rolling-window summary 保留
short halfyear、next halfyear 和 weekly 单周期视图三张四股池图；这里的 weekly 不是 4w rolling 诊断。

## 关键里程碑

| stage | conclusion |
| --- | --- |
| baseline / delay | 开盘短线信号为正；delay2 后仍有可学排序。 |
| post-open features | 开盘后盘口动态有增量，`09:31-09:40` 可以作为当前主样本域。 |
| dirty-tail diagnostics | raw short score 带“短正长负”的拥挤追涨 tail；可见信息能识别其中一部分。 |
| mentor re-scope | 从两模型 `alpha - risk` 切回直接训练 single mixed label。 |
| S/M/L mixed label | 固定 `w_long=0.30`，在 S/M/L 池内保住 short 并改善 next internal excess。 |
| feature/model sweep | 18m 小缓存上晋级 `soft_core_reg_light`，随后完成 36m 归档验证。 |
| 36m soft-core full year | `2024-01..2024-12` 已同步并归档；S/M/L short 和 next 池内 Top100 excess 均值均为正。 |
| 36m halfyear rolling mainline | 同一 feature/model 口径已完成 `2018H1..2024H2` 半年 folds；2020-2024 S/M/L 与 2018-2019 universe-only 均已归档。 |
| 2025 OOS extension | 同一 feature/model 口径已完成 `2025H1..2025H2`；S/M/L 池内 short 与 next excess 均为正。 |
| 2020-2025 rolling-window summary | 合并视角已归档三张核心图；S/M/L 池内 short 与 next excess 均为正，作为当前总结材料。 |
| 2022-2025 baseline | universe + `pool_L` 季度 excess/IC 和日度累计曲线已归档；`pool_L` short/next 均为正，universe next 为负，后续优化主看 `pool_L`。 |
| next mentor direction | 后续重点做强 `2022-2025` 信号，展示使用 universe + `pool_L`，验收主看 `pool_L`，继续走特征工程和模型优化。 |

`09:30` 是单独 regime：它强，但主要混合集合竞价结果、第一张开盘盘口快照、时间坐标和缺失/0 模式。
当前主优化放在 `09:31-09:40`。

`09:31-09:40` 有稳定 short alpha。raw decision-time baseline 的 short Top100 excess 为
`+22.21 bps`，同一 Top100 的 next-close excess 为 `-32.21 bps`。主线吸收 guard、clean target
和 learned risk layer 的诊断经验，改为直接训练 single mixed label；两模型 final score 留作对照证据。

## 资料入口

- 真实实验顺序、run、数字和 K8s 输出：见 [experiment_log.md](experiment_log.md)。
- 命令、K8s、artifact sync、股池读取：见 [runbook.md](runbook.md)。
- 代码和脚本索引：见 [project_map.md](project_map.md)。
