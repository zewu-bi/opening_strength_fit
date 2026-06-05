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
并使用轻度 LightGBM sampling / regularization。进入 36m 正式验证后，这个模型配置在说明文档和图表中
简称为 `baseline`；真实 run id 保留为 artifact 追溯键：
`lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1`。
它使用 `36m train -> next 1m test`，覆盖 `2024-01` 至 `2024-12`。当前已同步并归档
`2024-01` 至 `2024-12` 全年结果。
为了检查更早年份和更长 OOS 持有窗口的稳定性，已提交同一 `baseline` 口径的
`36m train -> next 6m test` 半年 rolling：
`lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1`，
覆盖 `2018H1` 至 `2024H2` 共 14 个半年 folds。

`baseline` 全年 S/M/L 池内 Top100 excess 为：

```text
pool_S: short +8.3 bps, next +7.7 bps
pool_M: short +9.3 bps, next +5.6 bps
pool_L: short +10.4 bps, next +4.4 bps
```

当前执行口径：

| item | current setting |
| --- | --- |
| sample slice | `09:31:00-09:40:00` |
| label | mixed label, `w_long=0.30` |
| feature/model | `baseline` (`soft_core_reg_light`) |
| training universe | A 股 `00/30.SZ` 和 `60/68.SH` full universe |
| selection masks | universe / `pool_S` / `pool_M` / `pool_L` |
| main metrics | short Rank IC、池内 Top100 excess；next close 作为 tail 诊断和 mixed-label 定权参考 |
| next validation | `baseline`，2024 全年 12 个 monthly rolling folds；run id `lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1`，已完成并归档 |
| robustness validation | 同一 `baseline`，`36m train -> next 6m test` 半年 rolling；run id `lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1`，覆盖 `2018H1..2024H2` |

训练和全量打分使用 full universe。`pool_S`、`pool_M`、`pool_L` 作为 TopN selection mask；
pool membership 作为输出标记，模型特征保持在 decision-time visible 信息集内。

## 研究口径

默认数据源和窗口：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
data window: 09:15:00 - 09:45:00
project sample window: 09:30:00 - 09:40:00 integer-minute decision points
current optimization slice: 09:31:00 - 09:40:00
```

外部候选股池来自 `lml.bzw@ssd/data/pool_{L,M,S}.parquet`，三份文件是
`date x symbol` bool 宽表，当前观察为嵌套池 `pool_S ⊂ pool_M ⊂ pool_L`。读取和配置方法见
[runbook.md](runbook.md#2-外部股池)。

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
| `selection mask` | universe / `pool_S` / `pool_M` / `pool_L` 是同一模型的切片维度。 |
| `next close` | mixed-label 定权和 dirty-tail 诊断口径。 |
| `replay` | 验证和交易约束诊断工具。 |

图表按 selection mask 分组或分面。baseline 一组通常有 `S/M/L` 三个柱子；baseline + 改进模型是
6 个柱子；叠加 rolling 月份时优先用 small multiples。

## 关键里程碑

| stage | conclusion |
| --- | --- |
| baseline / delay | 开盘短线信号为正；delay2 后仍有可学排序。 |
| post-open features | 开盘后盘口动态有增量，`09:31-09:40` 可以作为当前主样本域。 |
| dirty-tail diagnostics | raw short score 带“短正长负”的拥挤追涨 tail；可见信息能识别其中一部分。 |
| mentor re-scope | 从两模型 `alpha - risk` 切回直接训练 single mixed label。 |
| S/M/L mixed label | 固定 `w_long=0.30`，在 S/M/L 池内保住 short 并改善 next internal excess。 |
| feature/model sweep | 18m 小缓存上晋级 `soft_core_reg_light`，进入 36m 正式验证后命名为 `baseline`。 |
| 36m baseline full year | `2024-01..2024-12` 已同步并归档；S/M/L short 和 next 池内 Top100 excess 均值均为正。 |
| 36m halfyear rolling | 同一 `baseline` 口径已提交 `2018H1..2024H2` 半年 folds，用于稳健性和更早 OOS 检查。 |

`09:30` 是单独 regime：它强，但主要混合集合竞价结果、第一张开盘盘口快照、时间坐标和缺失/0 模式。
当前主优化放在 `09:31-09:40`。

`09:31-09:40` 有稳定 short alpha。raw decision-time baseline 的 short Top100 excess 为
`+22.21 bps`，同一 Top100 的 next-close excess 为 `-32.21 bps`。主线吸收 guard、clean target
和 learned risk layer 的诊断经验，改为直接训练 single mixed label；两模型 final score 留作对照证据。

## 资料入口

- 真实实验顺序、run、数字和 K8s 输出：见 [experiment_log.md](experiment_log.md)。
- 命令、K8s、artifact sync、股池读取：见 [runbook.md](runbook.md)。
- 代码和脚本索引：见 [project_map.md](project_map.md)。
