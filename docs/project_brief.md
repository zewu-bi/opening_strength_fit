# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 cross-sectional short-horizon alpha。样本粒度是
`trading day x symbol x opening decision time`，模型只能使用 decision point 当时及以前可见的信息，
预测“当前主动买入并短持有约一分钟”的收益 proxy，并检查信号在更长持有期和外部股池中的迁移表现。

项目窗口是 `09:30:00-09:40:00` 的整分钟 decision points。实验确认 `09:30` 更像特殊 opening
snapshot，因此当前优化子域是 `09:31:00-09:40:00` post-open decision points；这不是改变项目定义，
只是把特殊快照从当前主目标中旁路出来。

## 当前主线

当前主线是单模型 mixed label：

```text
short_label = xs_norm(持有约 1 分钟后用 VWAP 卖出的收益 | date, decision_time)
long_label  = xs_norm(同一买入价持有到第二天收盘的收益 | date, decision_time)
train_label = short_label + w_long * long_label
```

`short_label` 仍是主体，`w_long` 只取小权重，起点按 `0.10` 附近做窄扫。目标不是把模型练成
next-close selector，而是让一分钟短线信号带一点长线稳定性约束。

训练仍使用 full A-share universe。`pool_S`、`pool_M`、`pool_L` 只作为 TopN selection mask：
模型训练和全量打分不按池过滤，也不把 membership 当模型特征；指标在不同 mask 下分别汇总。

执行顺序：

| step | focus | gate |
| --- | --- | --- |
| 1 | 选择 mixed label 的 `w_long` | 在同一 selection mask 下看 short / next 的 Rank IC 和 Top100；short 不能明显受损，next tail 不能明显恶化。 |
| 2 | 做强 short / replay | 固定 `w_long` 后，主目标回到 short Rank IC、Top100 excess 和 replay，相比 raw short baseline 要有明确提升。 |
| 3 | 按 selection mask 切片 | universe / S / M / L 分别报同一组指标；pool 只限制 TopN 候选，不改变训练口径。 |

## 研究口径

默认数据源和窗口：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
data window: 09:15:00 - 09:45:00
project sample window: 09:30:00 - 09:40:00 integer-minute decision points
current optimization slice: 09:31:00 - 09:40:00
```

股票 universe 默认是 A 股 `00/30.SZ` 和 `60/68.SH`。外部候选股池来自
`lml.bzw@ssd/data/pool_{L,M,S}.parquet`，三份文件是 `date x symbol` bool 宽表，当前观察为嵌套池
`pool_S ⊂ pool_M ⊂ pool_L`。读取和配置方法见 [runbook.md](runbook.md#2-外部股池)。

短线收益 label：

```text
decision_t = sampled decision tick
entry_t    = decision_t 之后第 entry_tick_delay 个 tick
buy_price  = ask_price_1[entry_t]
sell_vwap  = VWAP(entry_t + 60s, entry_t + 120s)
label      = sell_vwap / buy_price - 1 - fee_bps / 10000
```

`entry_tick_delay` 是研究用成交代理，不等于真实成交。如果价格已经涨上去，真实挂在原位置的买单可能不会成交。
`09:40` 是正式 decision point；它的 label 使用到约 `09:41-09:42` 的 VWAP 是预期口径，不是出界。
ClickHouse 原始 tick 偶尔出现 6 秒间隔时，不默认视为中间 tick 缺失；通常表示上一条 3 秒 tick 所有字段未变。

主评估：

- `Rank IC`：同一 `date x decision_time` 横截面内的排序能力。
- `Top100 excess`：Top100 相对同横截面均值的 raw short label 超额。
- `replay`：先做短线可执行 proxy，不在第一阶段混入完整交易约束。
- `selection mask`：universe / `pool_S` / `pool_M` / `pool_L` 是切片维度，不是单独的评估指标。
- `next close / next-day close`：只在选择 `w_long` 和诊断 dirty tail 时与 short 同看；固定 `w_long` 后，
  后续信号增强主目标仍是 short。

图表口径：同一指标在多个 selection mask 下展示。baseline 一组至少 3 个柱子；baseline + 一个改进模型
至少 6 个柱子；如果再加 rolling 维度，很容易到 `2 models x 3 pools x 3 windows = 18` 个柱子。
优先用分组柱、按 pool 分面或 small multiples。

## 实验弧线

| stage | question | answer |
| --- | --- | --- |
| 小窗和 1y baseline | 开盘短线是否有方向性？ | 有。小窗 GBM Top20 `+41.92 bps`；1y GBM Top20 `+34.33 bps`。 |
| CPU LightGBM delay | 成交 delay 后信号是否还在？ | 在，但 delay 越长越弱；universe delay2 Top20 `+36.75 bps`。 |
| replay / horizon decay | 是否直接进入交易约束或日频？ | 暂缓。短线 replay 为正，但 close / next close 排序基本消失。 |
| post-open features | 开盘后盘口动态是否有增量？ | 有。`postopen_v2` Rank IC `0.1394`，Top100 `+14.01 bps`。 |
| `09:31-09:40` baseline | 排除特殊 `09:30` 后是否还能学？ | 能。`postopen_0931_0940_baseline` Rank IC `0.1360`，Top100 `+13.45 bps`。 |
| next tail 诊断 | raw short score 是否带隔夜回吐？ | 是。Top100 short excess `+22.21 bps`，next excess `-32.21 bps`。 |
| guard / clean target | 可见信息能否减少 next tail？ | 能，但代价明显；强 clean target 会把 short excess 打到 `+6.21 bps`。 |
| learned risk layer | 两模型 `alpha - risk` 是否可行？ | 可行但复杂；`gap_penalty_030_p80` rolling short / next `+21.20 / +7.84 bps`，归因指向开盘拥挤 tail。 |
| mentor re-scope | 下一步继续两模型还是直接定义 label？ | 直接训练单模型 mixed label；两模型路线封存为历史对照。 |

## 关键事实

`09:30` 是单独 regime：它强，但主要混合集合竞价结果、第一张开盘盘口快照、时间坐标和缺失/0 模式。
当前不围绕它优化。

`09:31-09:40` 有稳定 short alpha。raw post-open baseline 在 `09:31-09:40` 的 short Top100 excess
为 `+22.21 bps`，但同一 Top100 的 next-close excess 为 `-32.21 bps`。这定义了核心矛盾：
短线信号存在，但 raw label 会奖励一部分“短正长负”的拥挤追涨 tail。

可见信息 guard 证明 next tail 可被当前信息部分识别。手工 `next_flip_guard_10t` 使用 spread、turnover heat、
return chase、ask depth、depth imbalance 的横截面 rank 条件，能把 next excess 拉到 `+11.88 bps`，
但 short excess 降到 `+6.77 bps`。它是诊断和 teacher，不是最终规则。

把 guard 直接塞进模型并不自然生效。guard-fail 降权、显式 guard feature、硬 feature core 都没有让 Top100
自动变干净；raw short label 对追涨/热度 tail 的奖励仍然更强。

clean target 能改变排序，但过于防守。`guard_shrunk_target_050_v1` 的 short / next excess 为
`+14.55 / -20.98 bps`；`guard_shrunk_target_075_v1` 为 `+6.21 / +0.07 bps`。这说明长线风险约束有信息，
但不能用强惩罚直接替代 short alpha 目标。

learned risk layer 的经验是：两模型公式能工作，但解释成本高。`bad_tail` v1 太像 next-close selector；
`conditional_bad_tail` v1 学成了 short-alpha 强度 proxy；`alpha_conditioned_reversal` v2/v3 解决了部分问题，
并在 18m rolling 中通过。后续 attribution 显示，被 risk penalty 踢出的原始 Top100 股票更偏高
`preopen_turnover`、`preopen_volume` 和开盘成交增量，符合“开盘拥挤后 next 回吐”的 dirty-tail 画像。
逐月 rolling short-vs-next 证据归档在
`experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv`，可用
`scripts/plot_rolling_validation_tradeoff.py` 重画报告图。
复盘后，这条路线封存为“短+长目标有信息”的证据，而不是继续作为当前主实现。

## 当前不做

- 不把 `final_score = alpha_rank - lambda * gap_risk_rank` 作为当前主线。
- 不把 `w_long` 放大到让模型变成 next-close selector。
- 不按 S/M/L 股池过滤训练，也不把股池 membership 当特征；股池只做 TopN selection mask。
- 不在 short / replay 做强前，把 fee/slippage、多档容量、同股冷却和 T+1 overlay 混进训练目标。
- 不围绕特殊 `09:30` opening snapshot 做主优化。
- 不继续叠加 clean target、risk-shrunk target 和 risk penalty。

## 资料入口

- 真实实验顺序、run、数字和 K8s 输出：见 [experiment_log.md](experiment_log.md)。
- 命令、K8s、artifact sync、股池读取：见 [runbook.md](runbook.md)。
- 代码和脚本索引：见 [project_map.md](project_map.md)。
