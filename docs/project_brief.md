# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 cross-sectional short-horizon alpha。样本是
`trading day x symbol x opening decision time`，特征只使用 decision point 当时及以前可见的信息。

当前优化窗口是 `09:31:00-09:40:00`。`09:30` opening snapshot 单独看作 regime。

## 当前判断

目标：继续强化 `09:31-09:40` 开盘短期模型。我们的职责是做好开盘强势股短期模型；
mentor 提供的隔夜模型以 `pool_L` 股池形式进入验收。两者做 overlay：先用隔夜视角给出
`pool_L`，再在池内用短期模型选 Top100，最终看隔夜池内收益是否增强。

2026-06-23 mentor re-scope：Top100 仍保留为快速信号诊断，但后续不能只用固定 Top100
定义可接受性。下一阶段需要补三类生产化验收：风格暴露评测、风险暴露评测和容量约束组合。
容量口径从“选 Top100”推进到“给定目标容量的可承载组合”，例如 `10 亿` 资金规模，并显式约束
单票成交占比、可成交量/ADV 占比、单票权重、行业或风格集中度、换手和费用，防止容量指标本身
依赖过高参与率或过度集中。

当前主线是单模型 mixed label：

```text
short_label = xs_norm(持有约 1 分钟后用 VWAP 卖出的收益 | date, decision_time)
long_label  = xs_norm(同一买入价持有到第二天收盘的收益 | date, decision_time)
train_label = short_label + 0.30 * long_label
```

核心假设：短期模型在 full universe 上学到开盘强弱、承接、资金方向和微观结构优势；
隔夜模型给出的 `pool_L` 包含另一套中低频质量/隔夜收益视角。如果短期模型的排序能力真实存在，
它在 `pool_L` 内仍应有正向排序力，并通过 Top100 overlay 强化原本隔夜股池。

下一步方向是继续做强开盘短期模型至收敛，并显式做 price-regime / price-bucket 诊断与交互。
这里的 price-regime 是盘口生态干预：便宜股和贵股的一档集中度、tick/bps 尺度、queue 和成交冲击
不能默认共享同一套特征生效模式。hist-surprise、xs-relative 一类结果说明，下一阶段应优先做
相对 tick、bps、ratio、zscore、rank 和 per-symbol history 的尺度归一化特征。具体实验记录、
复盘理由和运行细节见 [experiment_log.md](experiment_log.md)。

固定研究流程：尝试新的特征工程或模型优化，在集群上按固定 rolling 口径重新训练，同步
pool-internal artifacts，然后先用固定两张验收图评估信号增量，再补暴露和容量评测，避免临时口径漂移。

验收口径：

| metric | expectation |
| --- | --- |
| universe short Rank IC | 提升；直接检验短期模型本身的排序能力。 |
| `pool_L` Top100 next internal excess | 提升；检验短期模型叠加到 mentor 股池后的 overnight overlay 效果。 |
| cumulative next net / market-relative alpha | 保持可解释、稳定；上 panel 同时显示全 A 股市场平均和 `pool_L` background。 |
| style exposure | 待补；检查 Top picks / capacity portfolio 是否只是风格偏置收益。 |
| risk exposure | 待补；重点检查市值、流动性、价格、波动、行业集中度等风险暴露。 |
| capacity-constrained portfolio | 待补；从固定 Top100 推进到目标资金规模组合，例如 `10 亿`，并约束单票参与率、ADV / 可成交量占比、权重和换手。 |
固定两张图：

```text
2022-2025 short rank IC和next pool_L 超额
2022-2025 池内Top100隔夜净收益累和
```

默认图上展示 baseline、hist_surprise 和 path_shape；也可以在保留 baseline 的前提下选择
1-3 个新的 comparison models 一起画。累计图上半 panel 额外显示全 A 股市场平均和
`pool_L` background；下半 panel 显示 `pool_L` background、baseline 和 comparison models
相对全 A 股市场平均的累计 alpha。

不再把 `pool_L` short Rank IC、short Top100 excess、universe next excess 或 next Rank IC
作为主验收项：short 端在 A 股 T+1 下不能直接交易，短期收益能力由 universe short Rank IC
概括；next 端模型本身不负责预测隔夜排序，只看 `pool_L` 内 overlay 后的 Top100 next excess。

## 当前证据

| run / batch | result |
| --- | --- |
| mixed-label selection | `w_long=0.30` 在 S/M/L 复核后固定。 |
| current baseline | `soft_core_reg_light`，36m rolling，集群侧 pool-internal analysis。 |
| 2020-2025 rolling summary | S/M/L 池内 short 和 next 均为正；`pool_L` short `+11.1 bps`，next `+13.3 bps`。 |
| 2022-2025 baseline | universe short `+16.8 bps`、next `-8.5 bps`；`pool_L` short `+8.6 bps`、next `+8.0 bps`。 |
| S/M/L pool tradeoff | 2022-2025 `pool_S/M/L` pool next 均值 `19.8 / 15.2 / 11.2 bps`，真实 pool 换手 `23.3% / 16.2% / 9.9%`，确认小池收益底子更高但换手成本也更高。 |
| first pilot sweep | `reg_strong`、`bagging`、`no_preopen_reg_mid` 均未超过 baseline。 |
| second batch, 9 runs | 历史四格口径下几乎无增量，说明宽泛特征族加减和轻量调参边际收益很低。 |
| cross-sectional relative features | `xs_relative_v1` 提升 universe short Rank IC，但 `pool_L` next overlay 变弱；带 recent weight 的交互组不作为纯特征结论。 |
| model ensemble | `model_ensemble_v1` 的 overlay next 和 short 侧表现均弱于 baseline，本轮不通过。 |
| fullxs feature batch | `hist_same_minute_surprise` short/next 同向改善；`path_shape_confirm` 主要改善 overlay next；`rank_label_regression` 说明 IC 高但 Top100 失败的路线不能直接接受。 |
| price / scale batch | `scale_norm` 曾是综合最好候选：`pool_L` short `+0.615 bps`、next `+0.480 bps` vs baseline；`price_scale_norm` short 更强但 next / capital-adjusted 累计净收益略弱。 |
| hist + path exact-union | `rank_centered` 是最新最好候选：universe short Rank IC `0.1531`，`pool_L` next `+9.45 bps`，优于 baseline 的 `0.1489` / `+7.97 bps`。 |
| feature audit / hygiene | `pool_L` grouped audit 和 baseline 276 hygiene 已归档；fullxs `hist_path` corr09 sensitivity 也已补齐，用于复核 hist_surprise/path_shape 相关簇。 |

完整实验顺序、run id、K8s 状态、归档路径和逐项数字见 [experiment_log.md](experiment_log.md)。

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
| main display | 短期模型 universe 排序 + `pool_L` overnight overlay |
| main metrics | universe short Rank IC；`pool_L` Top100 next internal excess |
| acceptance figures | `experiments/results/backtests/optimization_overlay_acceptance_2022_2025/` |
| current research focus | 继续做强开盘短期模型；优先 price-regime 干预和尺度归一化特征，并补风格/风险暴露与容量约束验收 |

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
| `capacity portfolio` | 在目标资金规模和成交约束下构造的可承载组合；后续用于替代固定 Top100 作为生产化验收口径。 |
| `selection mask` | universe / `pool_S` / `pool_M` / `pool_L` 的切片维度。 |
| `overlay` | mentor 隔夜模型给出 `pool_L`，开盘短期模型在池内选 Top100。 |
| `next close` | overlay 的隔夜验收收益；当前主看 `pool_L` Top100 next internal excess。 |
| `exposure audit` | 对选股或容量组合的风格、风险和集中度暴露做归因，判断收益是否来自目标 alpha 而非不可接受偏置。 |

## 里程碑

| stage | conclusion |
| --- | --- |
| baseline / delay | 开盘短线信号为正；delay2 后仍有可学排序。 |
| post-open features | 开盘后盘口动态有增量，`09:31-09:40` 是当前主样本域。 |
| dirty-tail diagnostics | raw short score 有短正长负 tail，后续改为 mixed label 主线。 |
| 2020-2025 mainline | S/M/L 池内 short 和 next 均为正。 |
| 2022-2025 baseline | universe + `pool_L` 集群侧分析已归档，后续信号增强聚焦这一窗口。 |
| 2022-2025 sweeps | 首轮和第二批常规增强尚未形成实质增量。 |
| current signal-enhancement phase | 常规特征/模型 sweep 增量变小；下一步验证价格生态分层和尺度归一化特征，并用固定两张图验收。 |
| mentor capacity / exposure re-scope | Top100 降为诊断口径；下一步补风格暴露、风险暴露和 `10 亿` 级容量约束组合验收。 |

## 入口

- 命令、K8s、artifact sync、股池读取：见 [runbook.md](runbook.md)。
- 实验记录和归档路径：见 [experiment_log.md](experiment_log.md)。
- 代码和脚本索引：见 [project_map.md](project_map.md)。
