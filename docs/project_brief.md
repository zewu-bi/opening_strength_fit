# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：只使用
decision point 当时及以前可见的集合竞价、盘口、成交和短期动量信息，预测“当前主动买入并短持有约一分钟”的
future gross return，并检查模型分数是否能稳定识别更强股票或更好入场时刻。

样本粒度固定为 `trading day x symbol x opening timestamp`。当前 60s label 是
microstructure proxy，不是 A 股 T+1 下的可交易收益；真实选股价值要用 longer-horizon label
或日频候选池 overlay 继续验证。

## 当前结论

已归档：

1. 数据与 label 闭环：ClickHouse `stock.tick`、A 股过滤、开盘窗口、字段标准化和 labeled
   research dataset 构造流程已打通。
2. Ridge / GBM baseline：`1m3d` 小窗口和 `1y_next_month` baseline 证明开盘高频截面信息有正向排序能力。
3. CPU LightGBM delay：delay0/1/2 普通 universe 与 strong 分支已完成；普通 universe 稳定强于
   strong candidate，delay 越长 IC 和 Top20 replay 越弱。
4. 标准执行约束 replay：`proxy_top20 -> cost -> tradable -> liquidity -> capacity_l3_1m -> capacity_l5_2m`
   已归档，delay1/delay2 universe 在基础 liquidity 和小容量 sweep 下仍为正。
5. Alpha horizon decay：固定 `09:30` opening score 到 close / next close 仍有弱正排序 IC，
   但 next close Top20 收益不稳定；`09:30-09:39` 简单平均后长周期排序基本消失。

阶段判断：

- opening high-frequency proxy signal 存在，且在 2022-01 单月上有稳定横截面排序能力。
- 成交延迟和容量是主要衰减来源；当前 opening replay 只能作为 proxy 压力测试。
- 手工 strong candidate filter 没有提升 alpha density，更适合作为候选/约束诊断。
- short-horizon alpha discovery 与 horizon decay 均可归档，下一步不应直接扩大 tick-level replay 资金规模。

## 下一步

| 路线 | 优先级 | 口径 |
| --- | --- | --- |
| 日频候选池内 opening score rerank / overlay | active | 使用固定 `09:30` 或最早可交易 opening score，对比 overlay 前后的日频候选收益、Rank IC、TopK 命中率、换手和未成交样本。 |
| 日频特征聚合 | secondary | 暂不主推 `09:30-09:39` 简单平均；可研究固定 09:30、早盘前几分钟最大值/分位数、衰减斜率、持续强势次数。 |
| 执行容量复盘 | secondary | 若继续看 tick-level replay，应做容量参数敏感性、目标资金下降、分批执行和盘口冲击建模。 |

## 数据和 Label

默认数据源：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
window: 09:15:00 - 09:45:00
sample: 09:30:00 - 09:40:00 integer-minute decision points
```

关键字段包括 `TradingDay`、`Symbol`、`ExchTimeOffsetUs`、`Volume`、`Turnover`、
`AskPrice1..10`、`BidPrice1..10`、`AskVolume1..10`、`BidVolume1..10` 和 `Status`。
项目内部统一标准化为 `date`、`symbol`、`timestamp`、`volume`、`turnover`、
`ask_price_1` 等 snake_case 列。

short-horizon proxy label：

```text
decision_t = 当前样本 tick
entry_t = decision_t 之后第 entry_tick_delay 个 tick
buy_price = ask_price_1[entry_t]
sell_vwap = VWAP(entry_t + 60s, entry_t + 120s)
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

已归档 Ridge/GBM baseline 使用旧 `entry_tick_delay = 0`；LightGBM 主执行口径是
`entry_tick_delay = 1`，delay0/2 只用于执行敏感性。更长周期衰减检查采用 delay2 opening score
作为保守口径。

## 特征和约束

X 只能使用 decision point 当时及以前可见的信息：

- 盘口结构：mid price、spread、一档/多档深度、买卖盘不平衡、档位 gap。
- 成交活跃度：短窗口 `volume` / `turnover` 增量、成交速度、成交 VWAP。
- 动量：相对昨收/开盘收益、短 tick return。
- 集合竞价：竞价累计量额、竞价末价、竞价价格区间、竞价不平衡。
- 交易约束：涨停距离、A 股 universe、交易状态、candidate filter。

训练/replay 边界：

| 约束 | 当前处理 |
| --- | --- |
| entry delay、horizon、universe/candidate、固定部署硬过滤 | 影响 label 或样本域时重训。 |
| fee/slippage | 固定入选交易的收益扣减，先在 replay 中压测。 |
| 状态、spread、tick 新鲜度、容量、entry 卖盘 sweep、TopN、同股冷却 | 不改模型排序，先 replay。 |
| T+1 | 当前 60s replay 不能解决；用 close / next close 或日频候选 overlay 验证。 |

标准 replay 使用同一批 predictions 逐步叠加成本、可交易性、流动性和小容量 3/5 档 sweep。
如果 prediction 很瘦，可通过 raw tick 或同 delay labeled context 补执行字段；缺关键字段默认报错。

## 模型与评估

历史 baseline 使用 Ridge regression 和 sklearn GBM；已归档主线使用 CPU LightGBM，普通 universe 与
opening-strength candidate 分支共享对应 delay 的 PVC labeled cache。GPU 只是显式配置能力，不是默认路径。

重点指标：

- `cross_section IC`：同一 decision point 下不同股票的排序能力。
- `symbol_day IC`：同一股票当天多个 opening tick 的排序能力。
- `score bucket`：收益是否随模型分数单调变化。
- `Top/Bottom`：TopN 与 BottomN 的 gross label mean、win rate 和 spread。
- `gross replay`：按模型分数选 TopN，用 label 或 context label 回放方向性与稳定性。

当前阶段结论以日频候选池 overlay 的直接验证为下一道门槛。
