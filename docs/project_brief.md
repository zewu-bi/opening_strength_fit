# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：只使用
decision point 当时及以前可见的集合竞价、盘口、成交和短期动量信息，预测“当前主动买入并短持有约一分钟”的
future gross return，并检查模型分数是否能稳定识别更强股票或更好入场时刻。

样本粒度固定为 `trading day x symbol x opening timestamp`。当前 60s label 是
microstructure proxy，不是 A 股 T+1 下的可交易收益。当前阶段只做一件事：把开盘后横截面信号做强。

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
- 当前阶段唯一主目标是把 opening signal 做强；交易约束、容量扩展和日频 overlay 暂不重要。
- 主评估改为 Rank IC 和 Top100 选股收益；Top20 不再作为主评估。
- 容量只保留 ask1 口径；多档 sweep、fee/slippage、同股冷却等不进入当前实验目标。
- 特征重点转向开盘后的盘口信息和 ask/bid 档位。集合竞价会随时间衰弱，可以保留但应轻微削弱，
  不能成为主要依赖。
- 手工 strong candidate filter 没有提升 alpha density，更适合作为候选/约束诊断。
- short-horizon alpha discovery 与 horizon decay 均可归档，下一步不应直接扩大 tick-level replay 资金规模。

## 当前 Baseline

使用 `lgbm_opening_1y_next_month_delay2` 旧模型重新按新口径评估。baseline 不看单个均值，而固定为
`09:30-09:40` 分钟曲线。短期信号用 delay2 60s VWAP proxy label，长期检查用 next close label。

| minute | short Rank IC | short Top100 excess bps | next close Rank IC | next close Top100 excess bps |
| --- | ---: | ---: | ---: | ---: |
| 09:30 | 0.196 | +49.0 | 0.039 | +42.7 |
| 09:31 | 0.085 | +16.6 | -0.026 | -28.6 |
| 09:32 | 0.087 | +16.6 | -0.021 | -16.9 |
| 09:33 | 0.127 | +19.1 | -0.026 | -38.8 |
| 09:34 | 0.138 | +18.3 | -0.026 | -36.5 |
| 09:35 | 0.142 | +20.7 | -0.026 | -37.8 |
| 09:36 | 0.143 | +21.4 | -0.034 | -17.5 |
| 09:37 | 0.140 | +18.3 | -0.033 | -24.5 |
| 09:38 | 0.163 | +21.6 | -0.026 | -34.7 |
| 09:39 | 0.148 | +13.8 | -0.035 | -52.9 |
| 09:40 | 0.127 | +16.8 | -0.037 | -42.7 |

结论：短期高频信号成立，`09:30` 最强且很稳，`09:33-09:40` 仍有明确排序和 Top100 超额；
next close 只有 `09:30` 有正超额，后续分钟转弱。后续实验先抬高 short Rank IC 和 short Top100 曲线，
同时用 next close 两条曲线做长期检查。

## 下一步

| 路线 | 优先级 | 口径 |
| --- | --- | --- |
| Fresh delay2 baseline training | active | 从 delay2 cache 重训当前特征模型，输出同样四张分钟曲线和 feature importance。 |
| Feature dependence audit | active | 评估 `preopen_*`、累计成交字段和盘口特征贡献；集合竞价可以有贡献，但应削弱且不能是主要依赖。 |
| 开盘后盘口特征增强 | active | 加强 ask/bid 档位 gap、深度斜率、ask1/bid1 queue 变化、depth imbalance 变化和成交冲击比例。 |
| 训练目标增强 | secondary | 尝试横截面 demean/zscore label 或排序目标，让训练目标更贴近 Rank IC 和 Top100。 |
| Next close sanity check | secondary | 不作为当前优化目标，只检查增强后的高频信号是否完全牺牲长期表现。 |

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

X 只能使用 decision point 当时及以前可见的信息。新主线优先增强开盘后盘口信息；集合竞价相关特征
可以保留，但必须监控模型依赖度，避免信号主要来自集合竞价：

- 盘口结构：mid price、spread、一档/多档深度、买卖盘不平衡、档位 gap。
- 档位动态：ask/bid 深度斜率、ask1/bid1 queue 变化、深度集中度、档位 gap 变化。
- 成交活跃度：开盘后短窗口 `volume` / `turnover` 增量、成交速度、成交 VWAP、成交冲击比例。
- 动量：相对开盘价、mid/ask/bid 短 tick return。
- 集合竞价：可作为辅助特征保留；通过 feature importance / permutation / ablation 检查权重是否过重。
- 交易约束：涨停距离、A 股 universe、交易状态、candidate filter。

训练/replay 边界：

| 约束 | 当前处理 |
| --- | --- |
| entry delay、horizon、feature set、universe/candidate、固定部署硬过滤 | 影响 label、特征或样本域时重训。 |
| Rank IC、Top100 选股收益 | 当前主评估。 |
| 容量 | 暂只看 ask1。 |
| fee/slippage、状态、spread、tick 新鲜度、同股冷却 | 信号增强后再 replay。 |
| next close | 当前只做 sanity check，不作为优化目标。 |

已归档 replay 使用同一批 predictions 逐步叠加成本、可交易性、流动性和小容量 3/5 档 sweep。
后续不把多档 sweep 作为主线目标，避免过早优化执行假设。

## 模型与评估

历史 baseline 使用 Ridge regression 和 sklearn GBM；已归档主线使用 CPU LightGBM，普通 universe 与
opening-strength candidate 分支共享对应 delay 的 PVC labeled cache。GPU 只是显式配置能力，不是默认路径。

重点指标：

- `cross_section IC`：同一 decision point 下不同股票的排序能力。
- `symbol_day IC`：同一股票当天多个 opening tick 的排序能力。
- `score bucket`：收益是否随模型分数单调变化。
- `Top/Bottom`：TopN 与 BottomN 的 gross label mean、win rate 和 spread。
- `Top100`：当前主选股收益口径；Top20 不再作为主评估。
- `feature dependence`：跟踪集合竞价、累计成交字段和开盘后盘口特征的模型贡献。

当前阶段结论以开盘后盘口特征增强和 feature dependence audit 为下一道门槛。
