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

四宫格进一步说明，baseline 应拆成两个区间理解：

- `09:30` 是单独的开盘快照区间。短期 Rank IC 和 Top100 超额都是全图最高，next close 也还有小幅正值。
  这个点更可能来自集合竞价结果和开盘第一张盘口快照，而不是后续连续竞价路径。
- `09:31-09:40` 的 short Rank IC 和 short Top100 excess 全部为正，说明短周期信号真实存在；
  但同一个 raw score 拿到 next close 后，Rank IC 和 Top100 excess 全部为负。这是横截面超额问题，
  不是市场方向问题。

当前诊断是：

```text
raw score = 能延续的开盘强势 + 短时间的交易拥挤
```

其中“能延续的开盘强势”是后续想保留的部分；“短时间的交易拥挤”主要对应价格、成交额/换手、开盘冲击、
追买这类特征。它短期有惯性，所以 short horizon 正；但隔夜以后追买力量消失，前面被推高的价格回落，
所以 next close 可能变负。

辅助诊断支持这个机制。按事后真实 short label 排序时，short winner 和 next close 仍是正相关；
负的 next close panel 主要来自可交易的 raw prediction score。对 `09:31-09:39` 的 raw score
做简单中性化后：

| 分数口径 | short Top100 excess | next close Top100 excess |
| --- | ---: | ---: |
| raw score | +18.4 bps | -32.0 bps |
| 去掉成交热度暴露 | +5.7 bps | -3.9 bps |
| 去掉价格 + 成交热度暴露 | +8.0 bps | +3.3 bps |

这个表不是最终策略，只是定位问题：raw score 里确实有一组“短期正、隔夜回吐”的 price / turnover
暴露。下一步不是放弃短期训练，而是把 score 做得更干净：保留能延续的开盘强势，削弱短时间交易拥挤。

## 09:30 机制解释

`09:30` 应该单独看成 opening snapshot regime，而不是后续连续竞价路径的证据。

采样器会取目标决策时间之后、5 秒 lag 以内的第一条 tick。在 delay2 baseline prediction set 里，
约 74.9% 的 `09:30` 行精确落在 `09:30:00`，剩下 25.1% 是 `09:30:01` 到 `09:30:04`
之间第一条可见 tick。

这个划分只是 timestamp 划分，不是信息来源划分。`09:30:00` 这一行本身已经可能包含集合竞价和开盘
结果的投影：开盘盘口、spread、十档深度、imbalance、累计成交、开盘价相对昨收、短 tick lookback，
以及显式的 `preopen_*` 聚合。`09:30:00` 之后几秒的行，还可能额外包含开盘后最早几秒的连续竞价成交。

最初的 baseline 里，`09:30` 能被路由到特殊子树，主因不一定是“很多 0”。最关键的是训练特征里有
`exch_time_offset_us`，它基本就是日内时间坐标。只要这个字段在，LGBM 不需要依赖大量 0，就可以按时间
阈值把 `09:30`、`09:31` 和后续分钟切成不同区间。feature audit 里这个字段也不是摆设，有 290 次 split。

0/NaN 模式是辅助证据，而不是主开关。baseline 原始训练特征有 121 个，本地 prediction 文件只保留了
其中 24 个上下文字段；在这 24 个可见字段上：

- `09:30` 整个 bucket：平均每行 1.70 个精确 0，1.06 个 NaN。因为训练前用
  `SimpleImputer(fill_value=0)`，所以模型实际看到约 2.76 / 24 个“0 或被填成 0”的特征。
- 严格 `09:30` 且 `decision_lag_seconds == 0`：平均约 3.04 / 24 个 0/NaN，median 是 3。
- `09:31` 严格 lag0：平均只有 0.70 / 24，median 是 0。

所以 `09:30` 的 0/NaN 分布确实比后面分钟特殊，但不是 baseline 里那种“几十个 postopen 差分全是
0”主导，因为最开始的 baseline 还没有 `postopen_*_diff_1m` 这些特征。

更完整地说，`09:30` 特殊路由可能来自几类信息叠加：

- `exch_time_offset_us` 直接告诉模型当前日内时间。
- `return_30t`、`volume_diff_30t`、`turnover_diff_30t` 这类 lookback 特征在 `09:30`
  缺失率更高，被 imputer 填 0。
- `volume_diff_1t/3t`、`turnover_diff_1t/3t` 在 `09:30` 有约 36.6% 是 0，比后续分钟高。
- `preopen_*` 平等贴在所有分钟上，但模型一旦先按时间切出 regime，就可以在 `09:30` 子树里强用它，
  在后面分钟弱用它。

这不是严格的未来函数，但对机制判断很“脏”：pooled LGBM 可能已经变成隐式分时模型。

`09:30` 强的机制可以再拆成四层：

1. `09:30` 是集合竞价结果刚释放的第一张快照。此时能看到的不只是一个价格，而是开盘盘口、十档深度、
   spread、imbalance、累计成交、开盘价相对昨收和 `preopen_*` 聚合，这些都是集合竞价撮合结果的投影。
2. `preopen_*` 虽然平等贴在 `09:30-09:40` 每个分钟上，本身不随分钟衰减，但模型可以先用时间坐标或
   `09:30` 的缺失/0 模式切出开盘区间，然后在 `09:30` 子树里重用集合竞价信息，在后续分钟降低权重。
3. `09:30` 的部分历史类特征不应解释成连续竞价历史。`return_*`、`volume_diff_*`、`turnover_diff_*`
   在开盘第一分钟如果有值，很多“之前的 tick”可能来自集合竞价或开盘前序列。
4. 当前短周期 label 是 delay2 后未来约 60 秒卖出，天然更适合捕捉开盘第一分钟的价格发现和订单簿重排。
   集合竞价 imbalance 在 `09:30` 还没有被连续竞价完全消化，所以 `09:30` 的 Rank IC 和 Top100
   超额最高是合理的；到 `09:31` 以后，集合竞价信息被交易掉一部分，连续竞价路径虽然更多，但噪声和
   短期回吐也更多。

当前工作结论：`09:30` 强，主要因为开盘事件特殊。它混合了集合竞价结果、第一张开盘盘口快照、
以及一些开盘 tick 历史特征；它不是干净证据，不能直接说明开盘后路径特征解释了 alpha。
后续机制测试应把 `09:30` 开盘快照模型和 `09:31-09:40` 开盘后路径模型分开，并且需要消融显式
`preopen_*`、隐式开盘印记特征、短 tick 回看和时间/区间字段。

`09:38` 的 short Rank IC 有一次回升，可能是另一个微观结构或样本现象；但 Top100 超额没有超过
`09:30`，所以主故事仍然是 `09:30` 这个开盘事件特殊。

## 下一步

| 路线 | 优先级 | 口径 |
| --- | --- | --- |
| `09:30` 区间拆分 | active | 把 `09:30` opening snapshot 和 `09:31-09:40` post-open path 分开评估。 |
| 去时间坐标机制测试 | active | 去掉 `exch_time_offset_us`，确认 `09:30` 不是靠显式时间坐标被模型单独路由。 |
| Heat-neutral / residual score | active | 对 `09:31-09:40` 削弱 price / turnover / opening-impact 暴露，同时保留 short Rank IC 和 Top100 超额。 |
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
