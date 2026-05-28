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
- mentor 反馈后，主线更准确地表述为：先把 short-horizon label / target 做强，尤其是排除特殊
  `09:30` 后的 `09:31-09:40` post-open decision points；特征裁剪、执行 replay、tick-window v3
  都是服务于这个目标的辅助项。
- 主评估改为 Rank IC 和 Top100 选股收益；Top20 不再作为主评估。
- 容量只保留 ask1 口径；多档 sweep、fee/slippage、同股冷却等不进入当前实验目标。
- 特征重点转向开盘后的盘口信息和 ask/bid 档位。集合竞价会随时间衰弱，可以保留但应轻微削弱，
  不能成为主要依赖。
- 手工 strong candidate filter 没有提升 alpha density，更适合作为候选/约束诊断。
- short-horizon alpha discovery 与 horizon decay 均可归档，下一步不应直接扩大 tick-level replay 资金规模。
- 最新 tail-guard 诊断显示，`09:31-09:40` baseline 的 next-close 负 Top100 tail 可以通过可见的
  spread / turnover-flow / chase / depth-balance guard 明显翻转；但这是同一测试月的 post-hoc sweep，
  需要跨月验证后才可进入训练或交易规则。
- clean target 验证了 guard 信息确实有效：二元 guard-shrunk target 的 penalty 越强，Top100 guard-pass
  越高、next-close 负值越收敛；但 short Top100 excess 同步下滑，说明把 risk 直接洗进 short label
  会牺牲一部分核心短周期 alpha。
- 连续 risk-shrunk target 保住了 short Rank IC / Top100，但没有让 Top100 进入足够干净的 guard 区间，
  next-close 仍偏负。它更像温和 target 正则，不像完整的 tail 风险建模。
- 当前下一条主线确定为：保留 baseline / raw-label 模型专心做强 short alpha，另建 learned risk layer
  或 reranker，在建模层面学习“短正长负”的 dirty risk，而不是继续叠加 clean target 和手工后处理。

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
mentor 反馈也支持这个方向：事后真正的 short winner 并不必然隔夜反转，winner 里仍有一批能隔夜延续。
所以机会不在于把 short label 和 overnight label 立刻混合，而在于让模型少学“短期被追买推上去、隔夜回吐”
的部分，多学真正强的 short signal。

## Mentor Feedback

2026-05-27 补充约束：

- ClickHouse 里偶尔出现 6 秒间隔，并不表示中间 tick 缺失；它通常表示上一条 3 秒 tick 的所有字段都没有变化。
  当前不应把这种间隔当作 stale/missing 数据剔除。以后做 raw tick window 时，可以把“长时间未变化”看成盘口稳定性信息。
- 真实交易不是简单 `entry_tick_delay`：如果价格已经涨上去，挂在原位置的买单可能不会成交。当前阶段不建模这个执行约束。
- 以后可以考虑 short label + overnight label 的复合目标，但当前不做，避免过早把问题切到日频 overlay。
- 当前主线仍是把 label 做强，而不是先做更复杂的 tick window 或执行 replay。
- 如果 `09:30` 真的是特殊 opening snapshot 区间，就先不围绕它优化。这个区间容量小，主要信号应从
  `09:31-09:40` post-open decision points 里找。
- 当前分数看起来依赖“短期正、隔夜负”的反转类信号，但历史诊断也显示真正的短期赢家仍可能隔夜为正。
  因此目标不是放弃 short label，而是构造更干净的 short target，让模型少奖励纯交易拥挤，多奖励可延续的强势。

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
当前主线不再围绕它优化。若以后需要机制复盘，再把 `09:30` 开盘快照模型和 `09:31-09:40`
开盘后路径模型分开，并消融显式 `preopen_*`、隐式开盘印记特征、短 tick 回看和时间/区间字段。

`09:38` 的 short Rank IC 有一次回升，可能是另一个微观结构或样本现象；但 Top100 超额没有超过
`09:30`，所以主故事仍然是 `09:30` 这个开盘事件特殊。

## 下一步

| 路线 | 优先级 | 口径 |
| --- | --- | --- |
| Baseline short-alpha spine | active | 以 `lgbm_delay2_postopen_0931_0940_baseline_v1` / raw-label post-open 模型作为 alpha 主干，继续衡量 short Rank IC 和 Top100 excess。 |
| Learned risk layer / reranker | active next | 单独训练 dirty-risk / next-flip 风险层，输入仍只用 decision point 当时可见信息；最终分数用 `alpha_score - penalty * learned_risk` 或 TopK rerank。 |
| Cross-month guard/risk validation | active next | 把 `next_flip_guard_10t` 和 dirty-risk proxy 放到滚动月份验证，确认不是 2022-01 单月 post-hoc。 |
| Post-open label baseline | completed | `09:31-09:40` 训练/评估口径已完成；同域上 Rank IC 和 Top100 raw mean 小幅好于旧统一模型。 |
| Heat-neutral target label | completed / mixed | 50% shrink 保住 short Top100，并收敛 next-close Rank IC，但 short Rank IC 下降、next-close Top100 更差，不能直接通过 gate。 |
| Post-open clean-score check | completed / mixed | 当前 cleaner target 和 hard feature core 都没有同时改善 short 与 next-close Top100。 |
| Feature core辅助实验 | completed / failed gate | 242 个核心特征降低了复杂度，但 short Rank IC/Top100 和 next-close 表现均弱于 `09:31-09:40` baseline。 |
| Top-tail guard sweep | completed / needs validation | `next_flip_guard_10t` 使 next-close Top100 excess 在 10 个分钟全正，但 short excess 降到约 +6 bps，需跨月验证。 |
| Heat-neutral target v2 | completed / modest improvement | 更温和、窄暴露 heat-neutral target 的 short Rank IC/Top100 略好于 baseline，next-close Top100 负值有所收敛但未翻正。 |
| Strong regularization | completed / failed gate | raw-label 强正则 LGBM 没有改善 short 或 next-close Top100。 |
| Clean target / risk-shrunk target | parked as diagnostic | 可作为 guard 有效性的证据保留；当前不继续作为主 alpha 模型目标，避免和 risk layer 重复惩罚。 |
| `09:30` 机制测试 | parked | 如果 `09:30` 确认是特殊 opening snapshot，就不作为主优化目标。 |
| True tick-window v3 | parked | 之后再把 raw tick 回看窗口嵌入 cache 构建；当前先不做。 |
| Short + overnight composite label | parked | 之后可考虑复合目标；当前只作为 sanity check，不进训练目标。 |

下一轮实验的工作定义：

```text
alpha_model = raw short-label post-open baseline
risk_model  = learned dirty-risk / next-flip layer
final_score = alpha_score - lambda * risk_score
```

第一版先把 alpha 和 risk 明确拆开。alpha 模型继续只服务 short-horizon signal strength；risk 模型使用同一时点
可见的 spread、turnover-flow、return-chase、depth、imbalance 以及模型分数上下文，学习哪些 short winner
更像隔夜会回吐的 dirty tail。手工 guard 只作为 weak label / teacher / validation baseline，不作为最终规则。
成功标准不是把 short Top100 压成保守池，而是在 short excess 保留明显强度的同时，让 Top100 guard-pass、
next-close Top100 excess 和跨月稳定性同步改善。

## 数据和 Label

默认数据源：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
window: 09:15:00 - 09:45:00
sample: 09:30:00 - 09:40:00 integer-minute decision points
```

`09:40` 是正式 decision point；它的 60s proxy label 使用到约 `09:41-09:42` 的 VWAP 是预期口径，
不是 label 出界或数据问题。后续若排除特殊开盘快照，应排除 `09:30`，不应默认排除 `09:40`。

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

已归档 Ridge/GBM baseline 使用旧 `entry_tick_delay = 0`；LightGBM 主执行口径经历过
delay0/1/2 对比，近期 post-open 主线使用 delay2 作为保守 proxy。`entry_tick_delay` 只是研究 label
代理，不等价于真实成交：若价格上行，真实挂单可能无法成交。当前阶段暂不把这个执行条件放进训练或 replay。

ClickHouse 原始 tick 偶尔存在 6 秒间隔时，不默认视为数据缺失；mentor 反馈指出这通常表示上一条 3 秒 tick
所有字段均无变化。当前 label 构造按可见 tick 序列处理，后续 raw tick window 可以把这种“不变化时长”
作为稳定性特征，而不是缺失修补问题。

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
