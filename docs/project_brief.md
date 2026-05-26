# 项目简介

## 目标

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：利用集合竞价与开盘后高频盘口/成交数据，在 `09:30-09:40` 构造 decision point，预测短周期未来收益强弱，并评估模型分数的横截面排序价值。

核心问题是：能否仅利用 decision point 当时及此前可见的盘口、成交、竞价与短期动量信息，预测“当前主动买入并持有约一分钟”的 future gross return，并稳定识别更强的交易机会。

样本粒度是 `trading day x symbol x opening timestamp`。模型既可以研究单股票开盘阶段的短周期强弱变化，也可以在同一 decision point 下进行多股票横截面排序。

注意：A 股 T+1 下，当前 60s label 不是最终可交易收益，而是 short-horizon proxy label，用于发现 opening microstructure 中是否存在具有 longer-horizon persistence 的高频信号。真实选股价值需要后续接日频/T+1 label 再验证。

---

## 进度管理

### 已完成 / 已归档

1. **数据与 label 闭环**
   已确认 ClickHouse `stock.tick`、A 股过滤、开盘窗口覆盖和字段口径；已形成可复用 labeled research dataset 构造流程。样本粒度固定为 `date x symbol x decision timestamp`，X 只使用 decision point 当时及以前可见信息。

2. **Ridge / GBM baseline**
   已完成 `1m3d` 小窗口和 `1y_next_month` 2021 训练、2022-01 测试 baseline。归档结果说明：在无成交延迟、无成本、ask1 理想成交的 proxy 条件下，开盘高频截面信息有正向排序能力。该阶段只证明 signal discovery 闭环，不代表真实可交易的 T+1 策略收益。

3. **CPU LightGBM delay 扩展**
   已用 PVC labeled cache 完成 delay0/1/2 的普通 universe 与 opening-strength strong 分支训练。普通 universe 分支稳定强于 strong candidate 分支；delay 越长，group rank IC 和 Top20 replay 均明显衰减。delay1 是主执行口径，delay2 作为更保守成交敏感性。

4. **标准执行约束 replay**
   已完成 `proxy_top20 -> cost -> tradable -> liquidity -> capacity_l3_1m -> capacity_l5_2m` 六场景 replay，并归档轻量 summary。delay1/delay2 universe 在基础可交易、liquidity 和小容量 3/5 档 sweep 约束下仍为正。

5. **更长周期衰减检查**
   已完成 delay2 opening score 的 horizon decay 检查，分别看固定 `09:30` cohort 与 `09:30-09:39` 十个开盘分钟平均。固定 `09:30` 的 Rank IC 从 1m 到 10m 逐步衰减，到 close / next close 仍有弱正排序；但 Top20 next close 收益不稳定且为负。简单平均整个开盘 10 分钟后，close / next close 的排序效果基本消失。

### 阶段结论

- 开盘 high-frequency proxy signal 存在，且在 2022-01 单月上有稳定横截面排序能力。
- 这个信号对成交延迟敏感：delay0 到 delay2 逐步衰减，但 delay2 universe 在无约束和基础 liquidity 约束下仍为正。
- 手工 strong candidate filter 没有提升 alpha density；它更像交易候选/约束诊断，不适合作为当前主训练样本域。
- 容量仍是主要瓶颈：资金从 100 万/cycle、entry 3 档 sweep 加到 200 万/cycle、entry 5 档 sweep 后，收益继续下降，但没有出现旧 1000 万/10 档压力测试那种不适合作为默认汇报口径的跳崖。
- 更长周期衰减检查说明：固定 `09:30` 的 opening score 可能有弱的日内/隔夜排序信息；把 `09:30-09:39` 简单平均成 stock-day 特征并不成立。
- Short-horizon alpha discovery 与 alpha horizon decay 均可归档；opening proxy 值得作为已有日频候选池的辅助排序信号继续验证，但还没有证明能直接转成可放大的独立实盘策略。

### 下一步

1. **已有日频候选池内的开盘重排序 / 辅助排序信号 - active**
   优先把 opening score 当作已有日频候选池里的二级排序信号，而不是独立选股策略。推荐从固定 `09:30` 或最早可交易开盘分数开始，验证它能否在上游日频候选股中提升最终选择、仓位排序或入场优先级；评估口径应直接比较 overlay 前后的日频候选收益、Rank IC 和命中率。

2. **日频特征聚合 - secondary**
   简单平均 `09:30-09:39` 的 opening score 暂不作为主方案。若需要沉淀成 stock-day feature，应优先研究固定 `09:30`、早盘前几分钟最大值/分位数、衰减斜率、持续强势次数等结构化聚合，而不是直接取十分钟均值。

3. **执行容量复盘 / 小容量策略诊断**
   如果后续仍关注 tick-level replay，需要优先做容量参数敏感性、目标资金下降、分批执行和更细的盘口冲击建模；这条线是 execution research，不再是当前 alpha discovery 主线。

---

## 数据

数据来自 ClickHouse：

```text
host: ch.db.prod.highfortfunds.com
table: stock.tick
default window: 09:15:00 - 09:45:00
```

其中：

- `09:15-09:30` 用于集合竞价特征；
- `09:30-09:40` 用于构造 decision point；
- `09:40` 之后用于覆盖尾部 short-horizon label。

关键字段包括：

```text
TradingDay
Symbol
ExchTimeOffsetUs
Volume
Turnover
AskPrice1..10
BidPrice1..10
AskVolume1..10
BidVolume1..10
Status
```

项目内部统一标准化为：

```text
date
symbol
timestamp
volume
turnover
ask_price_1
...
```

等列名。

---

## Label

当前 label 定义为：

- 在样本 tick 做决策；
- 默认延迟 `entry_tick_delay` 个 tick 后，用 `ask1` 买入；
- 从实际成交 tick 开始持有 60 秒；
- 再用后续 60 秒成交 VWAP 退出。

定义如下：

```text
decision_t = 当前样本 tick

entry_t = decision_t 之后第 entry_tick_delay 个 tick

buy_price = ask_price_1[entry_t]

sell_vwap =
    (turnover[entry_t+120s] - turnover[entry_t+60s])
    / (volume[entry_t+120s] - volume[entry_t+60s])

label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

其中：

- `volume` 为累计成交量，单位为股；
- `turnover` 为累计成交额，单位为元。

已归档 Ridge/GBM baseline 使用无成交延迟旧口径；已归档 LightGBM 主执行口径采用 `entry_tick_delay = 1`，并补充 delay0/2 敏感性。`entry_tick_delay` 会改变买入价和未来退出窗口，因此需要对应的 labeled cache；交易成本先不进 delay label，统一在 replay 中扣减。

```text
fee_bps = 0
entry_tick_delay = 1
```

做结果比较时，旧归档的 sklearn/Ridge `entry_tick_delay = 0` 和 LightGBM `entry_tick_delay = 1` 主口径需要分开看。delay0/2 只用于执行敏感性或上下界检查；更长周期衰减检查采用 delay2 作为保守 opening score 口径。

---

## X 特征

X 只能使用 decision point 当时及此前可见的信息。

当前重点特征包括：

- 开盘动量：短 tick return、相对开盘价/昨收涨跌幅；
- 成交活跃度：短期 `volume` / `turnover` 增量、成交速度；
- 盘口结构：spread、十档深度、买卖盘不平衡；
- 集合竞价：开盘前成交活跃度、竞价涨幅、竞价量额；
- 交易约束：涨停距离、流动性、交易状态。

---

## 交易约束

- 十档盘口只使用真实价格和挂量，不假设价格档按 `0.01` 连续，也不补虚构档位。
- 回测需要显式处理成本、滑点、成交容量/参与率、重复持仓或同股冷却。
- 组合选择需要约束 TopN、单票权重和未选满现金处理。
- 涨跌停、停牌、交易状态、spread 和流动性过滤应作为执行约束，不应混进未来信息。

训练/replay 边界：

- 改变 label 或训练样本域的口径进训练：`entry_tick_delay`、horizon、universe/candidate、label 有效性、固定部署硬过滤。
- 只改变下单、成交、成本或选股后的约束先放 replay：fee、slippage、spread、容量/参与率、交易状态、涨停距离、同股每日最多一次。
- 已归档 LightGBM 主线固定 delay1；其他执行约束在同一批 predictions 上统一压测，delay0/2 作为执行敏感性。

当前真实约束拆分如下：

| 约束 | 当前处理位置 | 说明 |
| --- | --- | --- |
| 成交延迟 | replay + 可选重训 | replay 可用 raw tick context 和 `--context-entry-tick-delay` 重算 realized label；如果希望模型训练目标也变成 delay label，再单独训练 delay run。 |
| 手续费/滑点 | replay | 对固定入选交易是确定性收益扣减，用同一批 predictions 施加 `--fee-bps` / `--slippage-bps` 即可；除非要用 net label 重训排序器，否则不需要立刻重训 fee。 |
| 交易状态 | label + replay | 训练配置用 `[filters].tradable_statuses` 约束 decision/entry label 有效性；replay 可再次要求 `status` / `entry_status`。 |
| 决策和 entry 新鲜度 | replay | `decision_lag_seconds` 控制目标整分钟到实际 decision tick 的延迟；`entry_max_tick_gap_seconds` 控制 decision 到 entry 路径里的相邻 tick 最大间隔。成交延迟本身用 `entry_delay_seconds` 单独审计。 |
| spread | replay / candidate | strong candidate 可硬过滤 `spread_bps`；正式成交压力测试用 replay 的 `--max-spread-bps`。 |
| 涨停距离 | replay | 如果上游提供 `limit_up_price` 并生成 `ask1_to_limit_up_bps`，replay 可显式跑 `limit_up_room_10s` 或 `--min-limit-up-room-bps`；没有该列时不要把它混进默认 replay 网格。 |
| 一档挂量 | replay | 用 `ask_volume_1` / `bid_volume_1` 做最低可见深度过滤，但这只是存在性检查，不等价于完整成交模型。 |
| entry 卖盘容量 | replay | 使用 prediction 自带或 replay `--context-input` 补齐的 `entry_ask_price_1..N` / `entry_ask_volume_1..N` 检查目标金额能否在 entry tick 的卖盘里成交；默认容量场景使用 3 档和 5 档 sweep，并用真实档位价格的 sweep VWAP 修正 label。 |
| 容量/参与率 | replay | 默认用 `turnover_diff_30t` 作为可见成交额 proxy，结合 `capital_per_cycle` 和 `max_participation_rate` 判断能否容纳单票目标金额。 |
| TopN/单票权重/现金 | replay | `top_n` 决定目标持仓数；未选满资金留现金；可用 `max_symbol_weight` 限制单票权重。 |
| 同股重复/冷却 | replay | `max_symbol_trades_per_day` 和 `symbol_cooldown_minutes` 防止同一股票在多个 opening cycle 中反复交易。 |
| A 股 T+1 | 后续 horizon / overlay | 当前 60s replay 不是可交易收益。close / next close 衰减已补充检查；真实选股价值还需要接已有库存或日频候选池 overlay 验证。 |

约束升级原则：

- delay0/1/2 LightGBM 标准 replay 网格已归档。后续新增执行约束时，继续沿用 `proxy_top20 -> cost -> tradable -> liquidity -> capacity_l3_1m -> capacity_l5_2m` 的递进顺序，区分基础可交易约束和小容量压力测试。
- 如果某个执行约束只是改变入选后的收益、容量或现金权重，继续放 replay。
- 如果某个约束会改变训练样本域、候选池定义或未来 label 本身，再创建新的 run config 重训。
- fee/slippage 默认先放 replay；只有在研究“扣费后最优排序”或成本显著改变 label 横截面顺序时，才单独训练 net-label 模型。
- Ask1/多档卖盘不足不要只藏在 slippage 里。slippage 表示平均执行劣化；entry 卖盘容量决定是否能成交、成交多少，以及是否要扫到更高档位。
- replay-only 约束应支持“瘦 prediction + context input”：prediction 提供 score/key，raw tick 或同 delay labeled context 补真实盘口、状态、容量和 realized label；缺关键字段默认报错。

---

## 模型与评估

历史 baseline 使用：

- Ridge regression
- sklearn GBM

已归档 short-horizon 主线使用 LightGBM 普通 universe 与 opening-strength candidate 过滤分支，主执行口径采用 `entry_tick_delay = 1`。更长周期衰减检查已用 5min、close、next close 等 label 补充验证。
当前正式路径优先用 CPU LightGBM 读取 PVC labeled cache；GPU 只是显式配置能力。PVC cache 目标路径是 `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet`，只有最终 `*.parquet` 落盘后才可用于训练；`.tmp.parquet`、lock 和 heartbeat 不算完成结果。当前本地实验注册表保留已完成的 Ridge/GBM baseline，以及 CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。

训练阶段使用 X 拟合 label；测试阶段仅使用 X 生成 `prediction`，再用事后 label 做评估。

重点指标包括：

- `cross_section IC`：同一 decision point 下不同股票的排序能力，回答“这一刻哪只股票更强”；
- `symbol_day IC`：同一股票同一天多个 opening tick 的排序能力，回答“这只股票今天哪个时刻更强”；
- `score bucket`：收益是否随模型分数单调变化；
- `Top/Bottom`：TopN 与 BottomN 的 gross label mean、win rate 与 spread；
- `gross replay`：按模型分数选 TopN，用 label 回放收益，仅观察方向性与稳定性。

当前高频 proxy 闭环与更长周期衰减检查已证明值得继续研究；后续结论要以日频候选池 overlay 的直接验证为准。
