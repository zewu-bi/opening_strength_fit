# 项目简介

## 目标

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：利用集合竞价与开盘后高频盘口/成交数据，在 `09:30-09:40` 构造 decision point，预测短周期未来收益强弱，并评估模型分数的横截面排序价值。

核心问题是：能否仅利用 decision point 当时及此前可见的盘口、成交、竞价与短期动量信息，预测“当前主动买入并持有约一分钟”的 future gross return，并稳定识别更强的交易机会。

样本粒度是 `trading day x symbol x opening timestamp`。模型既可以研究单股票开盘阶段的短周期强弱变化，也可以在同一 decision point 下进行多股票横截面排序。

注意：A 股 T+1 下，当前 60s label 不是最终可交易收益，而是 short-horizon proxy label，用于发现 opening microstructure 中是否存在具有 longer-horizon persistence 的高频信号。真实选股价值需要后续接日频/T+1 label 再验证。

---

## 当前 Baseline

第一版先把高频信号发现闭环跑通：

1. 数据 probe：确认 ClickHouse `stock.tick`、A 股过滤和开盘窗口覆盖正常。
2. 构建 labeled research dataset：按 `date x symbol x timestamp` 计算 short-horizon label，只保留当前及过去可见的 X。
3. Ridge / GBM baseline：用多组数值特征拟合 label，做 out-of-sample 测试；后续活跃研究主线切到 LightGBM，并用 labeled feature cache 降低重复 ClickHouse/feature 构造成本。
4. 简单 replay：测试阶段只用 model prediction 决策，用事后 label 做 gross replay；该 replay 仅用于观察信号方向性与稳定性，不代表真实可交易的 T+1 策略收益。

已归档的 `1y_next_month` baseline 说明：在无成交延迟、无成本、ask1 理想成交的 proxy 条件下，开盘高频截面信息有 alpha 和排序能力。当前阶段不接日频组合回测，也不把 tick-level opening score 聚合成日频组合结果。

---

## 后续研究路线

1. **Short-horizon alpha discovery**
   继续使用当前高频 proxy label，验证开盘短周期横截面 alpha 是否稳定，并观察真实交易约束下的衰减。当前重点是等待 delay cache 完整落盘，训练 LightGBM 普通/strong 的 `entry_tick_delay = 0/1/2` 分支，再用统一 replay 网格压测成本、状态、spread、容量、十档卖盘和同股重复约束。

2. **Alpha horizon decay / extension**
   构造 30s、60s、5min、close、next open、next close 等 label，研究 opening predictor 的 alpha decay curve 与 horizon persistence。

3. **Daily alpha feature / overlay**
   如果高频 predictor 存在 longer-horizon persistence，将 `09:30-09:40` 的 score 聚合成 stock-day feature，例如 `opening_strength_score`、`opening_score_mean/max`、`top_rank_count` 等，再接入日频模型或 portfolio optimizer。

当前 active work 是第 1 步。只有在 delay 和真实交易约束后仍保留稳定 alpha，才推进第 2、3 步。

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

已归档 baseline 使用无成交延迟旧口径；后续新实验按 delay 分支比较。`entry_tick_delay` 会改变买入价和未来退出窗口，因此需要重做 labeled cache；交易成本先不进 delay label，统一在 replay 中扣减。

```text
fee_bps = 0
entry_tick_delay = 0 / 1 / 2
```

做结果比较时，旧归档的 sklearn/Ridge `entry_tick_delay = 0` 和后续 LightGBM `entry_tick_delay = 0/1/2` 需要分开看。

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
- 当前先单独跑 delay0/delay1/delay2；其他约束在同一批 predictions 上统一压测。

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
| entry 卖盘容量 | replay | 使用 prediction 自带或 replay `--context-input` 补齐的 `entry_ask_price_1..10` / `entry_ask_volume_1..10` 检查目标金额能否在 entry tick 的卖盘里成交；十档 sweep 场景会用 sweep VWAP 修正 label。 |
| 容量/参与率 | replay | 默认用 `turnover_diff_30t` 作为可见成交额 proxy，结合 `capital_per_cycle` 和 `max_participation_rate` 判断能否容纳单票目标金额。 |
| TopN/单票权重/现金 | replay | `top_n` 决定目标持仓数；未选满资金留现金；可用 `max_symbol_weight` 限制单票权重。 |
| 同股重复/冷却 | replay | `max_symbol_trades_per_day` 和 `symbol_cooldown_minutes` 防止同一股票在多个 opening cycle 中反复交易。 |
| A 股 T+1 | 后续 horizon | 当前 60s replay 不是可交易收益。真实选股价值必须接 close、next open、next close 等 label，或假设已有库存做 overlay。 |

约束升级原则：

- 先用 delay0/1/2 predictions 跑 `proxy_top20 -> cost -> tradable(10s) -> tradable_5s -> liquidity -> capacity -> strict` replay 场景，观察 IC、bucket 和 replay 是否同步衰减。
- 如果某个执行约束只是改变入选后的收益、容量或现金权重，继续放 replay。
- 如果某个约束会改变训练样本域、候选池定义或未来 label 本身，再创建新的 run config 重训。
- fee/slippage 默认先放 replay；只有在研究“扣费后最优排序”或成本显著改变 label 横截面顺序时，才单独训练 net-label 模型。
- Ask1/十档卖盘不足不要只藏在 slippage 里。slippage 表示平均执行劣化；entry 卖盘容量决定是否能成交、成交多少，以及是否要扫到更高档位。
- replay-only 约束应支持“瘦 prediction + context input”：prediction 提供 score/key，raw tick 或同 delay labeled context 补真实盘口、状态、容量和 realized label；缺关键字段默认报错。

---

## 模型与评估

历史 baseline 使用：

- Ridge regression
- sklearn GBM

后续主线可使用 LightGBM 普通 universe 与 opening-strength candidate 过滤分支，并按 `entry_tick_delay = 0/1/2` 比较延迟衰减。
当前正式路径优先用 CPU LightGBM 读取 PVC labeled cache；GPU 只是显式配置能力，当前没有活跃 GPU run/job。PVC cache 目标路径是 `/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet`，只有最终 `*.parquet` 落盘后才可用于训练；`.tmp.parquet`、lock 和 heartbeat 不算完成结果。当前本地实验注册表只保留已完成的 Ridge/GBM baseline。

训练阶段使用 X 拟合 label；测试阶段仅使用 X 生成 `prediction`，再用事后 label 做评估。

重点指标包括：

- `cross_section IC`：同一 decision point 下不同股票的排序能力，回答“这一刻哪只股票更强”；
- `symbol_day IC`：同一股票同一天多个 opening tick 的排序能力，回答“这只股票今天哪个时刻更强”；
- `score bucket`：收益是否随模型分数单调变化；
- `Top/Bottom`：TopN 与 BottomN 的 gross label mean、win rate 与 spread；
- `gross replay`：按模型分数选 TopN，用 label 回放收益，仅观察方向性与稳定性。

如果当前高频 proxy 闭环无法稳定成立，则后续 horizon extension 与日频 overlay 暂不展开。
