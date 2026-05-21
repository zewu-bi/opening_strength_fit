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
3. Ridge / GBM baseline：用多组数值特征拟合 label，做 out-of-sample 测试；后续活跃研究主线收敛到 GBM / GBM strong。
4. 简单 replay：测试阶段只用 model prediction 决策，用事后 label 做 gross replay；该 replay 仅用于观察信号方向性与稳定性，不代表真实可交易的 T+1 策略收益。

---

## 后续研究路线

1. **Short-horizon alpha discovery**
   继续使用当前高频 proxy label，验证开盘短周期横截面 alpha 是否稳定，并观察 `entry_tick_delay`、fee、容量与成交约束下的衰减。

2. **Alpha horizon decay / extension**
   构造 30s、60s、5min、close、next open、next close 等 label，研究 opening predictor 的 alpha decay curve 与 horizon persistence。

3. **Daily alpha feature / overlay**
   如果高频 predictor 存在 longer-horizon persistence，将 `09:30-09:40` 的 score 聚合成 stock-day feature，例如 `opening_strength_score`、`opening_score_mean/max`、`top_rank_count` 等，再接入日频模型或 portfolio optimizer。

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

已归档 baseline 使用无成交延迟旧口径；后续新实验默认：

```text
fee_bps = 0
entry_tick_delay = 1
```

做结果比较时，旧归档的 `entry_tick_delay = 0` 和后续的 `entry_tick_delay = 1` 需要分开看。

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

---

## 模型与评估

历史 baseline 使用：

- Ridge regression
- sklearn GBM

后续主线继续使用普通 GBM 和 opening-strength candidate 过滤后的 GBM strong。

训练阶段使用 X 拟合 label；测试阶段仅使用 X 生成 `prediction`，再用事后 label 做评估。

重点指标包括：

- `cross_section IC`：同一 decision point 下不同股票的排序能力，回答“这一刻哪只股票更强”；
- `symbol_day IC`：同一股票同一天多个 opening tick 的排序能力，回答“这只股票今天哪个时刻更强”；
- `score bucket`：收益是否随模型分数单调变化；
- `Top/Bottom`：TopN 与 BottomN 的 gross label mean、win rate 与 spread；
- `gross replay`：按模型分数选 TopN，用 label 回放收益，仅观察方向性与稳定性。

如果当前高频 proxy 闭环无法稳定成立，则后续 horizon extension 与日频 overlay 暂不展开。
