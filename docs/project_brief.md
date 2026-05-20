# 项目简介

## 目标

`opening_strength_fit` 做开盘强势股的日内短周期 alpha：用集合竞价和开盘后高频盘口/成交数据，在 `09:30-09:40` 内构造样本，预测短持有期可交易收益，并评估模型分数能否转成交易价值。

样本粒度是 `trading day x symbol x opening timestamp`。主配置默认取 `09:30` 到 `09:40` 的整分钟决策点；模型既可以做单股票入场择时，也可以在多股票数据上做同一时刻的股票排序。

第一版主线先聚焦一个明确问题：

```text
在开盘后的若干决策时刻，能否用当时及此前可见的盘口/成交/竞价信息，
预测“当前主动买入并持有约一分钟”的期望收益，并挑出更值得交易的机会。
```

暂不把第一版目标扩展到复杂组合权重、复杂执行模拟或新模型族。先把数据、label、X/Y 数据集、Ridge baseline 和最简单交易转化跑通。

## 主线拆解

1. 数据 probe：确认 ClickHouse `stock.tick` 能读、schema 符合预期、A 股过滤和开盘窗口数据覆盖正常。
2. 构建 labeled research dataset：按 `date x symbol x timestamp` 计算 label，只保留当前时刻及此前可见的 X，并按日期分区落地。
3. 规则分数 baseline：用单个可解释特征当 prediction，检查方向性和可交易性底线。
4. Ridge baseline：用多组数值特征拟合 label，做本地 smoke 和 rolling out-of-sample 测试。
5. 交易转化：测试阶段只用 model prediction 决策，用真实发生后的 label 评估收益；先做 threshold / top-N / 单股择时这类简单策略。

当前不做的事：

- 不按未来 label 选股或排序。
- 不在第一版做复杂仓位优化。
- 不急着引入 LightGBM、ranking loss 或更复杂模型，除非 Ridge 和规则 baseline 已经证明数据口径值得继续。

## 数据

数据来自 ClickHouse：

```text
host: ch.db.prod.highfortfunds.com
table: stock.tick
default window: 09:15:00 - 09:45:00
```

默认读取窗口比样本窗口更长：`09:15` 用于集合竞价特征，`09:45` 用于覆盖尾部 label。比如 `09:40` 买入样本的 label 需要 `t+60s` 到 `t+120s` 的成交 VWAP，也就是会用到约 `09:42` 的累计成交字段。

参考代码中的关键字段：

- `TradingDay`: 交易日
- `Symbol`: 证券代码
- `ExchTimeOffsetUs`: 从 `00:00:00` 开始的交易所时间偏移，单位微秒
- `Volume`: 累计成交量(股)
- `Turnover`: 累计成交额(元)
- `AskPrice1..10`, `BidPrice1..10`: 十档卖盘/买盘价格
- `AskVolume1..10`, `BidVolume1..10`: 十档卖盘/买盘数量
- `Status`: 交易状态码

项目读取后会标准化为 `date`, `symbol`, `timestamp`, `volume`, `turnover`, `ask_price_1` 等内部列名。盘口档位只使用真实字段，不假设连续价格档。

## Label

当前 label 是当前 tick 主动买入，延迟一分钟开始卖出，用下一分钟成交 VWAP 退出：

```text
buy_price = ask_price_1[t]
sell_vwap = (turnover[t+120s] - turnover[t+60s])
            / (volume[t+120s] - volume[t+60s])
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

`volume` 单位是股，`turnover` 单位是元，二者都是累计字段。默认 `fee_bps = 0`，交易费用可在实验配置里打开。

## X 特征

X 只能使用样本时刻及此前可见的信息，不能使用任何未来字段。第一版重点使用以下可解释特征：

- 开盘动量：当前价格相对前几个 tick、开盘价或昨收价的涨跌幅，刻画开盘后是否继续走强。
- 成交速度：累计 `volume` / `turnover` 在最近若干 tick 的增量，刻画资金关注和交易活跃度。
- 盘口不平衡：买盘深度和卖盘深度的相对强弱，例如十档买量和十档卖量的差占总深度比例。
- spread：`ask_price_1 - bid_price_1` 及其 bps 版本，刻画交易成本和流动性。
- 集合竞价涨幅/量额：集合竞价阶段相对昨收的涨幅、竞价量和竞价额，刻画开盘前强弱。
- 涨停距离：当前卖一价距离涨停价的 bps，刻画强势程度和潜在交易约束。

规则 baseline 会把这些特征分别当作单一 prediction；Ridge baseline 会把多组数值特征一起作为 X。

## 模型和评估

训练阶段用 X 拟合 label；测试/交易阶段只能用 X 生成 `prediction`，再用事后 label 评估。

第一版模型是 Ridge regression baseline。评估分三层：

- 预测层：看 `prediction` 与 label 的相关性，包括按日聚合的 sanity check，以及更贴近本项目的 `cross_section` / `symbol_day` 分组 IC。
- 分组层：按 score bucket 看收益是否单调，高分组是否明显更好。
- 交易层：把 prediction 转成简单交易规则，检查 top-score 或 threshold 选出的机会是否有正收益。

这里的 `daily IC` 只是沿用通用命名，含义是“每天内部所有样本的 prediction-label 相关性”。本项目主指标更应关注：

- `cross_section IC`: 同一决策时刻，不同股票之间的排序能力，回答“这一刻买哪只”。
- `symbol_day IC`: 同一股票同一天多个开盘 tick 之间的排序能力，回答“这只股票今天什么时候买”。

## 第一版交易假设

第一版不使用未来 label 做交易，只使用模型分数：

```text
每个决策时刻生成 prediction
选择 prediction 最高的 top N，或选择 prediction 超过阈值的样本
按等权买入
持有到 label 定义中的退出窗口
用 sell_vwap 计算事后收益
```

这套策略的目的不是马上逼近实盘，而是验证模型是否能挑出更赚钱的“一分钟持有机会”。如果这个最小闭环站不住，后续复杂策略和模型都先不展开。
