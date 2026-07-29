# 全 A、终点对齐 backward 1m 路径 TCN 基准

该实验将分钟价格路径重新定义为窗口终点对齐的历史收益：

```text
r_h(t) = LastPrice(t) / LastPrice(t - h trading minutes) - 1
```

缓存同时物化 1m/10m/60m 三个同步通道，但本实验完全屏蔽 10m/60m，只使用 1m
截面排名路径。它是纯价格路径/动量基准，不包含 opening incumbent score、竞价或盘口特征。

## 固定口径

- 价格：每个交易分钟 endpoint 之前最后一个 `LastPrice` 状态；
- 时钟：09:31–11:30、13:01–15:00，共 240 个 endpoint；午休按交易分钟连续；
- 特征截止：14:47；1m 在 14:47 已经完整已知；
- 输入：全 A 内每个 endpoint 的 1m 收益截面排名和有效性 mask；
- universe：`train_universe = evaluation_universe = all_a`；
- 模型：plain residual TCN、MSE、hidden width 64；
- 窗口：约 33 个月拟合、3 个月验证、随后测试 6 个月；
- OOS：2022H1–2025H2，共 8 folds、969 个交易日；
- 选择：每日全 A score Top100，相对当日全 A 等权均值；
- 固定镜像：`sha256:128aa51a0db962496dc292b301b845f335c9aef1057b55f0c64d13c0b295bf13`。

## 结果

| universe | 日均 Rank IC | Top100 return | 基准 return | Top100 excess | 正半年 | 正月份 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全 A | 0.080573 | 29.94 bps | 4.48 bps | **25.45 bps** | 6/8 | 40/48 |
| pool_L 诊断 | 0.057215 | 23.07 bps | 11.91 bps | **11.16 bps** | 7/8 | 36/48 |

全 A 日超额中位数为 `21.42 bps`，631/969 日为正。逐日对股票 target 做单边
P95/P99 upper-tail cap 后，超额分别为 `9.32/24.02 bps`。2024 年全 A 超额为
`-8.83 bps/day`，因此该信号不能解释为稳定、可直接交易的日收益。

以上均为等权研究指标，未扣交易成本，未实施涨跌停、成交、容量和收盘集合竞价约束。

## 与旧前向执行路径的区别

旧缓存的 1m 值是从时点 t 向未来计算的 ask-to-future-VWAP 执行收益，并通过 14:47
尾部截断保证收盘前可知；它不是原始价格对自身过去时刻做差。旧口径全 A 超额为
`93.57 bps/day`，新 backward 口径降为 `25.45 bps/day`，下降 `68.12 bps`
（仅保留约 27%）。pool_L 诊断也从 `42.49` 降至 `11.16 bps/day`。

这说明旧结果的大部分来自前向执行收益路径的特殊头部结构，而不是普通的 backward-looking
1m 动量。新结果应作为后续 `path-only` 的正式控制组；opening 信号的独立价值必须由
`opening+path` 相对该控制组的增量来判断。
