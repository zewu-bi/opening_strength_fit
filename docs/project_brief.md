# Project Brief

> Last reviewed: 2026-07-31

## 目标

项目验证一个问题：分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除交易成本、容量约束和
执行限制后仍稳定的可交易超额。模型可以使用全 A 股训练，但只评价既定股池内部的增量，不把它包装成独立
universe 策略。

当前 `09:31-09:40` 信号阶段已经收束。下一阶段研究同一选股问题在日内的衰减：沿用既有样本、label、
feature、模型、股池、rolling OOS 和验收口径，仅将十分钟决策窗口替换为另外 2–3 个预先固定的日内窗口。
2026-07-22 至 2026-07-29 的全天序列/隔夜 TCN 尝试源于需求理解偏差，已整体封存并停止。
首个 `10:01-10:10` 窗口已于 2026-07-30 完成：short Rank IC 上升，但 `pool_L` short Top100
绝对收益由 `3.1805 bps` 降至 `0.1153 bps`，隔夜 Top100 超额与费用后累和也显著下降。稍晚窗口更像
稳定识别“少跌”股票，而非继续捕获正向头部收益；opening edge 的可交易部分在开盘后半小时内已经明显衰减。
2026-07-31 完成 corrected decision-state `09:31-09:40` 单变量复跑：decision sampling 改为目标时刻
已经可见的最后状态后，`pool_L` next excess 从旧基准的 `17.1714` 小幅提高到 `17.7934 bps`。该结果
现命名为 `opening_model`，并作为最新信号/模型基准；对应训练 cache 命名为 `opening_cache`。旧 v4 的
统一容量/refill 证据只作为策略层历史参考，等待在 `opening_model` 上重跑。

## 固定研究口径

| 项目 | 口径 |
| --- | --- |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 当前决策面 | `09:31:00-09:40:00`，每分钟一个决策点 |
| 可见性 | 特征、股池和执行过滤只使用决策时点可见信息 |
| 训练 universe | A 股 `00/30.SZ`、`60/68.SH` |
| 选择 universe | `pool_L`；S/M/universe 只作诊断 |
| 目标 | `xs_norm(short_return) + 0.30 × xs_norm(next_close_return)` |
| 验证 | `36m train -> next 6m` rolling OOS，覆盖 2022-2025 |
| canonical base cache | `opening_base` |
| canonical training cache | `opening_cache` |
| latest signal/model baseline | `opening_model` |
| downstream strategy reference | archived v4 multiden；待在 `opening_model` 上重跑 |
| ablation baseline | fixed-clock v4 auction-pruned control |
| historical overlay baseline | `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` |

rolling OOS 可用于模型和特征选择，因此不是 untouched final test。配置中的 `test_*` 表示 fold 内不参与拟合
的时间窗，不代表最终冻结测试集。

## 最新数据源合同

后续新实验只使用下列 corrected v6 数据血缘。三类产物都是最新权威来源，但不表示每个实验同时把三类
都作为训练 label：v6 base 始终提供因果 feature 与 1 分钟 short baseline；独立 3 分钟 short label
只在明确比较或替换 short component 的实验中按样本键 join；corrected next-close label 用于 mixed target
的 long component 和对应评估。所有路径均位于
`/mnt/output/opening_strength_fit/cache/`，年度文件以 `{year}` 分片。

| 决策窗口 | v6 base cache（feature + 1m short） | standalone 3m short label | corrected next-close label |
| --- | --- | --- | --- |
| `09:31-09:40` | `opening_2019_2025_label_v6_decision_clock_state_clock6_unique_base_mcap_lag1` | `opening_2019_2025_short_label_v6_clock6_0931_0940_h180_vwap60_v1` | `opening_2019_2025_next_close_decision_clock_state_clock6_0931_0940` |
| `10:01-10:10` | `opening_2019_2025_label_v6_decision_clock_state_clock6_1001_1010_from_start_auction_reuse_mcap_lag1` | `opening_2019_2025_short_label_v6_clock6_1001_1010_h180_vwap60_v1` | `opening_2019_2025_next_close_decision_clock_state_clock6_1001_1010` |
| `14:01-14:10` | `opening_2019_2025_label_v6_decision_clock_state_clock6_1401_1410_from_start_auction_reuse_mcap_lag1` | `opening_2019_2025_short_label_v6_clock6_1401_1410_h180_vwap60_v1` | `opening_2019_2025_next_close_decision_clock_state_clock6_1401_1410` |

共同口径如下：

- 样本键为 `date × symbol × decision_target_timestamp`，decision state 取逻辑时钟当时或此前最后可见状态，
  entry 使用 `decision_target_timestamp + 6s` 的最后可见盘口状态及其 `buy_price`；
- base 内置 1m short label：entry 后持有 `60s`，再以随后 `60s` 累计成交量/成交额差计算卖出 VWAP；
- standalone 3m short label：entry 后持有 `180s`，再以随后 `60s` 的成交 VWAP 卖出；
- corrected next-close label：使用同一 v6 base 行的 `clock+6s buy_price` 计算下一交易日收盘收益，不再读取
  legacy `opening_2013_2025_next_close_labels_v1`；
- `10:00` 和 `14:00` 只作为对应 later-window cache 的 from-start context，正式实验样本仍从
  `10:01` 和 `14:01` 开始；
- base 与 corrected next-close 三个窗口均为 `7/7 completed`；3m 是当前仍在构建的独立产物，只有对应
  年度 parquet、base/3m manifest、构建 trace、成功标记和 key/覆盖率检查全部通过后才允许消费；
- mixed target 必须从所选 short label 与同窗口 corrected next-close label 重新生成。旧 mixed target、
  已完成模型和历史评估不会因新 label 产出而自动更新，也不得作为新实验的数据输入模板。

## 当前结论

| 候选 | `pool_L` next excess | 决策 |
| --- | ---: | --- |
| mech328 v2 | `14.3174 bps`（fixed-clock 重算） | historical overlay baseline |
| fixed-clock v4 control | `16.8024 bps` | 单变量 ablation baseline |
| fixed-clock v4 multi-denominator | `17.1714 bps` | archived previous baseline |
| `opening_model` | **`17.7934 bps`** | 最新信号/模型基准；策略层待重跑 |
| 10:01-10:10 multi-denominator | `6.5491 bps` | completed decay checkpoint；不晋级 |

旧 v4 multiden 的 capacity-only、realistic no-refill、visible pre-trade refill fill 分别为
`100%/81.3916%/99.9970%`，累计资金净收益为 `9217.9/7433.4/8598.7 bps`。refill 相对 no-refill
增加 `1165.3 bps` 累计资金净收益，且成本后结果为正；这是旧 v4 当时晋级的依据，现在只作
`opening_model` 重跑前的 downstream 历史参考。
单边 P95 upper-tail cap 后的 `-8.59 bps` 只说明收益依赖可观测的正尾幅度；该口径保留为收益来源诊断，
不再作为晋级 gate。分期胜率、bootstrap、overlap 与集中度同样保留为风险画像，不自动否决候选。
`opening_model` 的当前入口见
[baseline evidence](../experiments/evidence/baselines/opening_model/)；旧 v4 策略层参考见
[strategy evidence](../experiments/evidence/backtests/strategy_acceptance_clock6_v4_multiden_2022_2025_v1/)。
10:01 窗口的可审阅结果见
[日内衰减验收包](../experiments/evidence/backtests/nn_delay6_clock_state_36m_2022_2025_w1001_1010_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/)。
短名、cache 口径和不可变来源映射见
[canonical registry](../experiments/canonical/opening.toml)。

## 验收逻辑

信号层依次检查：

1. `pool_L` Top100 next internal excess 与相对各窗口匹配 `pool_L` 的 fee-adjusted cumulative excess；
2. 年、半年、月和 decision clock 稳定性；
3. universe short Rank IC 与 Top1000 score-bucket 形状；
4. capacity、exposure 与费用后的累计曲线。

策略层进一步检查：

1. 每分钟因果入场、退出和持仓；
2. 同日资金预算、refill 与持仓重叠；
3. 手续费、价差、深度、参与率和冲击后的 PnL；
4. 集中度、单边/双边尾部敏感性、leave-one-out 与月块 bootstrap 风险画像。

当前晋级依据是同一因果 OOS lineage 下的成本后资本收益、容量和执行可行性。单边 P95/P99 cap、trim、
分期胜率、bootstrap 下界和 overlap 不设自动硬门槛；它们用于解释收益来源、风险预算和后续执行设计。
晋级当前 opening policy 不等同于完成全天持仓账本或批准实盘部署。

## 已知边界与下一步

- visible refill 是决策前过滤后的重新分配，不是收到真实订单失败回报后的二次下单；
- overlap 当前假设各开盘切片持有至 next close，没有通用现金复用账本；
- ask2-10 深度在现有输入中无有效信息；
- 旧 `09:31-09:40` opening policy 的点时涨停审计已完成：Top100 在决策时封板仅
  `1/969,000`，实际 `clock+6s` 入场无卖一为 `0`；距涨停 `100 bps` 内占 Top100
  `0.130%`，剔除重选只改变 `-0.025 bps` excess，不是当前不可买涨停驱动；
  但模型会显著选中随后在 D 日收盘涨停的股票（Top100 `4.677%`，`6.05x` 富集），
  该组贡献 `38.85 bps`，需与“信号时已封板”严格区分；
- GPU NN 复跑依赖 run/job 中记录的集群镜像和外部 cache。

后续按以下顺序推进：

1. 在 `opening_model` 上重跑 unified capacity/no-refill/visible-refill 策略验收；
2. `10:01-10:10` 已完成；继续完成提交前已固定的 `11:01-11:10` 与 `14:01-14:10`；
3. 使用与 `opening_model` 相同的 cache 构建、特征、目标、模型、训练 universe、`pool_L` 选择和
   rolling OOS；
4. 每个窗口独立生成所需的分钟样本并完整重训，只允许 `[sample]` 时钟和相应数据 lineage 变化；
5. 先用固定四图比较 Rank IC、Top100 excess、分期稳定性和费用后曲线；只有保留足够信号的窗口再进入
   capacity/realistic promotion audit；
6. 汇总“窗口时点/距开盘时间 → OOS 选股能力”，回答衰减速度和是否存在午后残余信号。

非目标包括改变现有 target、继续宽扫普通 MLP、把 Top100 等权收益当作容量收益，或把公司日频 API
当作分钟策略回测器。
