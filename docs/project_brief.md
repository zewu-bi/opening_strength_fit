# Project Brief

> Last reviewed: 2026-07-29

## 目标

项目验证一个问题：分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除交易成本、容量约束和
执行限制后仍稳定的可交易超额。模型可以使用全 A 股训练，但只评价既定股池内部的增量，不把它包装成独立
universe 策略。

当前 `09:31-09:40` 信号阶段已经收束。下一阶段研究同一选股问题在日内的衰减：沿用既有样本、label、
feature、模型、股池、rolling OOS 和验收口径，仅将十分钟决策窗口替换为另外 2–3 个预先固定的日内窗口。
2026-07-22 至 2026-07-29 的全天序列/隔夜 TCN 尝试源于需求理解偏差，已整体封存并停止。

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
| opening policy / incumbent | fixed-clock v4 auction-pruned multi-denominator |
| ablation baseline | fixed-clock v4 auction-pruned control |
| historical overlay baseline | `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` |

rolling OOS 可用于模型和特征选择，因此不是 untouched final test。配置中的 `test_*` 表示 fold 内不参与拟合
的时间窗，不代表最终冻结测试集。

## 当前结论

| 候选 | `pool_L` next excess | 决策 |
| --- | ---: | --- |
| mech328 v2 | `14.3174 bps`（fixed-clock 重算） | historical overlay baseline |
| fixed-clock v4 control | `16.8024 bps` | 单变量 ablation baseline |
| fixed-clock v4 multi-denominator | `17.1714 bps` | 晋级为当前 opening policy/incumbent |

multiden 的 capacity-only、realistic no-refill、visible pre-trade refill fill 分别为
`100%/81.3916%/99.9970%`，累计资金净收益为 `9217.9/7433.4/8598.7 bps`。refill 相对 no-refill
增加 `1165.3 bps` 累计资金净收益，且成本后结果为正，因此随 multiden 一并纳入当前 opening policy。
单边 P95 upper-tail cap 后的 `-8.59 bps` 只说明收益依赖可观测的正尾幅度；该口径保留为收益来源诊断，
不再作为晋级 gate。分期胜率、bootstrap、overlap 与集中度同样保留为风险画像，不自动否决候选。可审阅
结果见[四图验收包](../experiments/evidence/backtests/nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1/)
和 [strategy evidence](../experiments/evidence/backtests/strategy_acceptance_clock6_v4_multiden_2022_2025_v1/)。

## 验收逻辑

信号层依次检查：

1. `pool_L` Top100 next internal excess 与 market-relative alpha；
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

1. 预先固定另外 2–3 个互不重叠的十分钟日内窗口，避开午休边界；具体时钟在提交实验前写入 run config；
2. 使用与 incumbent 相同的 cache 构建、特征、目标、模型、训练 universe、`pool_L` 选择和 rolling OOS；
3. 每个窗口独立生成所需的分钟样本并完整重训，只允许 `[sample]` 时钟和相应数据 lineage 变化；
4. 用同一套 Rank IC、Top100 excess、分期稳定性、容量和执行指标与 `09:31-09:40` 基准比较；
5. 汇总“窗口时点/距开盘时间 → OOS 选股能力”，回答衰减速度和是否存在午后残余信号。

非目标包括改变现有 target、继续宽扫普通 MLP、把 Top100 等权收益当作容量收益，或把公司日频 API
当作分钟策略回测器。
