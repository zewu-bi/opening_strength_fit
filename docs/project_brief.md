# Project Brief

> Last reviewed: 2026-07-28

## 目标

项目验证一个问题：分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除交易成本、容量约束和
执行限制后仍稳定的可交易超额。模型可以使用全 A 股训练，但只评价既定股池内部的增量，不把它包装成独立
universe 策略。

当前 `09:31-09:40` 信号阶段已经收束。全天 1m/10m/60m 路径和 D→D+1 label 已完成，首个 all-A
1m TCN 暴露出明确的涨停捷径：模型主要识别信号时已经涨停的股票，而不是一般意义上的可交易强势延续。
下一阶段因此拆为两条路线：更早决策时点的可交易强势延续/后续涨停预测，以及 `14:47` 已有持仓的
隔夜留仓判断；执行层继续建立完整的持仓、退出、现金复用、成本和市场冲击账本。

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

全天 all-A 1m TCN 的表面 Top100 return/base/excess 为 `98.06/4.49/93.57 bps`，但 Top100
平均 `33.58%` 是 D 日涨停附近股票，该组贡献 `87.82 bps`，占原始 Top100 return 约 `89.6%`。
事后剔除并补足 100 只后，相对未过滤原池的全 A / `pool_L` excess 为 `10.48/-1.82 bps`；
若基准同步改成相同非涨停过滤池，则池内 excess 为 `12.76/≈0.00 bps`。前者是固定原 benchmark
的压力测试，后者才是重新定义候选池后的池内选股超额。
该结果保留为涨停延续/尾盘已有持仓诊断，不作为广义强势延续或收盘新开仓策略晋级证据。详见
[all-A 1m TCN evidence](../experiments/evidence/backtests/temporal_nn_36m_2022_2025_all_a_rank_1m_tcn_mse_v1/)。

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
- all-A 1m TCN 训练只要求当天任意时刻存在有效路径，日频 target 只要求 D、D+1 收盘价有效，
  没有要求 `14:47`/收盘时仍有卖盘或可成交；
- 旧 `09:31-09:40` opening policy 也可能遇到集合竞价已涨停或开盘数分钟内涨停的股票；现有 ask1、
  status、spread、depth 约束比全天日频 target 更接近执行，但仍需补做信号时涨停状态和收益贡献审计；
- GPU NN 复跑依赖 run/job 中记录的集群镜像和外部 cache。

全天 1m/10m/60m 因果窄标签和 `close→next-close` 日频主 label 均已完成 2019–2025 全量回填。
无模型分析确认 pool_L 的短期路径更像 head selector：`60m@09:30` Top100 超额在 2020–2025 各年为正，
但全截面 Rank IC 近零偏负。日级序列 cache 已发布到 PVC。后续不再把临近收盘 path-only Top100
直接解释为新开仓收益，而按以下顺序推进：

1. 审计 opening Top100 在 `09:31-09:40` 的“已封死 / 有卖盘 / 后续才涨停 / 未涨停”贡献；
2. 在更早的多个决策时点要求实际 ask/status/depth 可入场，并以实际入场价构造 D+1 目标；
3. 信号时已封死股票从普通强势候选池移出，信号后才涨停的股票保留为预测成功；
4. 在过滤后的 universe 内重算 target rank 并完整重训，不能用事后过滤代替；
5. 涨停延续另建排队/封单/后续卖出成交量模型，未成交资金必须留现金；
6. `14:47` 模型只在已有持仓的隔夜留仓/减仓场景下单独验收。

非目标包括继续宽扫普通 MLP、使用收盘后才完成的数据预测 D 收盘入场、把 Top100 等权收益当作容量收益，
或把公司日频 API 当作分钟策略回测器。
