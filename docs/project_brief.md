# Project Brief

> Last reviewed: 2026-07-22

## 目标

项目验证一个问题：分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除交易成本、容量约束和
执行限制后仍稳定的可交易超额。模型可以使用全 A 股训练，但只评价既定股池内部的增量，不把它包装成独立
universe 策略。

当前 `09:31-09:40` 信号阶段已经收束。下一阶段将决策面扩展到全天分钟级，并建立完整的持仓、退出、现金
复用、成本和市场冲击账本。

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
| incumbent | `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` |

rolling OOS 可用于模型和特征选择，因此不是 untouched final test。配置中的 `test_*` 表示 fold 内不参与拟合
的时间窗，不代表最终冻结测试集。

## 当前结论

| 候选 | `pool_L` next excess | 决策 |
| --- | ---: | --- |
| mech328 v2 incumbent | `14.3174 bps`（fixed-clock 重算） | 保留稳定底座 |
| fixed-clock v4 control | `16.8024 bps` | canonical challenger |
| fixed-clock v4 multi-denominator | `17.1714 bps` | 均值仅增 `0.3690 bps`，分期胜率不足，归档 |

control 的 capacity-only、realistic no-refill、visible pre-trade refill fill 分别为
`100%/80.7803%/99.9969%`，累计资金净收益为 `8989.8/7183.6/8431.1 bps`。refill 的增益真实，
但 P95 winsor 后为 `-9.33 bps`，且 overlap 与月块下界恶化，因此 refill 保留为标准验收机制，当前策略
不晋级。可审阅摘要和 trace 位于 `experiments/evidence/`。

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
4. 集中度、缩尾、leave-one-out 与月块 bootstrap。

## 已知边界与下一步

- visible refill 是决策前过滤后的重新分配，不是收到真实订单失败回报后的二次下单；
- overlap 当前假设各开盘切片持有至 next close，没有通用现金复用账本；
- ask2-10 深度在现有输入中无有效信息；
- GPU NN 复跑依赖 run/job 中记录的集群镜像和外部 cache。

下一步使用更简单的 fixed-clock control 扩展全天因果 label/score，并把现有 acceptance 工具嵌入完整策略
账本。非目标包括继续宽扫普通 MLP、用事后全天平均 score 验收、把 Top100 等权收益当作容量收益，或把
公司日频 API 当作分钟策略回测器。
