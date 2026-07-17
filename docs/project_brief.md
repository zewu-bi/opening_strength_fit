# Project Brief

> Last reviewed: 2026-07-17

本文件是当前研究决策 memo，只保留目标、固定口径、候选状态、验收标准和下一步。实验数字与历史见
[experiment_log.md](experiment_log.md)，操作命令见 [runbook.md](runbook.md)，代码位置见
[project_map.md](project_map.md)。

## 目标

在 `pool_L` 内使用分钟级、因果可见的 opening score 做 overlay 排序和调仓，最终验证成本后、容量约束下
的可交易超额。模型可以在 full universe 上训练和打分，但策略只评价 pool 内增量，不追求脱离股池的
独立 universe 选股。

当前 `09:31-09:40` 信号筛选阶段已经收束。`auction_fresh_pruned_grouped_gated_v2_mech_v3`
已通过 Top100 收益验收，且 capacity/realistic 后仍高于既有 capacity anchors；但 short Rank IC 低于
mech328 v2/gated v2，收益仍明显依赖右尾，暂不替换 incumbent。下一阶段
优先对通过收益验收的候选做 refill/完整策略回放、exposure/overlap 与尾部稳健性复核；再把样本扩展到全天
分钟频决策序列。策略侧以现有 selected-order realistic replay 为起点，补成完整、因果、可交易的
`pool_L` overlay 组合回测，并以成本后 PnL 验收。

当前 `2022-2025` 的 `36m train -> next 6m` folds 是研究期 rolling OOS validation，允许用于模型、
特征、target 和候选晋级选择；模型尚未冻结，项目尚未定义 final untouched test。配置和代码中的
`test_*` 仅表示每个 fold 内不参与拟合的 out-of-time evaluation window，不表示最终测试集。

## 固定研究口径

| 项目 | 口径 |
| --- | --- |
| 数据 | 分钟数据来自 ClickHouse `stock.tick`；cap-enriched 线额外使用 `stock.daily_bar_jy` 的严格上一交易日市值/股本；新的 canonical lineage 使用固定 +6 秒逻辑 entry 和边界时刻最后已知状态，auction-fresh/conservative 分别作为严格 gap-gate 与 tick-count 历史对照；本地 parquet 仅作等价输入或调试 |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 当前决策面 | `09:31:00-09:40:00`，整数分钟 |
| 可见性 | 每个 decision point 只使用当时及以前的信息 |
| 训练 universe | A 股 `00/30.SZ`、`60/68.SH` full universe |
| 选择 universe | `pool_L` 为主；universe、`pool_S`、`pool_M` 仅作诊断 |
| 训练目标 | `xs_norm(short_return) + 0.30 × xs_norm(next_close_return)` |
| 历史基线 | `soft_core_reg_light` |
| 当前 incumbent | `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` |

`pool_S ⊂ pool_M ⊂ pool_L`，来自 `lml.bzw@ssd/data/pool_{S,M,L}.parquet`。默认只限制选择，
不限制训练；保守复现实验可使用前一交易日股池。

## 候选状态

| 角色 | 模型 | 用途 |
| --- | --- | --- |
| overlay incumbent | `mech328_v2_robust_zscore` | 当前 `pool_L` next overlay 最强、稳定性已过线 |
| targeted challenger | `mech328_v3_histavg_activity` | 在途的 ratio-style 无量纲化诊断；不做特征值层截面 zscore，保留 NN train-set global zscore；完成前不改变 incumbent |
| archived causal-data challenger | `auction_fresh_pruned_grouped_gated_v2_mech_v3` | Top100 next 收益验收胜出：`pool_L` next excess `16.97 bps`、Top100 8bps net cumulative `9893.9 bps`；capacity acceptance `9244.5 bps`、realistic `7323.9 bps`，但 P95 winsor 后接近/低于 0，暂不替换 incumbent |
| canonical cache challenger | `delay6_clock_state_cap_unique_ticks` | 固定 +6 秒逻辑 entry；entry/sell 边界读取最后已知状态并记录 source time/state age；保留确定性 revision 去重和截面 readiness；base 在途，target/model 等待 |
| coverage-control lineage | `conservative_cap_unique_ticks` | 旧 tick-count 覆盖率语义 + T-1 cap/share + 基础竞价变换；base 已完成，仅作对照，未提交的 target 已 superseded |
| short-ranking anchor | `xs_rank_inplace`、`deep_gelu_huber` | 解释 short Rank IC 上界 |
| structured anchors | `grouped_gated`、`grouped_gated_v2` | 对照收益高度与稳定性 |
| capacity anchors | `mlp_base`、`deep_gelu_mse`、LGBM pruned | 已完成容量/暴露或 realistic replay 的对照 |
| rejected direction | `symbol_zscore`、`mech328_v1`、NN+LGBM rankblend | 已有负面证据，不继续宽扫 |

完整指标和归档路径只记录在 [experiment_log.md](experiment_log.md)。

当前边界：mech328 v2 是 `pool_L` 的 conditional-mean / right-tail head overlay，不是 Top1000
单股 fine ranker；复核中的单股 Rank IC 轻微为负，而 100-name 平滑桶的 group IC 为正。auction-fresh
的同类复核显示 capacity/realistic 结果仍强，但 P95 右尾缩尾后收益接近或低于 0；后续必须用 refill、
完整资金预算与跨状态稳健性证明收益不是少数右尾驱动。

## 晋级标准

信号阶段保留以下回归 gate：

1. `pool_L` Top100 next internal excess 与 market-relative next alpha；
2. 分年、半年、月份和 decision clock 的稳定性；
3. universe short Rank IC 作为辅助解释，不作为 pool overlay 的否决项；
4. capacity、exposure 和 Top100 曲线作为诊断，不单独决定晋级。

策略阶段的主 gate 前移为：

1. 每分钟因果入场、持仓和退出；
2. refill 后的实际资金利用率、同日资金预算与持仓重叠；
3. 手续费、滑点、盘口深度、参与率和交易冲击后的 PnL；
4. 换手、容量、集中度、回撤以及跨年份/市场状态稳定性。

候选晋级记录必须包含：candidate 与 incumbent、唯一改动、数据/cache/pool 版本、样本与 label、
关键 gate、状态、artifact/trace 路径和失败解释。smoke/debug run 不需要完整证据包；写入本 brief 的
结论必须具备完整记录。

## 已有执行证据及边界

2026-07-03 已完成 LGBM 328 与 `mlp_base` 的 first-pass realistic acceptance；2026-07-17 补完
auction-fresh pruned 的同口径 replay。三者都从 capacity-selected child orders 出发，加入日内单票权重、
状态、价差、盘口深度、最小订单和整手约束。它证明执行约束会显著降低简单 capacity acceptance 的
累计收益；auction-fresh 在该约束下仍高于 LGBM/MLP anchors，但右尾依赖更清楚。

这仍不是完整策略回测：

- 被约束掉的订单不会用更低排名股票 refill；
- 每个 decision point 的成交额约束没有汇总成同日真实 turnover budget；
- 没有完整的持仓、退出、资金复用和冲击路径；
- ask-level 归因只有 ask1 有有效深度，ask2-10 为零，不能解释成真实十档可成交性。

因此下一阶段是补齐已有链路，不是从零开始“做真实回测”。

## 下一步

1. 等 fixed-clock +6s 七个年度 base 完成并审计逐日覆盖率/state age，随后构建 mixed-w030 target，
   用同规格 auction-pruned 模型与 auction-fresh v4 做唯一变量 A/B；新的数据实验只使用该 lineage。
2. 对 `auction_fresh_pruned_grouped_gated_v2_mech_v3` 做 refill/完整策略回放、overlap、右尾稳健性和
   market-state 拆解；只有这些复核不推翻收益质量时才进入 incumbent 对照。
3. 等 `mech328_v3_histavg_activity` 完成后，按同一 runbook gate 与 mech328 v2、gated v2、
   auction-fresh pruned v1 做四线对照。
4. 将决策面从开盘 `09:31-09:40` 扩展为全天分钟频决策序列；扩展时继续保证每个 decision point
   只使用当时及以前可见信息。
5. 构建完整、因果、可交易的 `pool_L` overlay 组合回测：覆盖候选 refill、持仓与退出、全日资金预算、
   成本、容量和市场冲击，并以成本后 PnL 验收。

## 非目标

- 不重启两模型 `alpha_rank - λ × gap_risk_rank`；该路线已封存。
- 不用完整 `09:31-09:40` 的事后 mean score 做正式日频验收。
- 不把 Top100 等权收益冒充容量组合收益。
- 不把公司日频 API 当作分钟级策略的天然验收器。
- 不继续做无假设的 plain MLP、wide-deep、低正则或统一 symbol-zscore 宽扫。
- 不以脱离 pool 的独立 universe 策略为当前目标。
