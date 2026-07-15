# Project Brief

> Last reviewed: 2026-07-10

本文件是当前研究决策 memo，只保留目标、固定口径、候选状态、验收标准和下一步。实验数字与历史见
[experiment_log.md](experiment_log.md)，操作命令见 [runbook.md](runbook.md)，代码位置见
[project_map.md](project_map.md)。

## 目标

在 `pool_L` 内使用分钟级、因果可见的 opening score 做 overlay 排序和调仓，最终验证成本后、容量约束下
的可交易超额。模型可以在 full universe 上训练和打分，但策略只评价 pool 内增量，不追求脱离股池的
独立 universe 选股。

当前 `09:31-09:40` 信号筛选阶段已经收束。下一阶段以现有 selected-order realistic replay 为起点，
补成真正的组合回测，并把样本从开盘 10 个 decision points 扩展到更长历史和全天分钟序列。

## 固定研究口径

| 项目 | 口径 |
| --- | --- |
| 数据 | ClickHouse `stock.tick`；本地 parquet 仅作等价输入或调试 |
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
| targeted challenger | `mech328_v3_histavg_activity` | 在途的 ratio-style 无量纲化诊断；完成前不改变 incumbent |
| short-ranking anchor | `xs_rank_inplace`、`deep_gelu_huber` | 解释 short Rank IC 上界 |
| structured anchors | `grouped_gated`、`grouped_gated_v2` | 对照收益高度与稳定性 |
| capacity anchors | `mlp_base`、`deep_gelu_mse`、LGBM pruned | 已完成容量/暴露或 realistic replay 的对照 |
| rejected direction | `symbol_zscore`、`mech328_v1`、NN+LGBM rankblend | 已有负面证据，不继续宽扫 |

完整指标和归档路径只记录在 [experiment_log.md](experiment_log.md)。

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

2026-07-03 已完成 LGBM 328 与 `mlp_base` 的 first-pass realistic acceptance：从 capacity-selected
child orders 出发，加入日内单票权重、状态、价差、盘口深度、最小订单和整手约束。它证明执行约束会
显著降低简单 capacity acceptance 的累计收益。

这仍不是完整策略回测：

- 被约束掉的订单不会用更低排名股票 refill；
- 每个 decision point 的成交额约束没有汇总成同日真实 turnover budget；
- 没有完整的持仓、退出、资金复用和冲击路径；
- ask-level 归因只有 ask1 有有效深度，ask2-10 为零，不能解释成真实十档可成交性。

因此下一阶段是补齐已有链路，不是从零开始“做真实回测”。

## 下一步

1. 冻结 `mech328_v2_robust_zscore` 为 incumbent；只把 v3 当作有明确假设的 challenger。
2. 将 selected-order replay 改为 pool 内完整组合构造：约束失败后 refill，并统一同日资金与成交预算。
3. 建立分钟级 outcome path：1/3/5/10 分钟收益、顺逆向 excursion、可成交退出价、next-close path
   和成本后收益；scalar target 从 path 派生，策略直接读取 path。
4. 显式实现持仓、退出、持仓重叠、资金复用、成本、冲击、涨跌停和停牌规则。
5. 扩展历史与全天 decision points，再按年份、行情状态、流动性和波动切分 OOS。

## 非目标

- 不重启两模型 `alpha_rank - λ × gap_risk_rank`；该路线已封存。
- 不用完整 `09:31-09:40` 的事后 mean score 做正式日频验收。
- 不把 Top100 等权收益冒充容量组合收益。
- 不把公司日频 API 当作分钟级策略的天然验收器。
- 不继续做无假设的 plain MLP、wide-deep、低正则或统一 symbol-zscore 宽扫。
- 不以脱离 pool 的独立 universe 策略为当前目标。
