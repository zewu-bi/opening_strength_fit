# Project Brief

本文件只回答当前研究怎么走：目标、固定口径、验收 gate 和下一步。完整实验流水、
run id、数字和归档路径见 [experiment_log.md](experiment_log.md)；命令见
[runbook.md](runbook.md)；代码索引见 [project_map.md](project_map.md)。

## 当前目标

继续强化 `09:31-09:40` 开盘短期模型。模型只使用 decision point 当时及以前可见的信息，
在 full universe 上训练和打分；mentor 的隔夜视角以 `pool_L` 股池进入验收。正式展示不是直接
交易 short-horizon return，而是检查短期模型在 `pool_L` 内选 Top100 后，是否增强 next-close
overlay。

当前主线仍是单模型 mixed label：

```text
short_label = xs_norm(约 1 分钟持有收益 | date, decision_time)
long_label  = xs_norm(同一买入价到次日收盘收益 | date, decision_time)
train_label = short_label + 0.30 * long_label
```

## 固定口径

| item | setting |
| --- | --- |
| data source | ClickHouse `stock.tick` |
| sample slice | `09:31:00-09:40:00` integer-minute decision points |
| training universe | A 股 `00/30.SZ`、`60/68.SH`、full universe |
| label | mixed label, `w_long = 0.30` |
| current baseline | archived `soft_core_reg_light` |
| selection masks | universe / `pool_S` / `pool_M` / `pool_L` only at TopN selection |
| primary gates | universe short Rank IC; `pool_L` Top100 next internal excess |
| acceptance figures | `experiments/results/backtests/optimization_overlay_acceptance_2022_2025/` |

`pool_S ⊂ pool_M ⊂ pool_L`，来自 `lml.bzw@ssd/data/pool_{S,M,L}.parquet`。Top100 仍是
研发诊断口径；生产化验收要看容量约束组合。

## 当前判断

- 2022-2025 baseline 已固定：universe short 有稳定排序力，`pool_L` overlay next 为正。
- 常规 LGBM 调参、宽泛特征族加减、简单模型 ensemble 的边际收益很低。
- 有效增量集中在尺度处理和历史/路径类特征：`hist_same_minute_surprise`、`path_shape_confirm`、
  `scale_norm`、`hist_path_rank_centered`。
- `hist_path_rank_centered` 是当前信号强度标尺：universe short Rank IC 和 `pool_L` next excess
  都高于 baseline。
- `hist_path_pruned_highdup` 是当前更干净的生产化候选：删除 26 个高重复 hist/path 特征后，
  信号基本保留，并已补 Top100 exposure、size/industry exposure 和 split20 capacity first-pass。

## 验收 Gate

| gate | 通过含义 |
| --- | --- |
| universe short Rank IC | 短期模型本身更强。 |
| `pool_L` Top100 next internal excess | 叠加 mentor 股池后，隔夜 overlay 更强。 |
| cumulative next net / market-relative alpha | 改善不是少数点造成，累计曲线可解释。 |
| exposure audit | 收益画像可解释，未退化为不可接受的风格、行业或流动性押注。 |
| capacity portfolio | 给定资金规模下，能在参与率、单票权重和集中度约束内装入。 |

不再把 `pool_L` short excess、universe next excess 或 next Rank IC 作为主 gate。short 端不能直接
变成 A 股 T+1 交易收益；next 端由 mentor 股池负责基础收益，短期模型只负责池内 overlay。

## 已知生产化事实

`hist_path_pruned_highdup` 的 first-pass 验收结论：

- 行为画像偏 opening activity / turnover heat，且 spread 偏低；这符合开盘强势股 alpha 机制。
- 市值偏中大，不是小票拥挤；行业超配电子、电力设备、计算机，低配机械设备、基础化工，但不是单行业押注。
- `10 亿 / 20` 的 split20 容量口径下，每个 `date x clock` 目标 `5000 万`，`9690/9690` 截面全满。
- 固定 Top100 不是容量组合：主口径平均吃到 top124，p95 top161，最深 top291；生产组合应允许向后取票并加行业/风格约束。

## 下一步目标

1. 在当前 causal feature set 上训练 NN 单模型，用同一 rolling、pool-internal 和 acceptance 口径比较
   baseline、`hist_path_rank_centered`、`hist_path_pruned_highdup`。
2. 只有 NN 单模型通过主 gate 后，才推进 NN + LGBM ensemble；否则不继续堆 ensemble。
3. 同步补生产化验收：ask-depth 约束、行业/风格上限、capacity-constrained next net 和
   market-relative alpha。
4. 可并行做一个低风险 hygiene 对照：按审计建议 hard-drop 17 个基础高重复/坏语义特征，只从模型
   feature list 移除，不物理删除 cache 或回测所需字段。

## 非目标

- 不重启两模型 `alpha_rank - lambda * gap_risk_rank` 路线；它已作为历史证据封存。
- 不用完整 `09:31-09:40` mean score 压成日频分数做正式验收；那会引入未来信息。
- 不把公司 API 回测当作当前高频 overlay 的天然验收器；若需要，另建因果 score adapter。
