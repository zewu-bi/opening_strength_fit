# Project Brief

本文件只回答当前研究怎么走：目标、固定口径、验收 gate 和下一步。不放命令、run 索引或完整
结果表。完整实验流水、run id、数字和归档路径见 [experiment_log.md](experiment_log.md)；命令见
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
| primary gates | universe short Rank IC; `pool_L` next internal excess; market-relative next alpha |
| acceptance surfaces | Top100 equal-weight diagnostics; 10亿 capacity acceptance; exposure audit |

`pool_S ⊂ pool_M ⊂ pool_L`，来自 `lml.bzw@ssd/data/pool_{S,M,L}.parquet`。Top100 仍是
研发诊断口径；10亿容量收益由 capacity acceptance 按 `allocated_notional` 加权 next-close label 计算。

## 当前判断

- 2022-2025 baseline 已固定：universe short 有稳定排序力，`pool_L` overlay next 为正。
- 常规 LGBM 调参、宽泛特征族加减、简单模型 ensemble 的边际收益很低。
- 有效增量集中在尺度处理和历史/路径类特征：`hist_same_minute_surprise`、`path_shape_confirm`、
  `scale_norm`、`hist_path_rank_centered`。
- `hist_path_rank_centered` 是当前信号强度标尺：universe short Rank IC 和 `pool_L` next excess
  都高于 baseline。
- `hist_path_pruned_highdup` 是当前更干净的生产化候选：删除 26 个高重复 hist/path 特征后，
  信号基本保留，并已补 Top100 exposure、size/industry exposure 和 split20 capacity first-pass。
- 容量验收不复用 Top100 等权收益；capacity audit 只看 fill/depth，capacity acceptance 才算收益。
- NN 已完成第一轮、第二轮、MSE neighborhood 和结构化 grouped NN 的前三个归档，不再处于
  “是否值得跑”的阶段。`grouped_gated_gelu_mse` / `grouped_residual_gelu_mse` /
  `grouped_cross_gelu_mse` 的 `pool_L` Top100 next overlay 均超过 `mlp_base` 和
  `deep_gelu_mse`，其中 `grouped_gated_gelu_mse` 最高；`deep_gelu_huber` 仍是 short / next
  Rank IC 最强候选。
- NN overlay 的主任务是 `pool_L` Top100 selector：模型先在 full universe 上训练和打分，
  但核心验收是池内 Top100 排序后的 next internal excess / acceptance / capacity / exposure；
  universe next 只作辅助诊断，不作为否定 overlay selector 的主 gate。
- `deep_gelu_mse` 已补 10亿 split20 capacity acceptance、Top100 core exposure 和
  size/industry exposure，暴露画像仍是 activity / turnover heat + 中大市值 +
  电子/电力设备/计算机，但强度低于 LGBM pruned，不构成新的不可接受押注；新的 grouped
  overlay challenger 需要复用同一组 capacity / exposure gate。
- NN + LGBM 328 rankblend 相对 LGBM 有改善，但没有打过已有 NN 候选；当前不把 LGBM rankblend
  作为主晋级方向。
- 当前决策点转为候选收敛：在 `deep_gelu_mse` / `base_plus_mse` 的 Top100 next overlay、
  已做容量验收的 `mlp_base` incumbent、以及 `deep_gelu_huber` 的排序力之间做 tradeoff；
  `deep_gelu_mse` 已补 market-relative / 暴露 / 容量验收，下一步是最终候选取舍或小规模互补 blend。

## 验收 Gate

| gate | 通过含义 |
| --- | --- |
| universe short Rank IC | 短期模型本身更强。 |
| `pool_L` Top100 next internal excess | 叠加 mentor 股池后，隔夜 overlay 更强。 |
| cumulative next net / market-relative alpha | 改善不是少数点造成，且相对 full-market / pool baseline 可解释。 |
| exposure audit | 收益画像可解释，未退化为不可接受的风格、行业或流动性押注。 |
| capacity portfolio | 给定资金规模下，能在参与率、单票权重和集中度约束内装入。 |

不再把 `pool_L` short excess、universe next excess 或 next Rank IC 作为主 gate。short 端不能直接
变成 A 股 T+1 交易收益；next 端由 mentor 股池负责基础收益，短期模型只负责池内 overlay。

## 已知生产化事实

`hist_path_pruned_highdup` 的 first-pass 生产化验收结论：

- 行为画像偏 opening activity / turnover heat，且 spread 偏低；这符合开盘强势股 alpha 机制。
- 市值偏中大，不是小票拥挤；行业超配电子、电力设备、计算机，低配机械设备、基础化工，但不是单行业押注。
- `10 亿 / 20` 的 split20 容量口径下，每个 `date x clock` 目标 `5000 万`，`9690/9690` 截面全满。
- 固定 Top100 不是容量组合：主口径平均吃到 top124，p95 top161，最深 top291；生产组合应允许向后取票并加行业/风格约束。

NN 单模型第一轮验收结论：

- `mlp_base`、`mlp_shallow_fast`、`mlp_wide_huber` 已完成 2022-2025 rolling metrics、pool-internal
  analysis 和 NN vs LGBM 328 acceptance 归档。
- 相对 `hist_path_pruned_highdup` 的 `pool_L` summary（short `9.2080`、next `8.8643`、
  short Rank IC `0.140789`、next Rank IC `0.002830`），三个 MLP 的 short / next excess 都更高。
- `mlp_base` 的 `pool_L` next excess 为 `12.4320 bps`，最适合作为 overlay 收益候选；
  `mlp_wide_huber` 的 universe short Rank IC 为 `0.162945`，最适合作为排序力候选。
- NN acceptance 图已经包含 cumulative next net / market-relative alpha，不再只是 fixed Top100
  excess 表。

NN 第二轮扫参 / ensemble 结论：

- 6 个任务（NN+LGBM rankblend、deep GELU Huber、SiLU wide low-drop、compact Huber、wide-deep h64、
  wide-deep h128 Huber）已完成训练、pool-internal analysis、artifact sync 和 acceptance 图归档。
- `deep_gelu_huber` 是排序力冠军：universe short Rank IC `0.164169`，`pool_L` short / next Rank IC
  `0.150744 / 0.015041`，均高于 `mlp_wide_huber`。
- `silu_wide_lowdrop` 是本轮最接近 `mlp_base` 的 next overlay 候选：`pool_L` next excess
  `11.9652 bps`，但仍低于 `mlp_base` 的 `12.4320 bps`。
- `nn_lgbm_rankblend` 的 `pool_L` next excess `10.4098 bps`，相对 LGBM 328 有增量，但低于
  `mlp_base`、`silu_wide_lowdrop` 和 `deep_gelu_huber`，暂不晋级。
- `compact_huber` 被 `deep_gelu_huber` / `mlp_wide_huber` 支配；`wide_deep_h64` 排序力弱；
  `wide_deep_h128_huber` 的 `pool_L` next excess 低于 LGBM 328，直接淘汰。

NN MSE / structured neighborhood 结论：

- 4 个任务（base low-reg MSE、base plus MSE、deep GELU MSE、SiLU mid MSE）已完成训练、
  pool-internal analysis、artifact sync 和 audit。
- `deep_gelu_mse` / `base_plus_mse` 的 `pool_L` next excess 为 `12.9610 / 12.7478 bps`，
  均高于 `mlp_base` 的 `12.4320 bps`，成为新的 Top100 overlay challenger。
- 四个 MSE 邻域的 universe short Rank IC 约 `0.151`，明显低于 `deep_gelu_huber` 的 `0.164169`；
  排序力主线仍看 `deep_gelu_huber`。
- 结构化 grouped NN 前三组已归档：`grouped_residual`、`grouped_cross`、`grouped_gated`
  的 `pool_L` next excess 分别为 `13.4939 / 13.3557 / 13.8491 bps`，进一步超过
  `deep_gelu_mse`；说明语义特征组 encoder + 小型融合层比单纯加宽 MLP 更有效。
- 当前 grouped 分组的主要不足是过粗：328 特征按旧规则约分成 `liquidity 173`、
  `hist_path 52`、`price_momentum 41`、`other 30`、`activity 25`、`preopen 7`。
  下一版 `grouped_gated_v2` 改为机制分组 + per-group embedding dim，并写出 gate diagnostics：
  `preopen_auction`、`limit_price_state`、`book_depth_level`、`book_shape_spread_gap`、
  `book_imbalance_pressure`、`trade_activity`、`trade_price_impact`、`postopen_price_path`、
  `postopen_liquidity_change`、`historical_surprise`、`path_shape_confirmation`、`other`。
- `deep_gelu_mse` 已补正式 10亿 split20 capacity acceptance：capacity-only audit 为 `9690/9690`
  截面全满，平均 top depth `134.15`，p95 `188`，max `308`；8bps capacity cumulative net
  `7916.02 bps`，高于 `mlp_base` 的 `7656.99 bps`。
- `deep_gelu_mse` 已补 Top100 core exposure 和 size/industry exposure：activity max z / mean z 为
  `1.303 / 0.956`，低于 LGBM pruned 的 `1.526 / 1.081`；size max z 为 `0.366`，
  低于 LGBM pruned 的 `0.544`；行业仍超配电子/电力设备/计算机，但 top5 industry share
  `0.524`、effective industries `13.24`，略优于 LGBM pruned 的 `0.531 / 12.88`。

Hygiene 事实：

- baseline 276 特征 hygiene / correlation audit 已归档，主口径给出 17 个 hard-drop candidates；
  fullxs `hist_path` corr09 sensitivity 也已归档。该事项不再作为下一步待办。
- 这类 drop 只作用于模型 feature list；`ask_price_1` 等 cache / label / 回测所需基础字段不应物理删除。

## 下一步目标

下一阶段是候选收敛和 targeted NN 结构验证，而不是继续宽扫网络结构。

1. 主候选保留两条证据链：`grouped_gated_gelu_mse` / `grouped_residual_gelu_mse` 代表新的
   Top100 next overlay challenger；`deep_gelu_huber` 代表最高 short / next Rank IC。
   `mlp_base` 和 `deep_gelu_mse` 作为已有容量 / exposure 验收锚点保留。
2. 优先跑 `grouped_gated_v2_gelu_mse`：只改机制分组、per-group dim 和 gate diagnostics，
   保持 MSE、GELU、dropout、learning rate、rolling window、label 和验收口径不变；用它验证
   当前 grouped gated 的增量是否来自更合理的特征组容量分配。
3. 先给 `grouped_gated_gelu_mse` / `grouped_gated_v2_gelu_mse` 补 market-relative / 10亿 capacity / exposure /
   size-industry 验收；如果容量和暴露不过度退化，它将替代 `deep_gelu_mse` 成为 overlay 主候选。
4. 若继续做模型组合，优先测 `grouped_gated_gelu_mse` 或 v2 + `deep_gelu_huber` 的小规模 score/rank blend，
   确认 overlay 收益和排序力是否互补；不再优先做 NN + LGBM rankblend，除非有新的互补性证据。
5. 后续 NN 结构只做 targeted variants：v2 分组的 gated/residual fusion、少量 seed 和 loss 对照。
   避免继续扩大 plain MLP、wide-deep h128 或低正则扫参。

## 非目标

- 不重启两模型 `alpha_rank - lambda * gap_risk_rank` 路线；它已作为历史证据封存。
- 不用完整 `09:31-09:40` mean score 压成日频分数做正式验收；那会引入未来信息。
- 不把公司 API 回测当作当前高频 overlay 的天然验收器；若需要，另建因果 score adapter。
