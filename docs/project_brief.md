# Project Brief

本文件只回答当前研究怎么走：目标、固定口径、验收 gate 和下一步。不放命令、run 索引或完整
结果表。完整实验流水、run id、数字和归档路径见 [experiment_log.md](experiment_log.md)；命令见
[runbook.md](runbook.md)；代码索引见 [project_map.md](project_map.md)。

## 当前目标

当前 `09:31-09:40` 开盘短期 overlay 阶段收束：`grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse`
升级为当前 overlay final candidate。它仍按 decision point 当时及以前可见信息训练和打分，
主验收是 `pool_L` 内部重排后的 next internal excess 和 market-relative alpha。该模型不是脱离
pool 的独立选股策略；full universe 训练和打分只是为了给池内 overlay 提供可比 score。

下一阶段转向更真实的 pool overlay 回测，而不是继续做单纯 Top100 overlay 追高。策略研究需要保持
分钟级因果口径：每个分钟点只使用当时及以前信息，在 pool 内显式建模入场、持仓、退出、换手、
成本、容量和交易冲击；同时把研究区间从开盘 10 分钟扩展到全天上百个决策点，形成对应的日内
特征序列和信号序列。

当前 overlay final candidate 仍沿用单模型 mixed label：

```text
short_label = xs_norm(约 1 分钟持有收益 | date, decision_time)
long_label  = xs_norm(同一买入价到次日收盘收益 | date, decision_time)
train_label = short_label + 0.30 * long_label
```

后续 label 不再只是一两个标量 target，而要沉淀成分钟频时序 label/path features。每个
`date x symbol x decision_time` 都应保存入场后 1/3/5/10 分钟收益、最大顺/逆向 excursion、
可成交退出价、next-close 路径和成本后收益。模型训练可以继续读 scalar target，但策略层必须能看到
完整、因果对齐的分钟级 outcome path。

## 固定口径

| item | setting |
| --- | --- |
| data source | ClickHouse `stock.tick` |
| sample slice | `09:31:00-09:40:00` integer-minute decision points |
| training universe | A 股 `00/30.SZ`、`60/68.SH`、full universe |
| label | mixed label, `w_long = 0.30` |
| current baseline | archived `soft_core_reg_light` |
| selection masks | primary evaluation inside `pool_L`; universe / `pool_S` / `pool_M` only as diagnostics |
| primary gates | `pool_L` next internal excess; market-relative next alpha; pool-internal ranking stability |
| diagnostics | Top100 equal-weight curves; universe short Rank IC; optional capacity / exposure audits |

`pool_S ⊂ pool_M ⊂ pool_L`，来自 `lml.bzw@ssd/data/pool_{S,M,L}.parquet`。Top100 仍是
研发诊断口径；最终研究对象是 pool 内 overlay 的可交易超额。已有 capacity acceptance 和 exposure
audit 是有用诊断，但不是当前候选晋级的必选项。

下一阶段固定口径会扩展为真实 pool-overlay 回测口径：更长历史区间、全天分钟频 outcome path、
真实交易成本、容量、交易冲击、入场/退出可成交性、持仓重叠、资金复用和组合约束。现有
Top100 overlay gate 保留为信号诊断和候选解释，不再作为唯一晋级标准。

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
- NN 已完成第一轮、第二轮、MSE neighborhood 和结构化 grouped NN 归档，不再处于
  “是否值得跑”的阶段。`grouped_gated_gelu_mse` 是当前 `pool_L` Top100 next overlay
  高度候选；`grouped_gated_v2_gelu_mse` 的 next excess 略低但稳定性和 Rank IC 更好；
  `deep_gelu_huber` 仍是 short / next Rank IC 最强候选。
- NN overlay 的主任务是 `pool_L` Top100 selector：模型先在 full universe 上训练和打分，
  但核心验收是池内 Top100 排序后的 next internal excess 和 market-relative alpha；
  universe next 只作辅助诊断，不作为否定 overlay selector 的主 gate。
- `deep_gelu_mse` 已补 10亿 split20 capacity acceptance、Top100 core exposure 和
  size/industry exposure，暴露画像仍是 activity / turnover heat + 中大市值 +
  电子/电力设备/计算机，但强度低于 LGBM pruned，不构成新的不可接受押注；后续 grouped
  overlay challenger 可按需要复用同一组 capacity / exposure diagnostics。
- NN + LGBM 328 rankblend 相对 LGBM 有改善，但没有打过已有 NN 候选；当前不把 LGBM rankblend
  作为主晋级方向。
- 本阶段候选收敛结论：`grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse`
  是当前 `pool_L` Top100 next overlay 领先候选；`grouped_gated_v2_xs_rank_inplace_gelu_mse`
  是同 328 特征无量纲化里的 short IC 领先对照；`grouped_gated_gelu_mse` /
  `grouped_gated_v2_gelu_mse` 作为结构化 NN 收益高度和稳定性锚点保留；
  `deep_gelu_huber` 继续作为排序力锚点。`grouped_gated_v2_symbol_zscore_gelu_mse`
  已完成但不晋级。
- `grouped_gated_v2_xs_rank_inplace_gelu_mse` 已完成：同一批 328 特征名、同 gated v2 架构、
  不追加 `xs_rel_` / `norm_` 列；它把 universe short Rank IC 从 gated v2 的 `0.157623`
  提到 `0.161260`，但 `pool_L` next excess 只从 `13.2768` 提到 `13.7351 bps`，说明纯 rank
  对短期排序更干净，但会压掉一部分可用于 next overlay 的状态幅度。
- `grouped_gated_v2_mech328_xs_rank_gelu_mse` 机制化 v1 已完成：价格类转 bps/tick，
  股数类部分乘价格转 notional pressure，金额/计数做单调压缩，最后做截面 rank。它提高 short
  排序力，universe short Rank IC `0.160371`、`pool_L` short Rank IC `0.149266`，但主
  overlay gate 退化，`pool_L` next excess 只有 `11.7491 bps`，低于 gated v2 的
  `13.2768 bps` 和 mech328 v2 的 `14.3174 bps`。该版保留为负面对照，因为它混淆“股数压力”
  和“资金压力”，且最终 rank 压掉了幅度状态。
- `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` 已完成，是当前更合理的机制化无量纲
  328 实验：
  同 328 特征名、同 gated v2 架构；价格转 tick/bps，成交量优先转历史同分钟比例或 share 尺度，
  turnover 保持资金金额语义，盘口总深度保留 notional depth，盘口 level queue 转 side-depth share，
  最后在同一 `date x decision_time` 截面做 robust z-score。它的目标是去掉物理单位，但保留高低成交额、
  深浅盘口、相对活跃度这些本应可见的状态。结果上 `pool_L` next excess 达 `14.3174 bps`、
  Top100 8bps next net cumulative 达 `8508.0 bps`，均高于原 gated v2 和 xs-rank；但
  universe short Rank IC `0.154160` 低于 gated v2 / xs-rank，因此它是 overlay 收益候选，不是
  short 排序力候选。
- 本阶段判断更新：`grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` 升级为当前 overlay
  final candidate；`xs_rank_inplace` 和 `deep_gelu_huber` 保留为排序力/短线对照，不替代它的
  overlay final candidate 位置。
- 下一阶段不再以继续增加 NN 变体为主。更关键的问题是：该信号在 `pool_L` 内能否通过真实回测
  转化成因果、可成交、可装入资金的 overlay 超额；以及从开盘 10 分钟扩展到全天上百个决策点后，
  分钟级特征序列和 label path 是否支持稳定的入场/退出规则。

## 验收 Gate

| gate | 通过含义 |
| --- | --- |
| `pool_L` Top100 next internal excess | 叠加 mentor 股池后，池内 overlay 更强。 |
| cumulative next net / market-relative alpha | 改善不是少数点造成，且相对 full-market / pool baseline 可解释。 |
| pool-internal stability | 分年、半年和月份上不依赖单一行情段。 |
| universe short Rank IC | 辅助判断短期排序力，不作为 pool overlay 的否决项。 |
| optional capacity / exposure diagnostics | 需要时解释收益画像、容量和交易约束，不作为当前候选的硬性晋级条件。 |

不再把 `pool_L` short excess、universe next excess 或 next Rank IC 作为主 gate。short 端不能直接
变成 A 股 T+1 交易收益；next 端由 mentor 股池负责基础收益，短期模型只负责池内 overlay。

下一阶段真实回测 gate 会前移到 pool 内可交易结果：分钟级因果 PnL、成本后收益、换手、成交容量、
交易冲击、持仓重叠、资金利用率和回撤。Top100 overlay 指标保留为信号解释和回归测试，
但不再单独决定策略是否晋级。

候选模型或 score blend 从 active challenger 晋级 final candidate 前，需要补齐同一证据包：

| item | requirement |
| --- | --- |
| candidate id | 对应 `experiments/runs/<run_id>.toml`，且 `run.id` 与文件名一致。 |
| baseline/incumbent | 明确比较对象，例如 `soft_core_reg_light`、`mlp_base` 或当前 final candidate。 |
| data/cache | 写明 labeled cache、next-close cache、pool 版本和日期范围。 |
| decision slice / label | 与本 brief 固定口径一致，写明 short/long/mixed 口径和 `w_long`。 |
| artifact roots | 写明 PVC 输出和本地 compact artifact 归档位置。 |
| gate evidence | 至少覆盖 `pool_L` Top100 next、market-relative 和稳定性；capacity / exposure 为选做诊断。 |

晋级记录模板：

```text
candidate:
baseline_or_incumbent:
decision:
  status: promote | keep_challenger | reject | needs_more_evidence
  date:
evidence:
  universe_short_rank_ic:
  pool_L_top100_next_excess:
  market_relative_next_alpha:
  exposure:
  capacity:
artifacts:
  metrics:
  pool_internal:
  market_relative:
  exposure:
  capacity:
notes:
```

宽扫、smoke、debug run 不需要填完整证据包；但如果结论要写入本 brief，就应补齐上述信息。
如果候选只在一个 gate 上领先，默认保持 challenger，不直接替代 incumbent。最终候选至少要有
一条清楚的失败解释路径：如果后续 live / replay 退化，能定位到排序力、股池 overlay、暴露或容量
中的哪一项。

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
- 旧 grouped 分组的主要不足是过粗：328 特征按旧规则约分成 `liquidity 173`、
  `hist_path 52`、`price_momentum 41`、`other 30`、`activity 25`、`preopen 7`。
  `grouped_gated_v2` 已改为机制分组 + per-group embedding dim，并写出 gate diagnostics：
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
- `grouped_gated_v2_symbol_zscore_gelu_mse` 已完成训练、pool-internal analysis 和 Top100
  acceptance；`pool_L` next excess `11.6661 bps`，低于 `grouped_gated_v2_gelu_mse`
  的 `13.2768 bps`、老 `grouped_gated_gelu_mse` 的 `13.8491 bps` 和 `mlp_base`
  的 `12.4320 bps`，且 short Rank IC 降到 `0.117035`。per-symbol train-window
  z-score 不作为候选推进。
- `grouped_gated_v2_xs_rank_inplace_gelu_mse` 已完成。它修复了 symbol-zscore 的 short IC
  退化，说明“无量纲化”本身不是错；问题在于按 symbol 历史 z-score 会洗掉横截面状态。
- `grouped_gated_v2_mech328_xs_rank_gelu_mse` 已完成。结论是“机制化后再纯 rank”没有成为
  overlay 候选：short IC 提升，但 `pool_L` next excess 和 Top100 8bps cumulative 均低于 gated v2。
- `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` 已完成，并在主 overlay gate 上优于
  gated v2 / xs-rank。当前结论是：若目标是 `pool_L` Top100 next overlay，优先机制化 v2；
  若目标是纯 short Rank IC，xs-rank 更强。

Hygiene 事实：

- baseline 276 特征 hygiene / correlation audit 已归档，主口径给出 17 个 hard-drop candidates；
  fullxs `hist_path` corr09 sensitivity 也已归档。该事项不再作为下一步待办。
- 这类 drop 只作用于模型 feature list；`ask_price_1` 等 cache / label / 回测所需基础字段不应物理删除。

## 下一步目标

下一阶段从候选收敛转向真实 pool-overlay 回测和全天样本扩展，而不是继续宽扫网络结构。

1. 冻结 `grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse` 为当前 overlay final candidate；
   `xs_rank_inplace`、`deep_gelu_huber`、`grouped_gated_gelu_mse` / `grouped_gated_v2_gelu_mse`
   和 `mlp_base` / `deep_gelu_mse` 只作为排序力、结构和容量锚点保留。
2. 构建真实 pool-overlay 回测：只在 pool 内做 overlay 排序/调仓，显式建模容量、交易冲击、
   成交量约束、停牌/涨跌停、成本、换手、持仓重叠和资金复用。
3. 扩大研究时间区间和日内决策面：优先从 `09:31-09:40` 扩展到全天上百个 decision points，
   为每个点生成对应的因果特征序列、score 序列和回测输入。
4. 把 label 做成分钟频时序特征：为每个 `date x symbol x decision_time` 保存后续分钟 return path、
   excursion、可成交退出价、next-close path 和成本后收益。训练 target 可以从 path 派生，
   策略回测必须直接读 path，避免用 full-window mean score 或事后汇总引入未来信息。
5. 扩大跨年份和跨市场状态验证：在更长训练/验证窗上按牛熊、流动性、波动和年份切分 OOS，
   但仍以 pool 内超额为核心目标。
6. 不继续推进 `grouped_gated_v2_symbol_zscore_gelu_mse`。它的 next Rank IC 略高，
   但主 gate 的 Top100 next excess、short Rank IC 和累计收益均退化；如果要降暴露，优先放到
   可选 exposure / capacity diagnostics 或真实回测的组合层处理。
7. 对同 328 特征归一化的当前判断：纯 xs-rank 是 short Rank IC 最强方向；mech328 v2 是
   `pool_L` Top100 next overlay 最强方向。后续若继续做 feature value normalization，优先采用
   “价格 tick/bps、成交量历史比例、turnover 金额、盘口 notional depth 和 queue share，再做截面
   robust z-score”的语义保留方案；mech v1 只作为机制化后再 rank 的负面对照。
8. 后续 NN 结构只做必要的 targeted variants 或策略层 score blend；避免继续扩大 plain MLP、
   wide-deep h128、低正则或统一 symbol z-score 扫参。

## 非目标

- 不重启两模型 `alpha_rank - lambda * gap_risk_rank` 路线；它已作为历史证据封存。
- 不用完整 `09:31-09:40` mean score 压成日频分数做正式验收；那会引入未来信息。
- 不把公司 API 回测当作当前高频策略的天然验收器；若需要，另建分钟级因果 score adapter。
- 不把脱离 pool 的独立 universe 策略作为当前目标；当前研究目标是 pool 内 overlay 超额。
