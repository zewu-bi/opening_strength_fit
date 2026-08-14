# Experiment Log

> Last reconciled: 2026-08-14
>
> Coverage: 2026-05-20 through 2026-08-14

本文件是实验事实账本，按时间记录假设、结果、状态和决策。当前研究口径见
[project_brief.md](project_brief.md)，执行命令见 [runbook.md](runbook.md)，更早的长记录保留在 Git 历史中。

## 证据规则

事实冲突时依次以 compact evidence、run TOML、Git/K8s trace、人工叙述为准。大型结果位于 PVC 或本地
mirror，不进入 Git；持久追溯依赖 run config、Job manifest、代码 revision、compact evidence 和本日志。
run 状态由 `osf-audit-experiments` 从 TOML 审计，不在文档中复制完整索引。

## 近期实验状态

| date | run / decision | status | impact |
| --- | --- | --- | --- |
| 2026-07-22 | fixed-clock v4 control / multiden + unified strategy acceptance | `completed` | multiden 曾晋级 opening policy，保留为旧数据链路对照 |
| 2026-07-30 | `10:01-10:10` 日内窗口重训 | `completed` | next excess `6.5491 bps`，确认较 09:31 明显衰减，不晋级 |
| 2026-07-31 | corrected `opening_model` 09:31-09:40 | `completed` | next excess `17.7934 bps`，成为最新信号基准；策略层仍需同数据重跑 |
| 2026-07-31 | corrected next-close 与 1m/3m label matrix | `completed` / 历史输入保留 | 旧实验继续按原输入复现，不与新数据版本混作严格单变量对照 |
| 2026-08-04 | 三窗口 raw source、350 features、基础 5-label 数据 | `completed` | 原始数据与特征/标签构建分层，combined label 仅作中间产物 |
| 2026-08-04 | 九个 horizon-split label | `completed`; 9 labels × 7 years | 后续实验从三窗口 × 1m/3m/5m 中选择；新版本为权威训练输入 |
| 2026-08-05 | corrected-label 三窗口 × 1m/3m NN 网格 | `completed`; 6 runs × 8 shards | 09:31 的短期/隔夜头部超额最强；10:01、14:01 在信号层归档，不晋级策略验收 |
| 2026-08-05 | 前两窗口 10m/1h/当日收盘 label | `completed`; 6 labels × 7 years | 10m/1h 用持有期后 60 秒 VWAP，收盘 label 用当日收盘价；next-close 逐 key 复用既有数据 |
| 2026-08-05 | 新分层数据 15-label max-10 NN 矩阵 | `superseded`; 15 runs × 8 shards | 只保留为 v6 同预算复现与 epoch 敏感性对照，不再作为当前结果口径 |
| 2026-08-06 | 15-label max-30 NN 矩阵 | `completed`; 15 runs × 8 shards；当前权威结果 | case 排序与正负方向稳健；1h/close 的较低 Top100 excess 是最终结果的一部分，不回退到 max-10 |
| 2026-08-11 | ds350 长 label、涨停尾部与严格 2026H1 留出诊断 | `completed`；compact evidence 已归档 | 未发现能解释信号的 selection-time future-label 泄露；收益高度依赖最终涨停尾部，且 ask1 深度显著限制可成交容量 |
| 2026-08-12 | 未来信息专项审计与 hard-cutoff smoke | 审计 `completed`；完整 kill tests `prepared/not run` | 确认 exchange time 与 receipt time 边界不一致；是否晚于真实生产截止时点尚未确认 |
| 2026-08-13 | 09:31 1m/close label 捷径诊断 | `completed`; 3 variants × 2 labels × 8 shards | 1m label 基本连续；原 close label + MSE 明显学习涨停状态，无涨跌停训练基本消除 close Top100 涨停富集 |
| 2026-08-14 | 11:01 raw/features/1m/3m/5m/close 数据 | `completed`; 5 datasets × 7 years，逐年配对验证通过 | PVC 已形成可直接训练的 350-feature 与 horizon-label 数据 |
| 2026-08-14 | 11:01 1m/3m 涨停依赖与四窗口归因 | `completed`; 4 runs × 8 shards | 11:01 无涨跌停 1m/3m 富集降至 `1.19x/1.05x`，短期超额保持 `7.82/8.01 bps` |
| 2026-08-14 | close 目标后续：2026H1 non-limit/rank 网格 | `prepared/not run` | 配置、support Jobs 和分析脚本已归档；没有训练结果，不进入当前结论 |
| 2026-08-14 | 治理分支集成与项目冻结 | `completed`；无 active run | 合并模块边界、契约、最新实验与 compact evidence；二分类探索未纳入；最终 CI `400 passed, 6 skipped`、覆盖率 `58.63%`，smoke、registry audit 与 contracts 全绿；此后不再安排训练 |

## 决策时间线

### 2026-05：建立信号与因果口径

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 05-20~21 | 小窗 GBM → 一年训练/次月测试 | decision Rank IC `0.1426 → 0.1831` | opening short 排序值得扩样本；不解释为可交易结果 |
| 05-22~26 | 加真实 entry delay、扩 horizon | delay2 后 short 仍有效，close/next-close 快速衰减 | 固定保守 delay；暂停直接 T+1 主线 |
| 05-26~27 | postopen、no-preopen、XS 处理 | postopen v2 / xs-demean 改善 short，next 仍弱 | 固定 `09:31-09:40` 样本域 |
| 05-28~29 | hard guard、learned/conditional risk | guard 改善 next 但损伤 short；两层公式局部有效 | guard 只作诊断，进入跨月验证 |

### 2026-06：固定 label、扩展 OOS、收敛特征

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 06-02~05 | risk layer → mixed label → 36m rolling | `w_long=.30` 在三池兼顾 short/next | 固定 single mixed label 与 36m→6m OOS |
| 06-05~11 | 扩 2020-2025 OOS、LGBM sweep | pool 内 next 较稳；常规调参无稳定增量 | 停止局部 LGBM 调参 |
| 06-12~23 | hist/path、scale、feature hygiene | rank-centered union 较强；重复簇可删 | 保留 hist/path 精确并集，drop 只作用模型输入 |
| 06-25~26 | 328-feature prune、capacity、首批 NN | 容量截面可填满；MLP next `12.4320 bps` | 328 特征成为干净基线，NN 进入结构扫描 |

### 2026-07：NN 收敛与执行约束

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 07-02~10 | NN 架构、MSE、grouped gated、机制化 | mech328 v2 next `14.3174 bps`；symbol z-score 失败 | grouped-gated + MSE 成为模型主线 |
| 07-15~17 | auction-fresh、T-1 reference、因果重建 | auction-pruned next `16.9692 bps`，realistic `7323.9 bps` | 通过收益 gate，后由 fixed-clock v4 supersede |
| 07-17~22 | fixed `clock+6s`、350-feature multiden | v4 multiden next `17.1714 bps`，Top100 fee8 `9891.7 bps` | 接受 fixed-clock 数据口径和 multiden continuation |
| 07-22~23 | capacity/no-refill/visible-refill 统一验收 | multiden refill fill `99.9970%`，net `8598.7 bps` | 尾部、分期和 overlap 保留为诊断，不设自动否决阈值 |
| 07-29 | 复核 full-day temporal 路线 | 需求应是固定十分钟窗口横向比较 | 归档全天序列/TCN 路线，回到既有范式 |
| 07-30 | 10:01-10:10 同口径重训 | short IC 提高，next 和费后收益显著下降 | 作为衰减 checkpoint，不替换 opening policy |
| 07-31 | corrected decision-state 09:31 重训 | next `17.7934 bps`，8/8 半年为正 | `opening_model` 晋级最新信号基准 |
| 07-31 | canonical naming reset | source lineage 不变，逻辑入口与长 run id 分离 | 使用 `opening_base/cache/model` 短入口 |

## 当前决策记录

### `opening_model` 09:31-09:40 信号验收（2026-07-31，已完成）

唯一变化是 decision sampling：从目标时刻后 5 秒内首条更新，改为目标时刻已可见的最后状态。target、
350 features、模型、seed、`pool_L`、rolling OOS 与 `clock+6s` entry 均不变；47,333,103 行 OOS
prediction 全量匹配 label。

| metric | archived v4 | corrected v6 |
| --- | ---: | ---: |
| universe short Rank IC | 0.156418 | **0.158330** |
| `pool_L` next Rank IC | 0.007140 | **0.007839** |
| `pool_L` next Top100 excess | 17.1714 bps | **17.7934 bps** |
| positive next months / half-years | 37/48; 8/8 | **38/48; 8/8** |
| Top100 fee8 cumulative | 9891.7 bps | **10193.0 bps** |

结果小幅全面提高，但只赢 `26/48` 月，属于同一信号的因果修正，不是强分期胜利。Decision：接受为最新
信号基准；在其上重跑统一策略验收前，不继承 v4 的策略晋级结论。来源映射见
[canonical opening registry](../experiments/canonical/opening.toml)。

### 10:01-10:10 日内窗口衰减（2026-07-30，已完成）

只修改十分钟窗口及对应 cache/label lineage；44,993,233 行 prediction 全量匹配同窗口 label。

| metric | 09:31-09:40 | 10:01-10:10 |
| --- | ---: | ---: |
| universe short Rank IC | 0.156418 | **0.253118** |
| short Top100 absolute return | **3.1805 bps** | 0.1153 bps |
| `pool_L` next Top100 excess | **17.1714 bps** | 6.5491 bps |
| positive next months / half-years | **37/48; 8/8** | 29/48; 6/8 |
| Top100 fee8 cumulative | **9891.7 bps** | 2708.5 bps |

10:01 更稳定地识别“跌得较少”的股票，但 opening edge 的隔夜与费后部分明显衰减。Decision：归档为
completed decay checkpoint，不做 downstream promotion audit，也不替换 09:31 policy。

### Corrected-label 三窗口 × 1m/3m 网格（2026-08-05，已完成）

三组预先固定十分钟窗口各自使用同窗口 corrected feature/label lineage；除 short label horizon 为 1m
或 3m 外，350 features、grouped-gated NN、seed、`pool_L`、36m rolling / 6m OOS 与 corrected
next-close 分量保持一致。六组训练均完成 8/8 shard，pool-internal 分析全量匹配 prediction 与 label。

| window | short horizon | universe short Rank IC | `pool_L` short Top100 excess | `pool_L` overnight Top100 excess |
| --- | --- | ---: | ---: | ---: |
| 09:31-09:40 | 1m | 0.160081 | 11.8129 bps | 18.0814 bps |
| 09:31-09:40 | 3m | 0.108995 | **12.9162 bps** | **20.2087 bps** |
| 10:01-10:10 | 1m | 0.255019 | 7.7006 bps | 6.6667 bps |
| 10:01-10:10 | 3m | 0.185261 | 7.8869 bps | 8.3935 bps |
| 14:01-14:10 | 1m | **0.378929** | 7.6493 bps | 1.0904 bps |
| 14:01-14:10 | 3m | 0.316359 | 8.0766 bps | 2.3101 bps |

窗口后移时，短期 IC 单调升高，但短期头部超额下降、隔夜头部超额强烈单调衰减。3m 相对 1m 在三个
窗口都降低短期 IC（平均 `-0.0611`），但提高短期超额（平均 `+0.5723 bps`）和隔夜超额（平均
`+1.6912 bps`）。Decision：若目标是当前经济指标，09:31-09:40 / 3m 是六格首选；10:01 和
14:01 明显落后，按 runbook 停在信号层归档，不进入 capacity / realistic promotion audit。

### 15-label max-30 权威结果与 max-10 对照（2026-08-06，已完成）

在新分层数据的 15-label max-10 基线上，仅把训练预算从 max 10 / patience 2 改为 max 30 /
patience 3；模型、feature、label、seed、rolling split 和 OOS 区间保持不变，结果写入独立目录。120/120
fold 完成：平均训练 `22.91` epochs，107/120 的 best epoch 大于 10，36/120 的 best epoch 为 30；
相对 epoch 10，best validation loss 平均再下降 `0.9121%`。这证明 10 不是由原验证准则确定的收敛点。

当前 15-label 矩阵的结果口径固定为 max-30；max-10 只用于 v6 同预算复现和 epoch 敏感性对照。
更低的训练期随机行 validation loss 没有转化成更高的时间外经济指标，以下回落本身就是 max-30
权威结果的一部分，而不是退回 max-10 的理由。pool_L 的 max-30 相对 max-10 平均变化如下：

| horizon | short Top100 excess Δ | next Top100 excess Δ | short Rank IC Δ | next Rank IC Δ |
| --- | ---: | ---: | ---: | ---: |
| 1m | +0.0390 bps | -0.2984 bps | +0.001274 | +0.000773 |
| 3m | +0.0366 bps | -0.3638 bps | +0.000593 | +0.000524 |
| 5m | -0.0957 bps | -0.5149 bps | +0.000113 | +0.000256 |
| 10m | -0.8302 bps | -0.9267 bps | +0.000055 | +0.002418 |
| 1h | -2.0263 bps | -2.6935 bps | -0.002383 | +0.000052 |
| close | -3.3656 bps | -4.2528 bps | -0.003210 | -0.002036 |

15/15 case 在两个预算下的 pool_L short/next excess 都保持为正，case 排序也高度一致：short/next
Rank IC 的 Spearman 分别为 `0.9964/0.9393`，short/next excess 的 Spearman 为 `0.9000/0.9250`。
因此窗口衰减、1m/3m 相对关系和信号正负等研究结论稳健；1h/close 后续引用和下游分析必须使用
max-30 的较低绝对 Top100 收益。Decision：max-30 是当前权威训练结果；max-10 降为历史对照。
后续可以把 early stopping validation 改为训练窗口尾部的时间块，但这属于下一版方法改进，不改变
本批结果以 max-30 为准。

### 未来信息专项审计（2026-08-12，审计已归档）

代码、PVC 与 ClickHouse 重放未发现 feature/label 的交易所时间直接重叠，但确认 raw cache 使用
`LocalTimeStamp` 去重后未将其保留到 Parquet。2025 每月首个交易日的 615,810 个决策状态中，
`68.4791%` 在标记决策时点后到达，延迟 p95 `1.6620s`、最大 `1.8567s`；全部早于当前 `t+6s`
买入点。`-2s` 单日 hard-cutoff 重建使 99.963% 样本至少一个特征变化，但只证明时钟边界重要，
不证明生产流程存在未来信息泄露。2022-2024 缺少有效 receipt timestamp，同日 Pool-L 的生成时点也
没有仓库内血缘。

Decision：审计状态固定为“未发现明确泄露，但仍存在未排除风险”。完整 hard-cutoff、raw-safe 与
feature-family kill tests 仅准备了配置和 Job，尚未执行，不能填报 IC 或涨停富集结果。完整证据、风险
清单和可运行入口见 [leakage audit](leakage_audit.md)。

### 09:31 1m/close label 捷径诊断（2026-08-13，已完成）

固定 `09:31-09:40`、DS350 grouped-gated NN、36m rolling / 6m OOS、max 30 epochs 和
`pool_L` Top100，只比较 1m 与 close 两个 horizon 的三组变体：每日截面 3σ target 截断、训练时
删除 `UpdownLimitStatus != 0` 的约 `2.7%` 样本、以及去除 next-close 分量的纯短label。验证集始终
保留完整 `pool_L`。完整表、原始汇报截图与机器可读数据见
[结果表 CSV](../experiments/evidence/backtests/ds350_w0931_limit_shortcut_ablation_2022_2025_v1/three_experiment_summary.csv)、
[贡献拆分 CSV](../experiments/evidence/backtests/ds350_w0931_limit_shortcut_ablation_2022_2025_v1/limit_contribution_summary.csv)
及原始汇报截图
[结果表](../experiments/evidence/backtests/ds350_w0931_limit_shortcut_ablation_2022_2025_v1/企业微信截图_17865898491793.png)、
[贡献拆分](../experiments/evidence/backtests/ds350_w0931_limit_shortcut_ablation_2022_2025_v1/企业微信截图_17865898257957.png)。

| 实验 | Label | IC | Label超额 | 收盘超额 | 次日收盘超额 |
| --- | --- | ---: | ---: | ---: | ---: |
| 3σ截断 | 1m | 0.15219 | 12.02 | 18.02 | 17.63 |
| 3σ截断 | close | 0.02806 | 18.97 | 18.97 | 18.16 |
| 无涨跌停训练 | 1m | 0.15122 | 11.34 | 19.33 | 14.35 |
| 无涨跌停训练 | close | **0.03606** | 22.18 | 22.18 | 14.33 |
| 纯短label | 1m | 0.14923 | 12.09 | 17.56 | 15.05 |
| 纯短label | close | 0.02276 | **22.84** | **22.84** | **19.88** |

3σ截断和纯短 close 模型的 Top100 最终涨停率为 `3.6900%/3.7961%`，相对完整 `pool_L`
的 `0.9611%` 基准富集 `3.84x/3.95x`；无涨跌停 close 模型降至 `1.1512%`、即 `1.20x`。
后者的 `22.18 bps` 收盘超额中，`16.20 bps` 来自非最终涨停股，说明删除少量涨跌停训练样本
确实改变了 close 模型的优化方向；次日收盘超额同步回落到 `14.33 bps`。

纯短label仍富集涨停，排除了 next-close/隔夜分量单独造成偏好的解释。label 分布审计进一步显示：
1m 中普通股与最终涨停股高度重叠，三组 1m Label 超额均以非涨停贡献为主；close 中最终涨停股均值
约 `739.47 bps`，普通股约 `-5.75 bps`，约 `1.06%` 的最终涨停样本占 mixed close target
平方量的 `17.07%`。这里的平方量对应零预测基准 MSE，不是每个 epoch 的 residual loss 实测占比。
3σ虽将该占比降至 `8.93%`，但 target Top100 涨停占比仍约 `30.98%`，所以单纯压幅度没有改变
涨停状态位于 label 顶部的排序结构。

无涨跌停 1m 模型持有到收盘的 `19.33 bps` 中仍有 `12.98 bps` 来自最终涨停，说明短期强势与
后来收盘涨停存在真实相关性；但其 1m Label 超额的 `10.05 / 11.34 bps` 来自非涨停股，不能据此
否定 1m label 的连续排序能力。Top100 只是尾部评估，会放大全样本 MSE 已学到的方向，不参与训练
loss。因此本轮支持的是 close label/loss 捷径，并非未来特征泄漏的直接证据。

Decision：本轮归档完成；1m label 维持可用，收盘持有收益需同时报告涨停归因。现有 mixed-MSE
close 模型不再单独作为连续 close alpha 证据；后续以无涨跌停 close 为基线，对照 robust loss 与
有界 rank/percentile close target，不再只调整标准差截断倍数。

### 四窗口 1m/3m 涨停依赖衰减（2026-08-14，已完成）

补齐 `11:01-11:10` DS350 1m/3m 的 Baseline 与无涨跌停训练，共 4 组 36m rolling / 6m OOS、
max 30 NN，每组 8/8 fold 完成。训练过滤仅作用于训练 frame：首个 36m fold 从 `28,971,750`
行降至 `28,190,820` 行，删除 `780,930` 行（约 `2.70%`）；验证和 `pool_L` Top100 仍使用完整样本。
四窗口富集统一除以 2022-2025 日度 `pool_L` 最终涨停率 `0.961149%`，避免不同模型分母漂移。
探索阶段已经完成的 09:31/10:01 10m support runs 按后续口径排除，不进入本节结果或当前候选集合。

| 窗口 | Baseline 1m/3m 富集 | 无涨跌停 1m/3m 富集 | 无涨跌停 Label 超额 |
| --- | ---: | ---: | ---: |
| 09:31-09:40 | `4.80x / 4.85x` | `2.59x / 2.12x` | `11.34 / 11.94 bps` |
| 10:01-10:10 | `2.68x / 2.93x` | `1.82x / 1.69x` | `8.48 / 8.41 bps` |
| 11:01-11:10 | `1.67x / 1.70x` | `1.19x / 1.05x` | `7.82 / 8.01 bps` |
| 14:01-14:10 | `0.71x / 0.74x` | `0.54x / 0.46x` | `7.75 / 8.17 bps` |

11:01 Baseline 1m/3m 的短期涨停贡献仅为 `0.33/7.77` 与 `0.50/7.92`；无涨跌停进一步降至
`0.24/7.82` 与 `0.32/8.01`。持有到收盘时，Baseline 涨停贡献占比约 `70%/64%`，无涨跌停为
`43%/34%`。因此窗口后移后，涨停依赖不只是绝对 bps 下降，其占总超额比例也下降；11:01 无涨跌停
3m 已几乎消除数量富集，同时保留全部短期排序收益。14:01 的 `<1x` 表示 Top100 低配最终涨停股，
不能单独解释为负 alpha。

1m 的排序主体对训练过滤不敏感：09:31/10:01/14:01 的全截面 Rank IC 分别从
`0.1475/0.2512/0.3739` 变为 `0.1512/0.2555/0.3778`；四窗口 1m 总超额变化分别为
`11.96→11.34`、`8.52→8.48`、`7.77→7.82`、`7.63→7.75 bps`。非最终涨停贡献在四个窗口均
略有上升，说明过滤主要移除了 Top100 尾部的涨停偏好，没有移除普通股票上的主要短期收益来源。
这一证据足以支持“存在非涨停排序能力”，但尚未直接验证非涨停股票各预测分位严格平滑单调。

Decision：近开盘窗口存在最强的涨停状态捷径，但 1m/3m 的主要短期收益并不依赖该捷径；后续短周期
实验默认保留完整训练集并继续报告涨停归因，只有 close 路线将无涨跌停训练作为方法基线。完整两表、
机器可读数据见
[涨停贡献表](../experiments/evidence/backtests/ds350_four_window_limit_tables_v1/table_1_limit_contribution_ratio.csv)、
[涨停率与富集表](../experiments/evidence/backtests/ds350_four_window_limit_tables_v1/table_2_limit_rate_enrichment.csv)
及 [metrics](../experiments/evidence/backtests/ds350_four_window_limit_tables_v1/metrics.json)。

### Ordinary 328 mech v3 cap-cache（2026-07-21，已完成）

最终完成的是 `mech328_v3_capcache_896` 单 shard 归因性重跑，而非早期 `histavg_activity` 任务：

- `pool_L` short / next excess：`11.1004 / 16.3318 bps`；
- short / next Rank IC：`0.138516 / 0.006657`；
- next 正月：`38/48`。

它强于 mech328 v2、略低于 auction-pruned。Decision：作为普通 328 对照保留，不再补跑新 cache 主线。

### Fixed-clock +6s state cache 与 auction-pruned 验收（2026-07-17~22，已完成）

`stock.tick` 只在状态变化时写行，因此“向后数两条更新”不能代表固定 6 秒。新口径把执行时钟与源状态
时间拆开：

```toml
[labels]
entry_alignment = "clock_state"
entry_clock_delay_seconds = 6
future_alignment = "clock_state"
require_entry_after_cross_section_ready = true
```

entry 和 sell 边界均做 backward point-in-time lookup，并记录 source timestamp/state age；同时间戳 revision
先确定性去重。7 个年度 base、7 个 mixed target 及两组 8-shard 训练均完成。

| run | short Rank IC | next Rank IC | next excess bps | next 正月 | fee8 cumulative bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| mech328 v2 | 0.154160 | **0.008054** | 14.3174 | **39/48** | 8508.0 |
| v4 control, 325 features | 0.156070 | 0.006411 | 16.8024 | 38/48 | 9713.0 |
| v4 multiden, 350 features | **0.156418** | 0.007140 | **17.1714** | 37/48 | **9891.7** |

multiden 相对 control 仅增加 `0.3690 bps`，月度胜 `25/48`，不是稳定逐期 A/B 胜出。Decision：接受
fixed-clock 因果口径；multiden 作为 continuation candidate，control 保留为单变量基线。

### Unified strategy acceptance toolkit 与 fixed-clock v4 复核（2026-07-22，已完成）

统一工具在同一候选集比较 `capacity_only`、`realistic_no_refill` 和
`visible_pretrade_refill`。refill 只使用决策时点可见信息继续下探，不观察真实失败后再下单。

| run / policy | mean fill | cumulative capital net bps | bootstrap P05 bps |
| --- | ---: | ---: | ---: |
| control capacity-only | 100.0000% | 8989.8 | 805.8 |
| control no-refill | 80.7803% | 7183.6 | 724.9 |
| control visible refill | 99.9969% | 8431.1 | 209.2 |
| multiden capacity-only | 100.0000% | 9217.9 | 1067.3 |
| multiden no-refill | 81.3916% | 7433.4 | 1017.6 |
| multiden visible refill | 99.9970% | **8598.7** | 497.0 |

Decision（2026-07-23）：refill、overlap、tail 和 bootstrap 都是必跑诊断，但不设置自动否决阈值。
multiden 成本后收益为正且 refill 相对 no-refill 增加 `1165.3 bps`，因此当时晋级 opening policy；这不等于
完整持仓账本或实盘批准。

### Conservative cap + unique-tick cache（2026-07-16，base 已完成、target 已终止）

该对照在旧 cache 语义上增加严格 T-1 cap/share、基础竞价变换，并按
`date × symbol × exchange timestamp` 确定性去重。2019-04-19 的 39,115 个样本中 38,740 个有效，
其中 `92.51%` 恰为 6 秒，其余会漂移到更晚真实 tick。7 个年度 base 已全量重建；fixed-clock 确认后，
mixed target 未提交，路线标记为 superseded。

### Mech328 v3 cap-cache challenger（2026-07-16 重提记录；07-21 已完成）

该 challenger 固定 328 特征与 grouped-gated-v2/MSE，只将值变换切到 strict ratio-style v3；不追加截面
z-score，训练入口仍做 global z-score。旧 cache 缺市值/股本导致 reference fallback 和内存问题，低内存
重提后来停止，最终由上文 `mech328_v3_capcache_896` 完成。早期提交状态不再视为活动任务。

### Auction-fresh causal-prune acceptance archive（2026-07-15~17，已归档）

该独立 lineage 使用严格 T-1 cap/share、因果 freshness/readiness 和 pruned feature 输入。base、mixed target、
8 个训练 shard 与 pool analysis 全部完成。

| metric | auction-fresh | mech328 v2 | gated v2 |
| --- | ---: | ---: | ---: |
| universe short Rank IC | 0.150489 | 0.154160 | **0.157623** |
| `pool_L` next excess bps | **16.9692** | 14.3174 | 13.2768 |
| positive next months | 38/48 | **39/48** | **39/48** |
| Top100 fee8 cumulative bps | **9893.9** | 8508.0 | 8003.8 |

| downstream metric | auction-fresh | LGBM 328 | MLP base |
| --- | ---: | ---: | ---: |
| capacity cumulative net bps | **9244.5** | 6009.2 | 7657.0 |
| realistic mean fill | 0.8073 | 0.8022 | **0.8093** |
| realistic cumulative net bps | **7323.9** | 4705.3 | 6113.0 |

Decision：通过 Top100、capacity 和 first-pass realistic 收益 gate；short IC 退化与右尾依赖记录为诊断。
该 lineage 随后由 fixed-clock v4 supersede。

### Historical overlay baseline：mech328 v2

| metric | gated v2 | XS rank | mech328 v1 | mech328 v2 |
| --- | ---: | ---: | ---: | ---: |
| universe short Rank IC | 0.157623 | **0.161260** | 0.160371 | 0.154160 |
| `pool_L` next excess bps | 13.2768 | 13.7351 | 11.7491 | **14.3174** |
| positive next months | 39/48 | 37/48 | 33/48 | **39/48** |
| Top100 fee8 cumulative bps | 8003.8 | 8225.9 | 7263.6 | **8508.0** |

Decision at the time：mech328 v2 为 overlay incumbent，XS rank 为 short 排序锚；之后被 fixed-clock v4
multiden supersede，当前只作历史基线。

### First-pass realistic replay

| model | mean fill | min fill | mean next net bps | cumulative net bps |
| --- | ---: | ---: | ---: | ---: |
| LGBM pruned 328 | 0.802248 | 0.596093 | 9.7116 | 4705.3 |
| MLP base | 0.809290 | 0.635732 | 12.6171 | 6113.0 |
| auction-fresh pruned | 0.807326 | 0.596713 | 15.1165 | 7323.9 |

限制：selected-order replay 不 refill、不维护完整同日预算/持仓/退出，且缺 ask2-10 上下文。它是统一策略
验收的前身，不是完整策略回测。

## 状态与归档

Run status 只允许 `prepared`、`queued`、`running`、`completed`、`canceled`、`superseded`；未知状态必须使 audit 失败。
训练 run 完成时应有 metrics，artifact-only run 按 kind 验收 summary、trace 和成功标记。

本项目于 2026-08-14 冻结。`prepared/not run` 仅表示入口可复现，不表示排队或计划执行；所有未完成项
按现状封存。若未来重新开启研究，应先取消仓库归档，再更新 TOML status、trace 和本日志。
