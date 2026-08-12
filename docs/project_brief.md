# Project Brief

> Last reviewed: 2026-08-11

## 目标

项目研究分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除成本、容量和执行限制后仍稳定的
超额。模型可使用全 A 股训练，但只评价既定股池内的增量。

当前 `09:31-09:40` 基准为 `opening_model`。旧 corrected-label 血缘上的三窗口 × 1m/3m 网格已经
完成：窗口越晚，short Rank IC 越高，但可交易头部尤其隔夜超额越弱；六格中经济指标最强的是
`09:31-09:40 / 3m`。

新 ds350 分层数据上的 15-label / max-30 矩阵也已完成 120 个 rolling OOS fold，并登记为
`opening_label_matrix`。它是当前权威研究矩阵，不是 canonical 信号晋级：`opening_model` 仍是完成既定
信号评价的当前基准，二者不得用同一个“最新模型”概念混写。

## 固定研究口径

| 项目 | 口径 |
| --- | --- |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 当前决策面 | `09:31:00-09:40:00`，每分钟一个决策点 |
| 可见性 | feature、股池和执行过滤只使用决策时点可见信息 |
| 训练 universe | A 股 `00/30.SZ`、`60/68.SH` |
| 选择 universe | `pool_L`；S/M/universe 只作诊断 |
| 目标 | `xs_norm(short_return) + 0.30 × xs_norm(next_close_return)` |
| 验证 | `36m train -> next 6m` rolling OOS，覆盖 2022-2025 |
| canonical base/cache/model | `opening_base` / `opening_cache` / `opening_model` |
| canonical research matrix | `opening_label_matrix`（diagnostic only） |
| downstream reference | archived v4 multiden；待在 `opening_model` 上重跑 |
| ablation / historical baseline | fixed-clock v4 control / mech328 v2 |

rolling OOS 可用于模型和特征选择，不是 untouched final test。

## 最新数据源合同

后续新实验将原始数据、feature 和 label 分层：ClickHouse 只用于构建 PVC raw-source cache；训练前由
同窗口 raw cache 生成 350 features，并从下列 15 个最终 label 中选择一个。每个 label 单独存放在一个
PVC 目录，表中列出目录名；公共前缀为 `/mnt/output/opening_strength_fit/datasets/`。

| 决策窗口 | 1m label | 3m label | 5m label |
| --- | --- | --- | --- |
| `09:31-09:40` | `opening_0931_0940_labels_h1m_v2` | `opening_0931_0940_labels_h3m_v2` | `opening_0931_0940_labels_h5m_v2` |
| `10:01-10:10` | `opening_1001_1010_labels_h1m_v2` | `opening_1001_1010_labels_h3m_v2` | `opening_1001_1010_labels_h5m_v2` |
| `14:01-14:10` | `opening_1401_1410_labels_h1m_v2` | `opening_1401_1410_labels_h3m_v2` | `opening_1401_1410_labels_h5m_v2` |

首两个窗口各有 3 个长持有期 label：

| 决策窗口 | 10m VWAP | 1h VWAP | 当日收盘价 |
| --- | --- | --- | --- |
| `09:31-09:40` | `opening_0931_0940_labels_h10m_v1` | `opening_0931_0940_labels_h1h_v1` | `opening_0931_0940_labels_hclose_v1` |
| `10:01-10:10` | `opening_1001_1010_labels_h10m_v1` | `opening_1001_1010_labels_h1h_v1` | `opening_1001_1010_labels_hclose_v1` |

- 每个实验使用同窗口 `opening_<window>_features_350` 和一个最终 label；不得跨窗口或混用 target。
- label schema 为 3 个样本键加 `label_short`、`label_next_close`、`target_label`；NaN 表示无效。
- `target_label = xs_zscore(label_short) + 0.30 × xs_zscore(label_next_close)`。
- 中间 `opening_<window>_labels_1m_3m_5m_next_mixed` 只用于 lineage/回滚，不直接训练。
- 长持有期中间目录 `opening_{0931_0940,1001_1010}_labels_10m_1h_close_next` 同样只用于
  lineage、审计和拆分，不直接训练。
- 长持有期入场沿用决策时钟状态后 `+6s` 的 Ask1；10m/1h 在持有期结束后用 60 秒累计
  `Turnover/Volume` VWAP 退出，当日收盘使用 `close_reference.ClosePrice`。
- 长持有期 `label_next_close` 逐 key 复用既有 `opening_<window>_labels_h1m_v2`，不重新计算。
- 原 9 个短周期 label 的 63 个年度文件和新增 6 个长持有期 label 的 42 个年度文件均已通过
  schema、行数和 `_SUCCESS` 检查。

上述 9 个短周期与 6 个长持有期，共 15 个当前可用的新训练 label。9 个短周期 label 不是旧 6 个
1m/3m target cache 的逐值复制：旧 key 全部包含于新数据，但 key 覆盖、close tie-break 和计算精度不同；
当前旧实验保留原输入完成或复现，后续实验改用上述矩阵。

### 旧 corrected v6 兼容血缘

2026-07-31 启动的旧 1m/3m 实验继续读取各自 run TOML 中固定的 base、standalone 3m、corrected
next-close 和 target cache。六组 3×2 网格已经完成并归档；它们只用于历史复现，不再作为新实验模板。

## 当前结论

| 候选 | `pool_L` next excess | 决策 |
| --- | ---: | --- |
| mech328 v2 | `14.3174 bps` | historical overlay baseline |
| fixed-clock v4 control | `16.8024 bps` | ablation baseline |
| fixed-clock v4 multi-denominator | `17.1714 bps` | archived previous baseline |
| `opening_model` | **`17.7934 bps`** | 最新信号/模型基准；策略层待重跑 |
| 10:01-10:10 multi-denominator | `6.5491 bps` | decay checkpoint；不晋级 |
| corrected 3×2：09:31 / 3m | **`20.2087 bps`** | 六格经济指标最强；旧兼容血缘结论 |
| corrected 3×2：10:01 / 1m~3m | `6.6667~8.3935 bps` | 信号层归档；不晋级 |
| corrected 3×2：14:01 / 1m~3m | `1.0904~2.3101 bps` | 信号层归档；不晋级 |
| ds350 max-30：09:31 / 1m~close | `17.6132~20.6595 bps` next excess | 当前权威 label 矩阵；待策略验收 |
| ds350 max-30：10:01 / 1m~close | `7.1724~10.7120 bps` next excess | 窗口衰减确认；诊断保留 |
| ds350 max-30：14:01 / 1m~5m | `1.1508~2.7231 bps` next excess | 明显衰减；不进入晋级队列 |

`opening_model` 的当前证据见
[baseline evidence](../experiments/evidence/baselines/opening_model/)，不可变来源见
[canonical registry](../experiments/canonical/opening.toml)。完整历史数字见
[experiment log](experiment_log.md)。

15-label 的逐 case 指标见
[max-30 matrix](../experiments/evidence/backtests/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/matrix_summary.csv)。
不同 horizon 改变了监督目标，表中最好数字不能直接解释为单变量模型提升。

## 验收逻辑

信号层固定检查：

1. `pool_L` Top100 next internal excess 与费用后累和；
2. 年、半年、月和 decision clock 稳定性；
3. universe short Rank IC 与 Top1000 bucket；
4. capacity、exposure 和成本敏感性。

策略层补充因果入场/退出、同日预算与 overlap、成交约束和尾部/集中度。单边 cap、trim、bootstrap、
overlap 等用于风险解释，不设自动通过或否决门槛。

## 已知边界与下一步

- visible refill 是决策前重分配，不是真实失败回报后的二次下单；
- 尚无通用的全天持仓、退出和现金复用账本；
- ask2-10 深度在现有输入中无有效信息；
- GPU 复跑依赖 run/Job 中记录的镜像和外部 cache。
- 当前 350-feature NN 优先使用高显存 GPU 驻留快路径；80/96GB 节点一次保存约 48GiB 训练 tensor。
  代码保留可显式选择的 host/vectorized 路径，但权威矩阵配置不允许显存不足时静默 fallback。完整矩阵的
  历史主机内存峰值约 202GiB，因此正式资源口径仍为 8 CPU、256GiB request、1 GPU。

后续顺序：

1. 从 `opening_label_matrix` 的 09:31 候选中预注册少量候选，完成统一
   capacity/no-refill/visible-refill、exposure 和 strategy acceptance；
2. 只有完整验收通过后，才创建新的 `opening_model` source run 并原子更新 canonical/evidence/docs；
3. 后续新实验直接使用分层后的 `features_350 + horizon label` 数据，不再现场拼旧 target cache；
4. 10:01/14:01 保留为衰减诊断，不在没有新假设时重复消耗训练资源。
