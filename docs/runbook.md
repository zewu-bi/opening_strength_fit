# Runbook

本文只保留可执行流程。实验数字见 [experiment log](experiment_log.md)，目录契约见
[experiments/README](../experiments/README.md)。

## 1. 本地复现

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e ".[dev]"
.venv/bin/python -m pip check

make smoke
make ci
make contracts
```

`make smoke` 使用 `examples/smoke/` 的 12 行 tracked CSV，运行完整的 labeled input → Ridge →
prediction/metrics 流程，不访问 ClickHouse、S3 或 PVC。输出在 `output/smoke/`。

## 2. 外部数据

正式研究支持三类输入：

```text
data.source = clickhouse    查询原始 tick
data.source = labeled_pvc   读取已构建 labeled frame
data.source = path          读取本地 parquet/csv
```

连接信息只放 `.env` 或集群 Secret，字段模板见 `.env.example`。不要提交密码、token、原始 tick、股票池、
prediction、模型文件或 cache。

数据契约：

- 样本键为 `date × symbol × decision_target_timestamp`；
- `clock_state` entry 保存逻辑执行时钟和此前最后可见状态，并记录 state age；
- 日频市值/股本 enrichment 必须严格使用上一交易日；
- 股票池默认只过滤 selection，训练仍使用 full universe；严格敏感性可用前一交易日 membership；
- cache 发布必须同时具备 parquet、manifest 和 ready marker；临时文件或 lock 不代表完成。

## 3. 正式实验闭环

1. 复制最接近的 `experiments/runs/<run_id>.toml`，只改变待验证变量。
2. 更新 `run.id`、description、status、输入 lineage 和输出目录。
3. 根据代码 revision 构建不可变镜像，渲染并保存 Job manifest。
4. dry-run、提交、观察直到所有 shard 和 `_SUCCESS` 完成。
5. 运行 pool-internal 或对应 audit/acceptance。
6. 同步结果到 ignored mirror，再选择 compact evidence。
7. 运行 contracts，核对 trace，更新 experiment log。

失败和负面实验也保留配置，并使用 `canceled` 或 `superseded`，不要删除或覆盖。

## 4. Cache 与 label

```bash
osf-probe-clickhouse-data --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
osf-build-labeled-cache --config experiments/runs/<cache_run_id>.toml
osf-build-target-label-cache --config experiments/runs/<target_run_id>.toml
osf-build-next-close-labels --config experiments/runs/<next_close_run_id>.toml
```

决策特征与 label 边界必须分别声明对齐语义：

- `sample.decision_alignment = "clock_state"`：在每个逻辑决策时钟读取此前最后已知状态，并记录
  `decision_source_timestamp` 和 `decision_state_age_seconds`；无新物理 tick 不等于状态缺失。
- `sample.decision_alignment = "next_tick"` 加 `decision_max_lag_seconds = 5`：历史兼容口径，只保留
  目标时刻后 5 秒内出现新 tick 的股票分钟。
- `labels.entry_alignment = "clock_state"` 只控制入场边界，不能替代 decision alignment。corrected
  fixed-clock cache 的 entry 必须锚定 `decision_target_timestamp + entry_clock_delay_seconds`，而不是
  source tick 加 delay。

旧物理 tick-delay 和 forward-5s 配置只为历史复现与已生成的日内窗口对照保留。大型 cache 仅在 PVC；
Git 中保留构建代码、run config、Job、schema/fingerprint 语义和关键 trace。

## 5. 镜像与 Job

代码、依赖或 Dockerfile 变化时构建完整镜像；只有 run TOML 变化且基础镜像已包含所有入口时才使用
overlay。GPU 镜像必须显式固定与集群 CUDA 匹配的 Torch wheel。

```bash
IMAGE=<registry>/<repository>:<immutable-tag>

docker build \
  --build-arg CACHE_BUST=<version> \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t "$IMAGE" .

osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --image "$IMAGE" \
  --sharded

<kubectl-wrapper> apply --dry-run=client \
  -f experiments/jobs/<run_id>_sharded_job.yaml
<kubectl-wrapper> apply \
  -f experiments/jobs/<run_id>_sharded_job.yaml
```

观察任务：

```bash
osf-rolling-job-status --config experiments/runs/<run_id>.toml
osf-rolling-job-status --config experiments/runs/<run_id>.toml --tail 160
```

Job `Complete` 不等于输出完整；逐 shard 检查 `_SUCCESS`、metrics 和 trace。

## 6. 分析与 evidence

训练后先做 pool-internal 分析，再同步：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --analysis --image <cpu-image>

osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --all

osf-audit-experiments --require-metrics
osf-check-project-contracts
```

`--all` 将原始同步结果放在 ignored 的 `experiments/results/` 或 `output/`，并把允许的 compact 文件复制到
tracked `experiments/evidence/`。策略验收只记录 summary、bootstrap、leave-one-out、trace 和成功标记，
不记录逐日/逐决策表。

正式信号比较固定检查：short IC 与 next excess、费用后累计曲线、Top1000 平滑分桶、Top1000 收益区间
分布。对应入口为 `osf-plot-optimization-direction-comparison` 和
`experiments/scripts/run_top1000_rank_bucket_diagnostics.py`。

当前 canonical multiden 的四图、compact plot data 和 trace 可从本地 ignored mirror 统一刷新：

```bash
make evidence-four-figures
```

目标目录为 `experiments/evidence/backtests/<multiden-run-id>/`。control 只在前两图中保留为 ablation
baseline；multiden 是当前 opening policy/incumbent。

## 7. 容量与策略验收

```text
prediction
  -> osf-audit-capacity
  -> osf-analyze-capacity-acceptance
  -> osf-extract-execution-context
  -> osf-analyze-realistic-acceptance
  -> osf-audit-strategy-acceptance
```

统一策略验收比较：

```text
capacity_only
realistic_no_refill
visible_pretrade_refill
```

当前 policy 晋级以同一因果 OOS lineage 下的成本后资本收益、容量约束和执行可行性为主。单边 P95/P99
upper-tail cap、trim、top day/symbol、leave-one-out、月块 bootstrap、overlap 和集中度必须保留在
evidence 中，但只作为收益来源与风险诊断，不设置自动通过/否决阈值。单边 cap 尤其不能替代双边异常值
敏感性或真实成交 haircut。multiden 的 visible pre-trade refill 已按此口径随当前 opening policy 晋级。

可复用配置：

```text
experiments/runs/strategy_acceptance_clock6_v4_multiden_2022_2025_v1.toml
experiments/runs/strategy_acceptance_clock6_v4_control_2022_2025_v1.toml  # comparison only
```

## 8. 日内窗口衰减实验

以当前 incumbent run 为唯一模板。为每个预先固定的十分钟窗口新建 cache run 和训练 run，只修改
`[sample].start_time`、`end_time`、`decision_times` 以及由此变化的 cache lineage；label、feature、
模型、股池、rolling window、seed 和验收参数保持一致。不要构造全天序列或新的隔夜目标。

每个窗口独立完成 cache smoke、全量 cache、8-fold OOS、pool-internal 和同口径 acceptance，再把
`09:31-09:40` 与另外 2–3 个窗口按时点排序，报告 Rank IC、Top100 next excess、正半年/月比例、
容量和成本后结果的衰减。窗口的具体时钟在第一个 run config 提交前统一确定，避免看结果后移动窗口。

每个完成窗口都以对应训练 run id 写入 tracked evidence，至少保留固定四图、compact CSV、trace、
pool summary 和 SHA-256 manifest。固定四图已经显示 next excess、分期稳定性和 fee8 累和明显衰减的
窗口可以在信号层归档，不强制继续跑 downstream capacity/realistic promotion audit；该停止规则必须在
experiment log 中明确记录，不能解释为模型训练失败。

## 9. 常见故障

| 症状 | 检查 |
| --- | --- |
| container 找不到新配置/入口 | 镜像是否包含当前 revision；必要时 full build |
| cache 只有 tmp/lock | 等待 parquet、manifest、ready marker |
| completed 但 metrics 缺失 | shard `_SUCCESS`、同步参数和远端目录 |
| GPU kernel 不兼容 | Torch/CUDA wheel 与节点 compute capability |
| replay fill 异常 | execution context join、status/spread/depth 列和 trace |
| contract 报错 | run id/status、TOML/Job/entrypoint 对齐及 evidence 大小 |
