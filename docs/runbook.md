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
osf-build-daily-return-labels --config experiments/runs/<daily_return_run_id>.toml
```

新的 fixed-clock lineage 使用 `entry_alignment = "clock_state"`、显式 clock delay 和完整
cross-section readiness。旧物理 tick-delay 配置只为历史复现保留。大型 cache 仅在 PVC；Git 中保留构建
代码、run config、Job、schema/fingerprint 语义和关键 trace。

全天路径转日频 feature 时，主监督目标使用以 D 为 feature date、D+1 为 target date 的
`alpha_return_close_to_next_close`。其交易语义是 D 收盘集合竞价买入、D+1 收盘卖出；label 使用
`ClosePrice(D+1) / PreClosePrice(D+1) - 1`，以消除除权除息的机械缺口。它只持久化
`date/symbol/target_date/label` 四列；下一交易日必须由交易日历映射，不允许自然日 `+1`。

对应正式 cache 为：

```text
/mnt/output/opening_strength_fit/cache/daily_close_to_next_close_labels_v1/
```

同日收盘下单的 feature 必须在下单前完成。第一版固定 `14:50` 截止；按现有采样/退出定义，1m、10m、
60m 路径的最晚目标分钟分别约为 `14:47`、`14:38`、`13:48`。旧
`alpha_return_next_session_open_close` cache 只作诊断，不作现金股票 T+1 的可执行主目标。

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

## 8. 全天分钟级因果 label/cache

全天窄标签继续使用 `osf-build-labeled-cache`；`output_mode = "labels_only"` 时只查询必要 tick 字段，
并写出键加 1m/10m/60m 三个收益列。先运行一日 smoke，再启动年度任务：

```bash
osf-build-labeled-cache \
  --config experiments/runs/build_full_day_clock6_narrow_labels_smoke_20250102_v2.toml

hfcli kubectl --cluster research --namespace bizewu apply -f \
  experiments/jobs/build_full_day_clock6_narrow_labels_smoke_20250102_v2_job.yaml
```

逐日输出为 `date=YYYY-MM-DD/labels.parquet` 和 `summary.json`；两者都存在时自动断点复用。根目录的
`full_day_label_cache_manifest.json` 必须满足 `causal_timestamp_violations = 0`，最终成功标志为
`_SUCCESS`。字段口径、午休交易时钟和年度启动步骤见
[全天分钟级因果 label/cache V1](full_day_temporal_labels_v1.md)。

2025-01-02 窄表 smoke 已完成：`20,941,373` 条输入 tick 生成 `1,136,863 × 6`，240 个决策分钟齐全，
`7,270,214` 次因果时间戳比较为零违规，耗时 `5m21s`、峰值约 16GB。2019–2025 七个年度 Job
按年度并行运行，每年内部按日写分片并可断点续跑。

## 9. 常见故障

| 症状 | 检查 |
| --- | --- |
| container 找不到新配置/入口 | 镜像是否包含当前 revision；必要时 full build |
| cache 只有 tmp/lock | 等待 parquet、manifest、ready marker |
| completed 但 metrics 缺失 | shard `_SUCCESS`、同步参数和远端目录 |
| GPU kernel 不兼容 | Torch/CUDA wheel 与节点 compute capability |
| replay fill 异常 | execution context join、status/spread/depth 列和 trace |
| contract 报错 | run id/status、TOML/Job/entrypoint 对齐及 evidence 大小 |
