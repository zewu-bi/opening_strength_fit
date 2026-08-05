# Runbook

本文只保留可执行流程。研究判断见 [project brief](project_brief.md)，实验事实见
[experiment log](experiment_log.md)，目录契约见 [experiments README](../experiments/README.md)。

## 1. 本地复现

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e ".[dev]"
.venv/bin/python -m pip check

make smoke
make ci
make contracts
```

`make smoke` 使用 `examples/smoke/` 的小型输入，不访问 ClickHouse、S3 或 PVC。

## 2. 外部数据

```text
data.source = clickhouse    查询原始 tick
data.source = labeled_pvc   读取 PVC frame
data.source = path          读取本地 parquet/csv
```

连接信息只放 `.env` 或集群 Secret。密码、token、原始数据、prediction、模型和大型 cache 不进 Git。

数据门禁：

- 样本键为 `date × symbol × decision_target_timestamp`；
- decision、entry 和 future 读取逻辑时刻以前最后可见状态，entry 固定 `+6s`；
- 日频 enrichment 只用上一交易日；训练使用 full universe，股票池默认只过滤 selection；
- 发布物必须同时具备 parquet、manifest 和 `_SUCCESS`。

## 3. 正式实验闭环

1. 复制最接近的 run TOML，只改变待验证变量。
2. 更新 `run.id`、description、status、输入 lineage 和输出目录。
3. 构建不可变镜像，保存 Job manifest，dry-run 后提交。
4. 等待所有 shard 和 `_SUCCESS`，再运行分析/验收。
5. 同步 compact evidence，运行 contracts，更新 experiment log。

失败实验也保留配置，用 `canceled` 或 `superseded`。命名规则见
[canonical README](../experiments/canonical/README.md)。

## 4. Cache 与 label

```bash
osf-probe-clickhouse-data --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
osf-build-raw-source-cache --config experiments/runs/opening_<window>_raw_source.toml --year <YYYY>
osf-build-training-datasets --config experiments/runs/opening_<window>_features_350.toml --kind features --year <YYYY>
osf-build-training-datasets --config experiments/runs/opening_<window>_labels_5.toml --kind labels --year <YYYY>
osf-split-horizon-labels --config experiments/runs/opening_<window>_labels_horizon_split.toml --year <YYYY>
osf-build-long-label-raw-source --config experiments/runs/opening_0931_1010_long_label_raw_source_v1.toml --year <YYYY>
osf-build-long-horizon-labels --config experiments/runs/opening_<window>_labels_10m_1h_close.toml --year <YYYY>
osf-split-long-horizon-labels --config experiments/runs/opening_<window>_labels_10m_1h_close_split.toml --year <YYYY>
```

后续实验从一个最终 label 的独立目录取数：

```text
/mnt/output/opening_strength_fit/datasets/opening_{0931_0940,1001_1010,1401_1410}_labels_h{1,3,5}m_v2
/mnt/output/opening_strength_fit/datasets/opening_{0931_0940,1001_1010}_labels_{h10m,h1h,hclose}_v1
```

- 最终 label 发布时按 3 个样本键与同窗口 `opening_<window>_features_350` 对齐，并保持相同行顺序；
- 目标列为 `target_label`，NaN 行不训练；
- 消费前检查两侧 parquet、manifest、`_SUCCESS`、key 唯一性和 join 覆盖；
- 中间 `1m_3m_5m_next_mixed`、`10m_1h_close_next` 文件和旧 target cache 不作为新 run 输入。
- 长持有期 10m/1h 使用持有期结束后的 60 秒 VWAP，`hclose` 使用当日收盘价；三者均复用既有
  `label_next_close` 后构造相同权重的 mixed target。

训练直接传入同窗口的 feature 与最终 label，不再生成合并 cache：

```bash
osf-train \
  --config experiments/runs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1.toml \
  --feature-input /mnt/output/opening_strength_fit/datasets/opening_<window>_features_350 \
  --label-input /mnt/output/opening_strength_fit/datasets/opening_<window>_labels_<horizon>_<version> \
  --run-id <case_run_id> \
  --rolling-monthly --train-months 36 --test-months 6
```

发布阶段负责完整的 key 唯一性和覆盖检查。训练的 model-ready 快路径检查总行数并抽样核对三键，
随后按发布顺序只挂接 3 个窄 label 列；它不再重复执行全表 key 扫描、universe 正则过滤或通用特征清洗。
未显式启用该快路径的普通输入仍走完整 key join 和清洗后备逻辑。
当前矩阵运行三窗口 × 1m/3m/5m 和前两窗口 × 10m/1h/close，共 15 个 case、每个 8 个
半年 OOS 切分。每个 case 使用一个独立 Indexed Job（`completions: 8`），按 label 受控接力，
使全局最多同时运行 8 个训练 shard；不要把 15 个 case 合并成一个 120-shard Job。

这组 NN 必须使用向量化训练路径。训练集标准化后，完整 float32 feature/label tensor 一次搬到 GPU，
每个 epoch 只生成一个整块随机索引 tensor，再用 GPU `index_select` 切 batch；不得恢复
`TensorDataset -> Subset -> DataLoader` 的逐行 Python 取数。loss 只在每个 epoch 结束时同步回 CPU。
配置固定 `training_tensor_storage = "cuda_resident"`，显存不足时直接失败，不允许静默退回慢路径；
Job 只允许 `A100-80`、`H100-80` 和 `H20`，排除 32GB 5090。

2026-08-05 用最大 2025-07 fold 实测：36,119,812 个训练样本、350 个特征，训练 tensor
`51,288,133,040` bytes（47.77GiB），H20 显存稳定在 50,841MiB（约 49.65GiB），训练期 SM
连续约 83%–85%；
容器主机内存峰值 202.0GiB。完整 fold 从 Pod 启动到写完 645 万行预测共 9 分 15 秒，10 epochs、
best epoch 10。因而每 Pod 申请 `8 CPU / 256Gi / 1 GPU`、上限 `16 CPU / 384Gi / 1 GPU`；
内存不能按训练期常驻量下调，因为峰值出现在 dataframe 到标准化矩阵的过渡阶段。

四个清单覆盖全部 15 个 Job，默认全部 suspend；发布后由一个持久队列脚本逐个放行：

```bash
hfcli kubectl apply -f experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w0931_jobs.yaml
hfcli kubectl apply -f experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w1001_jobs.yaml
hfcli kubectl apply -f experiments/jobs/nn_ds350_label12_36m_grouped_gated_v2_mse_v1_w1401_jobs.yaml
hfcli kubectl apply -f experiments/jobs/nn_ds350_label15_36m_grouped_gated_v2_mse_v1_h5m_jobs.yaml

systemd-run --user --unit os-nn-ds350-label15-queue \
  --working-directory "$(pwd)" \
  experiments/scripts/run_ds350_label15_training_queue.sh
```

按最大 fold 外推，单个 label 的 8 个 shard 并行约 10 分钟，15 个 label 顺序队列约 2.5 小时；
把镜像拉取、共享 PVC 和节点差异计入后按 2.5–4 小时看待。日志中的 `torch_training_storage` 必须显示
`storage: cuda_resident`，每个 epoch 会输出 loss 和秒数，便于区分预处理与真正卡住。

历史 `next_tick`/forward-5s 只用于旧实验复现；新 cache 默认使用
`decision_alignment/entry_alignment/future_alignment = clock_state` 和 `entry_clock_delay_seconds = 6`。

## 5. 镜像与 Job

代码、依赖或 Dockerfile 变化时完整构建镜像；仅配置变化且基础镜像已含入口时才使用 overlay。

```bash
IMAGE=<registry>/<repository>:<immutable-tag>

docker build \
  --build-arg CACHE_BUST=<version> \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t "$IMAGE" .

osf-render-k8s-job --config experiments/runs/<run_id>.toml --image "$IMAGE" --sharded
<kubectl-wrapper> apply --dry-run=client -f experiments/jobs/<run_id>_sharded_job.yaml
<kubectl-wrapper> apply -f experiments/jobs/<run_id>_sharded_job.yaml
```

```bash
osf-rolling-job-status --config experiments/runs/<run_id>.toml
osf-rolling-job-status --config experiments/runs/<run_id>.toml --tail 160
```

Job `Complete` 后仍要检查 shard 输出、metrics 和 trace。

## 6. 分析与 evidence

```bash
osf-render-k8s-job --config experiments/runs/<run_id>.toml --analysis --image <cpu-image>
osf-sync-experiment-artifacts --config experiments/runs/<run_id>.toml --all
osf-audit-experiments --require-metrics
osf-check-project-contracts
```

原始同步结果留在 ignored mirror；Git 只保留 summary、trace、小型审计表和标准图。正式信号比较固定使用
short IC/Top100 excess、费用后累和、Top1000 分桶和收益分布四图。证据范围见
[evidence README](../experiments/evidence/README.md)。

## 7. 容量与策略验收

```text
prediction
  -> osf-audit-capacity
  -> osf-analyze-capacity-acceptance
  -> osf-extract-execution-context
  -> osf-analyze-realistic-acceptance
  -> osf-audit-strategy-acceptance
```

统一比较 `capacity_only`、`realistic_no_refill` 和 `visible_pretrade_refill`。晋级依据是同一因果 OOS
lineage 下的成本后资本收益、容量和执行可行性；tail、bootstrap、overlap 和集中度只作诊断。

## 8. 日内窗口衰减实验

以 `opening_model` 为模板；每个预先固定的十分钟窗口只修改 sample 时钟和对应数据 lineage，保持 label
定义、feature、模型、股池、rolling window、seed 和验收参数不变。每个窗口独立完成数据检查、训练、
pool-internal 和四图；明显落后的窗口可在信号层归档，停止规则写入 experiment log。

## 9. 常见故障

| 症状 | 检查 |
| --- | --- |
| container 找不到配置/入口 | 镜像是否包含当前 revision |
| cache 只有 tmp/lock | 等待 parquet、manifest、`_SUCCESS` |
| completed 但 metrics 缺失 | shard 成功标记、同步参数和远端目录 |
| GPU kernel 不兼容 | Torch/CUDA wheel 与节点能力 |
| replay fill 异常 | execution join、status/spread/depth 和 trace |
| contract 报错 | run id/status、TOML/Job/entrypoint 和 evidence 大小 |
