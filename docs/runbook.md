# Runbook

本文件只描述可执行流程。研究结论和数字见 [experiment_log.md](experiment_log.md)，当前目标见
[project_brief.md](project_brief.md)，入口所有权见 [project_map.md](project_map.md)。示例中的
`<run_id>`、`<image_tag>` 和路径必须替换，不把某次实验的临时值固化为通用操作。

## 1. 本地准备

```bash
cd ~/projects/opening_strength_fit
source .venv/bin/activate
python -m pip install -c requirements.lock -e ".[dev]"

python -m pip check
make ci
make contracts
```

`make ci` 运行 Ruff lint、format check 和 pytest；`make contracts` 校验 run/config/job/CLI 对齐。
提交集群任务前，二者都应通过。

需要 `hfcli` 提交或查看集群任务时，安装集群 extra：

```bash
python -m pip install -c requirements.lock -e ".[dev,cluster]"
```

本地连接信息放 `.env`，不要提交：

```bash
CLICKHOUSE_HOST='ch.db.prod.highfortfunds.com'
CLICKHOUSE_PORT='8123'
CLICKHOUSE_USER='...'
CLICKHOUSE_PASSWORD='...'
CEPH_LDAP_ID='...'
CEPH_LDAP_KEY='...'
```

## 2. 实验闭环

每个正式实验按同一顺序执行：

1. 从最接近的 `experiments/runs/*.toml` 复制新配置；只改变待验证变量。
2. 保证文件名、`[run].id` 和输出目录一致；状态从 `queued` 开始。
3. 选择镜像策略，渲染 Job，并做 client-side dry-run。
4. 提交、观察并等待完成；更新状态为 `running` / `completed`。
5. 运行 pool-internal 或对应 audit/acceptance。
6. 同步 compact artifacts，运行 experiment audit 与 project contracts。
7. 核对 trace 和主指标，再更新 experiment log；失败实验同样记录。

合法状态只有：

```text
queued -> running -> completed
                 \-> canceled
completed/queued -> superseded
```

不要引入 `submitted`、`done` 等同义状态。

## 3. 数据与缓存

### PVC 输出布局

新 run 使用 `experiments/config_templates/pvc_layout_v2.toml`，省略 `output.k8s_dir` 后由
`run.kind` 和 `model.name` 自动确定目录：

```text
runs/
├── models/{lightgbm,nn,gbm,ridge,ensemble}/
├── data/{labeled-cache,cache-transform,next-close-label-cache}/
├── analyses/<run-kind>/
└── legacy/untracked/
```

v2 rolling shard 使用 `fold_<start>_<end>`，年度 shard 使用 `fold_<year>-01_<year>-12`；显式
`k8s_dir` 的历史配置仍使用 legacy shard 命名，读取器兼容两代布局。迁移和回滚映射保存在 PVC
的 `.layout_migrations/`；活动任务目录必须等任务结束后再移动。

label cache 的规范目录/文件名统一为：

```text
opening_<range>_label_vN_<entry_semantics>_<base|mixed>_<enrichment>/
opening_<year>_label_vN_<entry_semantics>_<base|mixed>_<enrichment>.parquet
```

当前 lineage 中，`v1_tick2_physical` 是旧物理行 tick2，`v2_tick2_unique` 是去重后的 tick2，
`v3_tick2_gap5_ready` 是严格 gap/readiness 对照，`v4_clock6_state_unique` 是正式 fixed-clock +6 秒
语义。2026-07-17 的 layout v4 迁移只在同一 PVC 内 rename，不修改 parquet 内容。该 PVC 不可靠支持
symlink/hard-link/reflink，因此不保留旧路径 alias；所有 run config 和 Job manifest 已机械改写为规范路径。
空的失败目录、残留 heartbeat lock、退化成 0 字节文件的 alias，以及已由 v1 年度 mixed cache 完整
覆盖的 18 个月冗余单文件均已清理；操作记录位于
`/mnt/output/opening_strength_fit/.layout_migrations/`。

### 数据源检查

```bash
osf-probe-clickhouse-data \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD>
```

训练支持：

```text
data.source = clickhouse    直接查询 stock.tick
data.source = labeled_pvc   读取已构建 labeled cache
data.source = path          读取本地 parquet/csv
```

### 构建 labeled cache

```bash
osf-build-labeled-cache \
  --config experiments/runs/<cache_run_id>.toml
```

缓存完成条件是目标 parquet、manifest 和 ready marker 都存在；`.tmp`、lock 或 heartbeat 只表示仍在写。
不要并发重建同一路径。正式 run 使用 `cache.require_manifest = true`；旧 cache 只有在 read-only
模式下可暂时省略 manifest，新写入路径会先发布 manifest 再标记 ready。一旦存在 manifest，schema、
文件列和构建配置 fingerprint 不匹配会直接失败。

当源 `stock.tick` 存在同一股票、同一交易所时间戳的重复物理行，并希望 `entry_tick_delay` 按真实快照
而不是物理行计数时，仅在新 cache lineage 中启用：

```toml
[data]
tick_timestamp_deduplication = "latest_local_timestamp"
```

该步骤在特征和 label 构造前按 `date × symbol × timestamp` 只保留一行，优先最新本地接收时间，再按
累计成交状态和稳定 fingerprint 决定。重复行不消耗 delay tick，也不直接判 label 无效；例如 delay2
遇到重复时间戳会继续读取后面的第二个不同时间戳。未设置此字段的旧配置仍按物理行 `shift(-N)`，用于
保持历史 cache 的原样可复现性。

对省略未变化快照的 event/state 数据，正式 fixed-clock cache 不按“后续第 N 条更新”定义执行时间，
也不以单股票相邻更新间隔代理数据新鲜度。使用：

```toml
[labels]
entry_tick_delay = 2
entry_alignment = "clock_state"
entry_clock_delay_seconds = 6
future_alignment = "clock_state"
require_entry_after_cross_section_ready = true
```

`entry_timestamp` 表示逻辑执行时钟，`entry_source_timestamp` 表示该时钟之前最后一条可见状态；
两者之差写入 `entry_state_age_seconds`。sell start/end 使用同样的 backward point-in-time lookup。
clock-state 模式禁止同时设置 `entry_max_gap_seconds` 或 `max_future_gap_seconds`，避免再次把
“状态未变化”误判成缺失。若要判断 feed 是否真正 stale，应另查 heartbeat、sequence、抓取进程状态或
本地接收延迟。固定时钟不能替代同交易所时间戳去重：重复 revision 仍会改变盘口值和 lag/diff/path 特征。

### 严格上一交易日日频 enrichment

需要把公司规模字段保留在 labeled cache 供当前或后续任务使用时，在直接读取 ClickHouse 的 base-cache
配置中启用：

```toml
[daily_market_reference]
enabled = true
table = "stock.daily_bar_jy"
lag_sessions = 1
market_cap_unit_multiplier = 10000.0
share_unit_multiplier = 10000.0
```

该 enrichment 先找满足 `TradingDay < sample_date` 的最近交易日，再按 `symbol` 做 many-to-one join；
不得使用样本日记录，也不得在源表缺失时退回样本日。写入 cache 的业务字段为
`total_market_cap`、`float_market_cap`、`total_shares`、`float_shares`、`free_float_shares`；
`market_cap_reference_date` 和 `market_cap_reference_lag_sessions` 是审计 context，不进入模型特征。
`stock.daily_bar_jy` 的原始市值和股本单位均为万，当前统一乘 `10000` 转成元和股。
enrichment 字段写入 cache 不等于自动成为 raw model feature；由具体实验的 feature allowlist 和 value
transform 决定用途。当前 mechanismized v3 将市值/股本作为 support reference，用于把名义金额、盘口量和
成交量变成相对公司规模的无量纲值，但不把五个原始规模字段直接加入模型矩阵。

跨年度批量链路默认按 `base cache -> mixed/derived target cache -> training` 分阶段提交。下游配置中的
`wait_for_paths` 只是防止误读不完整产物的安全网，不是调度器；上游未完成时提前提交 target 或模型 Pod，
Pod 仍会按整 Pod request 参与资源调度，可能表现为 Pending 或 Running 后等待文件。若为了 overnight 排队
而主动预提交下游，必须在 experiment log 中记录这是 waiting job，并接受这段时间的资源排队/占用；正常流程仍
应在 base 完成后先检查 parquet、manifest、ready marker 和 reference-date 日志，再提交 target，所有
target 完成并审计后再提交模型。

### 派生 target 与 next-close label

```bash
osf-build-target-label-cache \
  --config experiments/runs/<target_cache_run_id>.toml

osf-build-next-close-labels \
  --config experiments/runs/<next_close_cache_run_id>.toml
```

feature hygiene 的 drop list 只影响模型输入，不能从 cache 物理删除 `ask_price_1`、entry context、
label 或回测依赖字段。

### 外部股池

```text
lml.bzw@ssd/data/pool_S.parquet
lml.bzw@ssd/data/pool_M.parquet
lml.bzw@ssd/data/pool_L.parquet
```

默认保持 full-universe 训练，只在选择时加 mask：

```toml
[stock_pool]
pool = "L"
filter_train = false
filter_selection = true
pool_date_lag_sessions = 0
```

严格 no-lookahead sensitivity 使用 `pool_date_lag_sessions = 1`。

## 4. 镜像

| 变更 | 镜像策略 |
| --- | --- |
| `src/`、依赖、Dockerfile 变化 | full build |
| 只有 `experiments/runs/*.toml` 变化，且 base 已含全部入口 | config overlay |
| GPU NN | full/overlay 镜像必须含匹配集群 CUDA 的固定 Torch build |
| CPU analysis/audit | 使用 CPU 镜像，不引用 GPU training 镜像 |

通用变量：

```bash
IMAGE_REPO=registry.corp.highfortfunds.com/bizewu/opening-strength-fit
VERSION=$(date +%Y%m%d)-<purpose>-v1
IMAGE=${IMAGE_REPO}:${VERSION}
```

`IMAGE` 必须是不可变 tag 或 digest；`osf-render-k8s-job` 不再默认使用 `:latest`。

Full build：

```bash
docker build \
  --build-arg CACHE_BUST=${VERSION} \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t ${IMAGE} .
docker push ${IMAGE}
```

GPU build：

```bash
docker build \
  --build-arg INSTALL_TORCH_CUDA=1 \
  --build-arg TORCH_PACKAGE='torch==<validated_version>' \
  --build-arg CACHE_BUST=${VERSION} \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) \
  -t ${IMAGE} .
docker push ${IMAGE}
```

Config overlay：

```bash
BASE_IMAGE=${IMAGE_REPO}:<verified_base_tag>
OVERLAY_IMAGE=${IMAGE_REPO}:${VERSION}
docker pull ${BASE_IMAGE}
```

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app/opening_strength_fit
RUN rm -rf experiments/runs
COPY experiments/runs ./experiments/runs
```

保存上段为临时 Dockerfile 后 build/push。渲染前确认目标 image digest 和所有新增 CLI 都存在。

## 5. 训练 Job

普通训练：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --image ${IMAGE}

hfcli kubectl --cluster research apply --dry-run=client \
  -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research apply \
  -f experiments/jobs/<run_id>_job.yaml
```

Rolling/sharded：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --sharded \
  --image ${IMAGE}

hfcli kubectl --cluster research apply --dry-run=client \
  -f experiments/jobs/<run_id>_sharded_job.yaml
hfcli kubectl --cluster research apply \
  -f experiments/jobs/<run_id>_sharded_job.yaml
```

同名 Job spec 变化时先删除再 apply：

```bash
hfcli kubectl --cluster research delete job <job_name> \
  --ignore-not-found -n bizewu
```

观察：

```bash
osf-rolling-job-status --config experiments/runs/<run_id>.toml
osf-rolling-job-status --config experiments/runs/<run_id>.toml --tail 160
hfcli kubectl --cluster research logs -n bizewu <pod_name> -f
hfcli kubectl --cluster research wait \
  --for=condition=complete job/<job_name> -n bizewu --timeout=24h
```

Indexed shard 必须各自写独立目录和 `_SUCCESS`；不要只凭 Job `Complete` 判断产物完整。

## 6. Pool-internal 分析

正式 TopN、Rank IC、稳定性和 SVG 在集群侧独立运行：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --analysis \
  --image ${CPU_IMAGE}

hfcli kubectl --cluster research apply --dry-run=client \
  -f experiments/jobs/<run_id>_pool_internal_analysis_job.yaml
hfcli kubectl --cluster research apply \
  -f experiments/jobs/<run_id>_pool_internal_analysis_job.yaml
```

标准输入/输出：

```text
predictions        /mnt/output/opening_strength_fit/runs/<category>/<run_id>/
next-close labels  /mnt/output/opening_strength_fit/cache/<label_cache>/
stock pools        lml.bzw@ssd/data/pool_{S,M,L}.parquet
analysis output    <run_output>/analysis/pool_internal_top100/
```

2020 年以前没有 S/M/L 覆盖，早期 shard 只配置 `pools = ["universe"]`。

## 7. 同步与验收图

同步 compact artifacts：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --all

osf-audit-experiments --require-metrics
osf-check-project-contracts
```

默认 `osf-audit-experiments` 只依赖 tracked run/job，可用于干净 checkout 和 CI；artifact sync 后使用
`--require-metrics`，再校验 ignored 的本地 metrics mirror 是否完整。

只有调试单个 shard 时才拉 prediction parquet：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --predictions --allow-partial
```

标准 overlay acceptance：

```bash
osf-plot-optimization-direction-comparison \
  --output-dir experiments/results/backtests/<comparison_id> \
  --baseline-run-id <incumbent_run_id> \
  --baseline-label <incumbent_label> \
  --direction <key>=<label>=<challenger_run_id> \
  --cumulative-relative-mode pool_l \
  --realized-fee-bps 8
```

### 固定四图验收

每个进入信号阶段正式比较的 candidate 必须保留下面四张图；四图使用同一 OOS 月份、同一
next-close label lineage 和 `pool_L`，不能用 Top100-only 探索图替代 Top1000 图。

| # | 固定产物 | 验收内容 | 生成入口 |
| --- | --- | --- | --- |
| 1 | `optimization_directions_overlay_acceptance.svg` | universe short Rank IC + `pool_L` Top100 next internal excess | `osf-plot-optimization-direction-comparison`；实现位于 `src/opening_strength_fit/optimization_acceptance_workflow.py` |
| 2 | `optimization_directions_net_alpha_cumulative.svg` | Top100 8 bps 累计净收益及相对 `pool_L` 的累计净收益差 | 同一 `osf-plot-optimization-direction-comparison` 调用 |
| 3 | `top1000_bucket_returns.svg` | Top1000 的 10/20/50 个平滑 score bucket 收益形状 | `experiments/scripts/run_top1000_rank_bucket_diagnostics.py` 默认模式 |
| 4 | `top1000_score_bucket_return_100bps_counts.svg` | Top1000 每100名一组的完整收益区间分布，用于识别中部重叠和左右尾差异 | `experiments/scripts/plot_top1000_score_bucket_return_histogram.py` |

先完成 pool-internal analysis 和 compact artifact sync，再生成前两张比较图。Top1000 两张图的数据
量更大，通常在集群读取 prediction/label 后只同步 compact CSV、SVG、PNG 和 trace。第三张图的标准
数据与图由下面的默认模式生成：

```bash
python experiments/scripts/run_top1000_rank_bucket_diagnostics.py \
  --prediction-root <prediction_root> \
  --next-label-root <next_close_label_root> \
  --pool-path lml.bzw@ssd/data/pool_L.parquet \
  --output-dir <rank_bucket_output> \
  --variant <candidate_label> \
  --run-id <candidate_run_id>
```

第四张图先按固定的十个100名桶和100 bps收益档生成计数：

```bash
python experiments/scripts/run_top1000_rank_bucket_diagnostics.py \
  --prediction-root <prediction_root> \
  --next-label-root <next_close_label_root> \
  --pool-path lml.bzw@ssd/data/pool_L.parquet \
  --output-dir <return_histogram_output> \
  --variant <candidate_label> \
  --run-id <candidate_run_id> \
  --top1000-bucket-return-histogram-only \
  --histogram-bin-width-bps 100
```

同步 `top1000_score_bucket_return_100bps_counts.csv` 后，使用固定画图脚本复画：

```bash
python experiments/scripts/plot_top1000_score_bucket_return_histogram.py \
  --histogram-csv <return_histogram_output>/top1000_score_bucket_return_100bps_counts.csv \
  --output-dir <return_histogram_output> \
  --variant <candidate_label>
```

第四张图的正式显示契约固定为单 panel、`x=[-3000, 3000] bps`、对数
`y=[10^2, 3×10^5]`、十条 Rank 1–100 至 Rank 901–1000 曲线。探索性改 bin 或坐标范围时必须换
产物名，不覆盖正式验收图。完成条件是四张 SVG、各自 plot-data/summary CSV 和 trace 均存在，并确认
candidate/incumbent 的月份、pool、label 和费用口径一致。当前可执行 Job 示例是
`experiments/jobs/top1000_rank_bucket_diag_auction_multiden_2022_2025_v1_job.yaml` 和
`experiments/jobs/top1000_score_bucket_return_histogram_auction_multiden_2022_2025_v1_job.yaml`。

累和图下方面板必须显式选择参考口径：`market` 是全 A 股市场平均，`pool_l` 是每条模型的
累计净收益减去同一数据口径下 `pool_L` 的累计净收益。模型与 `pool_L` 各自的手续费都计入；
trace 会记录所选模式、列名和定义。上方面板固定绘制 market、`pool_L` 和模型；
`pool_l` 模式的下方面板不画 `pool_L=0` 的橙色零信息线，`market` 模式则保留 pool 相对 market 的线。

主图只解释两件事：universe short Rank IC 与 `pool_L` next internal excess。累计图默认 Top100；
容量模式必须读 capacity acceptance daily summary，不能复用 Top100 等权收益。

## 8. Capacity audit 与 acceptance

链路：

```text
predictions
  -> osf-audit-capacity
  -> capacity_audit_selected.csv
  -> osf-analyze-capacity-acceptance
  -> capacity-weighted next-close return
```

Capacity audit 只构造组合和报告 fill/depth/participation，不计算收益。`target_notional` 是单个
`date × decision_time` 的目标资金；总资金按 N 个执行切片时，先除以 N。

```toml
[run]
kind = "capacity_audit"

[capacity_audit]
predictions = ["/mnt/output/opening_strength_fit/runs/<category>/<source_run_id>"]
pool = ["L"]
target_notional = 50000000
capacity_notional_col = "turnover_diff_30t"
max_participation_rate = 0.10
max_symbol_weight = 0.01
```

收益验收：

```toml
[run]
kind = "capacity_acceptance"

[capacity_acceptance]
selected_input = ["/mnt/output/opening_strength_fit/runs/analyses/capacity-audit/<capacity_run>/capacity_audit_selected.csv"]
label_input = ["/mnt/output/opening_strength_fit/cache/<next_close_cache>"]
label_col = "alpha_return_next_close"
fee_bps = 8
capacity_total_notional = 1000000000
```

## 9. Exposure audit

如果 prediction 不含市值/行业字段，先构建 keyed input：

```bash
osf-build-exposure-input \
  --predictions <prediction_root> \
  --pool L \
  --output <exposure_input.parquet>
```

再运行：

```bash
osf-audit-exposure \
  --predictions <predictions.parquet> \
  --exposure-input <exposure_input.parquet> \
  --output-dir <output_dir> \
  --pool L --top-n 100
```

Exposure audit 解释给定 TopN/组合的画像，不替代 capacity portfolio construction。

## 10. Realistic acceptance replay

该入口重放已经完成排序与 per-decision capacity allocation 的 child orders，并叠加执行约束：

```bash
osf-analyze-realistic-acceptance \
  --selected-input <capacity_audit_selected.csv> \
  --execution-input <execution_context.parquet> \
  --label-input <next_close_label_root> \
  --output-dir <output_dir> \
  --run-id <run_id> \
  --variant <label> \
  --capacity-total-notional 1000000000 \
  --fee-bps 8 \
  --max-daily-symbol-weight 0.005 \
  --min-child-notional 10000 \
  --round-lot-shares 100 \
  --price-col capacity_price \
  --status-col status \
  --tradable-status T0 \
  --tradable-status TRADE \
  --spread-bps-col spread_bps \
  --max-spread-bps 50 \
  --ask-depth-notional-col ask_depth_notional \
  --max-ask-depth-participation-rate 0.25
```

如上下文字段不在 prediction 中，先运行：

```bash
osf-extract-execution-context --config experiments/runs/<context_run_id>.toml
osf-ask-level-attribution --config experiments/runs/<attribution_run_id>.toml
```

当前实现是 selected-order replay：不会用更低排名股票 refill，也不会把每个 decision point 的成交额
合并成同日真实预算。读取结果时必须同时查看 trace 的 `modeling_note` 和 context 列完整性。

## 11. 故障处理

| 症状 | 处理 |
| --- | --- |
| `kubectl` 无 current-context | 始终用 `hfcli kubectl --cluster research ...` |
| `field is immutable` | 删除同名 Job 后重新 apply |
| 容器找不到新 config | TOML-only 做 overlay；代码/依赖变化做 full build，再 render |
| cache 只有 `.tmp`/lock/heartbeat | 等待 final parquet、manifest 与 ready marker |
| run `completed` 但 metrics missing | 同步 `--all`，检查 shard `_SUCCESS`，再 audit |
| GPU `no kernel image` | 使用与集群 GPU compute capability 匹配的固定 Torch/CUDA 镜像 |
| PVC API 被 RBAC 拒绝 | 从 manifest 的 claimName 和容器内 `/mnt/output` 检查 |
| realistic replay fill 异常 | 检查 execution context join、status/spread/depth 列和 trace |
| experiment audit 报未知 status | 改成五个合法状态之一，不扩展同义词 |
