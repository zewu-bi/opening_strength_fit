# opening_strength_fit Runbook

日常实验闭环：检查环境和 ClickHouse 数据，做极小本地 smoke，提交 K8s 训练，拉回 metrics/predictions，回测，分析，归档。

## 1. 环境准备

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

日常进入项目：

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate

set -a
. ./.env
set +a
```

只读预检：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

预检用于发现 config、Job YAML、metrics 和文档索引是否不同步。

## 2. 数据检查

先看 ClickHouse `stock.tick` 是否可读、schema 是否符合预期：

```bash
python scripts/probe_clickhouse_data.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

再看目标实验窗口的数据覆盖。这个命令只读聚合信息，不拉原始 tick：

```bash
python scripts/probe_clickhouse_data.py \
  --start-date 2021-01-01 \
  --end-date 2022-01-31 \
  --opening-window \
  --a-share-only \
  --year-layout
```

最后用多股票小窗口检查 tick 标准化、feature/label 构造、label 分布、决策点可交易性和横截面分组。只检查时不用保存；要跑本地 smoke 时加 `--labeled-output`：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

默认不要在本地为一年或多月窗口运行 `prepare_research_dataset.py`。ClickHouse 里是原始 tick，feature/label 需要 Python 计算，长窗口本地准备会很慢且占空间。正式长窗口训练直接用 `[data].source = "clickhouse"` 的 K8s Job；长窗口 label audit / rule baseline 也应放到集群或专门的小结果 Job，只把 CSV/metrics 拉回本地。

## 3. 本地 Smoke

本地 smoke 使用第 2 节保存的 multi-symbol labeled parquet，只确认训练、预测、评估和结果落盘能跑通。小样本结果不作研究结论。

训练 smoke：

```bash
python scripts/run_experiment.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --top-n 2 \
  --output-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke
```

本地训练 smoke 仍用 CPU config；GPU 是否可用通过第 5 节镜像 smoke 和 research 集群 Job 验证。

`run_experiment.py` 会训练模型、生成 `predictions.parquet`、写出 `metrics_by_year.csv`，并按 config 中的 evaluation 设置评估一次。这里用 `--top-n 2` 覆盖 config 的 `top_n=20`，避免 smoke 中每个横截面只有 5 个 symbol 时 top-score 变成全选。

查看 smoke metrics：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke
```

`summarize_opening_results.py` 只读取 `metrics_by_year.csv` 做摘要，不重算预测；本地 smoke 和后续集群拉回的年度 metrics 都用这一条查看。

正式 one-year / full-window opening 实验不要在本地准备 labeled dataset。使用 K8s Job 直接读 ClickHouse 原始 tick，在集群内计算 feature/label、训练和评估，然后只拉回 `metrics_by_year.csv`、`metrics_by_month.csv`、`predictions.parquet` 或开盘 replay 所需结果。

## 4. 新建实验

复制最接近的 config：

```text
lgbm delay1: experiments/runs/lgbm_opening_1y_next_month_delay1.toml
lgbm strong delay1: experiments/runs/lgbm_opening_1y_next_month_strong_delay1.toml
lgbm delay2: experiments/runs/lgbm_opening_1y_next_month_delay2.toml
lgbm strong delay2: experiments/runs/lgbm_opening_1y_next_month_strong_delay2.toml
```

至少修改：

```text
[run].id
[run].description
[run].status
[data].source
[data].tick_path
[window].test_start_date / test_end_date 或 test_start_month / test_end_month
[model]...
[evaluation].selection_mode
[output].local_dir
[output].k8s_dir
[k8s.resources]...
```

约定：

- `run.id` 必须等于 config 文件名。
- `status` 提交前写 `queued` 或 `running`，拉回 metrics 并确认后写 `completed`。
- 集群训练默认用 `[data].source = "clickhouse"`，通过 ClickHouse 读取训练窗口原始 tick，只把结果写到 PVC。
- 本地 smoke 可传 `--input` 或设置 `[data].tick_path` 使用小的 prepared parquet/cache。
- `[output].k8s_dir` 必须在 `/mnt/output/opening_strength_fit/` 下，且一个实验一个目录。
- `evaluation.selection_mode` 第一版用 `cross_section`。
- `[labels].entry_tick_delay` 控制决策后第几个 tick 成交；已归档结果是无延迟旧口径，后续新实验按 delay1/delay2 分支比较。

LightGBM GPU 主线还需要这些配置：

```toml
[model]
name = "lightgbm"
device_type = "gpu"
max_bin = 63
gpu_use_dp = false

[cache]
enabled = true
read = true
write = true
labeled_path = "/mnt/output/opening_strength_fit/cache/<delay>_labeled.parquet"

[k8s.node_selector]
mem_per_gpu_tier = "high"

[k8s.resources]
gpu_limit = "1"
```

说明：

- `device_type = "gpu"` 要求镜像里的 LightGBM 是 GPU 编译版。
- `max_bin = 63` 是 LightGBM GPU 常用设置，先作为主线默认。
- 普通 universe 和 strong candidate 可以共享同一个 delay 的 `labeled_path`；strong 过滤在 cache 读取后再做。
- `mem_per_gpu_tier = "high"` 会优先落到 A100/H100 这类适合单卡任务的节点。
- 渲染脚本会把 `gpu_limit` 同时写入 `requests` 和 `limits`，并加上 `has_gpu=true:NoSchedule` toleration。

## 5. Build 镜像

修改训练代码、config 或依赖后：

```text
改代码/config -> build 新 TAG -> push -> render Job YAML -> 确认 image -> apply
```

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-gpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

当前 Dockerfile 会从源码编译 GPU 版 LightGBM：

```text
pip install --no-binary=lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm
```

本地无 GPU 时可以用下面的 smoke 确认 GPU trainer 已编进镜像；输出里应出现
`This is the GPU trainer!!`，随后报 `No OpenCL device found` 属于本机无 GPU 的正常结果：

```bash
docker run --rm -i registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} python - <<'PY'
import numpy as np
from lightgbm import LGBMRegressor

x = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.0, 1.0, 2.0, 3.0])
try:
    LGBMRegressor(n_estimators=1, device_type="gpu", max_bin=63, verbosity=1).fit(x, y)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
PY
```

换了 `TAG` 后必须重新 render Job YAML。

## 6. 生成 Job YAML

单个 opening Job：

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/lgbm_opening_1y_next_month_delay1.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

输出路径：

```text
experiments/jobs/<run_id>_job.yaml
experiments/jobs/<run_id>_reader_job.yaml
```

确认镜像：

```bash
rg -n "image:" experiments/jobs/lgbm_opening_1y_next_month_delay1_job.yaml
```

确认 GPU YAML：

```bash
rg -n "nodeSelector|tolerations|nvidia.com/gpu|image:" experiments/jobs/lgbm_opening_1y_next_month_delay1_job.yaml
hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/lgbm_opening_1y_next_month_delay1_job.yaml
```

## 7. 提交和查看 Job

```bash
hfcli kubectl --cluster research delete job opening-strength-lgbm-opening-1y-next-month-delay1 --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/lgbm_opening_1y_next_month_delay1_job.yaml
hfcli kubectl --cluster research get jobs,pods -n bizewu -o wide
hfcli kubectl --cluster research logs -f job/opening-strength-lgbm-opening-1y-next-month-delay1 -n bizewu
```

reader Job：

```bash
hfcli kubectl --cluster research delete job opening-strength-read-lgbm-opening-1y-next-month-delay1 --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/lgbm_opening_1y_next_month_delay1_reader_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-read-lgbm-opening-1y-next-month-delay1 -n bizewu --timeout=300s
```

确认 GPU request 和节点：

```bash
hfcli kubectl --cluster research -n bizewu get pods -o go-template='{{range .items}}{{.metadata.name}}{{"\t"}}{{.spec.nodeName}}{{"\t"}}{{index (index .spec.containers 0).resources.requests "nvidia.com/gpu"}}{{"\n"}}{{end}}' | rg 'opening-strength'
```

排查要点：

- `field is immutable`: 删除同名 Job 后重新 apply。
- `FileNotFoundError: experiments/runs/*.toml`: 镜像里没有新 config，重新 build/push。
- `python: can't open file scripts/run_experiment.py`: Job 使用的镜像不是当前代码镜像，重新 build/push/render。
- `GPU Tree Learner was not enabled in this build`: 镜像里的 LightGBM 不是 GPU 编译版，检查 Dockerfile 是否用了 `USE_GPU=ON` 并重新 build/push。
- `No OpenCL device found`: 本地无 GPU smoke 会出现；如果在集群 GPU pod 里出现，检查 pod 是否申请到 `nvidia.com/gpu: "1"`、是否调度到 GPU 节点，以及节点驱动/OpenCL runtime。
- `No tick data path supplied`: 本地/path 模式没填 `[data].tick_path`，或本地 smoke 忘了传 `--input`。
- `missing ClickHouse credentials`: 集群 Job 没有 `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` 环境变量；配置 `[k8s].clickhouse_secret` 或检查平台注入。
- `input path does not exist`: 容器看不到该路径，检查 PVC、`mount_path` 和 `[data].tick_path`。

## 8. 拉回 Metrics

重复传 `--config` 可一次拉一个或多个实验：

```bash
python scripts/pull_k8s_metrics.py \
  --config experiments/runs/lgbm_opening_1y_next_month_delay1.toml
```

拉回文件写到：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
```

`output/k8s/metrics/` 是本地原始产物；`experiments/results/metrics/` 是收尾时归档的轻量证据。

## 9. 拉回 Predictions 和回测

拉回预测并合并：

```bash
python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/lgbm_opening_1y_next_month_delay1.toml \
  --output-dir output/predictions/lgbm_opening_1y_next_month_delay1
```

开盘短周期回测使用 predictions 中的 `decision_target_timestamp` 和 `label`；如果训练配置设置了
`entry_tick_delay`，这里的 `label` 已经是延迟成交后的收益。后续主线只继续 LightGBM 普通 universe 和 LightGBM strong。
默认每天本金从 1.0 开始，在 `09:30/09:32/09:34/09:36/09:38` 五个非重叠两分钟 cycle 上按预测分选
Top20；`--max-symbol-trades-per-day 1` 会阻止同一股票在同一天被重复选中，没选满的资金留作现金。

```bash
python scripts/run_opening_intraday_backtest.py \
  --run lgbm_delay1=output/predictions/lgbm_opening_1y_next_month_delay1/predictions_all.parquet \
  --run lgbm_strong_delay1=output/predictions/lgbm_opening_1y_next_month_strong_delay1/predictions_all.parquet \
  --fee-bps 5 \
  --slippage-bps 5 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_lgbm_delay1_constrained
```

新训练产物会把执行约束上下文列写进 predictions。拿到新 prediction 后，可以打开更严格约束：

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/predictions/gbm_opening_1y_next_month_delay1/predictions_all.parquet \
  --run gbm_strong=output/predictions/gbm_opening_1y_next_month_strong_delay1/predictions_all.parquet \
  --fee-bps 5 \
  --slippage-bps 5 \
  --tradable-status T0 \
  --tradable-status 20 \
  --tradable-status TRADE \
  --max-spread-bps 100 \
  --min-limit-up-room-bps 5 \
  --min-capacity-notional 100000 \
  --capital-per-cycle 10000000 \
  --max-participation-rate 0.05 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_1y_next_month_delay1_constrained
```

旧归档 predictions 没有 `status`、`spread_bps`、`turnover_diff_30t` 等上下文列；这些文件只能用于
成本、滑点和同股重复交易的复算。

## 10. 分析 Metrics

单个实验：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

比较已归档 full-window 实验：

```bash
python scripts/compare_opening_results.py
```

指定实验：

```bash
python scripts/compare_opening_results.py \
  --run gbm=output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

重点看：

```text
cross_section IC: group_rank_ic_mean
cross_section IC IR: group_rank_ic_ir
daily_rank_ic_mean
model_test_r2
top_score_mean_return
```

当前主配置的 `ic_mode` 是 `cross_section`。

## 11. 分析开盘短周期回测

重点看：

```text
mean_cycle_return_bps
cycle_win_rate
mean_day_final_return_bps
positive_day_rate
compounded_month_return
candidate_count
eligible_count
cash_weight
```

逐日曲线在 `output/reports/opening_intraday_top20_1y_next_month/daily_curves/`。

## 12. 收尾：记录和审计

一轮实验包括：训练完成、reader 合并完成、metrics 拉回、需要的 predictions/opening replay 分析完成，并且相关 config 的 `status` 已更新为 `completed`。

```bash
python scripts/record_experiment.py \
  --config experiments/runs/lgbm_opening_1y_next_month_delay1.toml
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

提交前确认：

- 新实验 config、Job YAML、metrics 记录和 opening replay 记录都落在对应目录。
- `docs/experiment_log.md` 写入实验状态或结论。
- `output/` 只保留本地运行产物，不提交大 parquet / pkl。
- 新 Job YAML 使用刚 build/push 的镜像 tag。
