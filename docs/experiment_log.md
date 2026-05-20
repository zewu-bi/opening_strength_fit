# Experiment Log

| run | status | notes |
| --- | --- | --- |
| `ridge_opening_full` | queued | 通用 opening Ridge full config。工作流已对齐 `xy_fit`：full config 本地 smoke、training/reader Job、`metrics_by_year.csv` 拉回、predictions 拉回、backtest 和 `record_experiment.py` 归档。 |
| `ridge_opening_1y_next_month` | running | Baseline A：全 A 股 universe，2021 训练、2022-01 测试，Ridge 横截面排序。 |
| `ridge_opening_1y_next_month_strong` | running | Baseline B：除开盘强势候选池过滤外，参数与 `ridge_opening_1y_next_month` 一致，用于检验强势/活跃样本内排序是否更稳定。 |
| `gbm_opening_1y_next_month` | running | 与 Baseline A 同口径，用 GBM 检验非线性模型是否优于 Ridge。 |
| `ridge_opening_1m_3d` | running | 小窗 Baseline A：2021-12 训练、2022-01-04 至 2022-01-06 测试，参数对齐 `ridge_opening_1y_next_month`。 |
| `ridge_opening_1m_3d_strong` | running | 小窗 Baseline B：2021-12 训练、2022-01-04 至 2022-01-06 测试，参数对齐 strong Ridge。 |
| `gbm_opening_1m_3d` | running | 小窗 GBM：2021-12 训练、2022-01-04 至 2022-01-06 测试，参数对齐 `gbm_opening_1y_next_month`。 |
