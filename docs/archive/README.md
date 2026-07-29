# Documentation archive

旧版逐次实验流水账已从当前工作树移除，避免与 `docs/experiment_log.md`、run TOML 和 evidence 重复。
当前结论只维护在 project brief 和 experiment log；实验参数与实际执行仍由 `experiments/runs/` 和
`experiments/jobs/` 保留。

需要审计旧叙述时，可从完整 Git 历史读取：

```bash
git log --all -- docs/archive
git show <revision>:docs/archive/<historical-file>
```

2026-07-22 至 2026-07-29 的全天分钟路径、隔夜日频目标和 TCN 路线因需求理解偏差而终止；其完整
配置、实现快照和证据保存在
[`experiments/archive/full_day_temporal_2026-07-22_2026-07-29/`](../../experiments/archive/full_day_temporal_2026-07-22_2026-07-29/)。
