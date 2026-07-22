# Documentation archive

旧版逐次实验流水账已从当前工作树移除，避免与 `docs/experiment_log.md`、run TOML 和 evidence 重复。
当前结论只维护在 project brief 和 experiment log；实验参数与实际执行仍由 `experiments/runs/` 和
`experiments/jobs/` 保留。

需要审计旧叙述时，可从完整 Git 历史读取：

```bash
git log --all -- docs/archive
git show <revision>:docs/archive/<historical-file>
```
