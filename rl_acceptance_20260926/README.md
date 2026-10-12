# 2026-09-26 实验验收与附加盘恢复

本次验收覆盖 `rl_scaleup_20260921` 的80项正式任务，以及 `rl_prepost_scaleup_20260922` 的104项扩展任务。

- 当前阶段报告：[INTERIM_REPORT.md](INTERIM_REPORT.md)。完整终验结束后生成 [FINAL_REPORT.md](FINAL_REPORT.md)。
- 首轮重算结果：`old/acceptance.json`；扩展轮已完成子集：`new/acceptance.json`。
- 完整扩展终验：`final/new/acceptance.json`；等待/执行状态：`final_watch.json`。
- 复制校验：`storage_migration.json` 和三个逐文件SHA-256清单。
- 最终存储布局：`storage_compatibility.json`；原始权重检查实测：`storage_guard_probe.json`。

完成档案和大型评价数组存放于 `/attached/remote-home1/Guilin/DRO/Codex/`，继续通过原实验路径访问。剩余训练所需父检查点及新输出检查点留在原实验根目录内，以满足冻结的外部权重检查。未修改训练源码、协议、seed、步数、选择规则或资源阈值。

训练仍由原tmux `rl-prepost` 管理。独立终验由tmux `rl-acceptance` 等待正式 `delivery.json=COMPLETE` 后执行：重新验证全部104项、输入隔离、预算、检查点、六项主检验，以及固定模型攻击器延长训练的次要比较。

```bash
cat /remote-home/Guilin/DRO/Codex/rl_acceptance_20260926/final_watch.json
tmux -S /remote-home/Guilin/DRO/Codex/rl_acceptance_20260926/acceptance.sock attach -t rl-acceptance
```

原训练的终态审计失败时，独立终验会记录 `NEEDS_REVIEW`，不会自动改参数或跳过审计。`PASS` 表示协议和数值验收通过；方法优势另按预注册统计门槛判定。
