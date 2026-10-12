# Gaussian ES 扩规模实验

- [理论与实验方案](docs/理论与实验方案.md)：目标、预算、预训练/后训练、调参、统计与验收。
- `protocol.json`：机器可读冻结配置。
- `artifacts/status.json`：运行状态；`artifacts/runs/`：各任务断点和日志。
- `HANDOFF.md`：tmux、恢复方式、实测速率及预计耗时。

独立分支，复用 v3.1 的真实 regret、Hungarian-L1、合法化与 REST；不修改已有研究结果。
