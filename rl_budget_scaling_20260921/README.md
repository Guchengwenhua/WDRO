# 纯 RL 训练规模实验，2026-09-21

[详细理论与预注册协议](docs/THEORY_PROTOCOL.md) · [机器协议](protocol.json) · [依赖队列](queue.json)

本实验检验训练不足、数据覆盖、预训练顺序和后训练规模。degree30，三个匹配训练种子，六条路线、18个训练任务，另18个测试任务。仅用学习策略自身拓扑的线长，无外部确定性求解器、最优标签或外部预训练。保留既有Actor架构、batch64和原始span线长语义。

所有源码、临时文件、输入副本、日志与结果均位于本目录；旧实验只读。正式结论以 `artifacts/report/REPORT.md` 和通过核验的 `complete.json` 为准。启动检查通过和正在训练都不表示科学实验已经完成。

## 后台运行

```bash
cd /remote-home/Guilin/DRO/Codex/rl_budget_scaling_20260921
tmux -S "$PWD/artifacts/operations/tmux.sock" new-session -d -s rl_budget_20260921 \
  "bash '$PWD/launch.sh' > '$PWD/artifacts/operations/scheduler.log' 2>&1"
```

查看状态：

```bash
python3 /remote-home/Guilin/DRO/Codex/rl_budget_scaling_20260921/status.py
tmux -S /remote-home/Guilin/DRO/Codex/rl_budget_scaling_20260921/artifacts/operations/tmux.sock attach -t rl_budget_20260921
```

训练进度在 `artifacts/runs/*/heartbeat.json`，完整队列在 `artifacts/operations/queue.json`，每任务控制台在 `artifacts/operations/*_s*.log`。

自动过程：资源准入 → 六条训练路线依赖串接 → 所有模型冻结 → 新测试统一推理 → 独立线长重算、统计、PNG/SVG/PDF曲线和中文结论。失败会写 `FAILED.json`/队列退出记录；容量不足等待，自己的任务可安全暂停并恢复。调度器重启会核验PID身份与锁，接管仍在运行的worker；不会重复启动。

资源策略：每GPU最多1任务，最多8卡；每worker3 CPU线程。三次连续GPU占用≤35%且空余≥8GiB接入，磁盘至少20GiB；只使用空闲容量，不干扰其他用户任务。无需人为填满显存，batch保持固定以免污染规模对照。

`artifacts/tests/` 内的短跑、恢复检查及端到端小型夹具均为诊断，不能与正式训练/新测试合并。夹具使用独立坐标命名空间。
