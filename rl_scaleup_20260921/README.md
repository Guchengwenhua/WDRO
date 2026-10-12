# RL 对抗搜索扩大训练（2026-09-21）

Sequential PPO / Best-of-K PPO，8 个新 seeds，degree 30。每阶段 2,048 updates、batch 64；含主动与简单搜索对照，共 80 项 GPU 任务。

* [详细理论与冻结方案](docs/THEORY_AND_PROTOCOL.md)
* [精确参数](protocol.json)
* [当前状态](docs/STATUS.md)
* 队列与实时 ETA：`artifacts/operations/queue.json`、`artifacts/operations/eta.json`
* 完成后的主结论：`artifacts/report/PRIMARY_REPORT.md`
* 完成后的工程交付状态：`artifacts/operations/delivery.json`

首次启动或服务器重启后的恢复命令：

```bash
bash /remote-home/Guilin/DRO/Codex/rl_scaleup_20260921/launch.sh
```

启动器会拒绝重复会话、已有活跃 worker 和源码/协议不匹配；保留 latest checkpoint 自动恢复。不要删除 source_freeze 或修改正式 protocol 来绕过检查。

查看 tmux（结束后会话自动退出，结果留在文件中）：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/rl_scaleup_20260921/artifacts/operations/scaleup.sock attach -t rl-scaleup
```

只读进度：

```bash
cat /remote-home/Guilin/DRO/Codex/rl_scaleup_20260921/artifacts/operations/eta.json
cat /remote-home/Guilin/DRO/Codex/rl_scaleup_20260921/artifacts/operations/queue.json
```

需要暂停时创建本目录 `artifacts/operations/PAUSE` 空文件。worker 在训练步边界保存退出，已进入测试的 worker 会完成本次测试；调度器保持运行并等待。删除 PAUSE 后自动恢复，暂停不消耗失败重试次数。

运行中每卡最多一个本实验 worker；本实验只向自己的子进程发送超时退出信号，不操作外部作业。
失败重试最多三次；遇到不可恢复错误时输出 NEEDS_REVIEW，并保留日志与 checkpoint。
全部模型、数据、临时文件、缓存、日志和 socket 均存于此目录。原研究代码只读复制，来源 hash 见 `artifacts/operations/origin.json`。
