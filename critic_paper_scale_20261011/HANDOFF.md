# 启动与交接

任务：原21项续到2048步，外加3项Critic校准、最多6项修正对照。科学参数和依赖在PROTOCOL.md、main/protocol.json、repaired/protocol.json及queue.json中。

```bash
cd /remote-home/Guilin/DRO/Codex/critic_paper_scale_20261011
bash launch.sh
```

tmux会话`critic-paper-scale-20261011`，socket为`artifacts/queue.sock`。

```bash
tmux -S /remote-home/Guilin/DRO/Codex/critic_paper_scale_20261011/artifacts/queue.sock list-sessions
cat /remote-home/Guilin/DRO/Codex/critic_paper_scale_20261011/artifacts/state.json
```

控制器日志`artifacts/controller.log`；逐任务日志`artifacts/<job_id>.attempt_<time>.log`；任务启动/退出、GPU接纳样本和命令在events.jsonl。每项的heartbeat记录当前步、实际GPU、当前segment吞吐与预计训练剩余时间。

- 主线：`artifacts/main/<family>_<arm>_s<seed>/`，包含resume_origin、latest、best、milestone、训练/验证日志和最终测试。
- 校准：`artifacts/calibration/s<seed>/`，标签块、Critic断点、holdout预测、results和qualified标志；只有通过者生成parent.pt。
- 修正配对：`artifacts/repaired/rest_<arm>_s<seed>/`。
- 自动结果表：RESULTS.md及artifacts/metrics.csv、contrasts.json；只纳入完成训练与测试的项。

若控制器意外退出但子进程还活跃，不重复启动；先核对state中的进程身份和心跳。需要停止时只向本控制器发送SIGTERM，它会让本队列子进程保存断点；切勿终止外部GPU任务。exit75为容量/主动暂停，非科学失败。恢复使用相同入口。

本轮无需一直保持聊天。按`/remote-home/Guilin/Skill.md`中“主动结束本轮对话；不要持续占用对话等待实验完成”的要求，正式日志确认开始推进后交接。启动进度和依据实际吞吐给出的初步估时保存在START_STATUS.json，之后以实时heartbeat为准。
