# 运行交接

任务：critic-paper-bank-20261011，共21次训练。参数见protocol.json，模型名单见inputs.json。

启动/恢复：

```bash
cd /remote-home/Guilin/DRO/Codex/critic_paper_bank_20261011
bash launch.sh
```

tmux 会话：`critic-paper-bank-20261011`，socket：`artifacts/queue.sock`。

```bash
tmux -S /remote-home/Guilin/DRO/Codex/critic_paper_bank_20261011/artifacts/queue.sock list-sessions
cat /remote-home/Guilin/DRO/Codex/critic_paper_bank_20261011/artifacts/state.json
```

控制器日志为artifacts/controller.log，各任务日志为artifacts/<family>_<arm>_s<seed>.attempt_<timestamp>.log。任务内部有heartbeat.json、train.jsonl、validation.jsonl、latest.pt、best.pt、test_best.npz、test_final.npz、results.json、complete.json。心跳记录实际GPU和当前更新数。启动身份与连续三次GPU接纳样本在events.jsonl。

控制器最多同时运行3项，有空闲容量才启动。失败不会标记complete，会保留退出码并继续其他任务；同一恢复命令跳过完成项并恢复未完成项。旧子进程活着时控制器拒绝重复启动；应先查看state记录的PID/start_ticks/boot_id及对应心跳，不要重复开队列或终止其他人的任务。

每8步保存完整可恢复状态。训练中断可能回退最多8次更新；评估中断重新从最终训练checkpoint评估。报告只读取完整结果，并明确列出已完成数量，不把排队、启动或初始验证称为完成。

每项结束自动更新RESULTS.md和artifacts/metrics.csv、contrasts.json。初始手工刷新命令为 `bash run_env.sh summarize.py`，只汇总已有实验数据。

实际存储位置：`/attached/remote-home1/Guilin/DRO/Codex/critic_paper_bank_20261011/artifacts`，在本目录通过artifacts符号链接访问。启动前附加盘约776GB可用，根盘约15GB可用。

首次启动的进度与初步耗时估计将根据正式训练日志记录于START_STATUS.json；早期吞吐估计不代表所有方法同速，尤其K3额外执行三个教师，Buffer包括CPU物理评估。

启动记录：23ICCAD初始validation的legal布尔数组触发NumPy分位数插值类型错误；已在统计入口统一转float64，未改变训练目标。修正后对本任务控制器发送SIGTERM，三个正在运行的任务按已有机制保存检查点并以75返回，再用相同命令恢复队列。最初state中的75属于主动保存断点暂停，不是训练数值失败；原错误/暂停日志均保留。该修正不是独立工程测试或新的实验分支。
