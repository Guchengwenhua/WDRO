# 运行、进度和恢复

## 后台流程

独立 tmux socket：`artifacts/operations/tmux.sock`；会话：`papers_wdro_20260915`。

流程为：CPU 数据准备/复验 → 冻结源码与数据 → 各 family 的 GPU smoke → 对应 ERM/WDRO 三seed正式训练 → 独立统计与验收。每 family 必须先通过完整batch64的GPU smoke，才能进入正式任务。

查看实时状态及估计：

```bash
cd /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915
bash run_env.sh eta.py
cat artifacts/operations/queue_state.json
```

查看会话和日志：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915/artifacts/operations/tmux.sock attach -t papers_wdro_20260915
```

按 `Ctrl-b` 再按 `d` 可离开会话，任务继续运行。总日志在 `artifacts/operations/pipeline.log`；调度、准入、每次失败与退出凭证在同一 operations 目录。

## 完成条件

- 4 个 GPU smoke 为运行预检；24 个 `scope=degree10_pilot` 才是正式训练。
- 各正式任务完成2048updates并保存验证选出的best、每net测试结果及平均预算LP。
- `artifacts/report/complete.json` 存在且独立审计通过。
- `artifacts/operations/exit.json` 为 `status=complete`，任务进程退出、锁释放。

如果 validation 最优仍是初始化，报告会原样记录；若 WDRO 没有优于 ERM，不能将工程完成解释为方法获胜。

## 资源策略

扫描全部8张GPU，以最近3次实际算力和显存决定准入，不按历史GPU编号固定排除卡。初始最低空闲显存=4GiB任务预算+2GiB余量；正式任务预算还取GPU smoke实际peak reserved+1GiB与4GiB的较大值，再加2GiB余量。连续3次利用率平均不超过80%、至少一次不超过70%、任一次不超过95%。每卡最多一个本分支worker，CPU最多16核，每任务2核，另受实际affinity约束。

如全部卡持续满算力，则状态为等待资源；空闲显存本身不触发向满负载卡继续塞任务。只对本分支的进程做容量退出或超时处理，绝不停止其他用户/其他实验的进程。

训练每32步和容量退出时保存完整状态。CUDA OOM保留最后完整checkpoint，不能保存失败forward已改变的BN/RNG并冒充完整更新。容量退出75无限退避，业务错误最多3次；失败后保留证据并阻止下游验收。

## 网络断开后的恢复

先读当前 `queue_state.json`、`heartbeat`、`exit.json` 并检查原 tmux 会话。会话尚在就直接查看，不再次启动。

如果调度器已退出且无完成凭证，可在同一路径、同一代码和协议下重新启动：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915/artifacts/operations/tmux.sock new-session -d -s papers_wdro_20260915 'bash /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915/pipeline.sh >> /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915/artifacts/operations/pipeline.log 2>&1'
```

若旧worker仍在，调度器核对PID/启动时钟/boot ID和内核flock后接管，避免重复运行。若冻结源码或协议不一致会拒绝恢复；需先查明原因，不能直接删除freeze/lock文件。

同一CPU运行的6步与2+4步恢复已有逐tensor等价性证据；跨CPU/CUDA模式被拒绝，同型号GPU间恢复保留状态但不承诺所有异构软硬件逐bit结果相同。

## ETA解释

`eta.py` 只读真实GPU预检和训练进度。计算剩余时间与等待容量时间分开；没有CUDA实测时明确标记“待CUDA实测”，不把CPU smoke或已排队任务说成GPU已完成。GPU准入数量变化时，wall-clock估计也会改变。
