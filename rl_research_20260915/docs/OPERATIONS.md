# 持久运行与恢复

## 范围

只运行本分支51任务，不修改 `rl_only_20260915` 等其他分支的源码、协议、检查点或进程。执行器使用独立文件锁和检查点；每32步保存Actor、价值网络、攻击器、优化器、历史对手池和随机状态。

调度最多4个worker，每GPU最多新增1个。每进程PyTorch分配上限为显存8%，准入至少6GiB空余，根盘保留20GiB。持续满载卡在至少3次采样后只允许占空比25%的低优先级共享任务，计算阶段间主动等待；这不是硬件GPU配额保证。没有预占剩余显存、终止或修改其他用户任务。实际准入、GPU UUID、PID启动时间及命令均记录。

## 查看

```bash
cd /remote-home/Guilin/DRO/Codex/rl_research_20260915
tmux -S artifacts/operations/research.sock list-sessions
cat artifacts/operations/queue.json
tail -n 20 artifacts/operations/pipeline.log
cat artifacts/operations/delivery.json
```

`running`/`waiting_resources`表示工作尚未完成；`failed`或`audit_failed`表示需要审查日志；不能将其标成DONE。最后会自动运行统计审计，失败会保留非零退出码和原因。

## 恢复

先核对上述tmux、queue和每个worker的 `heartbeat.json`。PID必须同时核对 `/proc/PID/stat` 启动时间；不要只用 `kill -0` 判断。活跃队列不得重复启动。

确认本分支旧scheduler和所有worker已退出、source/protocol未变化后：

```bash
cd /remote-home/Guilin/DRO/Codex/rl_research_20260915
tmux -S artifacts/operations/research.sock new-session -d -s rl_research 'cd /remote-home/Guilin/DRO/Codex/rl_research_20260915 && exec bash pipeline.sh >> artifacts/operations/pipeline.log 2>&1'
```

有效complete任务会跳过。未完成任务从最新checkpoint恢复，每任务最多3次尝试；科学设置不会因失败改变。完整的实际调用日志保留重放开销，统计分别报告有效预算与恢复额外调用。

## 自动交付

队列完成51项后执行 `report.py`，生成 `artifacts/report/REPORT.md`、`metrics.csv`、`comparisons.csv`、`audit.json`。`pipeline.sh`随后核对worker进程退出和锁释放，写入 `artifacts/operations/delivery.json`，并更新本分支 `STATUS.md`。报告保留所有不利方法/seed/分布，CI跨零不能解释为等价。

## 耗时

运行时间受其他作业与共享占空比影响。`artifacts/operations/eta.json`给出实测估计，不是完成承诺。源代码、协议或任务预算变更时必须另建研究版本，不能沿用已完成任务的凭证。
