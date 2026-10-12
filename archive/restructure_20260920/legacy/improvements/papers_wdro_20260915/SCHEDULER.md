# 后台实验队列

调度器仅依赖 Python 标准库和服务器 `nvidia-smi`，没有导入任何改良攻击器。

## 启动和状态

在本目录运行：

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python scheduler.py --manifest queue.json
```

由主流程启动后台 tmux 时，使用本目录下的独立 socket，并将输出写入 `artifacts/operations/scheduler.log`。
建议会话名 `papers_wdro_20260915`，socket 路径 `artifacts/operations/tmux.sock`。
完整示例需在 `queue.json` 和所引用代码全部冻结后执行：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915/artifacts/operations/tmux.sock new-session -d -s papers_wdro_20260915 'cd /remote-home/Guilin/DRO/Codex/improvements/papers_wdro_20260915 && exec /remote-home/Guilin/miniconda3/envs/py311/bin/python scheduler.py --manifest queue.json >>artifacts/operations/scheduler.log 2>&1'
```

- 实时状态：`artifacts/operations/queue_state.json`。
- GPU / CPU / 内存 / 磁盘快照：`resource_latest.json`、`resource_history.jsonl`。
- 调度器源文件冻结：`source_freeze.json`。
- 每个任务：`attempt_*.json`、`heartbeat_*.json`、`result_*.json`、`<task_id>.log`。
- 队列结束：`exit.json`；只有任务全部验收、子进程退出、后处理成功才写 `status: complete`。
- `--dry-run` 仅写资源与 `dry_run.json`，不启动 CUDA；`--once` 做一次状态恢复和资源采样后退出，已经启动的 worker 会保留。

## Manifest

```json
{
  "source_paths": ["protocol.json", "runner.py", "run_env.sh"],
  "tasks": [
    {
      "id": "example_d10_s101",
      "command": ["bash", "{root}/run_env.sh", "{root}/runner.py", "--family", "rest", "--method", "wdro", "--seed", "101", "--run-id", "example_d10_s101", "--device", "cuda"],
      "complete": "artifacts/runs/example_d10_s101/complete.json",
      "checkpoint": "artifacts/runs/example_d10_s101/latest.pt",
      "progress": "artifacts/runs/example_d10_s101/heartbeat.json",
      "depends_on": [],
      "cpu_threads": 2,
      "required_free_mib": 4096,
      "completion_checks": {"status": "complete", "scope": "degree10_pilot", "completed_updates": 2048, "protocol_sha256": "FILL_IN_FROZEN_HASH"}
    }
  ]
}
```

`command` 是参数数组，不经过 shell 字符串展开。只替换 `{root}`、`{gpu}` 和 `{cpu_threads}`。
`resume_args` 仅当 checkpoint 已存在时追加；runner 如本身自动恢复，可省略。
`CUDA_VISIBLE_DEVICES` 使用物理卡编号，因此 runner 内应使用逻辑 `cuda:0`。
`PAPERS_WDRO_CPU_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 等设置为该任务 CPU 核数，子进程使用独立 CPU affinity 与 nice 10。

完成文件需包含 `status: complete` 和 `artifacts`（相对完成文件目录的路径至 SHA-256 的非空映射）。
可选 `binding_checks` 校验完成文件中 `binding` 对象的指定字段。
后处理通过 manifest 的 `postprocess: {command: [...], complete: "..."}` 指定，验收接口相同。
测试用任务可明确设置 `verify_artifacts: false`；正式任务应保留默认 hash 校验。

## 资源和恢复规则

所有可见 GPU 每 20 秒采样，连续 3 次 GPU 利用率均不超过 95%、平均不超过 80%、至少一次不超过 70%，且最少空闲显存达到 `required_free_mib + 2048 MiB` 才准入。启动前再采样确认显存与瞬时利用率。
正式任务依赖 `gpu_smoke_<family>` 时，`required_free_mib` 自动提高至 `max(4096, ceil(已验收 smoke 的 cuda_max_reserved_mib + 1024))`（如 task 原需求更大也会保留），随后再加上述 2048 MiB 余量。
调度器重新核对 `metrics.json` 的已验收 SHA-256；缺少有效显存峰值或文件变化会阻止该任务准入。
实际准入需求、smoke 峰值和指标摘要记录在 attempt / eligible 元数据，冻结的 queue manifest 不被修改。
本分支每卡最多 1 个任务；默认最多 8 个任务和 16 个 CPU 核，实际还受 affinity / cgroup quota 限制。
GPU 空闲显存不代表空闲算力；不终止、挂起或改变其他用户的任务。

wrapper 与 runner 继承同一 task / GPU flock 文件描述符。调度器重启通过持久化 attempt、PID、`/proc` 启动时钟、Linux boot ID 与内核锁识别存活进程。
调度器收到 SIGTERM 会保留 worker；重新运行同一命令会恢复监督，避免重复启动。

本任务运行后如空闲显存连续 2 次低于 1 GiB，wrapper 会向**自己的**子进程组发 SIGTERM，请 runner 保存 checkpoint 后退出；120 秒宽限后仍未退出才对自己的进程组发 SIGKILL。
容量退避和 runner 主动退出 75 会等待 120 秒后无限重试，不计业务失败。
业务失败最多 3 次，随后保留日志并写失败报告；依赖失败的任务不运行。
默认 runner 进度文件 30 分钟不更新或单任务超过 24 小时触发自身终止保护，任务可按合理工作量覆盖超时。
CPU 繁忙时会受实际操作系统调度影响，ETA 应使用 runner 实测更新时长；等待其他用户释放 GPU 算力的时间无法预先保证。

## 验证

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python -m unittest discover -s tests -p test_scheduler.py -v
```

测试涵盖多采样准入、PID 复用拒绝、artifact 损坏拒绝、依赖环、checkpoint 恢复参数、继承内核锁、容量退出重试和调度器重启后不重复启动。
集成测试使用假的 `nvidia-smi` 和 CPU stand-in 子进程，不消耗 GPU，也不构成 GPU 实验验收。

## 实时 ETA 和后台查看

当前论文 runner 自动检测 `latest.pt`，因此其任务应省略 `resume_args`；CLI 只传 `--device cuda`，不要传不存在的 `--resume` 或 `--physical-gpu-index`。
当前 runner 完成文件把 `protocol_sha256`、`frozen_sha256` 放在顶层，应通过 `completion_checks` 检查，不用 `binding_checks`。
其 heartbeat 的 `time`、`update`、`total_updates` schema 与调度器按文件更新时间监控的方式兼容。

在分支目录查看动态 ETA：

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python eta.py
watch -n 30 /remote-home/Guilin/miniconda3/envs/py311/bin/python eta.py
```

JSON 供脚本读取，或明确保存当前快照：

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python eta.py --json
/remote-home/Guilin/miniconda3/envs/py311/bin/python eta.py --output artifacts/operations/eta_latest.json
```

`eta.py` 默认只读 `queue_state.json`、每个 run 的 heartbeat、progress、validation、actual_calls 和 metrics。
按 `family × method` 分别使用真实 CUDA 计时；优先正式 batch 的最近更新，数据不足时使用 GPU smoke 并明确标为 batch 外推。
CPU smoke 不参与 GPU 标定。训练、初始化及后续验证、best/initial 两次 test、33 候选攻评和未直接计时的 LP 余量分别列出。
输出单 GPU / 当前并行度 / 最多 8 worker 的持续算力情景，GPU 排队等待始终标为未知。
这些时间区间是工程估算，不是置信区间，不能保证其他用户未来的 GPU/CPU 占用。
