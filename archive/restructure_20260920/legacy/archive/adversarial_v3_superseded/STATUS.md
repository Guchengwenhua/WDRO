# 对抗生成框架 v3：运行状态

更新时间：2026-09-12 08:20:54 UTC。
当前：**A 固定模型验证 / failed**；A 完成 **11/30**，B 完成 **0/18**；本版本活动任务 0。
A：活动 0，待运行 19，失败待处理 0；B：活动 0，待运行 18，失败待处理 0。

累计真实greedy评分完成 4,269,056，GeoSteiner新求解完成 4,132,481；sampled主更新评分 0。
最近 15.0 秒：真实greedy评分 0.0 net/s，GeoSteiner新求解 0.0 net/s。磁盘可用 2635.2 GiB。

| GPU | 可用显存 GiB | 利用率 % | 本版本活动任务 |
|---|---:|---:|---:|
| 0 | 9.16 | 97 | 0 |
| 1 | 9.16 | 98 | 0 |
| 2 | 47.50 | 0 | 0 |
| 3 | 47.50 | 0 | 0 |
| 4 | 37.08 | 0 | 0 |
| 5 | 35.08 | 0 | 0 |
| 6 | 9.16 | 98 | 0 |
| 7 | 9.16 | 99 | 0 |

| 活动任务 | GPU | 当前环节 / 已提交进度 | 本进程时长 | 真实评分完成 | net/s | 心跳年龄 |
|---|---:|---|---:|---:|---:|---:|
| 暂无活动任务 | — | — | — | — | — | — |

当前异常：
- pipeline 阶段 fixed 失败，退出码 None

[冻结协议](protocol.json) · [实验计划](EXPERIMENT_PLAN.md) · [操作手册](RUNBOOK.md) · [pipeline状态](experiments/pipeline/state.json) · [原始采样](experiments/monitor/samples.jsonl) · [监控代码](adv_v3/monitor.py)

本页只读采样任务状态、PID、心跳和账目；每15秒保存采样，每60秒刷新正文。完成数要求任务complete.json为pass且对应进程退出；最终交付仍以report审计及pipeline完成文件为准。GPU任务数来自活跃任务的明确physical_gpu_index，不从进程环境推断。
