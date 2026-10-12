# 当前状态

**已按用户修订要求暂停。** 2026-09-15，用户明确排除 MST/几何下界等非学习型参考算法，后续工作转入 [rl_only_20260915](../../rl_only_20260915/README.md)。本分支没有完成科学验收，不应继续自动启动。

暂停时正式更新数：ERM-LB 196、Random-WDRO 47、ES-WDRO 70、Critic-WDRO 69。调度器及所有记录 worker/训练进程已退出，所有锁释放，10 个已有检查点均通过读取/有限值/绑定核验；冻结源码未改。证据：`artifacts/operations/pause_verified.json`。下方为暂停前的历史启动和估算信息。

2026-09-15 20:33:37 UTC：理论与实现完成；服务器 tmux 后台队列已启动，先执行 GPU smoke，再执行正式训练。

已通过：7 项几何与依赖排除测试、8 项调度测试、6 个方法的 CPU 端到端 smoke、CPU 自动报告独立重算、带故障注入的中断恢复一致性审计（354 张量、92 数组）。

正式科学验收仍待完成。CPU smoke 不作为训练效果证据。

2026-09-15 20:37 UTC 交接更新：6 个 GPU smoke 全部通过；正式任务正在 GPU 2、3、4、5 并行运行（Random-WDRO、ERM-LB、ES-WDRO、Critic-WDRO），队列无失败。已核实调度器与 worker 的 PID 启动身份、训练心跳和 latest.pt。

初步全批量实测：ERM 约 0.46 秒/更新，Random-WDRO 约 1.73 秒/更新。按四张相近可用 GPU，18 模型及评测预计还需 **4–6 小时**；这是早期吞吐量估算，负载与可用卡数变化会影响完成时间。动态进度与本次交接证据见 `artifacts/operations/queue.json`、`handoff.json`。预计后续可用容量增加时，队列最多使用 8 张满足余量和利用率条件的 GPU，每 worker 3 个 CPU 线程。

最新资源、任务与 ETA 以 `artifacts/operations/queue.json` 和 `resource_latest.json` 为准。训练报告在全部 18 个正式 GPU 模型完成后自动生成。

会话 `oracle_free_20260915`，独立 socket `artifacts/operations/oracle_free.sock`。断开 SSH 不影响服务器任务；按 OPERATIONS.md 中的查看命令核实会话和心跳，不重复启动队列。
