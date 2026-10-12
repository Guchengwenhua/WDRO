# 纯 RL 规模与机制扩展（2026-09-22）

详细方案见 [docs/THEORY_PROTOCOL.md](docs/THEORY_PROTOCOL.md)，冻结参数见 [protocol.json](protocol.json)。

本目录延续已验收的 `rl_budget_scaling_20260921`，只复制已验证的纯 RL 父 checkpoint；旧目录不修改。所有新增产物、缓存和临时文件均在本目录。

首批队列53项：14种延长/调参方案×3种子，3种模型容量×3种子，另有2个新种子的父模型预训练。验证集按预注册规则晋级，最终训练任务总数58–74项；E50和最多2个晋级变体到49,152步、共5个种子。容量实验独立使用201/202/203，若晋级则候选与BASE都到32,768步。

实际新增预算范围712,704–1,105,920次更新；采样轨迹50,331,648–75,497,472条，另有同数网络贪心baseline评分。复用的历史预训练不计新增成本，但比较模型总训练预算时必须计入。

主要产物：

- `artifacts/tests/preflight.json`：工程启动门槛，不能当作科学结果。
- `artifacts/lifecycle.json`：整体阶段与退出状态。
- `artifacts/operations/<stage>/queue.json`：资源、运行身份、排队和失败记录。
- `artifacts/runs/<id>/`：日志、逐步预算、验证曲线、latest/best及端点checkpoint。
- `artifacts/promotion.json`：只依据验证数据做出的不可变晋级决定。
- `artifacts/selection.json`：全部训练完成后的全局模型冻结，早于首次测试读取。
- `artifacts/eval/`：新测试集的逐net拓扑、长度、摘要。
- `artifacts/report/REPORT.md`：结束后自动报告；统计、预算、数值复算和图同时输出。

查询当前进度（无需连接GPU）：

```bash
bash /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/run_env.sh /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/status.py
```

tmux连接及最新预计时间见 [HANDOFF.md](HANDOFF.md)。主进程从 `launch.sh` 启动；恢复前先检查该tmux会话和现有进程，锁与已完成文件会阻止重复训练。

磁盘保留20GiB余量，目录上限28GiB；一旦不足，自身队列暂停等待。显存不足也保存检查点并等待。调度器不会终止其他任务，不改变冻结batch来塞入资源。日常无需人工逐步监控；机器重启、持续资源不足或重试耗尽时，需要根据生命周期/失败文件恢复。

`artifacts/tests/e2e_superseded_queue_order` 是为调整队列顺序而主动停止的早期微型工程试跑，保留诊断记录，不计入任何正式预算或结果。
