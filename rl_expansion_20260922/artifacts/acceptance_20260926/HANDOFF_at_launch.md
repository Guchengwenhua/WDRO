# 扩量实验交接

**正式运行中。** 2026-09-22 04:45 UTC 已在服务器tmux启动；04:47 UTC核验8张GPU均有实际训练更新，8个wrapper及trainer身份都存活、无失败、24个CPU线程分配无重叠，初始完整checkpoint均存在。逐项凭证见 `artifacts/operations/handoff_runtime.json`。

八卡热身实测每步约1.11–1.15秒。预计从交接时起：首批筛选还需 **24–30小时**，整轮（含自动晋级长训练和测试）还需 **36–72小时**。晋级数量和共享服务器资源变化会影响耗时，资源等待不设确定上限。当前目录约4GiB，全盘约80GiB空余；正式运行继续保留20GiB余量、目录上限28GiB。

首批队列共53项：51个筛选运行加2个提前训练的新种子父模型。整个方案58–74个训练运行，新增约71万–111万次更新、约5033万–7550万条采样轨迹，另有同数贪心baseline评分。最长到49,152步，晋级后的延长方案共有5个种子；架构对照使用另一组3种子。

工程启动门槛已通过：27项测试、CPU/GPU精确恢复、4种规模GPU测速、所有调参分支的小样本运行；最终独立微型流程完成18个训练、34个测试、4,224条测试线长独立复算及自动报告。此处是工程证据，**正式科学结果尚未产生**。

运行目录：`/remote-home/Guilin/DRO/Codex/rl_expansion_20260922`。

tmux socket：`/remote-home/Guilin/DRO/Codex/rl_expansion_20260922/artifacts/operations/expansion.sock`。

会话名：`rl_expansion_20260922`。管线PID在启动时为948449，以 `artifacts/lifecycle.json` 中的PID/start_ticks/boot_id及实际 `/proc` 为准。

```bash
tmux -S /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/artifacts/operations/expansion.sock attach -t rl_expansion_20260922
```

```bash
bash /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/run_env.sh /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/status.py
```

只有确认该会话及已登记调度进程不再运行时，才在同目录通过 `bash launch.sh` 恢复；活动worker会通过PID/start_ticks/boot_id、GPU锁和任务锁恢复识别。

断线后如果tmux仍在，直接连接或运行status命令即可。若会话和调度器均已退出，可用以下命令重新建立服务端会话；既有checkpoint、任务完成清单和worker身份会被恢复核验：

```bash
tmux -S /remote-home/Guilin/DRO/Codex/rl_expansion_20260922/artifacts/operations/expansion.sock new-session -d -s rl_expansion_20260922 -c /remote-home/Guilin/DRO/Codex/rl_expansion_20260922 'bash launch.sh > artifacts/operations/pipeline.log 2>&1'
```

后续无需维持本次对话。管线会自动完成筛选、验证集晋级、长训练、模型冻结、新测试和报告。完成标志为 `artifacts/lifecycle.json` 的 `status=complete`，同时存在通过审计的 `artifacts/report/complete.json`；不能仅凭tmux退出判断成功。科学报告将生成在 `artifacts/report/REPORT.md`。
