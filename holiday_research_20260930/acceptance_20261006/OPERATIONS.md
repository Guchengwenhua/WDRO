国庆实验运行状态回执

截至本目录 inventory.json 的 UTC+8 时间快照。原始部署 HANDOFF、PLAN、ACCEPTANCE_INVENTORY 和实验产物保留；本回执另存，不覆盖历史交接。

正式 bank 队列在 2026-10-01 05:43:51 UTC+8 结束，9/9 退出码0，failures为空；2000步日志连续，每个作业均保存 latest/best 的4族完整测试结果。初步 bank 10/10、滚动补充2/2、初步几何、有限扰动、几何扩展均有完成产物。读取进程列表时没有上述实验的活跃 worker 或 holiday controller。

总控制器 state.json 为 controller_error，active=null，jobs={}。2026-10-01 05:48:11 UTC+8 记录 TypeError：int() 收到 NoneType。调用链 controller.py existing_dependencies → owns_live → live；上游成功退出后 queue_status.json 正常设置 child_pid=null，代码 source.get('child_pid',0) 仍返回 None，随后 int(None) 异常。故上游完成并未触发后续启动；不是 GPU OOM、科学门控拒绝或训练预算耗尽。

两份 jobs.json 合计66项，45项bank（21训练阶段、21评估、2门控、1报告），21项geometry（13训练阶段含100步前缀、4评估、3门控、1报告）。全部完成标记缺失，两个 week_* artifacts 目录均为空，控制器事件中无 job_start。实际应归类为“未开始，控制器失败阻断”。训练阶段可续跑同一模型，不能当作独立模型数。

旧 acceptance_inventory.json 生成于 2026-09-30 23:48:09 UTC+8，早于正式队列完成，内含63项skipped_dependency和3项not_run，attempts均0；它对应早先 idle controller 停止时的快照。当前 state 中 dependency_queue 也是最后一次轮询留下的旧状态，不能用其中“lagged3_s103运行中”判断当前状态。当前上游状态以各自完成文件与queue_status为准。

未完成的科学内容包括 qualified-best3 的三种子2000更新，d10四方法10000更新及测试，d30 Geo/选中bank三种子训练与测试，全新几何dev/test筛选及门控，100步训练成本前缀、后续4方法×3种子2000更新，以及两条整周报告。当前没有真实negative_gate、paused训练或deadline-skipped实验结果。

历史日志另保留：控制器最初启动曾有换行字符串SyntaxError，后来可进入等待阶段；rolling的report.log保留一次汇总脚本SyntaxError，但ROLLING_COMPARISON.md/json已存在且可从逐实例结果复算，不能把旧汇总失败当作两次训练失败。初步bank旧closure记载过queue.py遮蔽标准库的导入失败，后续10个成功产物均保留。

本次只分析已生成数据、状态与日志，新增CPU统计和报告，没有重启控制器、修改实验代码或启动训练。原计划停止训练时间为2026-10-06 23:29 UTC+8，最终截止2026-10-07 23:29 UTC+8；截止时间尚未来临不等于进程仍在运行。若后续恢复，应先修复None PID处理，沿用已有成功产物和冻结门控，并重新评估剩余预算，不能将原整周计划视作仍有完整七天可用。

[逐项状态 CSV](/remote-home/Guilin/DRO/Codex/holiday_research_20260930/acceptance_20261006/unrun_jobs.csv)；[状态快照 JSON](/remote-home/Guilin/DRO/Codex/holiday_research_20260930/acceptance_20261006/inventory.json)；[控制器日志](/remote-home/Guilin/DRO/Codex/holiday_research_20260930/artifacts/controller.log)；[控制器源码](/remote-home/Guilin/DRO/Codex/holiday_research_20260930/controller.py)。
