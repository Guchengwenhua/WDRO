# 完整验收脚本预检

状态：PASS（预检通过，不代表18个任务已经完成）。

- 现有完整三种子数据的52个正式配对比较：独立bootstrap与报告实现最大差值0.0；随机种子20260915，重复2000次。
- 用现有六组CPU smoke数据在内存重放报告：metrics.csv、paired_comparisons.csv、attacks.csv、budget_and_runtime.csv、REPORT.md五文件逐字节一致。没有写入或覆盖冻结报告。
- metrics/paired字段与最终验收代码一致。complete.json包含生成时间，正确排除逐字节比较。
- statistics_review.py已完成--outdir/--require-complete：缺任务退出2、数值错误退出1、完整成功退出0；越界目录被拒绝。原partial快照保留。
- 当前脚本/协议哈希、时间与详细检查范围见full_auditor_preflight.json。统计脚本编辑已结束，可进行hash绑定。

本次没有模型前向、GPU执行或新增训练。完整后台验收仍须等待18/18正式任务、6/6 smoke、原调度器和工作进程退出、锁释放。
