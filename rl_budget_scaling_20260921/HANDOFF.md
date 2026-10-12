# 后台实验交接

状态：训练、测试和独立验收均已完成。验收结论为 `PASS_FOR_FROZEN_THREE_SEED_EXPLORATORY_PROTOCOL`，详见[独立验收报告](artifacts/acceptance_20260922/ACCEPTANCE.md)及[完成凭证](artifacts/acceptance_20260922/complete.json)。原启动交接记录保留在 `artifacts/operations/handoff.json`。

详细理论与冻结判据：[THEORY_PROTOCOL.md](docs/THEORY_PROTOCOL.md)。运行命令与恢复说明：[README.md](README.md)。

正式队列实际耗时7.9502小时，2026-09-22 02:01 UTC完成，符合启动时6–9小时的估计。18个训练和18个测试任务成功退出，正式运行锁均已释放。独立验收重算1,720,320条自身线长，保存值与重算值最大差为0；启动前21项测试与小型全流程仅作为工程检查保留。

自动执行报告位于 `artifacts/report/REPORT.md`，原完成凭证为 `artifacts/report/complete.json`，均保持不变。独立验收产物位于 `artifacts/acceptance_20260922/`。本轮支持基础训练不足是原因之一；继续ERM优于本次配置的GroupDRO后训练。结论仍限于三个种子和degree30合成测试，不外推为最优性或已追平旧主线。

查询：`python3 /remote-home/Guilin/DRO/Codex/rl_budget_scaling_20260921/status.py`

tmux任务已随队列完成退出，无需重新启动。原会话名为 `rl_budget_20260921`。
