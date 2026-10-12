# 本轮交接状态

快照时间：2026-09-15T22:07:14.968480+00:00。实时状态以 artifacts/operations/queue_state.json 为准。

- 详细理论、三个论文组件适配和基础 WDRO 训练/统计程序已冻结。
- 288 个源码/数据文件在交接前重新核验；CPU 集成、源点匹配反例、共享平均预算 LP、恢复等价与调度器恢复测试通过。
- GPU REST 与 Timing-RST 预检已完成并通过产物校验；峰值 reserved 分别为 134 MiB、736 MiB。它们是 6-step/batch64 预检，不是正式结论。
- 服务器 tmux 会话 papers_wdro_20260915 存活，24 次正式实验尚未完成；GPU 重新高负载时自动等待。
- 后续按 GPU 预检门禁自动启动各 family 的三seed ERM/WDRO 训练，最后独立重算统计/LP并验收。
- 已测 WDRO 更新约 4–6 秒/步，2048步单次训练约数小时。整批24次还有两个 family 待 GPU 标定；可用卡数量和排队时间不可预先保证，eta.py 自动更新。

理论见 [THEORY.md](THEORY.md)，运行/恢复命令见 [OPERATIONS.md](OPERATIONS.md)。时序任务上的效果不能解释为原纯线长 WDRO 的提升；本轮只是 degree10 任务迁移研究，不是论文完整复现。
