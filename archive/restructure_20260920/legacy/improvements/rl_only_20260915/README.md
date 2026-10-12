# 只用强化学习自身反馈的 REST / WDRO

本研究按用户修订要求，移除最优求解器、MST、几何上下界、Hungarian 匹配和外部预训练参考。保留的线长计算只是环境对 Actor 自己输出动作的测量，不构造参考答案。

## 方案与状态

- [详细理论及可行性分析](THEORY.md)：纯 RL 能否学好、Critic 到底学习什么、训练成本与误差的条件、六分支验证设计。
- [独立审查](THEORY_REVIEW.md)。
- [状态](STATUS.md)与[运行恢复](OPERATIONS.md)。
- [冻结协议](protocol.json)。
- [实际记录与统计范围](IMPLEMENTATION_NOTES.md)。

## 实验

| 分支 | 学习/搜索方式 |
|---|---|
| ERM-Critic | Actor 自身 sampled 线长 + 学習型 value 基线 |
| ERM-rollout | Actor 自身 sampled 线长 + 自身 greedy rollout 基线 |
| WDRO-RL | 学习型坐标对手 + 学习型 REST；最严格的双 RL 主路线 |
| WDRO-Random、ES、Critic | 同样没有外部参考的三种提案对照；提案器本身不都属于 RL |

degree30；每分支三个从头初始化 seed；每模型 2,048 更新、batch64。四个 WDRO 分支每中心 8 次自身 Actor 候选评分。所有模型共享同 seed 初始 Actor 和配对测试坐标。自贪心基线额外调用量单列。

Actor 优化 `mean(stop_gradient(L−b)·log π)`；Critic 预测当前策略的平均自身线长，不冒充最优长度。最终最优误差未知，但相同输入上的线长下降可以直接测量。

## 产物

- `artifacts/operations/queue.json`：任务/心跳/动态耗时。
- `artifacts/runs/{method}_s{seed}/`：训练日志、latest/best、坐标/动作拓扑/线长和哈希凭证。
- `artifacts/report/REPORT.md`：全部 GPU 任务结束后自动生成。
- `artifacts/report/metrics.csv`、`paired_comparisons.csv`、`attacks.csv`、`budget_and_runtime.csv`：完整统计。
- [CPU 工程检查](artifacts/cpu_report/REPORT.md)、[恢复一致性](artifacts/resume_audit/audit.json)：不代替正式科学验收。

之前的 [oracle_free_20260915](../oracle_free_20260915/) 因含几何参考下界已暂停；其检查点和日志保留，不混入本实验。
