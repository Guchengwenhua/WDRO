# 无 GeoSteiner 的 REST / WDRO 学习

目标：把训练、困难样本搜索、Critic 监督、选模和测试全部改成只依赖模型自身可行树长及便宜几何界的流程。所有网络从随机权重开始，不读取旧最优标签和旧模型。

## 阅读入口

- [详细理论方案](THEORY.md)：目标、下界证明、REINFORCE、四种 WDRO 搜索、实验协议和保证边界。
- [独立理论审查](THEORY_REVIEW.md)：替代目标的含义与容易误解的地方。
- [当前状态](STATUS.md)：正式 GPU 进展与时间估算。
- [冻结协议](protocol.json)：degree30，6 方法×3 seeds，2,048 更新/模型，batch64，WDRO 每中心8次实际评分。
- [运行与恢复](OPERATIONS.md)：tmux、队列、心跳、检查点和失败处理。

## 核心目标

令 M 为终端 Manhattan MST 长度，H 为包围盒宽高之和，B=max(H,M/2)。
无需精确求解器即可证明 B≤L★≤M。因此 q_B=L_eval/B−1 是历史 REST evaluator 相对最优值的保守上界；q_M=L_eval/M−1 可为负，作为 ERM 消融。

每个输入上的动作长度排序保持一致，跨输入权重、困难样本排名与 WDRO 目标会改变。此次验证是独立概念验证，不直接复现旧最优标签训练的结果。

## 自动产物

- `artifacts/runs/{method}_s{seed}/`：训练日志、权重、恢复状态、逐 net 坐标/拓扑/指标和完成凭证。
- `artifacts/operations/queue.json`：自动更新的调度状态和实测 ETA。
- `artifacts/report/REPORT.md`：全部正式任务完成后自动生成。
- `artifacts/report/metrics.csv`、`paired_comparisons.csv`、`attacks.csv`、`bound_diagnostics.csv`、`budget_and_runtime.csv`：完整统计。
- [CPU 工程验收](artifacts/cpu_report/REPORT.md)及[恢复一致性验收](artifacts/resume_audit/audit.json)：只证明工程流程，不代替 GPU 科学结果。

验收会从保存的 topology 独立重算全部测试/攻击树长，核对界、分数、候选择优、输入配对、工作量、哈希和锁。bootstrap 只提供本次三个训练种子和有限合成数据上的不确定性分析，不证明任意分布上的鲁棒性。
