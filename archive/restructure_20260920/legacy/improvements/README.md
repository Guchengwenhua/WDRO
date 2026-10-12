# 对抗样本与 Critic 改进研究

| 入口 | 内容 | 阅读入口 |
|---|---|---|
| [adversarial_v3_1](adversarial_v3_1/) | 已验收 ES/Critic/RL；链接到 `revisions/adversarial_v3_1_20260912/Codex` | [合并汇总](../docs/improvements/ADVERSARIAL_V3_1.md) |
| [critic_search_20260915](critic_search_20260915/) | 本次按 tmp2 的七个候选方向独立开展研究 | 该目录的理论、配置、状态和结果文件 |
| [oracle_free_20260915](oracle_free_20260915/) | 含 MST/几何下界的方案，已按用户修订要求暂停 | [理论方案](oracle_free_20260915/THEORY.md)、[暂停证据](oracle_free_20260915/STATUS.md) |
| [rl_only_20260915](rl_only_20260915/) | 仅使用 RL 自身动作线长反馈，无最优值、MST、上下界或匹配求解器 | [详细分析方案](rl_only_20260915/THEORY.md)、[状态](rl_only_20260915/STATUS.md) |
| [rl_only_vs_mainline_20260919](rl_only_vs_mainline_20260919/) | RL-only 与 GroupDRO、Mixed ERM、Hard Mining、原 ERM/WDRO 的 33 模型统一评测；两套完整合成测试集及共同窗口攻击，已完成验收 | [比较结果](rl_only_vs_mainline_20260919/RESULTS.md)、[训练条件差异](rl_only_vs_mainline_20260919/TRAINING_DIFFERENCES.md) |

早期 v3 因 greedy 批形状数值问题而被取代，入口在 [archive/adversarial_v3_superseded](../archive/adversarial_v3_superseded/)。它保留失败和纠错证据，不是当前已验收 v3.1 结果。

每个新研究使用独立子目录；共同父 Actor、数据和参考可以按身份核验后引用。需要分别报告代理/策略训练成本、真实候选评分预算和外层更新预算。
