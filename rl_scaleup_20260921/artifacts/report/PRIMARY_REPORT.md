# 扩大训练的预注册结论

工程与附加审计：PASS。

正数表示改善；八项单侧 seed 符号翻转检验共同做 Holm 校正。NOT_ESTABLISHED 包括无效应、效应不足或证据不足。

| 阶段 | 指标 | 方法 / 对照 | 改善 | 相对改善 | Holm p | 正向 seeds | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| fixed | lambda5__gain | bestof_ppo / reinforce | 0.017001 | 4.32% | 0.06250 | 7/8 | NOT_ESTABLISHED |
| coev | nominal_length | bestof_ppo / random | -0.018800 | -0.52% | 1.00000 | 5/8 | NOT_ESTABLISHED |
| coev | random32_length | bestof_ppo / random | -0.007694 | -0.18% | 1.00000 | 4/8 | NOT_ESTABLISHED |
| coev | es32_length | bestof_ppo / random | -0.011695 | -0.27% | 1.00000 | 4/8 | NOT_ESTABLISHED |
| fixed | lambda5__gain | sequential_ppo / reinforce | -0.005934 | -1.51% | 1.00000 | 2/8 | NOT_ESTABLISHED |
| coev | nominal_length | sequential_ppo / random | -0.105628 | -2.89% | 1.00000 | 3/8 | NOT_ESTABLISHED |
| coev | random32_length | sequential_ppo / random | -0.151758 | -3.51% | 1.00000 | 2/8 | NOT_ESTABLISHED |
| coev | es32_length | sequential_ppo / random | -0.150587 | -3.46% | 1.00000 | 2/8 | NOT_ESTABLISHED |

运输成本见 training_budget.csv；未满足 ρ 的情况必须保留。有限候选测试不提供连续 WDRO 保证。
完整次要对照、分布分项和描述性 95% CI 见 REPORT.md、metrics.csv、comparisons.csv。
纯线长、degree30、8 seeds、最后 checkpoint；不代表最优性 regret、跨 degree 或芯片设计泛化。
