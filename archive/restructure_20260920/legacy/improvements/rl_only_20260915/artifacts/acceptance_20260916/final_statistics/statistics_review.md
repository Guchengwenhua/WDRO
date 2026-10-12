# 独立统计审阅（完整冻结实验）

快照：2026-09-16T05:33:24Z；完成 18/18。
已覆盖六种方法各三个冻结训练种子；比较解释仍受样本、预算和区间限制。

| 方法 | 完成种子 | 初始平均长度 | 最终平均长度 | 攻击后长度 | 训练平均成本 |
|---|---|---:|---:|---:|---:|
| erm_critic | [101, 102, 103] | 10.626327 | 3.943987 | 4.546064 | 不适用 |
| erm_rollout | [101, 102, 103] | 10.626327 | 3.596621 | 4.210805 | 不适用 |
| wdro_random | [101, 102, 103] | 10.626327 | 3.676745 | 4.363553 | 0.040151 |
| wdro_es | [101, 102, 103] | 10.626327 | 3.691841 | 4.403248 | 0.040175 |
| wdro_critic | [101, 102, 103] | 10.626327 | 3.684321 | 4.259235 | 0.040108 |
| wdro_rl | [101, 102, 103] | 10.626327 | 3.683677 | 4.257888 | 0.040358 |

独立数值检查错误：0。

## 解释边界

- Finite-run dual updates are not a hard transport-budget constraint; inspect each run's mean_cost_minus_rho.
- Random32 attack maximizes raw length at lambda=0 over a coordinate window; it does not enforce the rho transport budget.
- mix_gauss test examples share one generated centroid layout; bootstrap intervals are conditional on this layout.
- Three training seeds do not support broad stability claims; intervals have no multiple-comparison correction.
- REST terminal-segment accounting may double-count collinear overlap; no optimality gap is measured.

均值对四个分布组及现有种子等权。长度由保存拓扑独立逐边重算；样本量与配对区间见同名前缀 JSON/CSV。
