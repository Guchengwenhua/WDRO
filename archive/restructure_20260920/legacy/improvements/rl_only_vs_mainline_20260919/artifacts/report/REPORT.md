# 纯 RL 与 GroupDRO 等主线模型：统一评测

**33 个冻结 Actor 的统一评测与独立数值核验通过。** 11 类方法各 3 个 seed，degree 30；本次不重训、不调用 GeoSteiner 或读取最优标签。

## 比较口径

- 主线五方法：ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST；新方案六方法：ERM-Critic、ERM-rollout、Random/ES/Critic/RL-WDRO。
- 所有方法在每套数据上收到逐字节相同的坐标，使用相同 REST Actor 架构、T=1 greedy、64 行固定前向及重复末行 padding。没有额外归一化或重新选模。
- 新方案测试集为全部 4,096 个固定窗口 nets；主线测试集为全部 40,000 个 bbox 归一化 nets。两套数据生成器/尺度处理不同，分别报告。
- Random32 是相同 512 个中心 × 32 个候选上的自身线长最大值；不能解释为 ρ 约束内最坏风险。
- 主线模型使用 DAC21 预训练起点、2,000×256 微调及 relative-regret 参考反馈；新模型为随机起点、2,048×64 自身线长训练。**这是现有模型比较，不能把差异全部归因于算法或去掉 GeoSteiner。**

## 汇总

均值对四分布及三 seed 等权，越低越好。P95 先在各 seed/分布内计算再平均；最差组取种子均值后四组最大值。

### 新方案完整测试集（4,096 nets）

| 方法 | mean 线长 | P95 平均 | 最差组 mean |
|---|---:|---:|---:|
| mainline:erm | 3.084834 | 3.433490 | 3.507427 |
| mainline:mixed_erm | 3.084183 | 3.433810 | 3.505154 |
| mainline:hard_mining | 3.083568 | 3.432579 | 3.505994 |
| mainline:group_dro | 3.084174 | 3.434870 | 3.505174 |
| mainline:wdro_rest | 3.083276 | 3.431698 | 3.506426 |
| rl_only:erm_critic | 3.943987 | 4.545034 | 4.364988 |
| rl_only:erm_rollout | 3.596621 | 4.077535 | 4.107012 |
| rl_only:wdro_random | 3.676745 | 4.179677 | 4.187895 |
| rl_only:wdro_es | 3.691841 | 4.203719 | 4.234431 |
| rl_only:wdro_critic | 3.684321 | 4.198411 | 4.165710 |
| rl_only:wdro_rl | 3.683677 | 4.178578 | 4.190785 |

### 主线完整测试集（40,000 nets）

| 方法 | mean 线长 | P95 平均 | 最差组 mean |
|---|---:|---:|---:|
| mainline:erm | 3.239463 | 3.581746 | 3.515353 |
| mainline:mixed_erm | 3.239578 | 3.582625 | 3.513270 |
| mainline:hard_mining | 3.238817 | 3.581046 | 3.514405 |
| mainline:group_dro | 3.239468 | 3.582328 | 3.513194 |
| mainline:wdro_rest | 3.238521 | 3.580404 | 3.514185 |
| rl_only:erm_critic | 4.077677 | 4.688992 | 4.381851 |
| rl_only:erm_rollout | 3.770805 | 4.244983 | 4.108175 |
| rl_only:wdro_random | 3.867872 | 4.384665 | 4.193896 |
| rl_only:wdro_es | 3.893699 | 4.409810 | 4.234875 |
| rl_only:wdro_critic | 3.840866 | 4.354771 | 4.176834 |
| rl_only:wdro_rl | 3.861522 | 4.346547 | 4.190901 |

### 新方案窗口攻击（512 centers × 32 candidates）

| 方法 | mean 线长 | P95 平均 | 最差组 mean |
|---|---:|---:|---:|
| mainline:erm | 3.492553 | 3.805394 | 3.757135 |
| mainline:mixed_erm | 3.491817 | 3.806737 | 3.752984 |
| mainline:hard_mining | 3.490733 | 3.804871 | 3.753998 |
| mainline:group_dro | 3.491598 | 3.809130 | 3.752885 |
| mainline:wdro_rest | 3.489738 | 3.797108 | 3.754105 |
| rl_only:erm_critic | 4.546064 | 5.037870 | 4.930930 |
| rl_only:erm_rollout | 4.210805 | 4.634175 | 4.538868 |
| rl_only:wdro_random | 4.363553 | 4.809382 | 4.661926 |
| rl_only:wdro_es | 4.403248 | 4.873298 | 4.741232 |
| rl_only:wdro_critic | 4.259235 | 4.709789 | 4.647675 |
| rl_only:wdro_rl | 4.257888 | 4.671127 | 4.625106 |

## 新方案相对 GroupDRO 的配对坐标比较

下表为方法减 GroupDRO 的四组平均差及探索性 95% 区间。跨两训练体系独立重采样各自的三个训练 seed，同时使用共同 net 重采样；没有把同名 seed 当成相同初始化。2,000 次 bootstrap，未做多重比较校正。

| 测试套件 | 新方法 | mean 差 | 相对均值差 | 95% 区间 |
|---|---|---:|---:|---|
| rl_test | rl_only:erm_critic | +0.859813 | +27.88% | [+0.482675, +1.070559] |
| rl_test | rl_only:erm_rollout | +0.512446 | +16.62% | [+0.431174, +0.647840] |
| rl_test | rl_only:wdro_random | +0.592571 | +19.21% | [+0.482535, +0.794096] |
| rl_test | rl_only:wdro_es | +0.607667 | +19.70% | [+0.414281, +0.830362] |
| rl_test | rl_only:wdro_critic | +0.600146 | +19.46% | [+0.431950, +0.924650] |
| rl_test | rl_only:wdro_rl | +0.599503 | +19.44% | [+0.441433, +0.856091] |
| mainline_test | rl_only:erm_critic | +0.838209 | +25.87% | [+0.499016, +1.028010] |
| mainline_test | rl_only:erm_rollout | +0.531337 | +16.40% | [+0.457142, +0.658582] |
| mainline_test | rl_only:wdro_random | +0.628404 | +19.40% | [+0.501947, +0.825442] |
| mainline_test | rl_only:wdro_es | +0.654231 | +20.20% | [+0.450940, +0.894592] |
| mainline_test | rl_only:wdro_critic | +0.601398 | +18.56% | [+0.453674, +0.886720] |
| mainline_test | rl_only:wdro_rl | +0.622053 | +19.20% | [+0.476751, +0.864496] |
| rl_attack32 | rl_only:erm_critic | +1.054466 | +30.20% | [+0.635258, +1.297309] |
| rl_attack32 | rl_only:erm_rollout | +0.719207 | +20.60% | [+0.594229, +0.950457] |
| rl_attack32 | rl_only:wdro_random | +0.871955 | +24.97% | [+0.674956, +1.243926] |
| rl_attack32 | rl_only:wdro_es | +0.911650 | +26.11% | [+0.599993, +1.316937] |
| rl_attack32 | rl_only:wdro_critic | +0.767637 | +21.99% | [+0.559888, +1.129735] |
| rl_attack32 | rl_only:wdro_rl | +0.766290 | +21.95% | [+0.640994, +0.991656] |

## 验收与边界

- 33 个验证集选出的 checkpoint 来源哈希、输入数据、源码、输出拓扑、连通性和调用账本均核验；保存的 1,995,840 条长度由另一种 NumPy 实现独立重算。
- 原 RL-only 两套结果复用前，对每个模型、每个分布/套件重放首 64 行并核对拓扑与线长；新主线测试集则对全部 33 个模型重新推理。
- 新增真实推理行 1,636,416（包含兼容性重放及可能的恢复重放）；复用保存结果 368,640 行。物理前向含 padding 另见预算表。
- 线长为历史 REST span 记账值，共线重叠可能重复计入；没有测量精确 regret、最优率或近似比。
- 本轮未更新任何 Actor 参数；未使用测试结果选 checkpoint、超参或方法。旧模型的训练历史仍包含参考算法，本次无求解器推理不改变其训练来源。
- 同架构/同推理/同样本消除了评测口径差异；预训练、训练样本量、学习率、目标及选模等差异仍存在。需另有同起点同预算训练消融才能回答方法本身的因果优劣。
- 三个训练 seed 数量有限；合成结果不自动推广至其他 degree 或公开芯片；窗口攻击不构成连续鲁棒性证明。

[逐 seed 指标](per_seed_metrics.csv) · [汇总](summary.csv) · [相对 GroupDRO/Mixed ERM/WDRO 的全部区间](comparisons.csv) · [实际推理预算](inference_budget.csv)
