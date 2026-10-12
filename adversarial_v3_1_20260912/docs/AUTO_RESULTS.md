# 自动结果汇总（待人工科学解读）

这些计数由已通过审计的 CSV 自动汇总；没有进行人工结论复核，也没有新增统计检验。

A 完成 30 个任务、1080 个评价格；B 完成 18 个任务、216 行评价统计。

| A 的等搜索预算配对（左减右） | 稳定正向条件数 | 稳定负向条件数 | 其余 |
|---|---:|---:|---:|
| es64 − random64 | 40 | 0 | 0 |
| es64 − cem64 | 18 | 0 | 22 |
| critic_calibrated4 − random_top4 | 12 | 7 | 21 |
| rl4 − random_top4 | 2 | 22 | 16 |
| critic_calibrated4 − critic_raw4 | 7 | 1 | 32 |
| critic_calibrated64 − random64 | 36 | 0 | 4 |
| rl64 − random64 | 22 | 0 | 18 |

B 有 0/18 个任务保留起始模型（best_update=0）；该结果保留在完整表中。

A 的正向只表示同父模型、degree、分布和 lambda 条件内的惩罚攻击分数更高；条件数量不能当作独立实验重复或全局胜率。训练费用、padding 和恢复额外调用另计。
B 仅验证 degree30 的最小一步策略交替，并非完整多步 MARL。CI 条件于本次三个 seed 和固定中心，未校正多重比较。有限搜索与合成数据不构成连续 Wasserstein 证书或 ISPD 外推证明。

工程交付通过不等同于方法有效；三路线的最终中文科学结论仍待人工复核。

- [完整自动提取表](../experiments/interpretation/all/extraction.json)
- [独立交付审计](../experiments/diagnostics/delivery_audit.json)
- [研究数据报告](RESULTS.md)
