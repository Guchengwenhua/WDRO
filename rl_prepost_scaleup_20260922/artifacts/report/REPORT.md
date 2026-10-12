# 预训练与后训练扩展：最终报告

工程/数值审计：PASS；完整任务 104/104。

正数为线长改善；主指标是两类独立攻击的四分布等权均值。六项单侧检验统一Holm校正，.5%效应和6/8正向seed门槛。

| 比较 | 方法 | 改善 | 相对改善 | Holm p | seeds | 判定 |
|---|---|---:|---:|---:|---:|---|
| post_only | bestof_ppo | 0.315726 | 7.49% | 0.02344 | 8/8 | SUPPORTED |
| pre_only_at_long_post | bestof_ppo | 0.071812 | 1.84% | 0.02344 | 7/8 | SUPPORTED |
| ppo_vs_random_at_long_pre_long_post | bestof_ppo | -0.000873 | -0.02% | 1.00000 | 5/8 | NOT_ESTABLISHED |
| post_only | sequential_ppo | 0.454941 | 10.50% | 0.02344 | 8/8 | SUPPORTED |
| pre_only_at_long_post | sequential_ppo | 0.050815 | 1.31% | 0.02344 | 8/8 | SUPPORTED |
| ppo_vs_random_at_long_pre_long_post | sequential_ppo | -0.000866 | -0.02% | 1.00000 | 3/8 | NOT_ESTABLISHED |

完整checkpoint曲线和分布/P95在metrics.csv；次要区间未校正，见secondary_comparisons.csv。
运输成本统计仅针对本任务新增训练片段；每run的inherited_budget.json记录父checkpoint中的计数。lineage_budget.csv按唯一继承源run记录完整历史阶段成本，包含源完成manifest的评价调用；共享祖先不重复计入此表。
更强固定模型搜索不等于路由器收益；degree30、自身线长、有限攻击，不提供最优性或连续WDRO保证。
旧轮test已被查看；本轮所有模型在新命名空间相同测试集评价，预算/最后checkpoint预先固定。
