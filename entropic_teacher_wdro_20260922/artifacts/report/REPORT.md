# 熵预算与无GeoSteiner研究结果

工程验收：PASS。训练116项，评价224项。

目标为自身动作线长；教师相对损失不是真正最优regret。有限候选期望预算满足不代表连续Wasserstein认证。

| 方法 | 对照 | 尾部相对改善 | 改善种子 | Holm p | 本协议科学门槛 |
|---|---|---:|---:|---:|---|
| soft_budget | hard | -0.200% | 2/8 | 1.00000 | 未通过 |
| anneal | soft_budget | -0.102% | 2/8 | 1.00000 | 未通过 |
| excess | soft_budget | 0.298% | 7/8 | 0.20312 | 未通过 |
| tail | soft_budget | -0.950% | 0/8 | 1.00000 | 未通过 |

全部degree、seed、方法、best/last和自然/独立攻击指标见 metrics.csv；统计和负结果见 audit.json。
